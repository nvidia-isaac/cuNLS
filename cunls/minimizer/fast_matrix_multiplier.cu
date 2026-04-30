/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/minimizer/fast_matrix_multiplier.h"
#include "cunls/minimizer/sparse_matrix.h"

#define WARP_SIZE 32

namespace cunls {

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ int BinarySearchDevice(const int *__restrict__ arr,
                                                  int lo, int hi, int target) {
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

// ============================================================================
// Position precomputation kernel (runs once during Initialize)
// ============================================================================

/**
 * One warp per input row. Precomputes output positions for every (a, b)
 * column pair using binary search, storing them in a flat map so that the
 * compute kernel can avoid repeated searches.
 *
 * Layout: position_map[(start + a) * max_nnz_per_row + b] = output flat index.
 */
__global__ void
PrecomputePositionsKernel(const int *__restrict__ input_row_offsets,
                          const int *__restrict__ input_col_ids,
                          int num_input_rows,
                          const int *__restrict__ output_row_offsets,
                          const int *__restrict__ output_col_ids,
                          int *__restrict__ position_map, int max_nnz_per_row) {
  extern __shared__ int s_cols[];

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_in_block = threadIdx.x / WARP_SIZE;

  if (warp_id >= num_input_rows)
    return;

  int *my_cols = s_cols + warp_in_block * max_nnz_per_row;

  int start = input_row_offsets[warp_id];
  int end = input_row_offsets[warp_id + 1];
  int L = end - start;

  for (int i = lane; i < L; i += WARP_SIZE) {
    my_cols[i] = input_col_ids[start + i];
  }
  __syncwarp();

  for (int a = 0; a < L; a++) {
    int col_a = my_cols[a];
    int out_start = output_row_offsets[col_a];
    int out_end = output_row_offsets[col_a + 1];
    int map_base = (start + a) * max_nnz_per_row;

    for (int b = lane; b < L; b += WARP_SIZE) {
      int col_b = my_cols[b];
      int pos = BinarySearchDevice(output_col_ids, out_start, out_end, col_b);
      position_map[map_base + b] = pos;
    }
  }
}

// ============================================================================
// Compute kernel
// ============================================================================

/**
 * One warp per input row. Loads values into shared memory, then for every
 * (a, b) pair looks up the precomputed output position from position_map
 * and accumulates val_a * val_b with atomicAdd.
 *
 * Block-sparsity optimization: for block-sparse Jacobians (state blocks up
 * to 16x16), the per-row nnz `L` is small (typically <= max_nnz_per_row).
 * The 32 warp lanes are decomposed into `kAPerWarp = 32 / kThreadsPerA`
 * sub-groups, each processing a different `a` value in parallel rather
 * than serializing them in the outer loop.  This halves (or more) the
 * outer-loop iteration count vs. the naive one-`a`-at-a-time scheme.
 *
 * Shared memory layout (per block):
 *   [warp0_vals ... warpN_vals]
 *
 * @tparam kThreadsPerA Power-of-two number of lanes that cooperate on a
 *                      single `a` value.  Must be >= max_nnz_per_row
 *                      (rounded up to a power of two), capped at 32.
 */
template <int kThreadsPerA>
__global__ void ComputeJtJKernel(const int *__restrict__ input_row_offsets,
                                 const float *__restrict__ input_values,
                                 int num_input_rows,
                                 const int *__restrict__ position_map,
                                 float *__restrict__ output_values,
                                 int max_nnz_per_row) {
  static_assert(kThreadsPerA > 0 && (kThreadsPerA & (kThreadsPerA - 1)) == 0,
                "kThreadsPerA must be a power of two");
  static_assert(kThreadsPerA <= WARP_SIZE, "kThreadsPerA must be <= 32");
  constexpr int kAPerWarp = WARP_SIZE / kThreadsPerA;

  extern __shared__ float s_vals[];

  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp_in_block = threadIdx.x / WARP_SIZE;

  if (warp_id >= num_input_rows) {
    return;
  }

  float *my_vals = s_vals + warp_in_block * max_nnz_per_row;

  int start = input_row_offsets[warp_id];
  int end = input_row_offsets[warp_id + 1];
  int L = end - start;

  for (int i = lane; i < L; i += WARP_SIZE) {
    my_vals[i] = input_values[start + i];
  }
  __syncwarp();

  // Lane decomposition: a_sub picks which `a` value within the kAPerWarp
  // group this lane handles; b_lane is the lane's column within the
  // kThreadsPerA group cooperating on that `a`.
  const int a_sub = lane / kThreadsPerA;
  const int b_lane = lane & (kThreadsPerA - 1);

  for (int a_base = 0; a_base < L; a_base += kAPerWarp) {
    int a = a_base + a_sub;
    if (a < L) {
      float val_a = my_vals[a];
      int map_base = (start + a) * max_nnz_per_row;
      for (int b = b_lane; b < L; b += kThreadsPerA) {
        int pos = position_map[map_base + b];
        atomicAdd(&output_values[pos], val_a * my_vals[b]);
      }
    }
  }
}

// Picks the smallest power-of-two >= max_nnz_per_row, capped at WARP_SIZE.
// Lower values let multiple `a` columns be processed in parallel within
// one warp; for max_nnz_per_row <= 16 this gives a 2x or better reduction
// in serialized outer-loop iterations.
static int PickThreadsPerA(int max_nnz_per_row) {
  int t = 1;
  while (t < max_nnz_per_row && t < WARP_SIZE) {
    t *= 2;
  }
  return t;
}

// ============================================================================
// Host helpers
// ============================================================================

static int GetMaxNnzPerRow(const Problem &problem) {
  int max_nnz = 0;
  for (const auto &batch : problem.GetResidualBatches()) {
    auto sizes = batch.GetFactorBatch()->StateBlockSizes();
    int nnz = 0;
    for (auto s : sizes)
      nnz += static_cast<int>(s);
    max_nnz = std::max(max_nnz, nnz);
  }
  return max_nnz;
}

static void ComputeBlockConfig(int max_nnz_per_row, int smem_per_element,
                               int &warps_per_block, int &block_size,
                               size_t &smem_bytes) {
  constexpr int kMaxSharedMem = 48 * 1024;
  int smem_per_warp = max_nnz_per_row * smem_per_element;
  warps_per_block =
      std::min(8, smem_per_warp > 0 ? kMaxSharedMem / smem_per_warp : 8);
  warps_per_block = std::max(warps_per_block, 1);
  block_size = warps_per_block * WARP_SIZE;
  smem_bytes = static_cast<size_t>(warps_per_block) * smem_per_warp;
}

// ============================================================================
// Structure-aware sparsity pattern computation
// ============================================================================

struct ExpandPair {
  int row_offset;
  int col_offset;
  int row_tangent;
  int col_tangent;
  int write_offset;
};

/**
 * One thread per block pair.  Atomically increments per-row non-zero counts
 * for all rows spanned by the block pair.
 */
__global__ void
ComputeBlockRowCountsKernel(const ExpandPair *__restrict__ pairs, int num_pairs,
                            int *__restrict__ row_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs)
    return;
  auto p = pairs[idx];
  for (int i = 0; i < p.row_tangent; i++) {
    atomicAdd(&row_counts[p.row_offset + i], p.col_tangent);
  }
}

/**
 * One thread block per block pair.  Writes dense column indices for the
 * sub-block into the CSR col_ids array using precomputed row offsets and
 * per-pair write offsets.
 */
__global__ void ExpandBlockPairsKernel(const ExpandPair *__restrict__ pairs,
                                       int num_pairs,
                                       const int *__restrict__ row_offsets,
                                       int *__restrict__ col_ids) {
  int pair_idx = blockIdx.x;
  if (pair_idx >= num_pairs)
    return;
  auto p = pairs[pair_idx];
  int total = p.row_tangent * p.col_tangent;
  for (int k = threadIdx.x; k < total; k += blockDim.x) {
    int i = k / p.col_tangent;
    int j = k % p.col_tangent;
    col_ids[row_offsets[p.row_offset + i] + p.write_offset + j] =
        p.col_offset + j;
  }
}

/**
 * Derives the J^T J sparsity pattern from the factor graph connectivity.
 * Each factor connecting state blocks A,B produces dense sub-blocks
 * (A,A), (A,B), (B,A), (B,B) in the Hessian.  Block pairs are collected
 * on the host, deduplicated via a hash set, and expanded into a CSR on
 * the GPU.
 *
 * Block-sparsity optimization: rather than push every (row, col, row_tang,
 * col_tang) tuple per factor (millions of entries) and then sort+unique,
 * we deduplicate on-the-fly with an unordered_set keyed on
 * (row_offset << 32 | col_offset).  Tangent sizes are constant per
 * column-offset and looked up after dedup.  All const-state-id D2H copies
 * are issued first and a single stream sync gates host processing.
 *
 * Uses buffer_ as scratch space to avoid runtime GPU memory allocation.
 */
void FastSparseMatrixMultiplier::ComputeOutputStructure(cudaStream_t stream,
                                                        const Problem &problem,
                                                        CSRSparseMatrix &output,
                                                        int num_cols) {
  struct BlockInfo {
    int col_offset;
    int tangent_size;
  };

  // Per-batch descriptor for O(1) pointer → column lookup via arithmetic.
  struct BatchDesc {
    const float *base;
    int ambient_size;
    int tangent_size;
    int num_blocks;
    std::vector<int> col_offsets; // [block_idx] → column, -1 if constant
  };

  const auto &state_batches = problem.GetStateBatches();
  std::vector<BatchDesc> batch_descs;
  batch_descs.reserve(state_batches.size());

  // Phase 1: issue all const-state-id D2H copies into one pinned buffer
  // and synchronize the stream once instead of once per batch.
  std::vector<size_t> const_offsets(state_batches.size() + 1, 0);
  for (size_t i = 0; i < state_batches.size(); i++) {
    const_offsets[i + 1] =
        const_offsets[i] + state_batches[i]->NumConstStateBlocks();
  }
  size_t total_const = const_offsets.back();

  if (total_const > 0) {
    if (pinned_buf_.size() < total_const) {
      pinned_buf_.resize(total_const);
    }
    for (size_t i = 0; i < state_batches.size(); i++) {
      size_t nc = state_batches[i]->NumConstStateBlocks();
      if (nc > 0) {
        THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
            pinned_buf_.data() + const_offsets[i],
            state_batches[i]->ConstStateIds(), nc * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
      }
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  // Phase 2: build per-batch column-offset tables.
  int last_col = 0;
  for (size_t i = 0; i < state_batches.size(); i++) {
    auto *batch = state_batches[i];
    BatchDesc bd;
    bd.base = batch->StateBlockDevicePtr(0);
    bd.ambient_size = static_cast<int>(batch->AmbientSize());
    bd.tangent_size = static_cast<int>(batch->TangentSize());
    bd.num_blocks = static_cast<int>(batch->NumStateBlocks());
    bd.col_offsets.assign(bd.num_blocks, 0);

    size_t nc = state_batches[i]->NumConstStateBlocks();
    std::vector<bool> is_const(bd.num_blocks, false);
    for (size_t k = 0; k < nc; k++) {
      int idx = pinned_buf_[const_offsets[i] + k];
      if (idx >= 0 && idx < bd.num_blocks) {
        is_const[idx] = true;
      }
    }

    for (int j = 0; j < bd.num_blocks; j++) {
      if (is_const[j]) {
        bd.col_offsets[j] = -1;
      } else {
        bd.col_offsets[j] = last_col;
        last_col += bd.tangent_size;
      }
    }
    batch_descs.push_back(std::move(bd));
  }

  // Pre-build a col_offset -> tangent_size map.  Tangent size is constant
  // per state batch, so every column offset has a fixed tangent.
  std::unordered_map<int, int> col_to_tangent;
  col_to_tangent.reserve(static_cast<size_t>(num_cols));
  for (const auto &bd : batch_descs) {
    for (int col : bd.col_offsets) {
      if (col >= 0) {
        col_to_tangent.emplace(col, bd.tangent_size);
      }
    }
  }

  // Lambda: resolve device pointer → BlockInfo via pointer arithmetic.
  auto resolve_ptr = [&](const float *ptr) -> BlockInfo {
    for (const auto &bd : batch_descs) {
      auto diff = ptr - bd.base;
      if (diff >= 0 &&
          diff < static_cast<ptrdiff_t>(bd.num_blocks) * bd.ambient_size) {
        int idx = static_cast<int>(diff / bd.ambient_size);
        int col = bd.col_offsets[idx];
        if (col >= 0) {
          return {col, bd.tangent_size};
        }
        return {-1, 0};
      }
    }
    return {-1, 0};
  };

  const auto &res_batches = problem.GetResidualBatches();
  const auto &state_ptrs = problem.GetStatePointers();

  // Estimate the number of unique (row, col) block pairs.  We don't know
  // the real count up front; reserving a healthy fraction of the worst
  // case avoids most rehashes for typical workloads.
  size_t total_pair_estimate = 0;
  for (size_t rb = 0; rb < res_batches.size(); rb++) {
    auto *factor = res_batches[rb].GetFactorBatch();
    size_t nb = factor->StateBlockSizes().size();
    total_pair_estimate += factor->NumFactors() * nb * nb;
  }

  // Hash-set dedup: O(1) insert per (row, col) pair vs. the previous
  // sort+unique on millions of 16-byte structs.  Keys pack
  // (row_offset, col_offset) into a single 64-bit integer.
  std::unordered_set<uint64_t> seen_pairs;
  seen_pairs.reserve(std::min<size_t>(total_pair_estimate,
                                      static_cast<size_t>(1) << 22));

  // Per-factor scratch.  16x16 block sparse means at most ~16 blocks per
  // factor; stack array avoids per-factor heap allocations.
  constexpr int kMaxBlocksPerFactor = 64;
  int local_cols[kMaxBlocksPerFactor];
  std::vector<int> local_cols_dyn;

  for (size_t rb = 0; rb < res_batches.size(); rb++) {
    auto *factor = res_batches[rb].GetFactorBatch();
    size_t nf = factor->NumFactors();
    size_t nb = factor->StateBlockSizes().size();
    const auto &h_ptrs = state_ptrs[rb];

    int *cols_ptr;
    if (nb <= kMaxBlocksPerFactor) {
      cols_ptr = local_cols;
    } else {
      local_cols_dyn.resize(nb);
      cols_ptr = local_cols_dyn.data();
    }

    for (size_t f = 0; f < nf; f++) {
      int nblocks = 0;
      const float *const *fp = h_ptrs.data() + f * nb;
      for (size_t b = 0; b < nb; b++) {
        auto info = resolve_ptr(fp[b]);
        if (info.col_offset >= 0) {
          cols_ptr[nblocks++] = info.col_offset;
        }
      }
      for (int a = 0; a < nblocks; a++) {
        uint64_t row_part = static_cast<uint64_t>(
                                static_cast<uint32_t>(cols_ptr[a]))
                            << 32;
        for (int b = 0; b < nblocks; b++) {
          uint64_t key =
              row_part | static_cast<uint32_t>(cols_ptr[b]);
          seen_pairs.insert(key);
        }
      }
    }
  }

  // Materialize into a sorted BlockPair vector for offset computation.
  // The unique count is O(num_state_blocks^2) in the worst case but
  // typically much smaller, so this sort dominates only at <<1% scale
  // compared to the previous sort over all (factor x nb x nb) entries.
  struct BlockPair {
    int row_offset;
    int col_offset;
    int row_tangent;
    int col_tangent;
  };

  std::vector<BlockPair> pairs;
  pairs.reserve(seen_pairs.size());
  for (uint64_t key : seen_pairs) {
    int row = static_cast<int>(key >> 32);
    int col = static_cast<int>(key & 0xFFFFFFFFu);
    auto row_it = col_to_tangent.find(row);
    auto col_it = col_to_tangent.find(col);
    int row_t = row_it != col_to_tangent.end() ? row_it->second : 0;
    int col_t = col_it != col_to_tangent.end() ? col_it->second : 0;
    pairs.push_back({row, col, row_t, col_t});
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const BlockPair &a, const BlockPair &b) {
              if (a.row_offset != b.row_offset) {
                return a.row_offset < b.row_offset;
              }
              return a.col_offset < b.col_offset;
            });

  // Compute per-pair write offsets: within a block row, each successive
  // block column starts after the previous one's tangent width.
  std::vector<ExpandPair> gpu_pairs(pairs.size());
  int prev_row = -1;
  int cum = 0;
  for (size_t k = 0; k < pairs.size(); k++) {
    if (pairs[k].row_offset != prev_row) {
      cum = 0;
      prev_row = pairs[k].row_offset;
    }
    gpu_pairs[k] = {pairs[k].row_offset, pairs[k].col_offset,
                    pairs[k].row_tangent, pairs[k].col_tangent, cum};
    cum += pairs[k].col_tangent;
  }

  static_assert(sizeof(ExpandPair) == 5 * sizeof(int));
  int num_pairs = static_cast<int>(gpu_pairs.size());
  size_t pairs_ints = static_cast<size_t>(num_pairs) * 5;

  buffer_.resize(pairs_ints + num_cols);
  auto *d_pairs = reinterpret_cast<ExpandPair *>(buffer_.data());
  int *row_counts = buffer_.data() + pairs_ints;

  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(d_pairs, gpu_pairs.data(),
                                      num_pairs * sizeof(ExpandPair),
                                      cudaMemcpyHostToDevice, stream));
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(row_counts, 0, num_cols * sizeof(int), stream));

  {
    constexpr int kBlockSize = 256;
    int grid = (num_pairs + kBlockSize - 1) / kBlockSize;
    ComputeBlockRowCountsKernel<<<grid, kBlockSize, 0, stream>>>(
        d_pairs, num_pairs, row_counts);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  output.row_offsets.resize(num_cols + 1);
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(output.row_offsets.data(), 0, sizeof(int), stream));

  {
    auto stream_policy = thrust::cuda::par_nosync.on(stream);
    thrust::device_ptr<int> counts_ptr(row_counts);
    thrust::device_ptr<int> offsets_ptr(output.row_offsets.data());
    thrust::inclusive_scan(stream_policy, counts_ptr, counts_ptr + num_cols,
                           offsets_ptr + 1);
  }

  if (pinned_buf_.size() < 1) {
    pinned_buf_.resize(1);
  }
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(pinned_buf_.data(), output.row_offsets.data() + num_cols,
                      sizeof(int), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  int total_nnz = pinned_buf_[0];

  output.col_ids.resize(total_nnz);
  output.values.resize(total_nnz);

  {
    constexpr int kBlockSize = 256;
    ExpandBlockPairsKernel<<<num_pairs, kBlockSize, 0, stream>>>(
        d_pairs, num_pairs, output.row_offsets.data(), output.col_ids.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

// ============================================================================
// FastSparseMatrixMultiplier
// ============================================================================

void FastSparseMatrixMultiplier::Initialize(cudaStream_t stream,
                                            const Problem &problem,
                                            const CSRSparseMatrix &input,
                                            CSRSparseMatrix &output) {
  if (input.row_offsets.size() <= 1 || input.values.empty()) {
    max_nnz_per_row_ = 0;
    return;
  }

  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, input, num_rows, num_cols, num_nonzeros);

  max_nnz_per_row_ = GetMaxNnzPerRow(problem);
  if (max_nnz_per_row_ == 0) {
    return;
  }

  ComputeOutputStructure(stream, problem, output, num_cols);

  // Precompute output positions for every input (a, b) pair so
  // that ComputeSquaredMatrix avoids binary searches entirely.
  {
    size_t map_size = static_cast<size_t>(num_nonzeros) * max_nnz_per_row_;
    position_map_.resize(map_size);

    int warps_per_block, block_size;
    size_t smem;
    ComputeBlockConfig(max_nnz_per_row_, static_cast<int>(sizeof(int)),
                       warps_per_block, block_size, smem);
    int grid = (num_rows + warps_per_block - 1) / warps_per_block;

    PrecomputePositionsKernel<<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.col_ids.data(), num_rows,
        output.row_offsets.data(), output.col_ids.data(), position_map_.data(),
        max_nnz_per_row_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

void FastSparseMatrixMultiplier::ComputeSquaredMatrix(
    cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &input,
    CSRSparseMatrix &output) {
  int num_input_rows = static_cast<int>(input.row_offsets.size()) - 1;
  if (num_input_rows <= 0 || max_nnz_per_row_ == 0 || output.values.empty()) {
    return;
  }

  THROW_ON_CUDA_ERROR(cudaMemsetAsync(
      output.values.data(), 0, output.values.size() * sizeof(float), stream));

  int warps_per_block, block_size;
  size_t smem;
  ComputeBlockConfig(max_nnz_per_row_, static_cast<int>(sizeof(float)),
                     warps_per_block, block_size, smem);
  int grid = (num_input_rows + warps_per_block - 1) / warps_per_block;

  // Dispatch to a kernel specialization sized to the per-row nnz so that
  // multiple `a` columns are processed in parallel within one warp.
  int threads_per_a = PickThreadsPerA(max_nnz_per_row_);
  switch (threads_per_a) {
  case 1:
    ComputeJtJKernel<1><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  case 2:
    ComputeJtJKernel<2><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  case 4:
    ComputeJtJKernel<4><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  case 8:
    ComputeJtJKernel<8><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  case 16:
    ComputeJtJKernel<16><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  default:
    ComputeJtJKernel<32><<<grid, block_size, smem, stream>>>(
        input.row_offsets.data(), input.values.data(), num_input_rows,
        position_map_.data(), output.values.data(), max_nnz_per_row_);
    break;
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
