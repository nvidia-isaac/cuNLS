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

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/minimizer/hessian_structure.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {
namespace {

constexpr int kBlockSize = 256;

/// Sorts last, so candidate pairs touching a constant state land in a trailing
/// segment that the expansion simply ignores.
constexpr uint64_t kInvalidPairKey = ~uint64_t(0);

int GridFor(size_t count) { return static_cast<int>((count + kBlockSize - 1) / kBlockSize); }

/**
 * One thread per (factor, block).  Resolves the factor's state pointer to a
 * global column offset by testing it against one state batch's storage range,
 * exactly like `col_ids_kernel` does for the triplet path.  Threads whose
 * pointer belongs to a different batch leave the entry untouched, so the caller
 * runs this once per state batch over a `-1`-initialized output.
 */
__global__ void ResolveFactorColumnsKernel(int num_entries, int num_blocks_per_factor,
                                           float const *const *__restrict__ state_pointers,
                                           const int *__restrict__ block_sizes,
                                           const float *__restrict__ batch_base, int ambient_dim,
                                           int tangent_dim, int num_state_blocks,
                                           const int *__restrict__ block_col_map,
                                           int *__restrict__ out_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_entries) {
    return;
  }

  // A factor's block slot only maps onto this batch if the tangent dims agree;
  // mirrors the guard in FillColIdsInJacobianBlock.
  int block = idx % num_blocks_per_factor;
  if (block_sizes[block] != tangent_dim) {
    return;
  }

  const float *ptr = state_pointers[idx];
  ptrdiff_t diff = ptr - batch_base;
  if (diff < 0 || diff >= static_cast<ptrdiff_t>(num_state_blocks) * ambient_dim) {
    return;
  }

  int block_index = static_cast<int>(diff / ambient_dim);
  int col = block_col_map[block_index];
  if (col >= 0) {
    out_cols[idx] = col;
  }
}

/**
 * One thread per (state block, tangent component).  Records the owning block's
 * tangent size at every column it spans, so a block pair can look up its tile
 * dimensions from the two column indices alone.
 */
__global__ void FillTangentAtColKernel(int num_entries, int tangent_dim,
                                       const int *__restrict__ block_col_map,
                                       int *__restrict__ tangent_at_col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_entries) {
    return;
  }
  int block = idx / tangent_dim;
  int component = idx - block * tangent_dim;
  int col = block_col_map[block];
  if (col >= 0) {
    tangent_at_col[col + component] = tangent_dim;
  }
}

/** One thread per candidate (factor, block_a, block_b). */
__global__ void BuildPairKeysKernel(int num_entries, int num_blocks,
                                    const int *__restrict__ factor_cols,
                                    uint64_t *__restrict__ keys) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_entries) {
    return;
  }

  int pairs_per_factor = num_blocks * num_blocks;
  int factor = idx / pairs_per_factor;
  int pair = idx - factor * pairs_per_factor;
  int block_a = pair / num_blocks;
  int block_b = pair - block_a * num_blocks;

  int col_a = factor_cols[factor * num_blocks + block_a];
  int col_b = factor_cols[factor * num_blocks + block_b];
  keys[idx] = (col_a < 0 || col_b < 0)
                  ? kInvalidPairKey
                  : ((static_cast<uint64_t>(col_a) << 32) | static_cast<uint32_t>(col_b));
}

/** Marks the first candidate of each run of equal keys. */
__global__ void MarkGroupStartsKernel(int num_valid, const uint64_t *__restrict__ sorted_keys,
                                      int *__restrict__ flags) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_valid) {
    return;
  }
  flags[idx] = (idx == 0 || sorted_keys[idx] != sorted_keys[idx - 1]) ? 1 : 0;
}

/** Turns an inclusive scan of the group-start flags into 0-based group ids. */
__global__ void DecrementKernel(int count, int *__restrict__ values) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    values[idx] -= 1;
  }
}

/**
 * One thread per sorted candidate; the group's representative writes out the
 * pair's row/column and tile dimensions.
 */
__global__ void GatherUniquePairsKernel(int num_valid, const uint64_t *__restrict__ sorted_keys,
                                        const int *__restrict__ flags,
                                        const int *__restrict__ group_id,
                                        const int *__restrict__ tangent_at_col,
                                        int *__restrict__ pair_row, int *__restrict__ pair_col,
                                        int *__restrict__ pair_row_tangent,
                                        int *__restrict__ pair_col_tangent) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_valid || flags[idx] == 0) {
    return;
  }
  uint64_t key = sorted_keys[idx];
  int row = static_cast<int>(key >> 32);
  int col = static_cast<int>(key & 0xFFFFFFFFu);
  int g = group_id[idx];
  pair_row[g] = row;
  pair_col[g] = col;
  pair_row_tangent[g] = tangent_at_col[row];
  pair_col_tangent[g] = tangent_at_col[col];
}

/**
 * One thread per block pair.  Atomically accumulates per-row non-zero counts
 * for every row the pair spans.
 */
__global__ void ComputeBlockRowCountsKernel(int num_pairs, const int *__restrict__ pair_row,
                                            const int *__restrict__ row_tangent,
                                            const int *__restrict__ col_tangent,
                                            int *__restrict__ row_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs) {
    return;
  }
  int row = pair_row[idx];
  int tiles = row_tangent[idx];
  int width = col_tangent[idx];
  for (int i = 0; i < tiles; i++) {
    atomicAdd(&row_counts[row + i], width);
  }
}

/**
 * One thread block per block pair.  Writes the dense column indices of the tile
 * into the CSR col_ids array.  Every row of the block row shares the same
 * `write_offset`, which is what makes the assembler's scatter map so small.
 */
__global__ void ExpandBlockPairsKernel(int num_pairs, const int *__restrict__ pair_row,
                                       const int *__restrict__ pair_col,
                                       const int *__restrict__ row_tangent,
                                       const int *__restrict__ col_tangent,
                                       const int *__restrict__ write_offset,
                                       const int *__restrict__ row_offsets,
                                       int *__restrict__ col_ids) {
  int pair = blockIdx.x;
  if (pair >= num_pairs) {
    return;
  }
  int row = pair_row[pair];
  int col = pair_col[pair];
  int height = row_tangent[pair];
  int width = col_tangent[pair];
  int offset = write_offset[pair];
  int total = height * width;
  for (int k = threadIdx.x; k < total; k += blockDim.x) {
    int row_in_block = k / width;
    int col_in_block = k - row_in_block * width;
    col_ids[row_offsets[row + row_in_block] + offset + col_in_block] = col + col_in_block;
  }
}

/** out[i] = in[i] / divisor. */
__global__ void DivideKernel(int count, int divisor, const int *__restrict__ in,
                             int *__restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = in[idx] / divisor;
  }
}

/**
 * One thread per block pair.  Accumulates per-block-row tile counts for every
 * block row the pair spans.
 */
__global__ void ComputeBlockRowTileCountsKernel(int num_pairs, int block_size,
                                                const int *__restrict__ pair_row,
                                                const int *__restrict__ row_tangent,
                                                const int *__restrict__ tiles_per_pair,
                                                int *__restrict__ block_row_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs) {
    return;
  }
  int block_row = pair_row[idx] / block_size;
  int rows = row_tangent[idx] / block_size;
  int width = tiles_per_pair[idx];
  for (int i = 0; i < rows; i++) {
    atomicAdd(&block_row_counts[block_row + i], width);
  }
}

/**
 * One thread block per block pair.  Writes the tile column indices of the pair
 * into the BSR col_ids array.  As in the scalar case, every block row of the
 * pair shares the same tile-valued `write_offset`.
 */
__global__ void ExpandBlockPairsBSRKernel(
    int num_pairs, int block_size, const int *__restrict__ pair_row,
    const int *__restrict__ pair_col, const int *__restrict__ row_tangent,
    const int *__restrict__ col_tangent, const int *__restrict__ write_offset,
    const int *__restrict__ row_offsets, int *__restrict__ col_ids) {
  int pair = blockIdx.x;
  if (pair >= num_pairs) {
    return;
  }
  const int block_row = pair_row[pair] / block_size;
  const int block_col = pair_col[pair] / block_size;
  const int rows = row_tangent[pair] / block_size;
  const int cols = col_tangent[pair] / block_size;
  const int offset = write_offset[pair];
  const int total = rows * cols;
  for (int k = threadIdx.x; k < total; k += blockDim.x) {
    int tile_row = k / cols;
    int tile_col = k - tile_row * cols;
    col_ids[row_offsets[block_row + tile_row] + offset + tile_col] = block_col + tile_col;
  }
}

/**
 * Pushes each group's row-relative write offset back to the candidate slot it
 * came from, giving the assembler `write_offsets[f * nb * nb + a * nb + b]`
 * with no search.
 */
__global__ void ScatterWriteOffsetsKernel(int num_candidates, int num_valid,
                                          const int *__restrict__ pair_order,
                                          const int *__restrict__ group_id,
                                          const int *__restrict__ group_write_offset,
                                          int *__restrict__ write_offsets) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_candidates) {
    return;
  }
  // Candidates at or past num_valid touch a constant state; the sentinel key
  // sorted them into the trailing segment.
  int value = idx < num_valid ? group_write_offset[group_id[idx]] : -1;
  write_offsets[pair_order[idx]] = value;
}

}  // namespace

void HessianStructureBuilder::BuildLayout(const Problem &problem) {
  const auto &residual_batches = problem.GetResidualBatches();
  layout_.assign(residual_batches.size(), HessianBatchLayout());

  size_t jacobian_cursor = 0;
  size_t residual_cursor = 0;
  size_t col_cursor = 0;
  size_t pair_cursor = 0;

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto *factor_batch = residual_batches[i].GetFactorBatch();
    auto block_sizes = factor_batch->StateBlockSizes();

    HessianBatchLayout &layout = layout_[i];
    layout.num_factors = static_cast<int>(factor_batch->NumFactors());
    layout.residual_dim = static_cast<int>(factor_batch->ResidualsSize());
    layout.num_blocks = static_cast<int>(block_sizes.size());
    layout.tangent_dim =
        static_cast<int>(std::accumulate(block_sizes.begin(), block_sizes.end(), size_t(0)));
    layout.jacobian_offset = jacobian_cursor;
    layout.residual_offset = residual_cursor;
    layout.col_offset = col_cursor;
    layout.pair_offset = pair_cursor;

    const size_t nf = layout.num_factors;
    jacobian_cursor += nf * layout.residual_dim * layout.tangent_dim;
    residual_cursor += nf * layout.residual_dim;
    col_cursor += nf * layout.num_blocks;
    pair_cursor += nf * layout.num_blocks * layout.num_blocks;
  }

  jacobian_values_size_ = jacobian_cursor;
  total_pairs_ = pair_cursor;
  factor_cols_.resize(col_cursor);
}

void HessianStructureBuilder::ResolveFactorColumns(cudaStream_t stream, const Problem &problem,
                                                   int num_cols, dvector<int> &tangent_at_col) {
  const auto &residual_batches = problem.GetResidualBatches();
  const auto &host_state_pointers = problem.GetStatePointers();

  if (factor_cols_.empty()) {
    return;
  }
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(factor_cols_.data(), 0xFF, factor_cols_.size() * sizeof(int), stream));

  tangent_at_col.resize(static_cast<size_t>(num_cols));
  if (num_cols > 0) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(tangent_at_col.data(), 0, tangent_at_col.size() * sizeof(int), stream));
  }

  // Staging buffers live only for this call.  State pointers are uploaded once
  // and reused across the state-batch loop below.
  dvector<float *> state_pointers(factor_cols_.size());
  std::vector<dvector<int>> block_sizes_device(residual_batches.size());
  for (size_t i = 0; i < residual_batches.size(); i++) {
    auto block_sizes = residual_batches[i].GetFactorBatch()->StateBlockSizes();
    std::vector<int> sizes_host(block_sizes.begin(), block_sizes.end());
    block_sizes_device[i].resize(sizes_host.size());
    block_sizes_device[i].CopyFromHost(sizes_host.data(), sizes_host.size());

    const size_t count = static_cast<size_t>(layout_[i].num_factors) * layout_[i].num_blocks;
    if (count > 0) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(state_pointers.data() + layout_[i].col_offset,
                                          host_state_pointers[i].data(), count * sizeof(float *),
                                          cudaMemcpyHostToDevice, stream));
    }
  }

  // A (factor, block) slot is owned by exactly one state batch, so each batch
  // overwrites only its own entries of the -1-initialized output.
  dvector<int> block_col_map;
  int last_col = 0;
  for (auto *state_batch : problem.GetStateBatches()) {
    ComputeStateBlockColumnOffsets(stream, last_col, state_batch, block_col_map);

    const size_t tangent_entries = state_batch->NumStateBlocks() * state_batch->TangentSize();
    if (tangent_entries > 0) {
      FillTangentAtColKernel<<<GridFor(tangent_entries), kBlockSize, 0, stream>>>(
          static_cast<int>(tangent_entries), static_cast<int>(state_batch->TangentSize()),
          block_col_map.data(), tangent_at_col.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }

    for (size_t i = 0; i < residual_batches.size(); i++) {
      const size_t count = static_cast<size_t>(layout_[i].num_factors) * layout_[i].num_blocks;
      if (count == 0) {
        continue;
      }
      ResolveFactorColumnsKernel<<<GridFor(count), kBlockSize, 0, stream>>>(
          static_cast<int>(count), layout_[i].num_blocks,
          state_pointers.data() + layout_[i].col_offset, block_sizes_device[i].data(),
          state_batch->StateBlockDevicePtr(0), static_cast<int>(state_batch->AmbientSize()),
          static_cast<int>(state_batch->TangentSize()),
          static_cast<int>(state_batch->NumStateBlocks()), block_col_map.data(),
          factor_cols_.data() + layout_[i].col_offset);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }

    last_col +=
        static_cast<int>((state_batch->NumStateBlocks() - state_batch->NumConstStateBlocks()) *
                         state_batch->TangentSize());
  }

  // state_pointers goes out of scope here; the kernels above are ordered behind
  // its upload on `stream`, so wait before the allocation is released.
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

int HessianStructureBuilder::DiscoverBlockPairs(cudaStream_t stream, const Problem &problem,
                                                int num_cols, dvector<int> &pair_fields,
                                                dvector<int> &pair_order, dvector<int> &pair_group,
                                                size_t &num_valid) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  BuildLayout(problem);
  // Every buffer below is scratch for this call only.  Holding them as members
  // would keep ~20 bytes per candidate pair alive for the whole solve, which on
  // a large SBA problem is hundreds of MiB against no reuse benefit.
  dvector<int> tangent_at_col;
  ResolveFactorColumns(stream, problem, num_cols, tangent_at_col);

  num_valid = 0;
  if (total_pairs_ == 0 || num_cols == 0) {
    return 0;
  }

  // ---- Candidate keys -----------------------------------------------------
  dvector<uint64_t> pair_keys(total_pairs_);
  pair_order.resize(total_pairs_);
  for (const HessianBatchLayout &layout : layout_) {
    const size_t count =
        static_cast<size_t>(layout.num_factors) * layout.num_blocks * layout.num_blocks;
    if (count == 0) {
      continue;
    }
    BuildPairKeysKernel<<<GridFor(count), kBlockSize, 0, stream>>>(
        static_cast<int>(count), layout.num_blocks, factor_cols_.data() + layout.col_offset,
        pair_keys.data() + layout.pair_offset);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // ---- Sort and segment ---------------------------------------------------
  thrust::device_ptr<uint64_t> keys(pair_keys.data());
  thrust::device_ptr<int> order(pair_order.data());
  thrust::sequence(stream_policy, order, order + total_pairs_);
  thrust::sort_by_key(stream_policy, keys, keys + total_pairs_, order);

  // Constant-state candidates carry the sentinel key and sorted to the tail.
  size_t num_invalid =
      static_cast<size_t>(thrust::count(stream_policy, keys, keys + total_pairs_, kInvalidPairKey));
  num_valid = total_pairs_ - num_invalid;

  int num_pairs = 0;
  dvector<int> group_flags(num_valid);
  pair_group.resize(num_valid);
  if (num_valid > 0) {
    MarkGroupStartsKernel<<<GridFor(num_valid), kBlockSize, 0, stream>>>(
        static_cast<int>(num_valid), pair_keys.data(), group_flags.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());

    // Inclusive, not exclusive.  An exclusive scan yields the correct id only
    // for elements that *start* a group and hands every duplicate the id of the
    // next group -- and duplicates are the whole point here, since every block
    // pair shared by several factors must resolve to one group.  Scan
    // inclusively (so the last element holds the group count) and shift down.
    thrust::device_ptr<int> flags(group_flags.data());
    thrust::device_ptr<int> groups(pair_group.data());
    thrust::inclusive_scan(stream_policy, flags, flags + num_valid, groups);

    if (pinned_buf_.size() < 2) {
      pinned_buf_.resize(2);
    }
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(pinned_buf_.data(), pair_group.data() + num_valid - 1,
                                        sizeof(int), cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    num_pairs = pinned_buf_[0];

    DecrementKernel<<<GridFor(num_valid), kBlockSize, 0, stream>>>(static_cast<int>(num_valid),
                                                                   pair_group.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // ---- Per-pair descriptors ----------------------------------------------
  // Layout: [row | col | row_tangent | col_tangent | write_offset]
  const size_t stride = static_cast<size_t>(num_pairs);
  pair_fields.resize(5 * stride);
  if (num_pairs > 0) {
    int *pair_row = pair_fields.data();
    int *pair_col = pair_row + stride;
    int *row_tangent = pair_col + stride;
    int *col_tangent = row_tangent + stride;

    GatherUniquePairsKernel<<<GridFor(num_valid), kBlockSize, 0, stream>>>(
        static_cast<int>(num_valid), pair_keys.data(), group_flags.data(), pair_group.data(),
        tangent_at_col.data(), pair_row, pair_col, row_tangent, col_tangent);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  return num_pairs;
}

void HessianStructureBuilder::Build(cudaStream_t stream, const Problem &problem, int num_cols,
                                    CSRSparseMatrix &output, bool want_scatter_maps) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  dvector<int> pair_fields;
  dvector<int> pair_order;
  dvector<int> pair_group;
  size_t num_valid = 0;
  const int num_pairs =
      DiscoverBlockPairs(stream, problem, num_cols, pair_fields, pair_order, pair_group, num_valid);

  output.row_offsets.resize(static_cast<size_t>(num_cols) + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(output.row_offsets.data(), 0,
                                      output.row_offsets.size() * sizeof(int), stream));
  write_offsets_.resize(want_scatter_maps ? total_pairs_ : 0);

  if (total_pairs_ == 0 || num_cols == 0) {
    output.col_ids.resize(0);
    output.values.resize(0);
    return;
  }

  const size_t stride = static_cast<size_t>(num_pairs);
  int *pair_row = pair_fields.data();
  int *pair_col = pair_row + stride;
  int *row_tangent = pair_col + stride;
  int *col_tangent = row_tangent + stride;
  int *write_offset = col_tangent + stride;

  dvector<int> row_counts(static_cast<size_t>(num_cols));
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(row_counts.data(), 0, static_cast<size_t>(num_cols) * sizeof(int), stream));

  if (num_pairs > 0) {
    // Pairs are sorted by (row, col), so a segmented scan over the block row
    // gives each tile's offset within the row.
    thrust::exclusive_scan_by_key(stream_policy, thrust::device_pointer_cast(pair_row),
                                  thrust::device_pointer_cast(pair_row) + num_pairs,
                                  thrust::device_pointer_cast(col_tangent),
                                  thrust::device_pointer_cast(write_offset));

    ComputeBlockRowCountsKernel<<<GridFor(num_pairs), kBlockSize, 0, stream>>>(
        num_pairs, pair_row, row_tangent, col_tangent, row_counts.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  {
    thrust::device_ptr<int> counts(row_counts.data());
    thrust::device_ptr<int> offsets(output.row_offsets.data());
    thrust::inclusive_scan(stream_policy, counts, counts + num_cols, offsets + 1);
  }

  if (pinned_buf_.size() < 2) {
    pinned_buf_.resize(2);
  }
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(pinned_buf_.data(), output.row_offsets.data() + num_cols,
                                      sizeof(int), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  const int total_nnz = pinned_buf_[0];

  output.col_ids.resize(static_cast<size_t>(total_nnz));
  output.values.resize(static_cast<size_t>(total_nnz));

  if (num_pairs > 0) {
    ExpandBlockPairsKernel<<<num_pairs, kBlockSize, 0, stream>>>(
        num_pairs, pair_row, pair_col, row_tangent, col_tangent, write_offset,
        output.row_offsets.data(), output.col_ids.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  if (want_scatter_maps) {
    ScatterWriteOffsetsKernel<<<GridFor(total_pairs_), kBlockSize, 0, stream>>>(
        static_cast<int>(total_pairs_), static_cast<int>(num_valid), pair_order.data(),
        pair_group.data(), write_offset, write_offsets_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // The scratch buffers go out of scope; the kernels above read them.
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void HessianStructureBuilder::Build(cudaStream_t stream, const Problem &problem, int num_cols,
                                    int block_size, BSRSparseMatrix &output,
                                    bool want_scatter_maps) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  if (block_size < 1 || num_cols % block_size != 0) {
    throw std::runtime_error(
        "HessianStructureBuilder: block size must divide the tangent dimension");
  }

  dvector<int> pair_fields;
  dvector<int> pair_order;
  dvector<int> pair_group;
  size_t num_valid = 0;
  const int num_pairs =
      DiscoverBlockPairs(stream, problem, num_cols, pair_fields, pair_order, pair_group, num_valid);

  const int num_block_rows = num_cols / block_size;
  output.block_size = block_size;
  output.num_block_rows = num_block_rows;
  output.max_tiles_per_row = 0;
  output.row_offsets.resize(static_cast<size_t>(num_block_rows) + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(output.row_offsets.data(), 0,
                                      output.row_offsets.size() * sizeof(int), stream));
  write_offsets_.resize(want_scatter_maps ? total_pairs_ : 0);

  if (total_pairs_ == 0 || num_cols == 0) {
    output.col_ids.resize(0);
    output.values.resize(0);
    return;
  }

  const size_t stride = static_cast<size_t>(num_pairs);
  int *pair_row = pair_fields.data();
  int *pair_col = pair_row + stride;
  int *row_tangent = pair_col + stride;
  int *col_tangent = row_tangent + stride;
  int *write_offset = col_tangent + stride;

  // Tile counts, not scalar column counts: a (row_tangent x col_tangent) pair
  // occupies (row_tangent / b) x (col_tangent / b) tiles.
  dvector<int> tiles_per_pair(stride);
  dvector<int> block_row_counts(static_cast<size_t>(num_block_rows));
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(block_row_counts.data(), 0, block_row_counts.size() * sizeof(int), stream));

  if (num_pairs > 0) {
    DivideKernel<<<GridFor(num_pairs), kBlockSize, 0, stream>>>(num_pairs, block_size, col_tangent,
                                                                tiles_per_pair.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());

    thrust::exclusive_scan_by_key(stream_policy, thrust::device_pointer_cast(pair_row),
                                  thrust::device_pointer_cast(pair_row) + num_pairs,
                                  thrust::device_pointer_cast(tiles_per_pair.data()),
                                  thrust::device_pointer_cast(write_offset));

    ComputeBlockRowTileCountsKernel<<<GridFor(num_pairs), kBlockSize, 0, stream>>>(
        num_pairs, block_size, pair_row, row_tangent, tiles_per_pair.data(),
        block_row_counts.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  if (num_block_rows > 0) {
    thrust::device_ptr<int> counts(block_row_counts.data());
    // Peak row length before the scan destroys it; the SpMV picks its schedule
    // from this.
    output.max_tiles_per_row = *thrust::max_element(stream_policy, counts, counts + num_block_rows);
    thrust::device_ptr<int> offsets(output.row_offsets.data());
    thrust::inclusive_scan(stream_policy, counts, counts + num_block_rows, offsets + 1);
  }

  if (pinned_buf_.size() < 2) {
    pinned_buf_.resize(2);
  }
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(pinned_buf_.data(),
                                      output.row_offsets.data() + num_block_rows, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  const int total_tiles = pinned_buf_[0];

  output.col_ids.resize(static_cast<size_t>(total_tiles));
  output.values.resize(static_cast<size_t>(total_tiles) * block_size * block_size);

  if (num_pairs > 0) {
    ExpandBlockPairsBSRKernel<<<num_pairs, kBlockSize, 0, stream>>>(
        num_pairs, block_size, pair_row, pair_col, row_tangent, col_tangent, write_offset,
        output.row_offsets.data(), output.col_ids.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  if (want_scatter_maps) {
    ScatterWriteOffsetsKernel<<<GridFor(total_pairs_), kBlockSize, 0, stream>>>(
        static_cast<int>(total_pairs_), static_cast<int>(num_valid), pair_order.data(),
        pair_group.data(), write_offset, write_offsets_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

}  // namespace cunls
