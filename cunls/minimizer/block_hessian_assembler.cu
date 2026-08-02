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

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/minimizer/block_hessian_assembler.h"
#include "cunls/minimizer/hessian_structure.h"
#include "cunls/minimizer/problem.h"

namespace cunls {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxWarpsPerBlock = 8;
constexpr size_t kMaxSharedBytes = 48 * 1024;

// ============================================================================
// Assembly kernel
// ============================================================================

/**
 * One warp per factor.
 *
 * Stages `J_f` (m x n, row-major) and `r_f` in shared memory, then has the 32
 * lanes cooperatively evaluate the n^2 entries of `H_f = J_f^T J_f` and the n
 * entries of `b_f = -J_f^T r_f`, scattering each with a single atomicAdd.
 *
 * Per-factor global atomics drop from `m * n^2` (one per residual row per
 * column pair) to `n^2 + n`, and consecutive lanes hit consecutive slots within
 * a block-pair run.
 *
 * The two storage layouts differ only in how a `(row, column)` pair becomes an
 * offset into the value array, so they share everything else:
 *  - scalar CSR: `row_offsets[row] + write_offset + column_in_block`
 *  - block BSR:  the enclosing tile is
 *    `row_offsets[row / b] + write_offset + column_in_block / b`, and the entry
 *    sits at `(row % b) * b + column_in_block % b` inside it.
 * `row / b`, `row % b` and their column counterparts are precomputed per local
 * column, so the inner loop stays free of integer division.
 *
 * @tparam kBlockStorage Selects BSR addressing.
 */
template <bool kBlockStorage>
__global__ void AssembleBlockHessianKernel(
    int num_factors, int residual_dim, int tangent_dim, int num_blocks, int block_size,
    const float *__restrict__ jacobians, const float *__restrict__ residuals,
    const int *__restrict__ block_of_col, const int *__restrict__ offset_in_block,
    const int *__restrict__ tile_of_col, const int *__restrict__ sub_of_col,
    const int *__restrict__ triangular_pairs, const int *__restrict__ factor_cols,
    const int *__restrict__ write_offsets, const int *__restrict__ row_offsets,
    float *__restrict__ hessian_values, float *__restrict__ rhs) {
  extern __shared__ int s_arena[];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int num_pairs = tangent_dim * (tangent_dim + 1) / 2;

  // Block-wide: these tables depend only on the batch's block layout, not on
  // the factor, so they are loaded once per thread block.
  int *s_block_of_col = s_arena;
  int *s_offset_in_block = s_block_of_col + tangent_dim;
  int *s_tile_of_col = s_offset_in_block + tangent_dim;
  int *s_sub_of_col = s_tile_of_col + tangent_dim;
  int *s_triangular = s_sub_of_col + tangent_dim;
  for (int i = threadIdx.x; i < tangent_dim; i += blockDim.x) {
    s_block_of_col[i] = block_of_col[i];
    s_offset_in_block[i] = offset_in_block[i];
    if (kBlockStorage) {
      s_tile_of_col[i] = tile_of_col[i];
      s_sub_of_col[i] = sub_of_col[i];
    }
  }
  for (int i = threadIdx.x; i < num_pairs; i += blockDim.x) {
    s_triangular[i] = triangular_pairs[i];
  }
  __syncthreads();

  const int factor = blockIdx.x * warps_per_block + warp_in_block;
  if (factor >= num_factors) {
    return;
  }

  // Per-warp arena: J (residual_dim*tangent_dim floats), r (residual_dim floats), row starts
  // (tangent_dim ints), global columns (tangent_dim ints), block-pair write offsets
  // (num_blocks*num_blocks ints).
  const int per_warp_ints =
      residual_dim * tangent_dim + residual_dim + 2 * tangent_dim + num_blocks * num_blocks;
  int *s_warp = s_arena + 4 * tangent_dim + num_pairs + warp_in_block * per_warp_ints;
  float *s_jacobian = reinterpret_cast<float *>(s_warp);
  float *s_residual = s_jacobian + residual_dim * tangent_dim;
  int *s_row_start = reinterpret_cast<int *>(s_residual + residual_dim);
  int *s_global_col = s_row_start + tangent_dim;
  int *s_write_offset = s_global_col + tangent_dim;

  const float *j_src = jacobians + static_cast<size_t>(factor) * residual_dim * tangent_dim;
  for (int i = lane; i < residual_dim * tangent_dim; i += kWarpSize) {
    s_jacobian[i] = j_src[i];
  }
  const float *r_src = residuals + static_cast<size_t>(factor) * residual_dim;
  for (int i = lane; i < residual_dim; i += kWarpSize) {
    s_residual[i] = r_src[i];
  }
  const int *wo_src = write_offsets + static_cast<size_t>(factor) * num_blocks * num_blocks;
  for (int i = lane; i < num_blocks * num_blocks; i += kWarpSize) {
    s_write_offset[i] = wo_src[i];
  }
  const int *cols_src = factor_cols + static_cast<size_t>(factor) * num_blocks;
  for (int p = lane; p < tangent_dim; p += kWarpSize) {
    int col = cols_src[s_block_of_col[p]];
    if (col < 0) {
      // Constant state block: this row and column of H_f are dropped, which
      // is the block-level equivalent of the col_id == -1 triplet filter.
      s_global_col[p] = -1;
      s_row_start[p] = -1;
    } else {
      int global_col = col + s_offset_in_block[p];
      s_global_col[p] = global_col;
      // `col` is a multiple of block_size, so the tile row is
      // col / b + offset / b and the in-tile row is offset % b.
      s_row_start[p] = kBlockStorage ? row_offsets[col / block_size + s_tile_of_col[p]]
                                     : row_offsets[global_col];
    }
  }
  __syncwarp();

  for (int p = lane; p < tangent_dim; p += kWarpSize) {
    if (s_global_col[p] < 0) {
      continue;
    }
    float acc = 0.f;
    for (int k = 0; k < residual_dim; k++) {
      acc = fmaf(s_jacobian[k * tangent_dim + p], s_residual[k], acc);
    }
    atomicAdd(&rhs[s_global_col[p]], -acc);
  }

  for (int idx = lane; idx < num_pairs; idx += kWarpSize) {
    const int packed = s_triangular[idx];
    const int p = packed >> 16;
    const int q = packed & 0xFFFF;
    if (s_row_start[p] < 0 || s_row_start[q] < 0) {
      continue;
    }
    float acc = 0.f;
    for (int k = 0; k < residual_dim; k++) {
      acc = fmaf(s_jacobian[k * tangent_dim + p], s_jacobian[k * tangent_dim + q], acc);
    }
    const int block_p = s_block_of_col[p];
    const int block_q = s_block_of_col[q];
    if (kBlockStorage) {
      const int tile_area = block_size * block_size;
      atomicAdd(&hessian_values[static_cast<size_t>(s_row_start[p] +
                                                    s_write_offset[block_p * num_blocks + block_q] +
                                                    s_tile_of_col[q]) *
                                    tile_area +
                                s_sub_of_col[p] * block_size + s_sub_of_col[q]],
                acc);
      if (p != q) {
        atomicAdd(
            &hessian_values[static_cast<size_t>(s_row_start[q] +
                                                s_write_offset[block_q * num_blocks + block_p] +
                                                s_tile_of_col[p]) *
                                tile_area +
                            s_sub_of_col[q] * block_size + s_sub_of_col[p]],
            acc);
      }
    } else {
      atomicAdd(&hessian_values[s_row_start[p] + s_write_offset[block_p * num_blocks + block_q] +
                                s_offset_in_block[q]],
                acc);
      if (p != q) {
        atomicAdd(&hessian_values[s_row_start[q] + s_write_offset[block_q * num_blocks + block_p] +
                                  s_offset_in_block[p]],
                  acc);
      }
    }
  }
}

/**
 * Picks the largest warp count that fits the per-warp staging arena in shared
 * memory, capped at kMaxWarpsPerBlock.
 */
int PickWarpsPerBlock(int residual_dim, int tangent_dim, int num_blocks, size_t &shared_bytes) {
  const size_t block_ints = static_cast<size_t>(4) * tangent_dim +
                            static_cast<size_t>(tangent_dim) * (tangent_dim + 1) / 2;
  const size_t warp_ints = static_cast<size_t>(residual_dim) * tangent_dim + residual_dim +
                           2 * static_cast<size_t>(tangent_dim) + num_blocks * num_blocks;
  const size_t warp_bytes = warp_ints * sizeof(int);
  const size_t block_bytes = block_ints * sizeof(int);

  int warps = kMaxWarpsPerBlock;
  if (warp_bytes > 0) {
    size_t budget =
        kMaxSharedBytes > block_bytes ? (kMaxSharedBytes - block_bytes) / warp_bytes : 0;
    warps = static_cast<int>(std::min<size_t>(warps, budget));
  }
  if (warps < 1) {
    throw std::runtime_error("BlockHessianAssembler: factor too large for shared-memory assembly");
  }
  shared_bytes = block_bytes + static_cast<size_t>(warps) * warp_bytes;
  return warps;
}

}  // namespace

void BlockHessianAssembler::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                                       CSRSparseMatrix &hessian) {
  auto range = profiler_domain_.CreateDomainRange("Initialize");
  num_cols_ = num_cols;
  block_size_ = 1;
  structure_builder_.Build(stream, problem, num_cols, hessian, /*want_scatter_maps=*/true);
  BuildPlans(problem);
}

void BlockHessianAssembler::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                                       int block_size, BSRSparseMatrix &hessian) {
  auto range = profiler_domain_.CreateDomainRange("Initialize");
  num_cols_ = num_cols;
  block_size_ = block_size;
  structure_builder_.Build(stream, problem, num_cols, block_size, hessian,
                           /*want_scatter_maps=*/true);
  BuildPlans(problem);
}

void BlockHessianAssembler::BuildPlans(const Problem &problem) {
  // The structure builder resolved each factor's state pointers to global
  // columns and segmented the block pairs; the row-relative write offsets the
  // assembler needs fell out of that segmentation, so nothing is recomputed
  // here.
  const auto &layout = structure_builder_.Layout();
  plans_.clear();
  plans_.resize(layout.size());

  for (size_t i = 0; i < layout.size(); i++) {
    BatchPlan &plan = plans_[i];
    plan.layout = layout[i];

    auto block_sizes = problem.GetResidualBatches()[i].GetFactorBatch()->StateBlockSizes();

    // Local column -> (block, offset within block).
    std::vector<int> block_of_col(plan.layout.tangent_dim);
    std::vector<int> offset_in_block(plan.layout.tangent_dim);
    int cursor = 0;
    for (int b = 0; b < plan.layout.num_blocks; b++) {
      for (size_t k = 0; k < block_sizes[b]; k++) {
        block_of_col[cursor] = b;
        offset_in_block[cursor] = static_cast<int>(k);
        cursor++;
      }
    }
    plan.block_of_col.resize(block_of_col.size());
    plan.block_of_col.CopyFromHost(block_of_col.data(), block_of_col.size());
    plan.offset_in_block.resize(offset_in_block.size());
    plan.offset_in_block.CopyFromHost(offset_in_block.data(), offset_in_block.size());

    // Upper-triangle enumeration of H_f, packed as (p << 16) | q.
    if (plan.layout.tangent_dim > 0xFFFF) {
      throw std::runtime_error("BlockHessianAssembler: factor tangent dimension exceeds 65535");
    }
    std::vector<int> triangular;
    triangular.reserve(static_cast<size_t>(plan.layout.tangent_dim) *
                       (plan.layout.tangent_dim + 1) / 2);
    for (int p = 0; p < plan.layout.tangent_dim; p++) {
      for (int q = p; q < plan.layout.tangent_dim; q++) {
        triangular.push_back((p << 16) | q);
      }
    }
    plan.triangular_pairs.resize(triangular.size());
    plan.triangular_pairs.CopyFromHost(triangular.data(), triangular.size());

    // Tile / in-tile decomposition of each local column, so the assembly
    // kernel never divides by the block size.  Meaningless (and unread) when
    // the Hessian is stored as scalar CSR.
    std::vector<int> tile_of_col(plan.layout.tangent_dim, 0);
    std::vector<int> sub_of_col(plan.layout.tangent_dim, 0);
    if (block_size_ > 1) {
      for (int p = 0; p < plan.layout.tangent_dim; p++) {
        tile_of_col[p] = offset_in_block[p] / block_size_;
        sub_of_col[p] = offset_in_block[p] % block_size_;
      }
    }
    plan.tile_of_col.resize(tile_of_col.size());
    plan.tile_of_col.CopyFromHost(tile_of_col.data(), tile_of_col.size());
    plan.sub_of_col.resize(sub_of_col.size());
    plan.sub_of_col.CopyFromHost(sub_of_col.data(), sub_of_col.size());
  }
}

void BlockHessianAssembler::Assemble(cudaStream_t stream, const Problem &problem,
                                     const float *jacobian_values, const float *residuals,
                                     CSRSparseMatrix &hessian, dvector<float> &rhs) {
  auto range = profiler_domain_.CreateDomainRange("Assemble");
  PrepareOutputs(stream, hessian.values, rhs);
  LaunchAssembly(stream, jacobian_values, residuals, hessian.row_offsets.data(),
                 hessian.values.data(), rhs.data(), /*block_storage=*/false);
}

void BlockHessianAssembler::Assemble(cudaStream_t stream, const Problem &problem,
                                     const float *jacobian_values, const float *residuals,
                                     BSRSparseMatrix &hessian, dvector<float> &rhs) {
  auto range = profiler_domain_.CreateDomainRange("Assemble");
  PrepareOutputs(stream, hessian.values, rhs);
  LaunchAssembly(stream, jacobian_values, residuals, hessian.row_offsets.data(),
                 hessian.values.data(), rhs.data(), /*block_storage=*/true);
}

void BlockHessianAssembler::PrepareOutputs(cudaStream_t stream, dvector<float> &values,
                                           dvector<float> &rhs) {
  rhs.resize(static_cast<size_t>(num_cols_));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(rhs.data(), 0, rhs.size() * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(values.data(), 0, values.size() * sizeof(float), stream));
}

void BlockHessianAssembler::LaunchAssembly(cudaStream_t stream, const float *jacobian_values,
                                           const float *residuals, const int *row_offsets,
                                           float *hessian_values, float *rhs, bool block_storage) {
  const int *factor_cols = structure_builder_.FactorCols().data();
  const int *write_offsets = structure_builder_.WriteOffsets().data();

  for (const BatchPlan &plan : plans_) {
    const HessianBatchLayout &layout = plan.layout;
    if (layout.num_factors == 0 || layout.tangent_dim == 0) {
      continue;
    }

    size_t shared_bytes = 0;
    int warps =
        PickWarpsPerBlock(layout.residual_dim, layout.tangent_dim, layout.num_blocks, shared_bytes);
    int threads = warps * kWarpSize;
    int grid = (layout.num_factors + warps - 1) / warps;

    if (block_storage) {
      AssembleBlockHessianKernel<true><<<grid, threads, shared_bytes, stream>>>(
          layout.num_factors, layout.residual_dim, layout.tangent_dim, layout.num_blocks,
          block_size_, jacobian_values + layout.jacobian_offset, residuals + layout.residual_offset,
          plan.block_of_col.data(), plan.offset_in_block.data(), plan.tile_of_col.data(),
          plan.sub_of_col.data(), plan.triangular_pairs.data(), factor_cols + layout.col_offset,
          write_offsets + layout.pair_offset, row_offsets, hessian_values, rhs);
    } else {
      AssembleBlockHessianKernel<false><<<grid, threads, shared_bytes, stream>>>(
          layout.num_factors, layout.residual_dim, layout.tangent_dim, layout.num_blocks, 1,
          jacobian_values + layout.jacobian_offset, residuals + layout.residual_offset,
          plan.block_of_col.data(), plan.offset_in_block.data(), plan.tile_of_col.data(),
          plan.sub_of_col.data(), plan.triangular_pairs.data(), factor_cols + layout.col_offset,
          write_offsets + layout.pair_offset, row_offsets, hessian_values, rhs);
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

}  // namespace cunls
