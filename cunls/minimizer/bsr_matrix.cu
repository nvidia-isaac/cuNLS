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

#include "cunls/common/helper.h"
#include "cunls/minimizer/bsr_matrix.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch.h"

namespace cunls {
namespace {

constexpr int kBlockSize = 256;

int GridFor(size_t count) { return static_cast<int>((count + kBlockSize - 1) / kBlockSize); }

/** One thread per block row; locates the diagonal tile and reads its diagonal. */
__global__ void ExtractBlockDiagonalKernel(int num_block_rows, int block_size,
                                           const int *__restrict__ row_offsets,
                                           const int *__restrict__ col_ids,
                                           const float *__restrict__ values,
                                           float *__restrict__ diagonal) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_block_rows) {
    return;
  }
  const int tile_area = block_size * block_size;
  for (int t = row_offsets[block_row]; t < row_offsets[block_row + 1]; t++) {
    if (col_ids[t] != block_row) {
      continue;
    }
    const float *tile = values + static_cast<size_t>(t) * tile_area;
    for (int k = 0; k < block_size; k++) {
      diagonal[block_row * block_size + k] = tile[k * block_size + k];
    }
    return;
  }
  // Structurally absent diagonal tile: the diagonal is zero there.
  for (int k = 0; k < block_size; k++) {
    diagonal[block_row * block_size + k] = 0.f;
  }
}

/**
 * @brief y = A * x for uniform block storage, one warp per block row.
 *
 * cuSPARSE's `cusparseSbsrmv` measured 3.6x slower than `csrmv_v3` on a
 * bundle-adjustment Hessian with 3x3 tiles, which would negate the point of
 * block storage, so the SpMV is written here.
 *
 * A warp per block row rather than a thread per row: bundle-adjustment Hessians
 * are wildly non-uniform -- a pose block row holds one tile per observation of
 * that camera (thousands) while a landmark row holds a handful -- so a
 * thread-per-row schedule leaves the few pose threads serializing for as long
 * as the whole kernel takes.  Lanes stride over the row's tiles instead, and a
 * butterfly reduction combines their partial `b`-vectors.
 *
 * @tparam kB Tile edge.
 */
template <int kB>
__global__ void BsrMultiplyWarpKernel(int num_block_rows, const int *__restrict__ row_offsets,
                                      const int *__restrict__ col_ids,
                                      const float *__restrict__ values, const float *__restrict__ x,
                                      float *__restrict__ y) {
  const int block_row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (block_row >= num_block_rows) {
    return;
  }

  const int end = row_offsets[block_row + 1];
  float acc[kB];
#pragma unroll
  for (int k = 0; k < kB; ++k) {
    acc[k] = 0.f;
  }

  for (int t = row_offsets[block_row] + lane; t < end; t += 32) {
    const float *tile = values + static_cast<size_t>(t) * kB * kB;
    const float *xs = x + static_cast<size_t>(col_ids[t]) * kB;
    float xv[kB];
#pragma unroll
    for (int l = 0; l < kB; ++l) {
      xv[l] = xs[l];
    }
#pragma unroll
    for (int k = 0; k < kB; ++k) {
#pragma unroll
      for (int l = 0; l < kB; ++l) {
        acc[k] = fmaf(tile[k * kB + l], xv[l], acc[k]);
      }
    }
  }

  // Butterfly rather than shfl_down so every lane ends with the totals and the
  // first kB lanes can write the output run coalesced.
#pragma unroll
  for (int k = 0; k < kB; ++k) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[k] += __shfl_xor_sync(0xFFFFFFFFu, acc[k], offset);
    }
  }
  if (lane < kB) {
    y[block_row * kB + lane] = acc[lane];
  }
}

/// Largest tile edge the generic BSR SpMV can accumulate per lane.
constexpr int kMaxSpMVBlockSize = 16;

/**
 * @brief Runtime-tile-edge fallback for block sizes without a specialization.
 *
 * Same schedule as BsrMultiplyWarpKernel, but the per-lane accumulator is a
 * fixed-size array, so the caller must reject edges above
 * kMaxSpMVBlockSize before dispatching here.
 */
__global__ void BsrMultiplyWarpGenericKernel(int num_block_rows, int block_size,
                                             const int *__restrict__ row_offsets,
                                             const int *__restrict__ col_ids,
                                             const float *__restrict__ values,
                                             const float *__restrict__ x, float *__restrict__ y) {
  const int block_row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (block_row >= num_block_rows) {
    return;
  }

  const int end = row_offsets[block_row + 1];
  float acc[kMaxSpMVBlockSize];
  for (int k = 0; k < block_size; ++k) {
    acc[k] = 0.f;
  }

  for (int t = row_offsets[block_row] + lane; t < end; t += 32) {
    const float *tile = values + static_cast<size_t>(t) * block_size * block_size;
    const float *xs = x + static_cast<size_t>(col_ids[t]) * block_size;
    for (int row_in_tile = 0; row_in_tile < block_size; ++row_in_tile) {
      float sum = 0.f;
      for (int col = 0; col < block_size; ++col) {
        sum = fmaf(tile[row_in_tile * block_size + col], xs[col], sum);
      }
      acc[row_in_tile] += sum;
    }
  }

  for (int k = 0; k < block_size; ++k) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[k] += __shfl_xor_sync(0xFFFFFFFFu, acc[k], offset);
    }
  }
  if (lane < block_size) {
    y[block_row * block_size + lane] = acc[lane];
  }
}

/**
 * @brief y = A * x, one thread per scalar row.
 *
 * The right schedule for near-uniform, short block rows -- a pose graph has a
 * handful of tiles per row, so a whole warp per row would leave most lanes idle
 * and pay for a reduction that spans mostly zeros.  Thread `i` walks row
 * `i % b` of every tile in block row `i / b`; the `b` consecutive threads
 * sharing a block row read each tile as one contiguous run and broadcast their
 * identical `col_ids` and `x` loads.
 *
 * @tparam kB Tile edge, or 0 to take it as a runtime argument.
 */
template <int kB>
__global__ void BsrMultiplyRowKernel(int num_rows, int runtime_block_size,
                                     const int *__restrict__ row_offsets,
                                     const int *__restrict__ col_ids,
                                     const float *__restrict__ values, const float *__restrict__ x,
                                     float *__restrict__ y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int edge = kB > 0 ? kB : runtime_block_size;
  const int block_row = row / edge;
  const int sub_row = row - block_row * edge;

  float acc = 0.f;
  const int end = row_offsets[block_row + 1];
  for (int t = row_offsets[block_row]; t < end; ++t) {
    const float *tile = values + static_cast<size_t>(t) * edge * edge + sub_row * edge;
    const float *xs = x + static_cast<size_t>(col_ids[t]) * edge;
    for (int l = 0; l < edge; ++l) {
      acc = fmaf(tile[l], xs[l], acc);
    }
  }
  y[row] = acc;
}

/**
 * @brief Launches the BSR SpMV, choosing a schedule from the row lengths.
 *
 * A warp per block row tolerates skew but wastes lanes on short rows; a thread
 * per scalar row is the opposite.  Bundle adjustment needs the former (pose
 * rows hold thousands of tiles), pose graphs the latter (every row holds a
 * handful), so the peak row length decides.
 */
void LaunchBsrMultiply(cudaStream_t stream, int num_block_rows, int block_size,
                       int max_tiles_per_row, const int *row_offsets, const int *col_ids,
                       const float *values, const float *x, float *y) {
  constexpr int kThreads = 128;
  constexpr int kSkewThreshold = 32;

  // The warp kernels accumulate a tile edge in a fixed per-lane array (kMaxSpMVBlockSize):
  // a larger edge takes the row kernel whatever the skew.
  if (max_tiles_per_row < kSkewThreshold || block_size > kMaxSpMVBlockSize) {
    const int num_rows = num_block_rows * block_size;
    const int grid = (num_rows + kThreads - 1) / kThreads;
    switch (block_size) {
#define LAUNCH_ROW(BVAL)                                                                     \
  case BVAL:                                                                                 \
    BsrMultiplyRowKernel<BVAL>                                                               \
        <<<grid, kThreads, 0, stream>>>(num_rows, BVAL, row_offsets, col_ids, values, x, y); \
    break
      LAUNCH_ROW(2);
      LAUNCH_ROW(3);
      LAUNCH_ROW(4);
      LAUNCH_ROW(6);
      LAUNCH_ROW(7);
      LAUNCH_ROW(15);
      LAUNCH_ROW(16);
#undef LAUNCH_ROW
      default:
        BsrMultiplyRowKernel<0><<<grid, kThreads, 0, stream>>>(num_rows, block_size, row_offsets,
                                                               col_ids, values, x, y);
        break;
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }

  const int warps_per_block = kThreads / 32;
  const int grid = (num_block_rows + warps_per_block - 1) / warps_per_block;
  switch (block_size) {
#define LAUNCH_WARP(BVAL)                                                                    \
  case BVAL:                                                                                 \
    BsrMultiplyWarpKernel<BVAL>                                                              \
        <<<grid, kThreads, 0, stream>>>(num_block_rows, row_offsets, col_ids, values, x, y); \
    break
    LAUNCH_WARP(2);
    LAUNCH_WARP(3);
    LAUNCH_WARP(4);
    LAUNCH_WARP(5);
    LAUNCH_WARP(6);
    LAUNCH_WARP(7);
    LAUNCH_WARP(8);
#undef LAUNCH_WARP
    default:  // at most kMaxSpMVBlockSize (larger edges took the row kernel)
      BsrMultiplyWarpGenericKernel<<<grid, kThreads, 0, stream>>>(
          num_block_rows, block_size, row_offsets, col_ids, values, x, y);
      break;
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

void MultiplyBSRByDenseVector(cudaStream_t stream, const BSRSparseMatrix &matrix,
                              const dvector<float> &x, dvector<float> &y) {
  y.resize(static_cast<size_t>(matrix.NumRows()));
  MultiplyBSRValuesByDenseVector(stream, matrix, matrix.values.data(), x.data(), y.data());
}

void MultiplyBSRValuesByDenseVector(cudaStream_t stream, const BSRSparseMatrix &structure,
                                    const float *values, const float *x, float *y) {
  if (structure.NumRows() == 0) {
    return;
  }
  LaunchBsrMultiply(stream, structure.num_block_rows, structure.block_size,
                    structure.max_tiles_per_row, structure.row_offsets.data(),
                    structure.col_ids.data(), values, x, y);
}

int ChooseHessianBlockSize(const Problem &problem) {
  int block_size = 0;
  for (const auto *state_batch : problem.GetStateBatches()) {
    const size_t num_free = state_batch->NumActiveStates() - state_batch->NumConstStates();
    if (num_free == 0) {
      continue;  // Contributes no columns, so its tangent size is irrelevant.
    }
    block_size = std::gcd(block_size, static_cast<int>(state_batch->TangentSize()));
    if (block_size == 1) {
      return 1;
    }
  }
  // The largest divisor of the common tangent size within kMaxSpMVBlockSize (e.g. 12 for 36, 5
  // for 25); 1 is scalar storage.
  for (int edge = std::min(block_size, kMaxSpMVBlockSize); edge > 1; --edge) {
    if (block_size % edge == 0) return edge;
  }
  return 1;
}

void ExtractDiagonal(cudaStream_t stream, const BSRSparseMatrix &matrix, dvector<float> &diagonal) {
  diagonal.resize(static_cast<size_t>(matrix.NumRows()));
  if (matrix.num_block_rows == 0) {
    return;
  }
  ExtractBlockDiagonalKernel<<<GridFor(matrix.num_block_rows), kBlockSize, 0, stream>>>(
      matrix.num_block_rows, matrix.block_size, matrix.row_offsets.data(), matrix.col_ids.data(),
      matrix.values.data(), diagonal.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
