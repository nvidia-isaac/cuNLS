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

#include <numeric>
#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/minimizer/bsr_matrix.h"
#include "cunls/minimizer/device_reduction.h"
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

/** One thread per block row; adds `scale * diagonal` to the diagonal tile. */
__global__ void AddScaledBlockDiagonalKernel(int num_block_rows, int block_size,
                                             const int *__restrict__ row_offsets,
                                             const int *__restrict__ col_ids, float scale,
                                             const float *__restrict__ diagonal,
                                             float *__restrict__ values) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_block_rows) {
    return;
  }
  const int tile_area = block_size * block_size;
  for (int t = row_offsets[block_row]; t < row_offsets[block_row + 1]; t++) {
    if (col_ids[t] != block_row) {
      continue;
    }
    float *tile = values + static_cast<size_t>(t) * tile_area;
    for (int k = 0; k < block_size; k++) {
      tile[k * block_size + k] += scale * diagonal[block_row * block_size + k];
    }
    return;
  }
}

/**
 * One thread per stored scalar entry.  `row_of_tile` maps a tile index to its
 * block row so the entry's global (i, j) can be recovered.
 */
__global__ void ScaleSymmetricBSRKernel(size_t num_values, int block_size,
                                        const int *__restrict__ row_of_tile,
                                        const int *__restrict__ col_ids,
                                        const float *__restrict__ scale,
                                        float *__restrict__ values) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_values) {
    return;
  }
  const int tile_area = block_size * block_size;
  const size_t tile = idx / tile_area;
  const int within = static_cast<int>(idx - tile * tile_area);
  const int row_in_tile = within / block_size;
  const int col_in_tile = within - row_in_tile * block_size;
  const int row = row_of_tile[tile] * block_size + row_in_tile;
  const int col = col_ids[tile] * block_size + col_in_tile;
  values[idx] *= scale[row] * scale[col];
}

/** One thread per tile; records the tile's block row. */
__global__ void FillRowOfTileKernel(int num_block_rows, const int *__restrict__ row_offsets,
                                    int *__restrict__ row_of_tile) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_block_rows) {
    return;
  }
  for (int t = row_offsets[block_row]; t < row_offsets[block_row + 1]; t++) {
    row_of_tile[t] = block_row;
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

/**
 * @brief Runtime-tile-edge fallback for block sizes without a specialization.
 *
 * Same schedule as BsrMultiplyWarpKernel; the accumulator is sized to the
 * largest edge ChooseHessianBlockSize can return.
 */
__global__ void BsrMultiplyWarpGenericKernel(int num_block_rows, int block_size,
                                             const int *__restrict__ row_offsets,
                                             const int *__restrict__ col_ids,
                                             const float *__restrict__ values,
                                             const float *__restrict__ x, float *__restrict__ y) {
  constexpr int kMaxBlockSize = 16;
  const int block_row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (block_row >= num_block_rows) {
    return;
  }

  const int end = row_offsets[block_row + 1];
  float acc[kMaxBlockSize];
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

  if (max_tiles_per_row < kSkewThreshold) {
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
    default:
      BsrMultiplyWarpGenericKernel<<<grid, kThreads, 0, stream>>>(
          num_block_rows, block_size, row_offsets, col_ids, values, x, y);
      break;
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief y = A * x for block storage, into a caller-owned buffer.
 *
 * Internal: the only caller is ComputeWeightedSquaredStepAsync below.
 */
void MultiplyBSRByDenseVector(cudaStream_t stream, const BSRSparseMatrix &matrix,
                              const dvector<float> &x, dvector<float> &y) {
  y.resize(static_cast<size_t>(matrix.NumRows()));
  if (matrix.NumRows() == 0) {
    return;
  }
  LaunchBsrMultiply(stream, matrix.num_block_rows, matrix.block_size, matrix.max_tiles_per_row,
                    matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
                    x.data(), y.data());
}

}  // namespace

int ChooseHessianBlockSize(const Problem &problem, int max_block_size) {
  int block_size = 0;
  for (const auto *state_batch : problem.GetStateBatches()) {
    const size_t num_free = state_batch->NumStateBlocks() - state_batch->NumConstStateBlocks();
    if (num_free == 0) {
      continue;  // Contributes no columns, so its tangent size is irrelevant.
    }
    block_size = std::gcd(block_size, static_cast<int>(state_batch->TangentSize()));
    if (block_size == 1) {
      return 1;
    }
  }
  if (block_size <= 1) {
    return 1;
  }
  while (block_size > max_block_size) {
    // Fall back to the largest divisor within budget; halving keeps the
    // divisibility invariant for the even sizes cuNLS actually uses and
    // otherwise bails out to scalar storage.
    if (block_size % 2 != 0) {
      return 1;
    }
    block_size /= 2;
  }
  return block_size;
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

void AddScaledDiagonal(cudaStream_t stream, float scale, const dvector<float> &diagonal,
                       const BSRSparseMatrix &matrix, BSRSparseMatrix &result) {
  CopyBSRSparseMatrix(stream, matrix, result);
  if (result.num_block_rows == 0) {
    return;
  }
  AddScaledBlockDiagonalKernel<<<GridFor(result.num_block_rows), kBlockSize, 0, stream>>>(
      result.num_block_rows, result.block_size, result.row_offsets.data(), result.col_ids.data(),
      scale, diagonal.data(), result.values.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ScaleSymmetric(cudaStream_t stream, BSRSparseMatrix &matrix, const dvector<float> &scale,
                    dvector<int> &row_of_tile) {
  if (matrix.values.empty()) {
    return;
  }
  // Scaling needs each entry's global row, which BSR does not store per tile.
  // The map is caller-owned so this stays allocation- and sync-free on the
  // solver's hot path.
  row_of_tile.resize(matrix.NumBlocks());
  FillRowOfTileKernel<<<GridFor(matrix.num_block_rows), kBlockSize, 0, stream>>>(
      matrix.num_block_rows, matrix.row_offsets.data(), row_of_tile.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  ScaleSymmetricBSRKernel<<<GridFor(matrix.values.size()), kBlockSize, 0, stream>>>(
      matrix.values.size(), matrix.block_size, row_of_tile.data(), matrix.col_ids.data(),
      scale.data(), matrix.values.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void CopyBSRSparseMatrix(cudaStream_t stream, const BSRSparseMatrix &input,
                         BSRSparseMatrix &output) {
  if (&input == &output) {
    return;
  }
  // Always copy the structure, never just the values.  Matching sizes do not
  // imply matching connectivity: a solver reused across a stream of problems
  // hits pairs with the same block-row and tile counts but different sparsity,
  // and reusing the previous pattern silently pairs new values with old
  // columns.
  output.block_size = input.block_size;
  output.num_block_rows = input.num_block_rows;
  output.max_tiles_per_row = input.max_tiles_per_row;

  output.row_offsets.resize(input.row_offsets.size());
  output.col_ids.resize(input.col_ids.size());
  if (!input.row_offsets.empty()) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(output.row_offsets.data(), input.row_offsets.data(),
                                        input.row_offsets.size() * sizeof(int),
                                        cudaMemcpyDeviceToDevice, stream));
  }
  if (!input.col_ids.empty()) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(output.col_ids.data(), input.col_ids.data(),
                                        input.col_ids.size() * sizeof(int),
                                        cudaMemcpyDeviceToDevice, stream));
  }

  output.values.resize(input.values.size());
  if (!input.values.empty()) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(output.values.data(), input.values.data(),
                                        input.values.size() * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
  }
}

void ComputeWeightedSquaredStepAsync(cudaStream_t stream, const BSRSparseMatrix &matrix,
                                     const dvector<float> &step, dvector<float> &scratch,
                                     float *d_out, float *d_partials) {
  MultiplyBSRByDenseVector(stream, matrix, step, scratch);
  DotProductToDevice(stream, step.data(), scratch.data(), step.size(), d_out, d_partials);
}

}  // namespace cunls
