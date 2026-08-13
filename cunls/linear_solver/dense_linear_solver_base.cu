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

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/dense_linear_solver_base.h"

namespace cunls {
namespace {

constexpr int kWarpSize = 32;
constexpr int kScatterThreads = 256;

/// @brief Scatters CSR values into a dense row-major matrix.
///
/// One warp per row; the warp's lanes stride over the row's non-zeros.  The
/// caller zeroes @p dense_matrix first, so entries not stored stay zero.
__global__ void ScatterCSRToDenseKernel(const int *__restrict__ row_offsets,
                                        const int *__restrict__ col_ids,
                                        const float *__restrict__ values, int num_rows,
                                        float *__restrict__ dense_matrix) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row >= num_rows) {
    return;
  }
  const int lane = threadIdx.x % kWarpSize;
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  float *const dense_row = dense_matrix + static_cast<size_t>(row) * num_rows;
  for (int idx = row_start + lane; idx < row_end; idx += kWarpSize) {
    dense_row[col_ids[idx]] = values[idx];
  }
}

/// @brief Scatters BSR tiles into a dense row-major matrix.
///
/// One CUDA block per block row.  The row's tiles and their entries are walked
/// as one flat range so a short row cannot leave most of the block idle, which
/// matters for bundle-adjustment Hessians where tile counts per row differ by
/// orders of magnitude.  Within a tile the flat index runs fastest over the
/// tile's columns, so consecutive threads write consecutive dense columns.
__global__ void ScatterBSRToDenseKernel(const int *__restrict__ row_offsets,
                                        const int *__restrict__ col_ids,
                                        const float *__restrict__ values, int block_size,
                                        int num_rows, float *__restrict__ dense_matrix) {
  const int block_row = blockIdx.x;
  const int row_begin = row_offsets[block_row];
  const int row_end = row_offsets[block_row + 1];
  const int tile_area = block_size * block_size;
  const int total = (row_end - row_begin) * tile_area;

  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int tile = row_begin + idx / tile_area;
    const int entry = idx % tile_area;
    const int k = entry / block_size;
    const int l = entry - k * block_size;
    const size_t dense_row = static_cast<size_t>(block_row) * block_size + k;
    const int dense_col = col_ids[tile] * block_size + l;
    dense_matrix[dense_row * num_rows + dense_col] =
        values[static_cast<size_t>(tile) * tile_area + entry];
  }
}

/**
 * @brief Scatters a CSR matrix into a zeroed dense row-major buffer.
 *
 * @param stream CUDA stream for the memset and kernel launch.
 * @param matrix Input CSR matrix; must be square and stored in full (both
 *        triangles), which is what HessianStructureBuilder produces.
 * @param[out] dense_matrix Destination, pre-allocated to `NumRows()^2`.
 */
void ScatterToDense(cudaStream_t stream, const CSRSparseMatrix &matrix,
                    dvector<float> &dense_matrix) {
  const int num_rows = static_cast<int>(matrix.NumRows());
  if (num_rows == 0) {
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(
      dense_matrix.data(), 0, static_cast<size_t>(num_rows) * num_rows * sizeof(float), stream));

  constexpr int kWarpsPerBlock = kScatterThreads / kWarpSize;
  const int blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  ScatterCSRToDenseKernel<<<blocks, kScatterThreads, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(), num_rows,
      dense_matrix.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Scatters a BSR matrix into a zeroed dense row-major buffer.
 *
 * One CUDA block per block row walks that row's tiles and writes their
 * `block_size^2` entries out.  Consecutive threads cover consecutive columns
 * within a tile row, so the dense writes coalesce.
 *
 * @param stream CUDA stream for the memset and kernel launch.
 * @param matrix Input BSR matrix; must be square and stored in full.
 * @param[out] dense_matrix Destination, pre-allocated to `NumRows()^2`.
 */
void ScatterToDense(cudaStream_t stream, const BSRSparseMatrix &matrix,
                    dvector<float> &dense_matrix) {
  const int num_rows = matrix.NumRows();
  if (num_rows == 0 || matrix.num_block_rows == 0) {
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(
      dense_matrix.data(), 0, static_cast<size_t>(num_rows) * num_rows * sizeof(float), stream));

  ScatterBSRToDenseKernel<<<matrix.num_block_rows, kScatterThreads, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(), matrix.block_size,
      num_rows, dense_matrix.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

bool DenseLinearSolverBase::InitializeCommon(cudaStream_t stream, size_t num_rows,
                                             const dvector<float> &rhs,
                                             const dvector<float> &result) {
  if (num_rows != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", num_rows, rhs.size());
    return false;
  }
  if (num_rows != result.size()) {
    LogError("LHS size: {} does not match result size: {}", num_rows, result.size());
    return false;
  }
  const size_t matrix_elements = num_rows * num_rows;
  if (dense_matrix_.size() != matrix_elements) {
    dense_matrix_.resize(matrix_elements);
  }
  EnsureBuffersSize(stream, num_rows);
  return true;
}

bool DenseLinearSolverBase::Initialize(cudaStream_t stream, const Problem & /*problem*/,
                                       const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                                       dvector<float> &result) {
  return InitializeCommon(stream, spd_matrix.NumRows(), rhs, result);
}

bool DenseLinearSolverBase::Initialize(cudaStream_t stream, const Problem & /*problem*/,
                                       const BSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                                       dvector<float> &result) {
  return InitializeCommon(stream, static_cast<size_t>(spd_matrix.NumRows()), rhs, result);
}

bool DenseLinearSolverBase::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                                  const dvector<float> &rhs, dvector<float> &result) {
  const size_t num_rows = spd_matrix.NumRows();
  if (!InitializeCommon(stream, num_rows, rhs, result)) {
    return false;
  }
  if (num_rows == 0) {
    return true;  // vacuously solved; nothing to launch
  }
  ScatterToDense(stream, spd_matrix, dense_matrix_);
  return FactorizeAndSolve(stream, static_cast<int>(num_rows), rhs, result);
}

bool DenseLinearSolverBase::Solve(cudaStream_t stream, const BSRSparseMatrix &spd_matrix,
                                  const dvector<float> &rhs, dvector<float> &result) {
  const size_t num_rows = static_cast<size_t>(spd_matrix.NumRows());
  if (!InitializeCommon(stream, num_rows, rhs, result)) {
    return false;
  }
  if (num_rows == 0) {
    return true;  // vacuously solved; nothing to launch
  }
  ScatterToDense(stream, spd_matrix, dense_matrix_);
  return FactorizeAndSolve(stream, static_cast<int>(num_rows), rhs, result);
}

}  // namespace cunls
