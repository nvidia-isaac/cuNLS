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

#include <cstdint>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/dense_linear_solver_base.h"

namespace cunls {
namespace {

constexpr int kWarpSize = 32;
constexpr int kScatterThreads = 256;

/// @brief slot[e] = row * n + col for every entry e of a CSR pattern (one warp per row).
template <typename Slot>
__global__ void DenseSlotsCSRKernel(const int *__restrict__ row_offsets,
                                    const int *__restrict__ col_ids, int num_rows,
                                    Slot *__restrict__ slot) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row >= num_rows) return;
  const int lane = threadIdx.x % kWarpSize;
  for (int idx = row_offsets[row] + lane; idx < row_offsets[row + 1]; idx += kWarpSize) {
    slot[idx] = static_cast<Slot>(row) * num_rows + col_ids[idx];
  }
}

/// @brief slot[e] = row * n + col for every value e of a BSR pattern (one CUDA block per block
/// row).
template <typename Slot>
__global__ void DenseSlotsBSRKernel(const int *__restrict__ row_offsets,
                                    const int *__restrict__ col_ids, int block_size, int num_rows,
                                    Slot *__restrict__ slot) {
  const int block_row = blockIdx.x;
  const int row_begin = row_offsets[block_row];
  const int tile_area = block_size * block_size;
  const int total = (row_offsets[block_row + 1] - row_begin) * tile_area;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int tile = row_begin + idx / tile_area;
    const int entry = idx % tile_area;
    const Slot row = static_cast<Slot>(block_row) * block_size + entry / block_size;
    slot[static_cast<size_t>(tile) * tile_area + entry] =
        row * num_rows + col_ids[tile] * block_size + entry % block_size;
  }
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
  if (!InitializeCommon(stream, spd_matrix.NumRows(), rhs, result)) return false;
  // The storage map: entry (row, col) of the pattern is dense element row * n + col.
  const int n = static_cast<int>(spd_matrix.NumRows());
  const size_t nnz = spd_matrix.NumNonZeros();
  wide_slots_ = static_cast<size_t>(n) * n > static_cast<size_t>(INT32_MAX);
  constexpr int kWarpsPerBlock = kScatterThreads / kWarpSize;
  const int blocks = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (wide_slots_) {
    entry_slot64_.resize(nnz);
    if (n > 0) {
      DenseSlotsCSRKernel<int64_t><<<blocks, kScatterThreads, 0, stream>>>(
          spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(), n, entry_slot64_.data());
    }
  } else {
    entry_slot_.resize(nnz);
    if (n > 0) {
      DenseSlotsCSRKernel<int><<<blocks, kScatterThreads, 0, stream>>>(
          spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(), n, entry_slot_.data());
    }
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

bool DenseLinearSolverBase::Initialize(cudaStream_t stream, const Problem & /*problem*/,
                                       const BSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                                       dvector<float> &result) {
  if (!InitializeCommon(stream, static_cast<size_t>(spd_matrix.NumRows()), rhs, result)) {
    return false;
  }
  const int n = spd_matrix.NumRows();
  const size_t num_values = spd_matrix.values.size();
  wide_slots_ = static_cast<size_t>(n) * n > static_cast<size_t>(INT32_MAX);
  if (wide_slots_) {
    entry_slot64_.resize(num_values);
    if (spd_matrix.num_block_rows > 0) {
      DenseSlotsBSRKernel<int64_t><<<spd_matrix.num_block_rows, kScatterThreads, 0, stream>>>(
          spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(), spd_matrix.block_size, n,
          entry_slot64_.data());
    }
  } else {
    entry_slot_.resize(num_values);
    if (spd_matrix.num_block_rows > 0) {
      DenseSlotsBSRKernel<int><<<spd_matrix.num_block_rows, kScatterThreads, 0, stream>>>(
          spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(), spd_matrix.block_size, n,
          entry_slot_.data());
    }
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

SolverMatrixStorage DenseLinearSolverBase::MatrixStorage() {
  if (wide_slots_) return {dense_matrix_.data(), nullptr, entry_slot64_.data()};
  return {dense_matrix_.data(), entry_slot_.data(), nullptr};
}

void DenseLinearSolverBase::ClearMatrixStorage(cudaStream_t stream) {
  if (dense_matrix_.size() == 0) return;
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(dense_matrix_.data(), 0, dense_matrix_.size() * sizeof(float), stream));
}

bool DenseLinearSolverBase::SolveStoredMatrix(cudaStream_t stream, const dvector<float> &rhs,
                                              dvector<float> &result) {
  const size_t num_rows = rhs.size();
  if (num_rows * num_rows != dense_matrix_.size() || result.size() != num_rows) {
    LogError("Dense solver: rhs / result size does not match the system it was initialized for");
    return false;
  }
  if (num_rows == 0) return true;
  return FactorizeAndSolve(stream, static_cast<int>(num_rows), rhs, result);
}

}  // namespace cunls
