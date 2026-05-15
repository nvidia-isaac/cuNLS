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

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/dense_cholesky_solver.h"

namespace cunls {
namespace {

constexpr int kWarpSize = 32;

// Writes the CSR matrix into a dense buffer in row-major order.  cuSOLVER
// (potrf, potrs) expects column-major layout, but this solver is only used
// for symmetric matrices (the Gauss-Newton Hessian J^T J), for which
// row-major == column-major (A == A^T).  Do NOT use this kernel for
// non-symmetric inputs without first transposing the layout.
__global__ void csr_to_dense_kernel(const int *__restrict__ row_offsets,
                                    const int *__restrict__ col_ids,
                                    const float *__restrict__ values,
                                    int num_rows,
                                    float *__restrict__ dense_matrix) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row >= num_rows) {
    return;
  }
  const int lane = threadIdx.x % kWarpSize;
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  float *const dense_row = dense_matrix + row * num_rows;
  for (int idx = row_start + lane; idx < row_end; idx += kWarpSize) {
    dense_row[col_ids[idx]] = values[idx];
  }
}

} // namespace

bool DenseCholeskySolver::Initialize(cudaStream_t stream,
                                     const Problem & /*problem*/,
                                     const CSRSparseMatrix &spd_matrix,
                                     const dvector<float> &rhs,
                                     dvector<float> &result) {
  const size_t matrix_size = spd_matrix.NumRows();
  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size,
             rhs.size());
    return false;
  }
  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size,
             result.size());
    return false;
  }
  EnsureBuffersSize(stream, matrix_size);
  return true;
}

bool DenseCholeskySolver::Solve(cudaStream_t stream,
                                const CSRSparseMatrix &spd_matrix,
                                const dvector<float> &rhs,
                                dvector<float> &result) {
  const size_t matrix_size = spd_matrix.NumRows();
  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size,
             rhs.size());
    return false;
  }
  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size,
             result.size());
    return false;
  }
  if (matrix_size == 0) {
    return true;
  }

  EnsureBuffersSize(stream, matrix_size);
  ConvertCSRToDense(stream, spd_matrix, dense_matrix_);

  const int n = static_cast<int>(matrix_size);
  auto handle =
      static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));

  THROW_ON_CUSOLVER_ERROR(
      cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, dense_matrix_.data(),
                       n, workspace_.data(),
                       static_cast<int>(workspace_.size()), dev_info_.data()));

  if (safety_checks_enabled_) {
    // Check devInfo from potrf before proceeding to potrs, since potrs would
    // overwrite it.  devInfo > 0 means the leading minor of order devInfo is
    // not positive-definite; devInfo < 0 means the devInfo-th parameter was
    // invalid.
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(),
                                        dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] != 0) {
      LogError("Cholesky factorization failed (devInfo = {}). Matrix is likely "
               "not positive-definite.",
               dev_info_pinned_[0]);
      return false;
    }
  }

  // potrs solves in-place on the RHS buffer, so copy rhs -> result first.
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(result.data(), rhs.data(),
                                      n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  THROW_ON_CUSOLVER_ERROR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                           dense_matrix_.data(), n,
                                           result.data(), n, dev_info_.data()));

  if (safety_checks_enabled_) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(),
                                        dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] < 0) {
      LogError("cusolverDnSpotrs reported invalid parameter at index {}.",
               -dev_info_pinned_[0]);
      return false;
    }
    if (dev_info_pinned_[0] > 0) {
      LogError("cusolverDnSpotrs failed (devInfo = {}).", dev_info_pinned_[0]);
      return false;
    }
  }

  return true;
}

void DenseCholeskySolver::EnsureBuffersSize(cudaStream_t stream, size_t n) {
  const size_t matrix_elements = n * n;
  if (dense_matrix_.size() != matrix_elements) {
    dense_matrix_.resize(matrix_elements);
  }
  if (dev_info_.size() != 1) {
    dev_info_.resize(1);
    dev_info_pinned_.resize(1);
  }

  if (n != last_n_ && n > 0) {
    auto handle =
        static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));
    int lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSpotrf_bufferSize(
        handle, CUBLAS_FILL_MODE_LOWER, static_cast<int>(n),
        dense_matrix_.data(), static_cast<int>(n), &lwork));
    workspace_.resize(static_cast<size_t>(lwork));
    last_n_ = n;
  }
}

void DenseCholeskySolver::ConvertCSRToDense(cudaStream_t stream,
                                            const CSRSparseMatrix &matrix,
                                            dvector<float> &dense_matrix) {
  const int num_rows = static_cast<int>(matrix.NumRows());
  if (num_rows == 0) {
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(
      dense_matrix.data(), 0,
      static_cast<size_t>(num_rows) * num_rows * sizeof(float), stream));

  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  csr_to_dense_kernel<<<blocks, kThreads, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
      num_rows, dense_matrix.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
