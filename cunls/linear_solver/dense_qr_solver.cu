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
#include "cunls/linear_solver/dense_qr_solver.h"

namespace cunls {
namespace {

constexpr int kWarpSize = 32;
constexpr float kDiagonalEpsilonAbs = 1e-7f;

/// Checks whether any diagonal element of an n x n column-major matrix has
/// absolute value below the threshold. Sets *status = 0 if any diagonal is
/// near-zero, 1 otherwise.
__global__ void check_diagonal_kernel(const float *__restrict__ matrix, int n,
                                      int lda, float threshold,
                                      int *__restrict__ status) {
  *status = 1;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (fabsf(matrix[i * lda + i]) <= threshold) {
      atomicExch(status, 0);
    }
  }
}

// Writes the CSR matrix into a dense buffer in row-major order.  cuSOLVER
// (geqrf, ormqr) and cuBLAS (trsm) expect column-major layout, but this
// solver is only used for symmetric matrices (the Gauss-Newton Hessian
// J^T J), for which row-major == column-major (A == A^T).  Do NOT use this
// kernel for non-symmetric inputs without first transposing the layout.
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

bool DenseQRSolver::Initialize(cudaStream_t stream, const Problem & /*problem*/,
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

bool DenseQRSolver::Solve(cudaStream_t stream,
                          const CSRSparseMatrix &spd_matrix,
                          const dvector<float> &rhs, dvector<float> &result) {
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
  auto cusolver =
      static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));

  // 1. QR factorization: A = Q R (in-place, R in upper triangle, Householder
  //    reflectors stored below the diagonal, scalars in tau).
  THROW_ON_CUSOLVER_ERROR(cusolverDnSgeqrf(
      cusolver, n, n, dense_matrix_.data(), n, tau_.data(), workspace_.data(),
      static_cast<int>(workspace_.size()), dev_info_.data()));

  if (safety_checks_enabled_) {
    // 2. Check R's diagonal for rank deficiency. geqrf always sets devInfo = 0
    //    on success, so we must explicitly inspect the diagonal of R.
    check_diagonal_kernel<<<1, 256, 0, stream>>>(
        dense_matrix_.data(), n, n, kDiagonalEpsilonAbs, dev_info_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());

    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(),
                                        dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] == 0) {
      LogError("QR factorization detected a (near-)zero diagonal in R. "
               "Matrix is rank-deficient.");
      return false;
    }
  }

  // 3. Copy rhs into work vector (ormqr and trsm operate in-place on it).
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(rhs_copy_.data(), rhs.data(),
                                      n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // 4. Apply Q^T to the rhs: y = Q^T * b.
  THROW_ON_CUSOLVER_ERROR(cusolverDnSormqr(
      cusolver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, dense_matrix_.data(), n,
      tau_.data(), rhs_copy_.data(), n, workspace_.data(),
      static_cast<int>(workspace_.size()), dev_info_.data()));

  // 5. Solve the upper-triangular system R x = y via cuBLAS trsm.
  auto cublas = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  const float alpha = 1.0f;
  THROW_ON_CUBLAS_ERROR(
      cublasStrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                  CUBLAS_DIAG_NON_UNIT, n, 1, &alpha, dense_matrix_.data(), n,
                  rhs_copy_.data(), n));

  // 6. Copy solution out.
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(result.data(), rhs_copy_.data(),
                                      n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  return true;
}

void DenseQRSolver::EnsureBuffersSize(cudaStream_t stream, size_t n) {
  const size_t matrix_elements = n * n;
  if (dense_matrix_.size() != matrix_elements) {
    dense_matrix_.resize(matrix_elements);
  }
  if (tau_.size() != n) {
    tau_.resize(n);
  }
  if (rhs_copy_.size() != n) {
    rhs_copy_.resize(n);
  }
  if (dev_info_.size() != 1) {
    dev_info_.resize(1);
    dev_info_pinned_.resize(1);
  }

  if (n != last_n_ && n > 0) {
    auto handle =
        static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));
    int geqrf_lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSgeqrf_bufferSize(
        handle, static_cast<int>(n), static_cast<int>(n), dense_matrix_.data(),
        static_cast<int>(n), &geqrf_lwork));

    int ormqr_lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSormqr_bufferSize(
        handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, static_cast<int>(n), 1,
        static_cast<int>(n), dense_matrix_.data(), static_cast<int>(n),
        tau_.data(), rhs_copy_.data(), static_cast<int>(n), &ormqr_lwork));

    int lwork = (geqrf_lwork > ormqr_lwork) ? geqrf_lwork : ormqr_lwork;
    workspace_.resize(static_cast<size_t>(lwork));
    last_n_ = n;
  }
}

void DenseQRSolver::ConvertCSRToDense(cudaStream_t stream,
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
