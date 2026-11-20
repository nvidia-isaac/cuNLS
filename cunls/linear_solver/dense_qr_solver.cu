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

constexpr float kDiagonalEpsilonAbs = 1e-7f;

/// Checks whether any diagonal element of an n x n column-major matrix has
/// absolute value below the threshold. Sets *status = 0 if any diagonal is
/// near-zero, 1 otherwise (one block).
__global__ void check_diagonal_kernel(const float *__restrict__ matrix, int n, int lda,
                                      float threshold, int *__restrict__ status) {
  bool singular = false;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    singular = singular || fabsf(matrix[i * lda + i]) <= threshold;
  }
  // One write after the vote: a thread setting 1 cannot overwrite another's 0.
  const int any_singular = __syncthreads_or(singular);
  if (threadIdx.x == 0) *status = any_singular ? 0 : 1;
}

}  // namespace

bool DenseQRSolver::FactorizeAndSolve(cudaStream_t stream, int n, const dvector<float> &rhs,
                                      dvector<float> &result) {
  auto cusolver = static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));

  // 1. QR factorization: A = Q R (in-place, R in upper triangle, Householder
  //    reflectors stored below the diagonal, scalars in tau).
  THROW_ON_CUSOLVER_ERROR(cusolverDnSgeqrf(cusolver, n, n, dense_matrix_.data(), n, tau_.data(),
                                           workspace_.data(), static_cast<int>(workspace_.size()),
                                           dev_info_.data()));

  if (safety_checks_enabled_) {
    // 2. Check R's diagonal for rank deficiency. geqrf always sets devInfo = 0
    //    on success, so we must explicitly inspect the diagonal of R.
    check_diagonal_kernel<<<1, 256, 0, stream>>>(dense_matrix_.data(), n, n, kDiagonalEpsilonAbs,
                                                 dev_info_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());

    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(), dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] == 0) {
      LogError(
          "QR factorization detected a (near-)zero diagonal in R. "
          "Matrix is rank-deficient.");
      return false;
    }
  }

  // 3. Copy rhs into result (ormqr and trsm operate in place on it).
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(result.data(), rhs.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // 4. Apply Q^T to the rhs: y = Q^T * b.
  THROW_ON_CUSOLVER_ERROR(cusolverDnSormqr(
      cusolver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, 1, n, dense_matrix_.data(), n, tau_.data(),
      result.data(), n, workspace_.data(), static_cast<int>(workspace_.size()), dev_info_.data()));

  // 5. Solve the upper-triangular system R x = y via cuBLAS trsm.
  auto cublas = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  const float alpha = 1.0f;
  THROW_ON_CUBLAS_ERROR(cublasStrsm(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, n, 1, &alpha, dense_matrix_.data(), n,
                                    result.data(), n));
  return true;
}

void DenseQRSolver::EnsureBuffersSize(cudaStream_t stream, size_t n) {
  if (tau_.size() != n) {
    tau_.resize(n);
  }
  if (dev_info_.size() != 1) {
    dev_info_.resize(1);
    dev_info_pinned_.resize(1);
  }

  if (n != last_n_ && n > 0) {
    auto handle = static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));
    int geqrf_lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSgeqrf_bufferSize(handle, static_cast<int>(n),
                                                        static_cast<int>(n), dense_matrix_.data(),
                                                        static_cast<int>(n), &geqrf_lwork));

    int ormqr_lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSormqr_bufferSize(
        handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, static_cast<int>(n), 1, static_cast<int>(n),
        dense_matrix_.data(), static_cast<int>(n), tau_.data(), dense_matrix_.data(),
        static_cast<int>(n), &ormqr_lwork));

    int lwork = (geqrf_lwork > ormqr_lwork) ? geqrf_lwork : ormqr_lwork;
    workspace_.resize(static_cast<size_t>(lwork));
    last_n_ = n;
  }
}

}  // namespace cunls
