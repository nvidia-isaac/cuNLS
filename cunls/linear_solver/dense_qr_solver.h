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

#pragma once

#include <cuda_runtime.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cusolver_helper.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/csr_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Dense GPU linear solver based on QR factorization via cuSOLVER.
 *
 * Converts the input CSR matrix to a dense matrix and solves A x = b via:
 *  1) QR factorization: A = Q R          (cusolverDnSgeqrf)
 *  2) Apply Q^T to rhs: y = Q^T b        (cusolverDnSormqr)
 *  3) Triangular solve:  R x = y          (cublasStrsm)
 *
 * Since the input matrix is symmetric, the row-major dense representation
 * produced by CSR conversion is identical to column-major, so no transpose
 * is required for the column-major cuSOLVER/cuBLAS APIs.
 *
 * QR factorization works for any non-singular square matrix (not limited
 * to SPD). Returns false from Solve() if the factorization reports an error
 * via devInfo.
 */
class DenseQRSolver : public CSRSparseLinearSolver {
public:
  /**
   * @brief Validates dimensions and pre-allocates internal buffers.
   *
   * @param stream CUDA stream used to query the cuSOLVER workspace size.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  bool Initialize(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  /**
   * @brief Converts CSR to dense and solves via QR factorization.
   *
   * The pipeline is:
   *   1. CSR -> dense conversion.
   *   2. cusolverDnSgeqrf  (in-place QR factorization).
   *   3. Copy rhs into a work vector.
   *   4. cusolverDnSormqr   (Q^T * b in-place).
   *   5. cublasStrsm        (R x = Q^T b upper-triangular solve).
   *   6. Copy result from work vector.
   *   7. Async copy of devInfo to pinned host, stream sync, host check.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false on dimension mismatch or singular matrix.
   */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
             const dvector<float> &rhs, dvector<float> &result) final;

private:
  void EnsureBuffersSize(cudaStream_t stream, size_t n);

  void ConvertCSRToDense(cudaStream_t stream, const CSRSparseMatrix &matrix,
                         dvector<float> &dense_matrix);

  cuSolverHandle cusolver_handle_;
  cuBLASHandle cublas_handle_;
  dvector<float> dense_matrix_;
  dvector<float> tau_;
  dvector<float> workspace_;
  dvector<float> rhs_copy_;
  dvector<int> dev_info_;
  pvector<int> dev_info_pinned_;
  size_t last_n_ = 0;
};

} // namespace cunls
