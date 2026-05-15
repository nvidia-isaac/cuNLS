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

#include "cunls/common/cusolver_helper.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/csr_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Dense GPU linear solver based on Cholesky factorization via cuSOLVER.
 *
 * Converts the input CSR symmetric positive-definite matrix to a dense matrix
 * and solves A x = b via:
 *  1) Cholesky factorization: A = L L^T  (cusolverDnSpotrf)
 *  2) Triangular solve using the factor   (cusolverDnSpotrs)
 *
 * Since the input matrix is symmetric, the row-major dense representation
 * produced by CSR conversion is identical to column-major, so no transpose
 * is required for the column-major cuSOLVER API.
 *
 * Returns false from Solve() if the matrix is not positive-definite (cuSOLVER
 * reports a non-zero devInfo from potrf).
 */
class DenseCholeskySolver : public CSRSparseLinearSolver {
public:
  /**
   * @brief Validates dimensions and pre-allocates internal buffers.
   *
   * @param stream CUDA stream used to query the cuSOLVER workspace size.
   * @param spd_matrix The SPD coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  bool Initialize(cudaStream_t stream, const Problem &problem,
                  const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                  dvector<float> &result) final;

  /**
   * @brief Converts CSR to dense and solves via Cholesky factorization.
   *
   * The pipeline is:
   *   1. CSR -> dense conversion.
   *   2. cusolverDnSpotrf  (in-place Cholesky factorization).
   *   3. devInfo check after potrf (if safety checks enabled).
   *   4. Copy rhs into result (potrs works in-place on B).
   *   5. cusolverDnSpotrs  (triangular solve).
   *   6. devInfo check after potrs (if safety checks enabled).
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The SPD coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false on dimension mismatch, non-SPD matrix
   *         (devInfo > 0 from potrf), or invalid parameter from potrs
   *         (devInfo < 0).
   */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
             const dvector<float> &rhs, dvector<float> &result) final;

private:
  void EnsureBuffersSize(cudaStream_t stream, size_t n);

  void ConvertCSRToDense(cudaStream_t stream, const CSRSparseMatrix &matrix,
                         dvector<float> &dense_matrix);

  cuSolverHandle cusolver_handle_;
  dvector<float> dense_matrix_;
  dvector<float> workspace_;
  dvector<int> dev_info_;
  pvector<int> dev_info_pinned_;
  size_t last_n_ = 0;
};

} // namespace cunls
