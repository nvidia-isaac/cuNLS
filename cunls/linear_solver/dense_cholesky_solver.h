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
#include "cunls/linear_solver/dense_linear_solver_base.h"

namespace cunls {

/**
 * @brief Dense GPU linear solver based on Cholesky factorization via cuSOLVER.
 *
 * Densifies the symmetric positive-definite coefficient matrix (from either
 * sparse layout; see DenseLinearSolverBase) and solves A x = b via:
 *  1) Cholesky factorization: A = L L^T  (cusolverDnSpotrf)
 *  2) Triangular solve using the factor   (cusolverDnSpotrs)
 *
 * Since the input matrix is symmetric, the row-major dense representation
 * produced by the scatter is identical to column-major, so no transpose is
 * required for the column-major cuSOLVER API.
 *
 * Returns false from Solve() if the matrix is not positive-definite (cuSOLVER
 * reports a non-zero devInfo from potrf).
 */
class DenseCholeskySolver : public DenseLinearSolverBase {
 protected:
  /** @copydoc DenseLinearSolverBase::EnsureBuffersSize */
  void EnsureBuffersSize(cudaStream_t stream, size_t n) final;

  /**
   * @brief Factorizes the dense matrix via Cholesky and solves.
   *
   * The pipeline is:
   *   1. cusolverDnSpotrf  (in-place Cholesky factorization).
   *   2. devInfo check after potrf (if safety checks enabled).
   *   3. Copy rhs into result (potrs works in-place on B).
   *   4. cusolverDnSpotrs  (triangular solve).
   *   5. devInfo check after potrs (if safety checks enabled).
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param n Matrix dimension.
   * @param rhs The right-hand side vector b.
   * @param result Output vector x.
   * @return true on success, false on a non-SPD matrix (devInfo > 0 from
   *         potrf) or an invalid parameter from potrs (devInfo < 0).
   */
  bool FactorizeAndSolve(cudaStream_t stream, int n, const dvector<float> &rhs,
                         dvector<float> &result) final;

 private:
  cuSolverHandle cusolver_handle_;
  dvector<float> workspace_;
  dvector<int> dev_info_;
  pvector<int> dev_info_pinned_;
  size_t last_n_ = 0;
};

}  // namespace cunls
