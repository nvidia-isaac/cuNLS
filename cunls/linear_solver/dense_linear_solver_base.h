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

#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

/**
 * @brief Shared machinery for the backends that densify before factorizing.
 *
 * DenseLDLTSolver, DenseCholeskySolver and DenseQRSolver all begin by
 * scattering the sparse coefficient matrix into an `n x n` dense buffer and
 * then never look at the sparse form again.  The sparse layout is therefore
 * invisible to them past the first kernel, which is why they accept *both*
 * scalar CSR and block BSR: scattering tiles is if anything simpler than
 * scattering CSR rows, since a tile carries its own `block_size x block_size`
 * geometry and needs no per-row offset indirection.
 *
 * Accepting both matters beyond tidiness.  The Hessian layout is chosen for
 * the problem, and a backend that declines block storage forces the whole
 * assembly onto scalar CSR (see NormalEquations::Initialize) — giving up the
 * block layout's smaller index array and the `BuildSystem` time that goes with
 * it.  A solver that discards the layout one kernel later has no reason to
 * impose that cost.  After this class, cuDSS is the only backend that does.
 *
 * The dense buffer holds the matrix row-major.  cuSOLVER and cuBLAS want
 * column-major, but every matrix reaching these solvers is a Gauss-Newton
 * Hessian and therefore symmetric, so the two layouts coincide.  Do not reuse
 * this class for a non-symmetric operator without transposing.
 *
 * Subclasses supply three things: workspace sizing, the factorize-and-solve
 * body, and nothing else.  Dimension validation, the dense scatter and the
 * layout dispatch all live here.
 */
class DenseLinearSolverBase : public SparseLinearSolver {
 public:
  /**
   * @copydoc SparseLinearSolver::SupportsBlockStorage
   *
   * Always true: the sparse layout does not survive the scatter into the dense
   * buffer, so there is nothing for either layout to be better at.
   */
  bool SupportsBlockStorage() const final { return true; }

  /** @copydoc SparseLinearSolver::Initialize */
  bool Initialize(cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  /** @copydoc SparseLinearSolver::Initialize */
  bool Initialize(cudaStream_t stream, const Problem &problem, const BSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  /** @copydoc SparseLinearSolver::Solve */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
             dvector<float> &result) final;

  /** @copydoc SparseLinearSolver::Solve */
  bool Solve(cudaStream_t stream, const BSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
             dvector<float> &result) final;

  /** @copydoc SparseLinearSolver::DisableSafetyChecks */
  void DisableSafetyChecks() final { safety_checks_enabled_ = false; }

 protected:
  /**
   * @brief Allocates the backend's working buffers for an `n x n` system.
   *
   * Called from both Initialize and Solve.  Implementations should resize only
   * when the requested size differs, so repeated calls at a fixed `n` are free.
   * @ref dense_matrix_ is sized by this class before the call, so subclasses
   * only need their own buffers.
   *
   * @param stream CUDA stream, for backends that query a library workspace size.
   * @param n Number of rows (and columns) of the dense system.
   */
  virtual void EnsureBuffersSize(cudaStream_t stream, size_t n) = 0;

  /**
   * @brief Factorizes @ref dense_matrix_ and solves against it.
   *
   * Called with @ref dense_matrix_ already populated with the full symmetric
   * matrix in row-major order and all buffers sized for `n`.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param n Matrix dimension; guaranteed positive.
   * @param rhs Right-hand side `b`.
   * @param result Output vector `x`, caller-allocated.
   * @return true on success, false on a detected numerical failure.
   */
  virtual bool FactorizeAndSolve(cudaStream_t stream, int n, const dvector<float> &rhs,
                                 dvector<float> &result) = 0;

  /** @brief Dense row-major copy of the coefficient matrix; `n * n` floats. */
  dvector<float> dense_matrix_;

  /** @brief Whether to run post-factorization numerical checks. */
  bool safety_checks_enabled_ = true;

 private:
  /**
   * @brief Common Initialize body: validate, size buffers.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param num_rows Scalar row count of the coefficient matrix.
   * @param rhs Right-hand side, checked against @p num_rows.
   * @param result Output vector, checked against @p num_rows.
   * @return true on success, false on dimension mismatch.
   */
  bool InitializeCommon(cudaStream_t stream, size_t num_rows, const dvector<float> &rhs,
                        const dvector<float> &result);
};

}  // namespace cunls
