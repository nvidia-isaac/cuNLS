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

namespace cunls {

class Problem;  // forward declaration; defined in cunls/minimizer/problem.h.

/**
 * @brief Base class for solvers of the sparse symmetric system `A x = b`.
 *
 * Derived classes implement specific strategies (cuDSS direct factorization,
 * dense pivoted LDLT / Cholesky / QR, block-Jacobi PCG).
 *
 * `A` arrives in one of two layouts, and every backend must accept at least
 * scalar CSR: Initialize and Solve are pure virtual for @ref CSRSparseMatrix
 * and defaulted for @ref BSRSparseMatrix.  A backend opts into the block form
 * by overriding @ref SupportsBlockStorage together with the two BSR overloads;
 * one that does not is never handed a BSR matrix, so its defaults are dead.
 * The layout is decided once per problem in @ref NormalEquations, and no
 * conversion between the two ever runs on the solve path.
 *
 * Initialize receives the originating @ref Problem so solvers can adapt
 * to its block / factor-graph structure (e.g.
 * @ref BlockSparsePCGSolver reads each state batch's @c TangentSize to
 * build its block-Jacobi preconditioner without a downcast at the call
 * site).  Solvers that don't care can simply ignore the argument.
 */
class SparseLinearSolver {
 public:
  /**
   * @brief Performs setup work for the linear system.
   *
   * Typically runs symbolic analysis of the sparsity pattern; some modes may
   * also perform an initial numerical factorization. Must be called once
   * whenever the matrix structure changes, before any call to Solve.
   *
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param stream     CUDA stream for asynchronous GPU operations.
   * @param problem    The originating optimization problem.  Solvers may use
   *                   its state-batch structure to specialize their setup
   *                   (e.g. derive a block-Jacobi preconditioner layout).
   *                   Pass a default-constructed @ref Problem when calling
   *                   on a raw matrix that wasn't produced by cuNLS factors.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs        The right-hand side vector b (size must equal matrix
   *                   rows).
   * @param result     Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Initialize(cudaStream_t stream, const Problem &problem,
                          const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                          dvector<float> &result) = 0;

  /**
   * @brief Solves a linear system Ax = b.
   *
   * Performs factorization and solve phases to compute the solution x.
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                     const dvector<float> &rhs, dvector<float> &result) = 0;

  /**
   * @brief Whether this backend can consume a block-stored matrix directly.
   *
   * The Hessian of a factor graph is naturally block structured, and assembling
   * it that way keeps one column index per tile instead of one per scalar
   * entry.  Backends that say yes get the block form.
   *
   * Backends that say no are not handed a converted copy — no BSR-to-CSR
   * expansion runs anywhere on the solve path.  Instead the Hessian is
   * assembled *natively* in scalar CSR for them (see NormalEquations), which is
   * their optimum: the block layout's saving is the index array, and
   * materializing scalar indices for a CSR-only backend would give that saving
   * straight back plus a per-iteration value permutation.
   *
   * The layout is therefore a property of the problem, and this method is a
   * veto, not a request.  Vetoing costs the block-storage delta only — the
   * smaller index array and the assembly time that comes with it — never the
   * much larger block-wise *assembly* win, which is layout-independent and
   * which every backend gets unconditionally.
   */
  virtual bool SupportsBlockStorage() const { return false; }

  /** @brief BSR counterpart of Initialize; only called when supported. */
  virtual bool Initialize(cudaStream_t stream, const Problem &problem,
                          const BSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                          dvector<float> &result) {
    return false;
  }

  /** @brief BSR counterpart of Solve; only called when supported. */
  virtual bool Solve(cudaStream_t stream, const BSRSparseMatrix &spd_matrix,
                     const dvector<float> &rhs, dvector<float> &result) {
    return false;
  }

  /**
   * @brief Advisory request to skip post-factorization safety checks.
   *
   * The dense backends copy a device-side status flag back to the host after
   * factorization and synchronize the stream to detect singular or
   * non-positive-definite matrices.  Skipping that removes a device-to-host
   * memcpy, a stream synchronization, and (for LDLT) in-kernel pivot checks,
   * which can be a significant fraction of the solve time for small systems.
   *
   * Backends with no such phase — cuDSS, which reports through its own status,
   * and the PCG solver, which has no factorization — ignore this.  It is a
   * hint, not a contract; see DenseLinearSolverBase for the implementation.
   */
  virtual void DisableSafetyChecks() {}

  /** @brief Virtual destructor for proper cleanup of derived solver instances.
   */
  virtual ~SparseLinearSolver() = default;
};
}  // namespace cunls
