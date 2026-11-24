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

#include <string>

#include "cunls/common/cudss_helper.h"
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

/**
 * @brief Configuration options for the cuDSS-based sparse linear solver: the
 * threading of its host-side operations.
 */
struct cuDSSLinearSolverOptions {
  int nthreads = 1;  ///< Number of threads for host-side operations.
  std::string threading_lib_path =
      "";  ///< Path to the threading library (empty disables multi-threading).
};

/**
 * @brief Sparse linear solver using NVIDIA's cuDSS library.
 *
 * This class provides a high-performance solver for sparse linear systems
 * Ax = b where A is symmetric positive definite. It uses NVIDIA's cuDSS
 * library for GPU-accelerated direct factorization.
 */
class cuDSSLinearSolver : public SparseLinearSolver {
 public:
  // This backend consumes CSR only; SupportsBlockStorage() stays false, so the
  // base class's block-storage overloads are never called on it.  The
  // using-declarations keep them visible rather than hidden by the CSR
  // overrides below.
  using SparseLinearSolver::Initialize;
  using SparseLinearSolver::Solve;

  /**
   * @brief Constructs a cuDSS linear solver.
   *
   * @param options Threading settings of the cuDSS host-side operations.
   */
  cuDSSLinearSolver(cuDSSLinearSolverOptions options = cuDSSLinearSolverOptions());

  /**
   * @brief Performs setup for the sparse linear system.
   *
   * Runs the cuDSS symbolic analysis phase (reordering) of the pattern of
   * @p spd_matrix; each Solve() factorizes the current values.
   *
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param stream CUDA stream for asynchronous GPU operations. Also used to
   *               lazily initialize the internal cuDSS handle on first call.
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  bool Initialize(cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  /**
   * @brief Solves a sparse SPD linear system Ax = b.
   *
   * Factorizes the current values (reusing the symbolic analysis of
   * Initialize()), then runs the solve phase.
   *
   * @p rhs and @p result must already be allocated with the same number of
   * elements as the number of rows in @p spd_matrix. The solver does not
   * resize them; a dimension mismatch causes the function to return false.
   *
   * @param stream CUDA stream for asynchronous GPU operations. Also used to
   *               lazily initialize the internal cuDSS handle if needed.
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x where the solution will be stored
   *               (size must equal matrix rows).
   * @return true on success, false if any dimension mismatch is detected.
   */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
             dvector<float> &result) final;

  /** @brief The values of the pattern of Initialize(), in its layout (the identity map). */
  SolverMatrixStorage MatrixStorage() final;

  /** @brief Every slot is a pattern value: nothing to clear. */
  void ClearMatrixStorage(cudaStream_t) final {}

  /** @brief Solve() with the pattern of Initialize() and the values in MatrixStorage(). */
  bool SolveStoredMatrix(cudaStream_t stream, const dvector<float> &rhs,
                         dvector<float> &result) final;

 private:
  /** @brief The factorization and solve phases for @p spd_matrix's structure and @p values. */
  bool SolveWithValues(cudaStream_t stream, const CSRSparseMatrix &spd_matrix, const float *values,
                       const dvector<float> &rhs, dvector<float> &result);

  /**
   * @brief The cuDSS handle, bound to @p stream; the first call also attaches
   * device_mem_pool_ to it (the handle is kept for the solver's lifetime).
   */
  void *Handle(cudaStream_t stream);

  const CSRSparseMatrix *pattern_ = nullptr;  ///< The pattern of Initialize() (not owned).
  dvector<float> values_;                     ///< MatrixStorage(): its values.

  cuDSSLinearSolverOptions options_;  ///< Solver configuration.

  // Declaration order is destruction order reversed: the pool outlives the handle,
  // the handle outlives the data made with it.
  cuDSSDeviceMemPool device_mem_pool_;  ///< Reusable pool for cuDSS allocations.
  cuDSSHandle cudss_handle_;            ///< Owns the cuDSS handle used for all solver phases.
  bool pool_attached_ = false;          ///< device_mem_pool_ is attached to cudss_handle_.
  cuDSSData cudss_data_;                ///< cuDSS data object storing internal solver state.

  cuDSSConfig cudss_config_;  ///< cuDSS configuration for solver parameters.
};
}  // namespace cunls
