/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "cunls/linear_solver/csr_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Configuration modes for cuDSS linear solver.
 *
 * Controls the trade-off between initialization time and solve time.
 */
enum class cuDSSLinearSolverMode {
  SlowInitFastSolve,  ///< Slower initialization, faster subsequent solves.
  FastInitSlowSolve,  ///< Faster initialization, slower subsequent solves (uses
                      ///< refactorization).
};

/**
 * @brief Configuration options for the cuDSS-based sparse linear solver.
 *
 * Controls the solver mode, threading, and library path used during
 * factorization and solve phases.
 */
struct cuDSSLinearSolverOptions {
  cuDSSLinearSolverMode mode =
      cuDSSLinearSolverMode::SlowInitFastSolve;  ///< Solver mode controlling
                                                  ///< the init/solve trade-off.
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
class cuDSSLinearSolver : public CSRSparseLinearSolver {
 public:
  /**
   * @brief Constructs a cuDSS linear solver.
   *
   * Initializes the solver with the specified cuDSS algorithm configuration.
   *
   * @param options Solver configuration controlling the initialization/solve
   *                trade-off and threading settings.
   */
  cuDSSLinearSolver(
      cuDSSLinearSolverOptions options = cuDSSLinearSolverOptions());

  /**
   * @brief Performs setup for the sparse linear system.
   *
   * Always runs the cuDSS symbolic analysis phase. For SlowInitFastSolve mode,
   * that is all; for FastInitSlowSolve mode, also performs an initial
   * factorization so that subsequent Solve calls can use cheaper
   * refactorization.
   *
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param handle cuDSSHandle_t provided by the caller (e.g. from cuDSSHandle).
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Initialize(void* handle, const CSRSparseMatrix& spd_matrix,
                          const dvector<float>& rhs,
                          dvector<float>& result) final;

  /**
   * @brief Solves a sparse SPD linear system Ax = b.
   *
   * For SlowInitFastSolve mode, performs a full factorization followed by the
   * solve phase. For FastInitSlowSolve mode, performs refactorization (reusing
   * the symbolic analysis from Initialize) followed by the solve phase.
   *
   * @p rhs and @p result must already be allocated with the same number of
   * elements as the number of rows in @p spd_matrix. The solver does not
   * resize them; a dimension mismatch causes the function to return false.
   *
   * @param handle cuDSSHandle_t obtained from Initialize.
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x where the solution will be stored
   *               (size must equal matrix rows).
   * @return true on success, false if any dimension mismatch is detected.
   */
  virtual bool Solve(void* handle, const CSRSparseMatrix& spd_matrix,
                     const dvector<float>& rhs, dvector<float>& result) final;

 protected:
  cuDSSLinearSolverOptions options_;  ///< Solver configuration.

  cuDSSDeviceMemPool device_mem_pool_;  ///< Reusable pool for cuDSS allocations.
  cuDSSData cudss_data_;  ///< cuDSS data object storing internal solver state.

  cuDSSConfig cudss_config_;  ///< cuDSS configuration for solver parameters.
};
}  // namespace cunls
