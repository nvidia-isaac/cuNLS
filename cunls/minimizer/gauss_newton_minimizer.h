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

#include <vector>

#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {

/**
 * @brief Summary statistics returned after optimization.
 *
 * Contains information about the optimization process including iteration count
 * and cost evolution.
 */
struct MinimizerSummary {
  /** @brief Number of iterations performed. */
  size_t num_iterations = 0;

  /** @brief Initial cost before optimization. */
  float initial_cost = 0.0f;

  /** @brief Final cost after optimization. */
  float final_cost = 0.0f;

  /** @brief Vector of costs from each iteration. */
  std::vector<float> iteration_costs;
};

/**
 * @brief Configuration options for the Gauss-Newton optimizer.
 *
 * Controls convergence criteria and iteration limits.
 */
struct MinimizerOptions {
  /** @brief Maximum number of Gauss-Newton iterations. Default: 50 */
  size_t max_num_iterations = 50;

  /**
   * @brief Convergence threshold for state updates.
   *
   * Optimizer terminates when the squared step norm falls below this threshold.
   * Default: 1e-6
   */
  float state_tolerance = 1e-6f;

  /**
   * @brief Convergence threshold for cost change.
   *
   * Optimizer terminates when the cost falls below this threshold.
   * Default: 1e-6
   */
  float cost_tolerance = 1e-6f;

  /**
   * @brief Type of sparse linear solver to use.
   *
   * Default: cuDSS
   */
  SparseLinearSolverType sparse_linear_solver_type =
      SparseLinearSolverType::cuDSS;

  /**
   * @brief Configuration for the sparse linear solver.
   *
   * Contains solver-specific configuration options. Defaults to cuDSS solver
   * with SlowInitFastSolve configuration and 1 thread.
   * To enable mutiple threads in cuDSS, please provide the full path to the
   * threading library in the cudss_solver_options.
   * e.g. .cudss_solver_options = cuDSSLinearSolverOptions{
   * .mode = cuDSSLinearSolverMode::SlowInitFastSolve,
   * .nthreads = 12,
   * .threading_lib_path = "/path/to/libcudss_mtlayer_gomp.so"
   * }
   */
  SparseLinearSolverConfig sparse_linear_solver_config = {
      .cudss_solver_options = cuDSSLinearSolverOptions()};
};

class GaussNewtonMinimizer {
 public:
  /**
   * @brief Constructs a Gauss-Newton optimizer.
   *
   * @param options Configuration options for the optimizer. Defaults to
   *                MinimizerOptions with standard values.
   */
  GaussNewtonMinimizer(const MinimizerOptions& options = MinimizerOptions());

  /**
   * @brief Virtual destructor for proper cleanup in derived classes.
   */
  virtual ~GaussNewtonMinimizer() = default;

  /**
   * @brief Deleted copy constructor
   */
  GaussNewtonMinimizer(const GaussNewtonMinimizer&) = delete;

  /**
   * @brief Deleted assigment operator
   */
  GaussNewtonMinimizer& operator=(const GaussNewtonMinimizer&) = delete;

  /**
   * @brief Deleted move constructor
   */
  GaussNewtonMinimizer(GaussNewtonMinimizer&&) = delete;

  /**
   * @brief Deleted move assigment operator
   */
  GaussNewtonMinimizer& operator=(GaussNewtonMinimizer&&) = delete;

  /**
   * @brief Solves the optimization problem.
   *
   * Minimizes the cost defined by the problem's factors using the Gauss-Newton
   * algorithm. The state values in the problem are updated in-place.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param problem The optimization problem to solve. State values are
   *                modified in-place during optimization.
   * @return Summary containing iteration count and cost statistics.
   */
  MinimizerSummary Minimize(cudaStream_t stream, Problem& problem);

 protected:
  /**
   * @brief Checks if convergence criteria are satisfied.
   *
   * Determines whether the optimization has converged based on step size,
   * cost reduction, and step quality. Also computes the step quality metric
   * (ratio of updated cost to current cost).
   *
   * @param stream CUDA stream for GPU operations.
   * @param updated_cost Cost after applying the step.
   * @param current_cost Cost before applying the step.
   * @param step State update step vector.
   * @param[out] step_quality Output argument for step quality metric
   *                          (updated_cost / current_cost).
   * @return True if converged, false otherwise.
   */
  virtual bool CheckConvergence(cudaStream_t stream, float updated_cost,
                                float current_cost, const dvector<float>& step,
                                float& step_quality);

  /**
   * @brief Determines if a step should be accepted.
   *
   * A step is accepted if it reduces the cost (step_quality < 1.0).
   *
   * @param step_quality Ratio of updated cost to current cost.
   * @return True if step should be accepted, false otherwise.
   */
  virtual bool AcceptStep(float step_quality);

  /**
   * @brief Determines if a step should be rejected.
   *
   * A step is rejected if it increases the cost (step_quality >= 1.0).
   *
   * @param step_quality Ratio of updated cost to current cost.
   * @return True if step should be rejected, false otherwise.
   */
  virtual bool RejectStep(float step_quality);

  /**
   * @brief Initializes internal data structures for optimization.
   *
   * Allocates and prepares residual vectors, Jacobian structures, and
   * state operations for the given problem.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem to initialize for.
   */
  virtual void Initialize(cudaStream_t stream, const Problem& problem);

  /**
   * @brief Builds the Gauss-Newton linear system.
   *
   * Computes residuals and Jacobian, converts to CSR format, and constructs
   * the linear system J^T J dx = -J^T r. The left-hand side (LHS) is the
   * approximate Hessian H = J^T J, and the right-hand side (RHS) is the
   * negative gradient -J^T r.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param minimizer_state Current minimizer state.
   * @param[out] lhs Output left-hand side matrix (H = J^T J).
   * @param[out] rhs Output right-hand side vector (-J^T r).
   */
  virtual void BuildSystem(cudaStream_t stream, const Problem& problem,
                           const MinimizerState& minimizer_state,
                           CSRSparseMatrix& lhs, dvector<float>& rhs);

  /**
   * @brief Updates states with the computed step.
   *
   * Computes updated_state = curr_state + step using state block
   * operations that respect state block manifolds.
   *
   * @param stream CUDA stream for GPU operations.
   * @param curr_state Current minimizer state.
   * @param step State update step.
   * @param[out] updated_state Output minimizer state after applying step.
   */
  void UpdateStates(cudaStream_t stream, const MinimizerState& curr_state,
                        const dvector<float>& step,
                        MinimizerState& updated_state);

  const MinimizerOptions options_;  ///< Optimizer configuration options.

  cuDSSHandle cudss_handle_;         ///< cuDSS handle for linear solver (destroyed after solver_ so data is freed first).
  SparseLinearSolverPtr solver_;     ///< Linear solver for Gauss-Newton system.
  void* solver_handle_ = nullptr;    ///< Opaque handle passed to the linear solver.
  SparseMatrixMultiplication gemm_;  ///< Matrix multiplication for H = J^T J.
  cuSPARSEHandle cusparse_handle_;   ///< cuSPARSE handle for sparse operations.

  StateBatchOps state_ops_;  ///< Operations on state batches.

  dvector<float> residuals_;  ///< Residual vector storage.

  SparseJacobian sparse_jacobian_;  ///< Jacobian in COO (triplet) format.
  CSRSparseMatrix csr_jacobian_;    ///< Jacobian in CSR format.
  dvector<int> csr_mapping_;        ///< Mapping from triplet to CSR indices.

  CSRSparseMatrix hessian_;  ///< Approximate Hessian H = J^T J.

  dvector<float> step_;  ///< State update step vector.

  dvector<uint8_t> buffer_;  ///< Temporary buffer for sparse operations.

  profiler::Domain profiler_domain_{
      "GaussNewtonMinimizer"};  ///< Profiling domain.
};

}  // namespace cunls
