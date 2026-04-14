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

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/pinned_vector.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"
#include "cunls/minimizer/sparse_matrix_multiplier.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {

/**
 * @brief Diagonal column scaling S for the normal equations.
 *
 * When not ``None``, the linear solve uses \f$S H S \, z = S b\f$ with
 * \f$b = -J^\top r\f$, then applies \f$\Delta x = S z\f$. ``None`` leaves
 * the standard system \f$H \Delta x = b\f$ unchanged.
 */
enum class ColumnScaling {
  /** No scaling (identity S). */
  None = 0,
  /** \f$S_{ii} = 1 / \sqrt{H_{ii}}\f$ with a floor on the diagonal. */
  HessianDiagonal = 1,
  /** \f$S_{jj} = 1 / \|J_{:,j}\|_2\f$ from the CSR Jacobian. */
  JacobianColumnNorm = 2,
};

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
   * @brief Maximum number of consecutive rejected steps before declaring
   *        convergence.
   *
   * When the optimizer rejects this many steps in a row (i.e., every trial
   * step increases cost or falls below the acceptance threshold), the
   * minimizer treats the current solution as converged because it can no
   * longer make progress.  Set to 0 to disable this criterion.
   * Default: 5
   */
  size_t max_consecutive_rejected_steps = 5;

  /**
   * @brief Type of sparse linear solver to use.
   *
   * Supported options:
   *  - cuDSS: GPU-accelerated sparse direct solver via NVIDIA's cuDSS library.
   *  - DenseLDLT: Converts the CSR matrix to dense and solves via a custom
   *    CUDA pivoted LDLT factorization. Suitable for small-to-medium systems
   *    where the matrix fits in dense form.
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

  /**
   * @brief Strategy for computing the approximate Hessian J^T * J.
   *
   * - ``cuSPARSE``: uses cuSPARSE SpGEMM reuse API (transpose + multiply).
   *   Robust and well-tested; may allocate large internal work buffers.
   * - ``Fast``: fast warp-efficient CUDA kernels with bitmap-based
   *   sparsity pattern discovery. Exploits the Problem's factor layout
   *   for kernel tuning.
   *
   * Default: Fast.
   */
  SparseMatrixMultiplierType sparse_square_multiplier_type =
      SparseMatrixMultiplierType::Fast;

  /**
   * @brief Optional diagonal scaling of the GN/LM normal equations.
   *
   * See ColumnScaling. Default: None.
   */
  ColumnScaling column_scaling = ColumnScaling::None;

  /**
   * @brief Disable runtime safety checks inside the minimizer.
   *
   * When false, the minimizer enables all optional runtime validation
   * that can catch numerical problems early.  Currently this covers
   * post-factorization checks in the linear solver:
   *  - Cholesky: cuSOLVER devInfo after potrf and potrs (non-SPD or
   *    invalid-parameter detection).
   *  - QR: diagonal-of-R inspection for rank deficiency.
   *  - LDLT: in-kernel pivot and diagonal checks during factorization
   *    and solve.
   *
   * Future minimizer versions may add additional checks (e.g. NaN/Inf
   * detection in the state update, cost-increase guards).
   *
   * On failure the solver's Solve() returns false and a diagnostic is
   * emitted via LogError().  The minimizer treats a false return as a
   * fatal error and throws std::runtime_error.
   *
   * When true, every check listed above is skipped: no device-to-host
   * memcpy, no stream synchronization, and no in-kernel validation.
   * This can noticeably reduce per-iteration latency for small systems
   * but may produce silently incorrect results if the matrix is singular
   * or ill-conditioned.
   *
   * Default: true (safety checks disabled).
   */
  bool disable_safety_checks = true;
};

/**
 * @brief Gauss-Newton nonlinear least-squares optimizer.
 */
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
   * @brief Fused cost evaluation + convergence check with a single D2H + sync.
   *
   * Enqueues the cost reduction and all convergence-related reductions
   * (squared step norm, etc.) as async device kernels, then performs one
   * cudaMemcpyAsync + cudaStreamSynchronize to read all scalars at once.
   *
   * @param stream CUDA stream.
   * @param problem The optimization problem.
   * @param updated_state State after applying the step.
   * @param current_cost Cost before the step.
   * @param step Step vector.
   * @param[out] updated_cost Cost after applying the step.
   * @param[out] step_quality Step quality metric.
   * @return True if converged.
   */
  virtual bool EvaluateAndCheckConvergence(
      cudaStream_t stream, const Problem& problem,
      const MinimizerState& updated_state, float current_cost,
      const dvector<float>& step, float& updated_cost, float& step_quality);

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
  virtual void Initialize(cudaStream_t stream, Problem& problem);

  /**
   * @brief Builds the Gauss-Newton linear system.
   *
   * Computes residuals and Jacobian, converts to CSR format, and constructs
   * the linear system J^T J dx = -J^T r (or S H S z = S b with dx = S z when
   * column scaling is enabled). The member hessian_ holds the unscaled H;
   * lhs and rhs are scaled for the solver when applicable.
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

  /**
   * @brief Computes the total cost for the current state values.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param minimizer_state Current minimizer state.
   * @return Total cost (sum of per-factor costs from all residual batches).
   */
  float ComputeCost(cudaStream_t stream, const Problem& problem,
                    const MinimizerState& minimizer_state);

  /**
   * @brief Async version: enqueues cost reduction writing result to d_cost_out.
   * Caller must copy and sync to read the scalar.
   */
  void ComputeCostAsync(cudaStream_t stream, const Problem& problem,
                        const MinimizerState& minimizer_state,
                        float* d_cost_out);

 private:
  /**
   * @brief Applies diagonal column scaling to the normal-equation system.
   *
   * After the unscaled Hessian copy is in lhs and rhs holds b = -J^T r,
   * this may replace the solve target with S H S z = S b: fills column_scale_,
   * scales lhs symmetrically, and sets rhs_i *= S_i. When column_scaling is
   * None, returns immediately (hessian_ is unchanged and already separate).
   *
   * @param stream CUDA stream for GPU work.
   * @param[in,out] lhs Approximate Hessian H = J^T J (scaled in-place when enabled).
   * @param[in,out] rhs Right-hand side b (elementwise-scaled when enabled).
   */
  void ApplyColumnScalingToNormalEquations(cudaStream_t stream,
                                           CSRSparseMatrix& lhs,
                                           dvector<float>& rhs);

  /**
   * @brief Maps the scaled linear unknown z to the manifold tangent step dx = S z.
   *
   * The direct solver produces z from (S H S + damping) z = S b. This
   * multiplies step in-place by column_scale_. No-op when column scaling
   * is disabled or step is empty.
   *
   * @param stream CUDA stream for GPU work.
   * @param[in,out] step Solution vector from the linear solver; overwritten by dx.
   */
  void MapScaledLinearSolutionToTangentStep(cudaStream_t stream,
                                            dvector<float>& step);

  /**
   * @brief Builds Jacobian COO structure and resizes value buffer.
   */
  void InitializeJacobian(cudaStream_t stream, const Problem& problem);

 protected:
  const MinimizerOptions options_;  ///< Optimizer configuration options.

  SparseLinearSolverPtr solver_;     ///< Linear solver for Gauss-Newton system.
  SparseMatrixMultiplierPtr gemm_;   ///< Matrix multiplication for H = J^T J.
  cuSPARSEHandle cusparse_handle_;   ///< cuSPARSE handle for sparse operations.

  StateBatchOps state_ops_;  ///< Operations on state batches.

  dvector<float> residuals_;  ///< Residual vector storage.

  SparseJacobian sparse_jacobian_;    ///< Jacobian in COO (triplet) format.
  CSRSparseMatrix csr_jacobian_;      ///< Jacobian in CSR format.
  CSRMatrixDimensions jacobian_dims_; ///< Cached Jacobian dimensions.
  dvector<int> csr_mapping_;          ///< Mapping from triplet to CSR indices.

  CSRSparseMatrix hessian_;           ///< Approximate Hessian H = J^T J.
  CSRMatrixDimensions hessian_dims_;  ///< Cached Hessian dimensions.

  /// Diagonal S when column_scaling is enabled; size = number of tangent DOFs.
  dvector<float> column_scale_;

  dvector<float> step_;  ///< State update step vector.

  dvector<uint8_t> buffer_;  ///< Temporary buffer for sparse operations.

  /// Device staging buffer for async scalar reductions (cost, step norm, etc.).
  dvector<float> d_scalars_;
  /// Pinned host mirror for async D2H of scalar results.
  PinnedVector<float> h_scalars_;
  /// Scratch buffer for partial sums used by reduction kernels.
  dvector<float> d_reduce_partials_;

  /// Working normal-equation system; retained across Minimize calls to preserve
  /// capacity.
  CSRSparseMatrix lhs_work_;
  dvector<float> rhs_work_;
  MinimizerState current_state_;
  MinimizerState updated_state_;

  profiler::Domain profiler_domain_{
      "GaussNewtonMinimizer"};  ///< Profiling domain.
};

}  // namespace cunls
