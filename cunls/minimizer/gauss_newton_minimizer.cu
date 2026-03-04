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

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <numeric>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

/**
 * @brief Constructs a Gauss-Newton optimizer.
 *
 * Initializes the optimizer with the provided options and creates the
 * appropriate sparse linear solver based on the solver type specified
 * in the options.
 *
 * @param options Configuration options for the optimizer.
 */
GaussNewtonMinimizer::GaussNewtonMinimizer(const MinimizerOptions& options)
    : options_(options),
      solver_(
          CreateCSRSparseLinearSolver(options_.sparse_linear_solver_type,
                                      options_.sparse_linear_solver_config)) {}
/**
 * @brief Initializes the residual vector to the correct size.
 *
 * Computes the total number of residuals across all residual batches in the
 * problem and resizes the residual vector accordingly.
 *
 * @param problem The optimization problem.
 * @param[out] residuals Residual vector to initialize.
 */
void InitializeResiduals(const Problem& problem, dvector<float>& residuals) {
  size_t residuals_size = 0;
  const auto& residual_batches = problem.GetResidualBatches();
  for (const auto& rb : residual_batches) {
    const auto& factor_batch = rb.GetFactorBatch();
    residuals_size += factor_batch->NumFactors() * factor_batch->ResidualsSize();
  }
  if (residuals.size() != residuals_size) {
    residuals.resize(residuals_size);
  }
}

/**
 * @brief Initializes the Jacobian structure in COO (triplet) format.
 *
 * Builds the sparse structure (row and column indices) for the Jacobian matrix
 * and allocates storage for the Jacobian values. The structure is built based
 * on the problem's factors and state batches.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param[out] coo_jacobian Jacobian in COO format to initialize.
 */
void InitializeJacobian(cudaStream_t stream, const Problem& problem,
                        SparseJacobian& coo_jacobian) {
  BuildTripletSparseStructure(stream, problem, coo_jacobian.structure);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  size_t jacobian_size = coo_jacobian.structure.col_ids.size();
  if (coo_jacobian.values.size() != jacobian_size) {
    coo_jacobian.values.resize(jacobian_size);
  }
}

/**
 * @brief Computes the total cost for the current state values.
 *
 * Evaluates all factors in the problem with the given minimizer state
 * and sums the resulting costs. Each factor batch is evaluated
 * independently, and the costs are accumulated.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 * @return Total cost (sum of squared residuals from all factors).
 */
float GaussNewtonMinimizer::ComputeCost(cudaStream_t stream,
                                        const Problem& problem,
                                        const MinimizerState& minimizer_state) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  const auto& state_pointers = minimizer_state.GetStatePointers();

  const auto& residual_batches = problem.GetResidualBatches();
  size_t max_residual_dim = 0;
  size_t total_num_cost_elements = 0;
  for (const auto& rb : residual_batches) {
    const auto& factor_batch = rb.GetFactorBatch();
    size_t n = factor_batch->NumFactors();
    total_num_cost_elements += n;
    max_residual_dim = std::max(
        n * factor_batch->ResidualsSize(), max_residual_dim);
  }

  size_t buffer_floats = total_num_cost_elements + max_residual_dim;
  if (buffer_.size() < buffer_floats * sizeof(float)) {
    buffer_.resize(buffer_floats * sizeof(float));
  }
  float* cost_ptr = reinterpret_cast<float*>(buffer_.data());
  float* residuals_ptr = reinterpret_cast<float*>(buffer_.data()) + total_num_cost_elements;

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto& rb = residual_batches[i];
    auto ptrs = state_pointers[i].data();
    rb.Evaluate(cost_ptr, residuals_ptr, nullptr, ptrs, stream);
    const auto& factor_batch = rb.GetFactorBatch();
    cost_ptr += factor_batch->NumFactors();
  }

  float total_cost = 0;
  if (total_num_cost_elements > 0) {
    thrust::device_ptr<float> cost_vec_ptr(reinterpret_cast<float*>(buffer_.data()));
    total_cost = thrust::reduce(stream_policy, cost_vec_ptr,
                               cost_vec_ptr + total_num_cost_elements);
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return total_cost;
}

/**
 * @brief Computes residuals and Jacobian for the current states.
 *
 * Evaluates all factor batches to compute residual values and their
 * Jacobian matrices. The residuals are stored in a dense vector, and the
 * Jacobian is stored in COO (triplet) sparse format.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 * @param[out] residuals Output residual vector.
 * @param[out] coo_jacobian Output Jacobian in COO format.
 */
void ComputeResidualAndJacobian(cudaStream_t stream, const Problem& problem,
                                const MinimizerState& minimizer_state,
                                dvector<float>& residuals,
                                SparseJacobian& coo_jacobian) {
  const auto& state_pointers = minimizer_state.GetStatePointers();
  const auto& residual_batches = problem.GetResidualBatches();
  float* residuals_ptr = residuals.data();
  float* jacobian_ptr = coo_jacobian.values.data();

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto& rb = residual_batches[i];
    auto ptrs = state_pointers[i].data();

    rb.Evaluate(nullptr, residuals_ptr, jacobian_ptr, ptrs, stream);

    const auto& factor_batch = rb.GetFactorBatch();
    size_t num_residuals =
        factor_batch->NumFactors() * factor_batch->ResidualsSize();
    residuals_ptr += num_residuals;

    auto block_sizes = factor_batch->StateBlockSizes();

    size_t params = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
    jacobian_ptr += num_residuals * params;
  }
}

/**
 * @brief Builds the Gauss-Newton linear system J^T J dx = -J^T r.
 *
 * This method:
 * 1. Computes residuals and Jacobian
 * 2. Converts Jacobian from COO to CSR format
 * 3. Computes the approximate Hessian values H = J^T J (structure precomputed)
 * 4. Computes the right-hand side -J^T r
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 * @param[out] lhs Output left-hand side matrix (H = J^T J).
 * @param[out] rhs Output right-hand side vector (-J^T r).
 */
void GaussNewtonMinimizer::BuildSystem(cudaStream_t stream,
                                       const Problem& problem,
                                       const MinimizerState& minimizer_state,
                                       CSRSparseMatrix& lhs,
                                       dvector<float>& rhs) {
  auto range = profiler_domain_.CreateDomainRange("BuildSystem");
  ComputeResidualAndJacobian(stream, problem, minimizer_state, residuals_,
                             sparse_jacobian_);

  // Copy values from triplet Jacobian to precomputed CSR structure using
  // mapping
  ConvertTripletToCSRValues(stream, sparse_jacobian_, csr_mapping_,
                            csr_jacobian_);

  gemm_.ComputeSquaredMatrix(stream, csr_jacobian_, hessian_);
  CopyCSRSparseMatrix(stream, hessian_, lhs);
  auto handle = cusparse_handle_.GetHandle(stream);
  ComputeRHS(stream, handle, csr_jacobian_, residuals_, rhs, buffer_);
}

/**
 * @brief Updates states with the computed step.
 *
 * Applies the state update step to the current state, producing the
 * updated state. Uses state block operations that respect state
 * block manifolds (e.g., for rotations, quaternions, etc.).
 *
 * @param stream CUDA stream for GPU operations.
 * @param curr_state Current minimizer state.
 * @param step State update step vector.
 * @param[out] updated_state Output minimizer state after applying step.
 */
void GaussNewtonMinimizer::UpdateStates(cudaStream_t stream,
                                        const MinimizerState& curr_state,
                                        const dvector<float>& step,
                                        MinimizerState& updated_state) {
  auto range = profiler_domain_.CreateDomainRange("UpdateStates");
  std::vector<const float*> x_ptrs;
  for (const auto& params : curr_state.GetStates()) {
    x_ptrs.push_back(params.data());
  }

  std::vector<float*> x_plus_delta_ptrs;
  for (auto& params : updated_state.GetStates()) {
    x_plus_delta_ptrs.push_back(params.data());
  }

  state_ops_.Plus(stream, x_ptrs, step, x_plus_delta_ptrs);
}

/**
 * @brief Initializes internal data structures for optimization.
 *
 * Prepares residual vectors, Jacobian structures, and state operations
 * for the given problem. Also precomputes the Hessian (J^T J) sparsity
 * structure to avoid recomputing it on each iteration.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem to initialize for.
 */
void GaussNewtonMinimizer::Initialize(cudaStream_t stream,
                                      const Problem& problem) {
  auto range = profiler_domain_.CreateDomainRange("Initialize");
  InitializeResiduals(problem, residuals_);
  InitializeJacobian(stream, problem, sparse_jacobian_);
  state_ops_.Preprocess(stream, problem.GetStateBatches());

  // Initialize the cuDSS handle for the linear solver
  solver_handle_ = cudss_handle_.GetHandle(stream);

  // Convert Jacobian triplet structure to CSR once. The structure doesn't
  // change across iterations; only values are updated via the mapping.
  auto handle = cusparse_handle_.GetHandle(stream);

  {
    auto r1 =
        profiler_domain_.CreateDomainRange("ConvertTripletStructureToCSR");
    ConvertTripletStructureToCSR(stream, handle, sparse_jacobian_.structure,
                                 csr_jacobian_, csr_mapping_, buffer_);
  }

  gemm_.Initialize(stream, csr_jacobian_, hessian_);
}

/**
 * @brief Checks if convergence criteria are satisfied.
 *
 * Convergence is determined by:
 * 1. Step size: squared step norm < state_tolerance
 * 2. Cost reduction: updated_cost < cost_tolerance
 * 3. Step quality: step_quality >= 1.0 (cost increased)
 *
 * Also computes step_quality = updated_cost / current_cost as a metric for
 * step acceptance/rejection.
 *
 * @param stream CUDA stream for GPU operations.
 * @param updated_cost Cost after applying the step.
 * @param current_cost Cost before applying the step.
 * @param step State update step vector.
 * @param[out] step_quality Output step quality metric (updated_cost /
 * current_cost).
 * @return True if converged, false otherwise.
 */
bool GaussNewtonMinimizer::CheckConvergence(cudaStream_t stream,
                                            float updated_cost,
                                            float current_cost,
                                            const dvector<float>& step,
                                            float& step_quality) {
  auto range = profiler_domain_.CreateDomainRange("CheckConvergence");
  float squared_step = ComputeSquaredStep(stream, step);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  LogMessage("Squared step = {}", squared_step);
  step_quality = updated_cost / current_cost;

  LogMessage("Step quality = {}", step_quality);

  if (squared_step < options_.state_tolerance ||
      updated_cost < options_.cost_tolerance || step_quality >= 1) {
    return true;
  }

  return false;
}

/**
 * @brief Determines if a step should be accepted.
 *
 * A step is accepted if it reduces the cost, i.e., step_quality < 1.0.
 *
 * @param step_quality Ratio of updated cost to current cost.
 * @return True if step should be accepted (cost reduced), false otherwise.
 */
bool GaussNewtonMinimizer::AcceptStep(float step_quality) {
  return step_quality < 1.f;  // Cost was reduced
}

/**
 * @brief Determines if a step should be rejected.
 *
 * A step is rejected if it increases the cost, i.e., step_quality >= 1.0.
 *
 * @param step_quality Ratio of updated cost to current cost.
 * @return True if step should be rejected (cost increased), false otherwise.
 */
bool GaussNewtonMinimizer::RejectStep(float step_quality) {
  return !AcceptStep(step_quality);
}

/**
 * @brief Solves the optimization problem using the Gauss-Newton algorithm.
 *
 * The main optimization loop:
 * 1. Initialize data structures
 * 2. Compute initial cost
 * 3. For each iteration:
 *    a. Build linear system J^T J dx = -J^T r
 *    b. Solve for step dx
 *    c. Update states: x_new = x + dx
 *    d. Compute new cost
 *    e. Check convergence (step norm, cost, predicted reduction)
 *    f. Accept or reject step; if max_consecutive_rejected_steps
 *       consecutive rejections occur, declare convergence
 * 4. Copy final state values back to problem
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem to solve. State values are
 *                modified in-place during optimization.
 * @return Summary containing iteration count and cost statistics.
 */
MinimizerSummary GaussNewtonMinimizer::Minimize(cudaStream_t stream,
                                                Problem& problem) {
  auto range = profiler_domain_.CreateDomainRange("Minimize");
  MinimizerSummary summary;

  // Initialize internal data structures
  Initialize(stream, problem);

  // No optimizable states (all constant): nothing to solve
  if (state_ops_.NumReducedStates() == 0) {
    MinimizerState current_state(stream, problem);
    summary.initial_cost = ComputeCost(stream, problem, current_state);
    summary.final_cost = summary.initial_cost;
    LogMessage("No optimizable states; skipping solver.");
    return summary;
  }

  // Create minimizer state snapshots for current and updated states
  MinimizerState current_state(stream, problem);
  MinimizerState updated_state(stream, problem);

  // Compute initial cost
  summary.initial_cost = ComputeCost(stream, problem, current_state);
  summary.final_cost = summary.initial_cost;
  LogMessage("Initial cost = {}", summary.initial_cost);

  // Early exit if already converged
  if (summary.initial_cost < options_.cost_tolerance) {
    return summary;
  }

  // Build initial linear system
  CSRSparseMatrix lhs;
  dvector<float> rhs;
  BuildSystem(stream, problem, current_state, lhs, rhs);

  step_.resize(rhs.size());

  {
    // Perform symbolic analysis (solver_handle_ set in Initialize)
    auto sa_range =
        profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    bool success = solver_->Initialize(solver_handle_, lhs, rhs, step_);
    if (!success) {
      std::string str = "Failed to initialize linear solver";
      LogError(str);
      throw std::runtime_error(str);
    }
  }

  // Main optimization loop
  summary.num_iterations = 0;
  size_t consecutive_rejected = 0;
  for (; summary.num_iterations < options_.max_num_iterations;
       summary.num_iterations++) {
    auto it_range = profiler_domain_.CreateDomainRange("Iteration");
    LogMessage("Iteration #{}", summary.num_iterations);

    summary.iteration_costs.push_back(summary.final_cost);

    {
      auto solve_range = profiler_domain_.CreateDomainRange("Solve");
      // Solve linear system: J^T J dx = -J^T r
      bool success = solver_->Solve(solver_handle_, lhs, rhs, step_);
      if (!success) {
        std::string str = "Failed to solve linear system";
        LogError(str);
        throw std::runtime_error(str);
      }
    }

    // Apply step: x_new = x + dx
    UpdateStates(stream, current_state, step_, updated_state);

    // Evaluate cost with updated states
    float cost = ComputeCost(stream, problem, updated_state);

    LogMessage("Current cost = {}, updated cost = {}", summary.final_cost,
               cost);

    // Check convergence criteria
    float step_quality;
    bool converged =
        CheckConvergence(stream, cost, summary.final_cost, step_, step_quality);
    LogMessage("Current step quality = {}", step_quality);

    if (converged) {
      LogMessage("Optimization converged");
      if (cost <= summary.final_cost) {
        summary.final_cost = cost;
        current_state.Copy(stream, updated_state.GetStates());
      }
      break;
    }

    // Reject step if it increases cost
    if (RejectStep(step_quality)) {
      LogMessage("Reject step");
      consecutive_rejected++;
      if (options_.max_consecutive_rejected_steps > 0 &&
          consecutive_rejected >= options_.max_consecutive_rejected_steps) {
        LogMessage("Converged: {} consecutive rejected steps",
                   consecutive_rejected);
        break;
      }
    } else {
      // Accept step and update state
      LogMessage("Accept step");
      consecutive_rejected = 0;
      AcceptStep(step_quality);
      summary.final_cost = cost;
      current_state.Copy(stream, updated_state.GetStates());
    }

    // Rebuild linear system for next iteration
    BuildSystem(stream, problem, current_state, lhs, rhs);
  };

  LogMessage("Optimization finished");

  // Copy final state values back to problem
  Copy(stream, current_state, problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  return summary;
}
}  // namespace cunls