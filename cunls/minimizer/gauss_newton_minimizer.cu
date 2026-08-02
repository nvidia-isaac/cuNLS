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

#include <numeric>
#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/residual_batch.h"
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
GaussNewtonMinimizer::GaussNewtonMinimizer(const MinimizerOptions &options)
    : options_(options),
      solver_(CreateCSRSparseLinearSolver(options_.sparse_linear_solver_type,
                                          options_.sparse_linear_solver_config)) {
  if (options_.disable_safety_checks) {
    solver_->DisableSafetyChecks();
  }
}
/**
 * @brief Initializes the residual vector to the correct size.
 *
 * Computes the total number of residuals across all residual batches in the
 * problem and resizes the residual vector accordingly.
 *
 * @param problem The optimization problem.
 * @param[out] residuals Residual vector to initialize.
 */
void InitializeResiduals(const Problem &problem, dvector<float> &residuals) {
  size_t residuals_size = 0;
  const auto &residual_batches = problem.GetResidualBatches();
  for (const auto &rb : residual_batches) {
    const auto &factor_batch = rb.GetFactorBatch();
    residuals_size += factor_batch->NumFactors() * factor_batch->ResidualsSize();
  }
  if (residuals.size() != residuals_size) {
    residuals.resize(residuals_size);
  }
}

void GaussNewtonMinimizer::ResizeFactorJacobians() {
  size_t num_floats = normal_equations_.JacobianValuesSize();
  if (factor_jacobians_.size() != num_floats) {
    factor_jacobians_.resize(num_floats);
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
void GaussNewtonMinimizer::ComputeCostAsync(cudaStream_t stream, const Problem &problem,
                                            const MinimizerState &minimizer_state,
                                            float *d_cost_out) {
  auto range = profiler_domain_.CreateDomainRange("ComputeCost");
  const auto &state_pointers = minimizer_state.GetStatePointers();

  const auto &residual_batches = problem.GetResidualBatches();
  size_t max_residual_dim = 0;
  size_t total_num_cost_elements = 0;
  size_t max_workspace_floats = 0;
  for (const auto &rb : residual_batches) {
    const auto &factor_batch = rb.GetFactorBatch();
    size_t n = factor_batch->NumFactors();
    total_num_cost_elements += n;
    max_residual_dim = std::max(n * factor_batch->ResidualsSize(), max_residual_dim);
    max_workspace_floats = std::max(max_workspace_floats, ResidualBatchWorkspaceNumFloats(n));
  }

  size_t buffer_floats = total_num_cost_elements + max_residual_dim + max_workspace_floats;
  if (buffer_.size() < buffer_floats * sizeof(float)) {
    buffer_.resize(buffer_floats * sizeof(float));
  }
  float *cost_ptr = reinterpret_cast<float *>(buffer_.data());
  float *residuals_ptr = reinterpret_cast<float *>(buffer_.data()) + total_num_cost_elements;
  float *workspace_ptr = residuals_ptr + max_residual_dim;

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto &rb = residual_batches[i];
    auto ptrs = state_pointers[i].data();
    rb.Evaluate(stream, workspace_ptr, residuals_ptr, ptrs, cost_ptr, nullptr);
    const auto &factor_batch = rb.GetFactorBatch();
    cost_ptr += factor_batch->NumFactors();
  }

  if (total_num_cost_elements > 0) {
    size_t partials_needed = ReducePartialCount(total_num_cost_elements);
    if (d_reduce_partials_.size() < partials_needed) {
      d_reduce_partials_.resize(partials_needed);
    }
    ReduceSumToDevice(stream, reinterpret_cast<float *>(buffer_.data()), total_num_cost_elements,
                      d_cost_out, d_reduce_partials_.data());
  } else {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_cost_out, 0, sizeof(float), stream));
  }
}

float GaussNewtonMinimizer::ComputeCost(cudaStream_t stream, const Problem &problem,
                                        const MinimizerState &minimizer_state) {
  if (d_scalars_.size() < 1) d_scalars_.resize(1);
  if (h_scalars_.size() < 1) h_scalars_.resize(1);

  ComputeCostAsync(stream, problem, minimizer_state, d_scalars_.data());

  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(h_scalars_.data(), d_scalars_.data(), sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return h_scalars_[0];
}

/**
 * @brief Computes residuals and Jacobian for the current states.
 *
 * Evaluates all factor batches to compute residual values and their Jacobian
 * matrices.  Both are dense per-factor blocks, concatenated across batches.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 * @param[out] residuals Output residual vector.
 * @param[out] jacobians Output per-factor dense Jacobian blocks.
 */
void ComputeResidualAndJacobian(cudaStream_t stream, const Problem &problem,
                                const MinimizerState &minimizer_state, dvector<float> &residuals,
                                PerFactorJacobians &jacobians, dvector<uint8_t> &buffer) {
  const auto &state_pointers = minimizer_state.GetStatePointers();
  const auto &residual_batches = problem.GetResidualBatches();
  size_t max_n = 0;
  for (const auto &rb : residual_batches) {
    max_n = std::max(max_n, rb.GetFactorBatch()->NumFactors());
  }
  size_t ws_floats = ResidualBatchWorkspaceNumFloats(max_n);
  if (buffer.size() < ws_floats * sizeof(float)) {
    buffer.resize(ws_floats * sizeof(float));
  }
  float *workspace_ptr = reinterpret_cast<float *>(buffer.data());

  float *residuals_ptr = residuals.data();
  float *jacobian_ptr = jacobians.data();

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto &rb = residual_batches[i];
    auto ptrs = state_pointers[i].data();

    rb.Evaluate(stream, workspace_ptr, residuals_ptr, ptrs, nullptr, jacobian_ptr);

    const auto &factor_batch = rb.GetFactorBatch();
    size_t num_residuals = factor_batch->NumFactors() * factor_batch->ResidualsSize();
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
void GaussNewtonMinimizer::ApplyColumnScalingToNormalEquations(cudaStream_t stream,
                                                               dvector<float> &rhs) {
  if (options_.column_scaling == ColumnScaling::None) {
    return;
  }

  // H_jj = ||J_{:,j}||^2, so scaling by the Hessian diagonal is the same as
  // scaling by the Jacobian column norms -- and no global Jacobian exists to
  // read the latter from.
  normal_equations_.ExtractHessianDiagonal(stream, column_scale_);
  InvertSqrtWithFloorInPlace(stream, column_scale_);

  normal_equations_.ScaleLhsSymmetric(stream, column_scale_);
  ElementwiseMultiplyInPlace(stream, rhs.data(), column_scale_.data(), rhs.size());
}

void GaussNewtonMinimizer::MapScaledLinearSolutionToTangentStep(cudaStream_t stream,
                                                                dvector<float> &step) {
  if (options_.column_scaling == ColumnScaling::None || step.empty()) {
    return;
  }
  ElementwiseMultiplyInPlace(stream, step.data(), column_scale_.data(), step.size());
}

void GaussNewtonMinimizer::BuildSystem(cudaStream_t stream, const Problem &problem,
                                       const MinimizerState &minimizer_state) {
  auto range = profiler_domain_.CreateDomainRange("BuildSystem");
  ComputeResidualAndJacobian(stream, problem, minimizer_state, residuals_, factor_jacobians_,
                             buffer_);
  normal_equations_.Assemble(stream, problem, factor_jacobians_.data(), residuals_.data(),
                             rhs_work_);
  ApplyColumnScalingToNormalEquations(stream, rhs_work_);
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
void GaussNewtonMinimizer::UpdateStates(cudaStream_t stream, const MinimizerState &curr_state,
                                        const dvector<float> &step, MinimizerState &updated_state) {
  auto range = profiler_domain_.CreateDomainRange("UpdateStates");
  std::vector<const float *> x_ptrs;
  for (const auto &params : curr_state.GetStates()) {
    x_ptrs.push_back(params.data());
  }

  std::vector<float *> x_plus_delta_ptrs;
  for (auto &params : updated_state.GetStates()) {
    x_plus_delta_ptrs.push_back(params.data());
  }

  state_ops_.Plus(stream, x_ptrs, step, x_plus_delta_ptrs);
}

/**
 * @brief Initializes internal data structures for optimization.
 *
 * Prepares residual vectors, Jacobian structures, and state operations for
 * the given problem. Also precomputes the Hessian (J^T J) sparsity structure
 * to avoid recomputing it on each iteration.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem to initialize for.
 */
void GaussNewtonMinimizer::Initialize(cudaStream_t stream, Problem &problem) {
  auto range = profiler_domain_.CreateDomainRange("Initialize");
  InitializeResiduals(problem, residuals_);
  state_ops_.Preprocess(stream, problem.GetStateBatches());

  // The Hessian pattern comes straight from the factor graph; no global
  // Jacobian is involved.
  normal_equations_.Initialize(stream, problem, static_cast<int>(state_ops_.NumReducedStates()),
                               solver_->SupportsBlockStorage());
  ResizeFactorJacobians();
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
bool GaussNewtonMinimizer::CheckConvergence(cudaStream_t stream, float updated_cost,
                                            float current_cost, const dvector<float> &step,
                                            float &step_quality) {
  auto range = profiler_domain_.CreateDomainRange("CheckConvergence");

  if (d_scalars_.size() < 1) d_scalars_.resize(1);
  if (h_scalars_.size() < 1) h_scalars_.resize(1);
  size_t partials_needed = ReducePartialCount(step.size());
  if (d_reduce_partials_.size() < partials_needed) {
    d_reduce_partials_.resize(partials_needed);
  }

  ComputeSquaredStepAsync(stream, step, d_scalars_.data(), d_reduce_partials_.data());
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(h_scalars_.data(), d_scalars_.data(), sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  float squared_step = h_scalars_[0];
  LogMessage("Squared step = {}", squared_step);
  step_quality = updated_cost / current_cost;

  LogMessage("Step quality = {}", step_quality);

  if (squared_step < options_.state_tolerance || updated_cost < options_.cost_tolerance ||
      step_quality >= 1) {
    return true;
  }

  return false;
}

bool GaussNewtonMinimizer::EvaluateAndCheckConvergence(cudaStream_t stream, const Problem &problem,
                                                       const MinimizerState &updated_state,
                                                       float current_cost,
                                                       const dvector<float> &step,
                                                       float &updated_cost, float &step_quality) {
  auto range = profiler_domain_.CreateDomainRange("EvaluateAndCheckConvergence");

  constexpr size_t kSlots = 2;  // [0] = cost, [1] = squared_step
  if (d_scalars_.size() < kSlots) d_scalars_.resize(kSlots);
  if (h_scalars_.size() < kSlots) h_scalars_.resize(kSlots);

  // Enqueue cost reduction (async, result stays on device)
  ComputeCostAsync(stream, problem, updated_state, d_scalars_.data());

  // Enqueue squared step reduction (async, result stays on device)
  size_t partials_needed = ReducePartialCount(step.size());
  if (d_reduce_partials_.size() < partials_needed) {
    d_reduce_partials_.resize(partials_needed);
  }
  ComputeSquaredStepAsync(stream, step, d_scalars_.data() + 1, d_reduce_partials_.data());

  // Single D2H + single sync
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(h_scalars_.data(), d_scalars_.data(), kSlots * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  updated_cost = h_scalars_[0];
  float squared_step = h_scalars_[1];

  LogMessage("Squared step = {}", squared_step);
  step_quality = updated_cost / current_cost;
  LogMessage("Step quality = {}", step_quality);

  return (squared_step < options_.state_tolerance || updated_cost < options_.cost_tolerance ||
          step_quality >= 1);
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
bool GaussNewtonMinimizer::RejectStep(float step_quality) { return !AcceptStep(step_quality); }

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
MinimizerSummary GaussNewtonMinimizer::Minimize(cudaStream_t stream, Problem &problem) {
  auto range = profiler_domain_.CreateDomainRange("Minimize");
  MinimizerSummary summary;

  // Initialize internal data structures
  Initialize(stream, problem);

  // No optimizable states (all constant): nothing to solve
  if (state_ops_.NumReducedStates() == 0) {
    current_state_.Recreate(stream, problem);
    summary.initial_cost = ComputeCost(stream, problem, current_state_);
    summary.final_cost = summary.initial_cost;
    LogMessage("No optimizable states; skipping solver.");
    return summary;
  }

  // Create minimizer state snapshots for current and updated states
  current_state_.Recreate(stream, problem);
  updated_state_.Recreate(stream, problem);

  // Compute initial cost
  summary.initial_cost = ComputeCost(stream, problem, current_state_);
  summary.final_cost = summary.initial_cost;
  LogMessage("Initial cost = {}", summary.initial_cost);

  // Early exit if already converged
  if (summary.initial_cost < options_.cost_tolerance) {
    return summary;
  }

  // Build initial linear system
  BuildSystem(stream, problem, current_state_);

  step_.resize(rhs_work_.size());

  {
    // Perform symbolic analysis on the requested CUDA stream.
    auto sa_range = profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    bool success = normal_equations_.InitializeSolver(stream, *solver_, problem, rhs_work_, step_);
    if (!success) {
      std::string str = "Failed to initialize linear solver";
      LogError(str);
      throw std::runtime_error(str);
    }
    // THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  // Main optimization loop
  summary.num_iterations = 0;
  size_t consecutive_rejected = 0;
  for (; summary.num_iterations < options_.max_num_iterations; summary.num_iterations++) {
    auto it_range = profiler_domain_.CreateDomainRange("Iteration");
    LogMessage("Iteration #{}", summary.num_iterations);

    summary.iteration_costs.push_back(summary.final_cost);

    {
      auto solve_range = profiler_domain_.CreateDomainRange("LinearSolve");
      bool success = normal_equations_.Solve(stream, *solver_, rhs_work_, step_);
      if (!success) {
        std::string str = "Failed to solve linear system";
        LogError(str);
        throw std::runtime_error(str);
      }
    }

    MapScaledLinearSolutionToTangentStep(stream, step_);

    UpdateStates(stream, current_state_, step_, updated_state_);

    // Fused: cost reduction + squared step in one D2H + sync
    float cost;
    float step_quality;
    bool converged = EvaluateAndCheckConvergence(stream, problem, updated_state_,
                                                 summary.final_cost, step_, cost, step_quality);

    LogMessage("Current cost = {}, updated cost = {}", summary.final_cost, cost);
    LogMessage("Current step quality = {}", step_quality);

    if (converged) {
      LogMessage("Optimization converged");
      if (cost <= summary.final_cost) {
        summary.final_cost = cost;
        current_state_.Copy(stream, updated_state_.GetStates());
      }
      break;
    }

    if (RejectStep(step_quality)) {
      LogMessage("Reject step");
      consecutive_rejected++;
      if (options_.max_consecutive_rejected_steps > 0 &&
          consecutive_rejected >= options_.max_consecutive_rejected_steps) {
        LogMessage("Converged: {} consecutive rejected steps", consecutive_rejected);
        break;
      }
    } else {
      LogMessage("Accept step");
      consecutive_rejected = 0;
      AcceptStep(step_quality);
      summary.final_cost = cost;
      current_state_.Copy(stream, updated_state_.GetStates());
    }

    BuildSystem(stream, problem, current_state_);
  };

  LogMessage("Optimization finished");

  // Copy final state values back to problem
  Copy(stream, current_state_, problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  return summary;
}
}  // namespace cunls
