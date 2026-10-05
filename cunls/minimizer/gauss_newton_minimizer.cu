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

#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/types.h"
#include "cunls/factor/constraint_factor_batch.h"
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
      solver_(CreateSparseLinearSolver(options_.sparse_linear_solver_type,
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
    residuals_size += factor_batch->NumActiveFactors() * factor_batch->ResidualsSize();
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
    size_t n = factor_batch->NumActiveFactors();
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
    CheckEvaluate(rb.Evaluate(stream, workspace_ptr, residuals_ptr, ptrs, cost_ptr, nullptr), i);
    const auto &factor_batch = rb.GetFactorBatch();
    cost_ptr += factor_batch->NumActiveFactors();
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
 * Per residual batch, either the FactorBatch's analytic Jacobian is used
 * directly (via ResidualBatch::Evaluate, which also applies any registered
 * loss function), or a finite-difference Jacobian is built via
 * `numeric_diff_builder_` from the raw residual-only evaluation, with loss
 * scaling then applied via `ResidualBatch::ApplyLoss` so both paths see
 * identical loss handling.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 * @param[out] residuals Output residual vector.
 * @param[out] jacobians Output per-factor dense Jacobian blocks.
 */
void GaussNewtonMinimizer::ComputeResidualAndJacobian(cudaStream_t stream, const Problem &problem,
                                                      const MinimizerState &minimizer_state,
                                                      dvector<float> &residuals,
                                                      PerFactorJacobians &jacobians,
                                                      dvector<uint8_t> &buffer) {
  const auto &state_pointers = minimizer_state.GetStatePointers();
  const auto &residual_batches = problem.GetResidualBatches();
  size_t max_n = 0;
  for (const auto &rb : residual_batches) {
    max_n = std::max(max_n, rb.GetFactorBatch()->NumActiveFactors());
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
    const auto &factor_batch = rb.GetFactorBatch();

    JacobianMode mode = problem.JacobianModeFor(i, options_.jacobian_mode);
    if (mode == JacobianMode::kAnalytic) {
      CheckEvaluate(rb.Evaluate(stream, workspace_ptr, residuals_ptr, ptrs, nullptr, jacobian_ptr),
                    i);
    } else {
      // Raw (pre-loss) residual + finite-difference Jacobian, then apply any
      // registered loss function to both in place -- exactly mirrors what
      // ResidualBatch::Evaluate would have done after an analytic
      // FactorBatch::Evaluate call.
      CheckEvaluate(factor_batch->Evaluate(residuals_ptr, nullptr, ptrs, stream), i);
      numeric_diff_builder_.Compute(stream, problem, i, minimizer_state, residuals_ptr,
                                    jacobian_ptr, options_.numeric_diff_options);
      if (rb.GetLossFunction() != nullptr) {
        rb.ApplyLoss(stream, workspace_ptr, residuals_ptr, nullptr, jacobian_ptr);
      }
    }

    size_t num_residuals = factor_batch->NumActiveFactors() * factor_batch->ResidualsSize();
    residuals_ptr += num_residuals;

    auto block_sizes = factor_batch->StateSizes();

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
  HoldActiveBounds(stream, minimizer_state, rhs_work_);
}

void GaussNewtonMinimizer::HoldActiveBounds(cudaStream_t stream,
                                            const MinimizerState &minimizer_state,
                                            dvector<float> &rhs) {
  if (!state_ops_.HasBounds() || rhs.empty()) return;
  // Projected Gauss-Newton: the components at a bound that the steepest
  // descent direction (rhs = -gradient, scaled by positive factors) pushes
  // outward leave the system. Their rows and columns are zeroed and the
  // diagonal is kept, so their step is exactly zero and the others solve the
  // reduced system. Only the held rows are touched.
  std::vector<const float *> x_ptrs;
  for (const auto &params : minimizer_state.GetStates()) x_ptrs.push_back(params.data());
  state_ops_.BoundMask(stream, x_ptrs, rhs, bound_mask_);
  normal_equations_.ZeroMaskedLhsRowsColumns(stream, bound_mask_);
  ElementwiseMultiplyInPlace(stream, rhs.data(), bound_mask_.data(), rhs.size());
}

std::vector<size_t> GaussNewtonMinimizer::StructureSignature(const Problem &problem) const {
  std::vector<size_t> signature;
  signature.push_back(problem.NumProblems());
  for (const StateBatch *batch : problem.GetStateBatches()) {
    signature.push_back(reinterpret_cast<size_t>(batch));
    signature.push_back(batch->NumActiveStates());
    signature.push_back(batch->NumConstStates());
  }
  const auto &residual_batches = problem.GetResidualBatches();
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    signature.push_back(reinterpret_cast<size_t>(residual_batches[i].GetFactorBatch()));
    signature.push_back(residual_batches[i].GetFactorBatch()->NumActiveFactors());
    signature.push_back(problem.NumStatePointers(i));
  }
  return signature;
}

void GaussNewtonMinimizer::SolveStep(cudaStream_t stream) {
  auto solve_range = profiler_domain_.CreateDomainRange("LinearSolve");
  auto solve = [&]() {
    if (!normal_equations_.Solve(stream, *solver_, rhs_work_, step_)) {
      std::string str = "Failed to solve linear system";
      LogError(str);
      throw std::runtime_error(str);
    }
  };
  solve();
  if (!state_ops_.HasBounds()) return;
  // Active-set refinement: a free component at a bound (its gradient points
  // inward) can still get an outward step through the coupling with the
  // others; projecting that step away would spoil the descent direction.
  // Hold such components too and solve again (rarely more than one pass). In
  // real-time mode exactly max_bound_refinements passes, without the
  // read-back of the count.
  const bool fixed = call_.fixed_iterations;
  const int max_passes = static_cast<int>(options_.max_bound_refinements);
  std::vector<const float *> x_ptrs;
  for (const auto &params : current_state_.GetStates()) x_ptrs.push_back(params.data());
  if (bound_count_.size() < 1) bound_count_.resize(1);
  for (int pass = 0; pass < max_passes; ++pass) {
    state_ops_.BoundMask(stream, x_ptrs, step_, bound_step_mask_);
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(bound_count_.data(), 0, sizeof(int), stream));
    HoldOutwardSteps(stream, bound_step_mask_, bound_mask_, bound_extra_, bound_count_.data());
    if (!fixed) {
      int count = 0;
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&count, bound_count_.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      if (count == 0) return;
      LogMessage("Bounds: {} more components held", count);
    }
    normal_equations_.ZeroMaskedLhsRowsColumns(stream, bound_extra_);
    ElementwiseMultiplyInPlace(stream, rhs_work_.data(), bound_extra_.data(), rhs_work_.size());
    solve();
  }
}

void GaussNewtonMinimizer::ProjectToBounds(cudaStream_t stream, MinimizerState &minimizer_state) {
  if (!state_ops_.HasBounds()) return;
  std::vector<float *> x_ptrs;
  for (auto &params : minimizer_state.GetStates()) x_ptrs.push_back(params.data());
  state_ops_.ProjectToBounds(stream, x_ptrs);
}

/**
 * @brief Updates states with the computed step.
 *
 * Applies the state update step to the current state, producing the
 * updated state. Uses state operations that respect state
 * manifolds (e.g., for rotations, quaternions, etc.).
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
  // Cheap host-only guard (no device work): sizes within capacities,
  // connectivity covering the active factors, at least one active factor.
  problem.CheckSizes();
  // Connectivity, sizes and contents may have been rewritten since the last
  // solve: expand index tables, and re-plan numeric-diff batches whose
  // connectivity changed.
  problem.PrepareStatePointers(stream);
  for (size_t i = 0; i < problem.GetResidualBatches().size(); ++i) {
    if (problem.JacobianModeFor(i, options_.jacobian_mode) == JacobianMode::kNumeric) {
      numeric_diff_builder_.Refresh(problem, i);
    }
  }
  InitializeResiduals(problem, residuals_);
  state_ops_.Preprocess(stream, problem.GetStateBatches());

  // The Hessian pattern comes straight from the factor graph; no global
  // Jacobian is involved.
  normal_equations_.Initialize(stream, problem, static_cast<int>(state_ops_.NumReducedStates()),
                               solver_->SupportsBlockStorage());
  ResizeFactorJacobians();
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

bool GaussNewtonMinimizer::WouldRejectStep(float step_quality) const {
  return !(step_quality < 1.f);
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
MinimizerSummary GaussNewtonMinimizer::Minimize(cudaStream_t stream, Problem &problem) {
  MinimizeCallOptions call;
  call.max_num_iterations = options_.max_num_iterations;
  call.max_line_search_steps = options_.max_line_search_steps;
  return Minimize(stream, problem, call);
}

MinimizerSummary GaussNewtonMinimizer::Minimize(cudaStream_t stream, Problem &problem,
                                                const MinimizeCallOptions &call) {
  if (!call.constraints_managed) {
    for (const auto &rb : problem.GetResidualBatches()) {
      if (dynamic_cast<const ConstraintFactorBatchBase *>(rb.GetFactorBatch()) != nullptr) {
        throw std::invalid_argument(
            "GaussNewtonMinimizer / LevenbergMarquardtMinimizer: the problem has constraint "
            "factor batches; solve it with AugmentedLagrangianMinimizer");
      }
    }
  }
  call_ = call;
  auto range = profiler_domain_.CreateDomainRange("Minimize");
  MinimizerSummary summary;
  if (call_.problem_at_cap != nullptr) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(call_.problem_at_cap, 0, problem.NumProblems() * sizeof(int), stream));
  }

  // Structure setup, unless the caller vouches for an unchanged structure and
  // the sizes agree.
  std::vector<size_t> signature = StructureSignature(problem);
  const bool reuse =
      call_.reuse_structure && structure_problem_ == &problem && signature == structure_signature_;
  if (!reuse) {
    structure_problem_ = nullptr;  // invalid until the setup below completes
    solver_ready_ = false;
    partition_ready_ = false;
    Initialize(stream, problem);
  }
  BeginCall();

  // No optimizable states (all constant): nothing to solve
  if (state_ops_.NumReducedStates() == 0) {
    current_state_.Recreate(stream, problem);
    summary.initial_cost = ComputeCost(stream, problem, current_state_);
    summary.final_cost = summary.initial_cost;
    LogMessage("No optimizable states; skipping solver.");
    return summary;
  }

  if (problem.NumProblems() > 1 || call_.fixed_iterations) {
    MinimizerSummary batched = MinimizeBatched(stream, problem);
    structure_problem_ = &problem;
    structure_signature_ = std::move(signature);
    return batched;
  }

  // Create minimizer state snapshots for current and updated states
  current_state_.Recreate(stream, problem);
  updated_state_.Recreate(stream, problem);
  ProjectToBounds(stream, current_state_);

  // Compute initial cost
  summary.initial_cost = ComputeCost(stream, problem, current_state_);
  summary.final_cost = summary.initial_cost;
  LogMessage("Initial cost = {}", summary.initial_cost);

  // Early exit if already converged; the problem still gets the states
  // projected onto their bounds.
  if (summary.initial_cost < options_.cost_tolerance) {
    if (state_ops_.HasBounds()) {
      Copy(stream, current_state_, problem);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    return summary;
  }

  // Build initial linear system
  BuildSystem(stream, problem, current_state_);

  step_.resize(rhs_work_.size());

  if (!solver_ready_) {
    // Perform symbolic analysis on the requested CUDA stream.
    auto sa_range = profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    bool success = normal_equations_.InitializeSolver(stream, *solver_, problem, rhs_work_, step_);
    if (!success) {
      std::string str = "Failed to initialize linear solver";
      LogError(str);
      throw std::runtime_error(str);
    }
    solver_ready_ = true;
  }
  structure_problem_ = &problem;
  structure_signature_ = std::move(signature);

  // Main optimization loop
  summary.num_iterations = 0;
  size_t consecutive_rejected = 0;
  bool stopped = false;  // left the loop on convergence, not at the iteration cap
  for (; summary.num_iterations < call_.max_num_iterations; summary.num_iterations++) {
    auto it_range = profiler_domain_.CreateDomainRange("Iteration");
    LogMessage("Iteration #{}", summary.num_iterations);

    summary.iteration_costs.push_back(summary.final_cost);

    SolveStep(stream);

    MapScaledLinearSolutionToTangentStep(stream, step_);

    UpdateStates(stream, current_state_, step_, updated_state_);

    // Fused: cost reduction + squared step in one D2H + sync
    float cost;
    float step_quality;
    bool converged = EvaluateAndCheckConvergence(stream, problem, updated_state_,
                                                 summary.final_cost, step_, cost, step_quality);

    LogMessage("Current cost = {}, updated cost = {}", summary.final_cost, cost);
    LogMessage("Current step quality = {}", step_quality);

    // Line search: a step that does not decrease the cost is shortened along
    // the same direction; the first shorter step that decreases it is taken.
    // With line search, any step that decreases the cost is taken, also one the
    // step-quality rule would reject; the damping is left as it is (raising it
    // on every such step makes Levenberg-Marquardt crawl in curved valleys).
    if (call_.max_line_search_steps > 0) {
      float shortened_cost = cost;
      bool shortened = false;
      for (size_t k = 0; k < call_.max_line_search_steps && !(shortened_cost < summary.final_cost);
           ++k) {
        ScaleInPlace(stream, step_.data(), 0.5f, step_.size());
        UpdateStates(stream, current_state_, step_, updated_state_);
        shortened_cost = ComputeCost(stream, problem, updated_state_);
        shortened = true;
      }
      if (shortened_cost < summary.final_cost && (shortened || WouldRejectStep(step_quality))) {
        LogMessage("Line search: step taken, cost = {}", shortened_cost);
        consecutive_rejected = 0;
        summary.final_cost = shortened_cost;
        current_state_.Copy(stream, updated_state_.GetStates());
        BuildSystem(stream, problem, current_state_);
        continue;
      }
    }

    if (converged) {
      LogMessage("Optimization converged");
      if (cost <= summary.final_cost) {
        summary.final_cost = cost;
        current_state_.Copy(stream, updated_state_.GetStates());
      }
      stopped = true;
      break;
    }

    if (RejectStep(step_quality)) {
      LogMessage("Reject step");
      consecutive_rejected++;
      if (options_.max_consecutive_rejected_steps > 0 &&
          consecutive_rejected >= options_.max_consecutive_rejected_steps) {
        LogMessage("Converged: {} consecutive rejected steps", consecutive_rejected);
        stopped = true;
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
  if (!stopped && call_.problem_at_cap != nullptr) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(call_.problem_at_cap, 1, sizeof(int), stream));
  }

  // Copy final state values back to problem
  Copy(stream, current_state_, problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  return summary;
}

BatchedStepControlParams GaussNewtonMinimizer::BatchedParams() const {
  BatchedStepControlParams params;
  params.levenberg_marquardt = false;
  params.state_tolerance = options_.state_tolerance;
  params.cost_tolerance = options_.cost_tolerance;
  params.max_consecutive_rejected_steps = static_cast<int>(options_.max_consecutive_rejected_steps);
  return params;
}

/**
 * @brief Minimize with per-subproblem step control.
 *
 * The same iteration as Minimize(), with every decision taken per subproblem
 * on the device: each subproblem accepts or rejects its own step (its states
 * are copied from the trial state only when accepted), keeps its own damping
 * and leaves the iteration at its own convergence. The linear system is solved
 * for all subproblems together; converged subproblems keep their states. One
 * 2-float read-back per iteration: the total cost and the number of active
 * subproblems; the loop ends when none is active.
 */
MinimizerSummary GaussNewtonMinimizer::MinimizeBatched(cudaStream_t stream, Problem &problem) {
  MinimizerSummary summary;
  current_state_.Recreate(stream, problem);
  updated_state_.Recreate(stream, problem);
  ProjectToBounds(stream, current_state_);
  if (!partition_ready_) {
    partition_.Build(stream, problem, current_state_, state_ops_.NumReducedStates());
    partition_ready_ = true;
  }
  batched_ = true;
  struct Reset {
    bool &flag;
    ~Reset() { flag = false; }
  } reset{batched_};

  if (d_scalars_.size() < 2) d_scalars_.resize(2);
  if (h_scalars_.size() < 2) h_scalars_.resize(2);
  auto read_back = [&]() {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(h_scalars_.data(), d_scalars_.data(), 2 * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    return std::make_pair(h_scalars_[0], static_cast<size_t>(h_scalars_[1]));
  };
  // ComputeCostAsync leaves the per-factor costs at the start of buffer_.
  auto factor_costs = [&]() { return reinterpret_cast<const float *>(buffer_.data()); };

  const bool fixed = call_.fixed_iterations;
  ComputeCostAsync(stream, problem, current_state_, d_scalars_.data());
  partition_.AccumulateFactorCosts(stream, factor_costs(), partition_.Cost());
  partition_.InitStepControl(stream, options_.cost_tolerance, BatchedInitialLambda(),
                             d_scalars_.data());
  float cost = std::numeric_limits<float>::quiet_NaN();
  size_t active = partition_.NumProblems();
  if (!fixed) {
    std::tie(cost, active) = read_back();
    LogMessage("Initial cost = {}, {} of {} subproblems active", cost, active,
               partition_.NumProblems());
  }
  summary.initial_cost = cost;
  summary.final_cost = cost;
  if (active == 0) {
    // Nothing to iterate; the problem still gets the states projected onto
    // their bounds (the read-back above already synchronized the stream).
    if (state_ops_.HasBounds()) {
      Copy(stream, current_state_, problem);
      if (!fixed) THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    return summary;
  }

  BuildSystem(stream, problem, current_state_);
  step_.resize(rhs_work_.size());
  if (!solver_ready_) {
    auto sa_range = profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    if (!normal_equations_.InitializeSolver(stream, *solver_, problem, rhs_work_, step_)) {
      std::string str = "Failed to initialize linear solver";
      LogError(str);
      throw std::runtime_error(str);
    }
    solver_ready_ = true;
  }

  BatchedStepControlParams params = BatchedParams();
  params.line_search = call_.max_line_search_steps > 0;
  for (summary.num_iterations = 0; summary.num_iterations < call_.max_num_iterations;) {
    auto it_range = profiler_domain_.CreateDomainRange("Iteration");
    summary.iteration_costs.push_back(summary.final_cost);
    SolveStep(stream);
    MapScaledLinearSolutionToTangentStep(stream, step_);
    UpdateStates(stream, current_state_, step_, updated_state_);

    partition_.ResetAccumulators(stream);
    ComputeCostAsync(stream, problem, updated_state_, d_scalars_.data());
    partition_.AccumulateFactorCosts(stream, factor_costs(), partition_.NewCost());
    // Line search per subproblem: halve the steps that do not decrease their
    // subproblem's cost, re-evaluate, until none is left or the budget is spent.
    // Real time: every step of the budget, predicated on the device (a
    // subproblem that needs no shortening gets a scale of 1).
    for (size_t k = 0; k < call_.max_line_search_steps; ++k) {
      partition_.MarkLineSearch(stream, d_scalars_.data());
      if (!fixed) {
        float num_shortened = 0.f;
        THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&num_shortened, d_scalars_.data(), sizeof(float),
                                            cudaMemcpyDeviceToHost, stream));
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (num_shortened == 0.f) break;
      }
      partition_.ScaleRows(stream, partition_.StepScale(), step_.data(), step_.data());
      UpdateStates(stream, current_state_, step_, updated_state_);
      ComputeCostAsync(stream, problem, updated_state_, d_scalars_.data());
      partition_.AccumulateFactorCosts(stream, factor_costs(), partition_.NewCost());
    }
    partition_.AccumulateRows(stream, step_.data(), nullptr, nullptr, partition_.StepSquared());
    AccumulatePredictedReduction(stream);
    partition_.StepControl(stream, params, d_scalars_.data());
    partition_.CopyAccepted(stream, problem, updated_state_, current_state_);
    summary.num_iterations++;
    if (!fixed) {
      std::tie(cost, active) = read_back();
      summary.final_cost = cost;
      LogMessage("Iteration #{}: cost = {}, {} subproblems active", summary.num_iterations, cost,
                 active);
      if (active == 0) break;
    }
    if (summary.num_iterations < call_.max_num_iterations)
      BuildSystem(stream, problem, current_state_);
  }

  if (call_.problem_at_cap != nullptr) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(call_.problem_at_cap, partition_.Active(),
                                        partition_.NumProblems() * sizeof(int),
                                        cudaMemcpyDeviceToDevice, stream));
  }
  Copy(stream, current_state_, problem);
  if (!fixed) THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return summary;
}
}  // namespace cunls
