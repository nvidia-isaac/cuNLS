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

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {
namespace {

using internal::MinimizerBounds;
using internal::MinimizerScratch;
using internal::MinimizerSystem;

/** Sizes the residual vector for the problem's active factors. */
void InitializeResiduals(const Problem &problem, dvector<float> &residuals) {
  size_t residuals_size = 0;
  for (const auto &rb : problem.GetResidualBatches()) {
    const auto &factor_batch = rb.GetFactorBatch();
    residuals_size += factor_batch->NumActiveFactors() * factor_batch->ResidualsSize();
  }
  if (residuals.size() != residuals_size) residuals.resize(residuals_size);
}

/**
 * Enqueues the per-factor costs at `state` (left at the start of
 * scratch.buffer, in residual-batch order) and, unless `d_total` is null,
 * their sum into `d_total`.
 */
void ComputeCostAsync(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
                      MinimizerScratch &scratch, float *d_total) {
  const auto &state_pointers = state.GetStatePointers();
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
  if (scratch.buffer.size() < buffer_floats * sizeof(float)) {
    scratch.buffer.resize(buffer_floats * sizeof(float));
  }
  float *costs = reinterpret_cast<float *>(scratch.buffer.data());
  float *cost_ptr = costs;
  float *residuals_ptr = costs + total_num_cost_elements;
  float *workspace_ptr = residuals_ptr + max_residual_dim;

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto &rb = residual_batches[i];
    CheckEvaluate(rb.Evaluate(stream, workspace_ptr, residuals_ptr, state_pointers[i].data(),
                              cost_ptr, nullptr),
                  i);
    cost_ptr += rb.GetFactorBatch()->NumActiveFactors();
  }

  if (d_total == nullptr) return;
  if (total_num_cost_elements > 0) {
    size_t partials_needed = ReducePartialCount(total_num_cost_elements);
    if (scratch.d_partials.size() < partials_needed) scratch.d_partials.resize(partials_needed);
    ReduceSumToDevice(stream, costs, total_num_cost_elements, d_total, scratch.d_partials.data());
  } else {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_total, 0, sizeof(float), stream));
  }
}

/** The total cost at `state`, read back (synchronizes the stream). */
float ComputeCost(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
                  MinimizerScratch &scratch) {
  if (scratch.d_scalars.size() < 1) scratch.d_scalars.resize(1);
  if (scratch.h_scalars.size() < 1) scratch.h_scalars.resize(1);
  ComputeCostAsync(stream, problem, state, scratch, scratch.d_scalars.data());
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(scratch.h_scalars.data(), scratch.d_scalars.data(),
                                      sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return scratch.h_scalars[0];
}

/**
 * Residuals and per-factor Jacobian blocks at `state`. Per residual batch,
 * either the FactorBatch's analytic Jacobian (ResidualBatch::Evaluate, which
 * also applies any loss function), or a finite-difference Jacobian of the raw
 * residuals with the loss applied afterwards (ResidualBatch::ApplyLoss), so
 * both paths see identical loss handling.
 */
void ComputeResidualAndJacobian(cudaStream_t stream, const Problem &problem,
                                const MinimizerState &state, const MinimizerOptions &options,
                                MinimizerSystem &system, MinimizerScratch &scratch) {
  const auto &state_pointers = state.GetStatePointers();
  const auto &residual_batches = problem.GetResidualBatches();
  size_t max_n = 0;
  for (const auto &rb : residual_batches) {
    max_n = std::max(max_n, rb.GetFactorBatch()->NumActiveFactors());
  }
  size_t ws_floats = ResidualBatchWorkspaceNumFloats(max_n);
  if (scratch.buffer.size() < ws_floats * sizeof(float)) {
    scratch.buffer.resize(ws_floats * sizeof(float));
  }
  float *workspace_ptr = reinterpret_cast<float *>(scratch.buffer.data());
  float *residuals_ptr = system.residuals.data();
  float *jacobian_ptr = system.factor_jacobians.data();

  for (size_t i = 0; i < residual_batches.size(); i++) {
    const auto &rb = residual_batches[i];
    auto ptrs = state_pointers[i].data();
    const auto &factor_batch = rb.GetFactorBatch();
    if (problem.JacobianModeFor(i, options.jacobian_mode) == JacobianMode::kAnalytic) {
      CheckEvaluate(rb.Evaluate(stream, workspace_ptr, residuals_ptr, ptrs, nullptr, jacobian_ptr),
                    i);
    } else {
      CheckEvaluate(factor_batch->Evaluate(residuals_ptr, nullptr, ptrs, stream), i);
      system.numeric_diff.Compute(stream, problem, i, state, residuals_ptr, jacobian_ptr,
                                  options.numeric_diff_options);
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
 * Projected Gauss-Newton: the components at a bound that the steepest descent
 * direction (rhs = -gradient, scaled by positive factors) pushes outward leave
 * the system. Their rows and columns are zeroed and the diagonal is kept, so
 * their step is exactly zero and the others solve the reduced system.
 */
void HoldActiveBounds(cudaStream_t stream, StateBatchOps &state_ops, const MinimizerState &state,
                      MinimizerSystem &system, MinimizerBounds &bounds) {
  if (!state_ops.HasBounds() || system.rhs.empty()) return;
  std::vector<const float *> x_ptrs;
  for (const auto &params : state.GetStates()) x_ptrs.push_back(params.data());
  state_ops.BoundMask(stream, x_ptrs, system.rhs, bounds.mask);
  system.normal_equations.ZeroMaskedLhsRowsColumns(stream, bounds.mask);
  ElementwiseMultiplyInPlace(stream, system.rhs.data(), bounds.mask.data(), system.rhs.size());
}

/**
 * Solves the system into system.step, mapped back from the column scaling.
 * With bounds, also the active-set refinement: a free component at a bound
 * (its gradient points inward) can still get an outward step through the
 * coupling with the others; projecting that step away would spoil the descent
 * direction. Such components are held too and the system solved again (rarely
 * more than one pass); with `fixed`, exactly max_bound_refinements passes,
 * without the read-back of the count.
 */
void SolveStep(cudaStream_t stream, const MinimizerOptions &options, bool fixed,
               StateBatchOps &state_ops, const MinimizerState &state, MinimizerSystem &system,
               MinimizerBounds &bounds) {
  auto solve = [&]() {
    if (!system.normal_equations.Solve(stream, *system.solver, system.rhs, system.step)) {
      std::string str = "Failed to solve linear system";
      LogError(str);
      throw std::runtime_error(str);
    }
  };
  solve();
  if (state_ops.HasBounds()) {
    std::vector<const float *> x_ptrs;
    for (const auto &params : state.GetStates()) x_ptrs.push_back(params.data());
    if (bounds.count.size() < 1) bounds.count.resize(1);
    for (size_t pass = 0; pass < options.max_bound_refinements; ++pass) {
      state_ops.BoundMask(stream, x_ptrs, system.step, bounds.step_mask);
      THROW_ON_CUDA_ERROR(cudaMemsetAsync(bounds.count.data(), 0, sizeof(int), stream));
      HoldOutwardSteps(stream, bounds.step_mask, bounds.mask, bounds.extra, bounds.count.data());
      if (!fixed) {
        int count = 0;
        THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&count, bounds.count.data(), sizeof(int),
                                            cudaMemcpyDeviceToHost, stream));
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (count == 0) break;
        LogMessage("Bounds: {} more components held", count);
      }
      system.normal_equations.ZeroMaskedLhsRowsColumns(stream, bounds.extra);
      ElementwiseMultiplyInPlace(stream, system.rhs.data(), bounds.extra.data(), system.rhs.size());
      solve();
    }
  }
  // dx = S z.
  if (options.column_scaling != ColumnScaling::None && !system.step.empty()) {
    ElementwiseMultiplyInPlace(stream, system.step.data(), system.column_scale.data(),
                               system.step.size());
  }
}

/** Projects the bounded batches of a minimizer state onto their bounds. */
void ProjectToBounds(cudaStream_t stream, StateBatchOps &state_ops, MinimizerState &state) {
  if (!state_ops.HasBounds()) return;
  std::vector<float *> x_ptrs;
  for (auto &params : state.GetStates()) x_ptrs.push_back(params.data());
  state_ops.ProjectToBounds(stream, x_ptrs);
}

/** updated = current ⊞ step, on each state's manifold. */
void UpdateStates(cudaStream_t stream, StateBatchOps &state_ops, const MinimizerState &current,
                  const dvector<float> &step, MinimizerState &updated) {
  std::vector<const float *> x_ptrs;
  for (const auto &params : current.GetStates()) x_ptrs.push_back(params.data());
  std::vector<float *> x_plus_delta_ptrs;
  for (auto &params : updated.GetStates()) x_plus_delta_ptrs.push_back(params.data());
  state_ops.Plus(stream, x_ptrs, step, x_plus_delta_ptrs);
}

/** Host-side size signature of the problem's structure (reuse_structure). */
std::vector<size_t> StructureSignature(const Problem &problem) {
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

}  // namespace

namespace internal {

void SetUpStructure(cudaStream_t stream, Problem &problem, const MinimizerOptions &options,
                    StateBatchOps &state_ops, MinimizerSystem &system) {
  // Cheap host-only guard (no device work): sizes within capacities,
  // connectivity covering the active factors, at least one active factor.
  problem.CheckSizes();
  problem.PrepareStatePointers(stream);
  for (size_t i = 0; i < problem.GetResidualBatches().size(); ++i) {
    if (problem.JacobianModeFor(i, options.jacobian_mode) == JacobianMode::kNumeric) {
      system.numeric_diff.Refresh(problem, i);
    }
  }
  InitializeResiduals(problem, system.residuals);
  state_ops.Preprocess(stream, problem.GetStateBatches());
  // The Hessian pattern comes straight from the factor graph; no global
  // Jacobian is involved.
  system.normal_equations.Initialize(stream, problem,
                                     static_cast<int>(state_ops.NumReducedStates()),
                                     system.solver->SupportsBlockStorage());
  const size_t jacobian_floats = system.normal_equations.JacobianValuesSize();
  if (system.factor_jacobians.size() != jacobian_floats) {
    system.factor_jacobians.resize(jacobian_floats);
  }
}

void BuildSystem(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
                 const MinimizerOptions &options, StateBatchOps &state_ops, MinimizerSystem &system,
                 MinimizerBounds &bounds, MinimizerScratch &scratch) {
  ComputeResidualAndJacobian(stream, problem, state, options, system, scratch);
  system.normal_equations.Assemble(stream, problem, system.factor_jacobians.data(),
                                   system.residuals.data(), system.rhs);
  if (options.column_scaling != ColumnScaling::None) {
    // H_jj = ||J_{:,j}||^2: scaling by the Hessian diagonal is scaling by the
    // Jacobian column norms.
    system.normal_equations.ExtractHessianDiagonal(stream, system.column_scale);
    InvertSqrtWithFloorInPlace(stream, system.column_scale);
    system.normal_equations.ScaleLhsSymmetric(stream, system.column_scale);
    ElementwiseMultiplyInPlace(stream, system.rhs.data(), system.column_scale.data(),
                               system.rhs.size());
  }
  HoldActiveBounds(stream, state_ops, state, system, bounds);
}

}  // namespace internal

Minimizer::Minimizer(const MinimizerOptions &options) : options_(options) {
  system_.solver = CreateSparseLinearSolver(options_.sparse_linear_solver_type,
                                            options_.sparse_linear_solver_config);
  if (options_.disable_safety_checks) system_.solver->DisableSafetyChecks();
}

MinimizerSummary Minimizer::Minimize(cudaStream_t stream, Problem &problem) {
  internal::InnerSolve settings;
  settings.max_num_iterations = options_.max_num_iterations;
  settings.max_line_search_steps = options_.max_line_search_steps;
  settings.reuse_structure = options_.reuse_structure;
  return internal::InnerMinimize(*this, stream, problem, settings);
}

/**
 * The iteration of Minimizer (see the class comment), with every decision
 * taken per subproblem on the device: each subproblem takes or rejects its own
 * step (its states are copied from the trial states only when taken) and
 * leaves the iteration at its own convergence. The linear system is solved for
 * all subproblems together. One 2-float read-back per iteration (total cost,
 * active subproblems), none with fixed_iterations.
 */
MinimizerSummary internal::InnerMinimize(Minimizer &m, cudaStream_t stream, Problem &problem,
                                         const InnerSolve &inner) {
  if (!inner.constraints_managed) {
    for (const auto &rb : problem.GetResidualBatches()) {
      if (dynamic_cast<const ConstraintFactorBatchBase *>(rb.GetFactorBatch()) != nullptr) {
        throw std::invalid_argument(
            "GaussNewtonMinimizer / LevenbergMarquardtMinimizer: the problem has constraint "
            "factor batches; solve it with AugmentedLagrangianMinimizer");
      }
    }
  }
  auto range = m.profiler_domain_.CreateDomainRange("Minimize");
  const MinimizerOptions &options = m.options_;
  MinimizerSystem &system = m.system_;
  MinimizerScratch &scratch = m.scratch_;
  ProblemPartition &partition = m.partition_;
  MinimizerState &current = m.current_state_;
  MinimizerState &updated = m.updated_state_;
  const bool fixed = inner.fixed_iterations;
  MinimizerSummary summary;
  if (inner.problem_at_cap != nullptr) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(inner.problem_at_cap, 0, problem.NumProblems() * sizeof(int), stream));
  }

  // Structure setup, unless the caller vouches for an unchanged structure and
  // the sizes agree.
  std::vector<size_t> signature = StructureSignature(problem);
  internal::MinimizerStructure &structure = m.structure_;
  if (!(inner.reuse_structure && structure.problem == &problem &&
        signature == structure.signature)) {
    structure = {};  // invalid until the setup below completes
    auto setup_range = m.profiler_domain_.CreateDomainRange("SetUpStructure");
    SetUpStructure(stream, problem, options, m.state_ops_, system);
  }

  current.Recreate(stream, problem);
  updated.Recreate(stream, problem);

  // No optimizable states (all constant): nothing to solve.
  if (m.state_ops_.NumReducedStates() == 0) {
    summary.initial_cost = ComputeCost(stream, problem, current, scratch);
    summary.final_cost = summary.initial_cost;
    LogMessage("No optimizable states; skipping solver.");
    return summary;
  }

  ProjectToBounds(stream, m.state_ops_, current);
  if (!structure.partition_ready) {
    partition.Build(stream, problem, current, m.state_ops_.NumReducedStates());
    structure.partition_ready = true;
  }

  // d_scalars: the partition's read-back slots (total cost, active count, and
  // with one subproblem whether its step was taken).
  if (scratch.d_scalars.size() < 3) scratch.d_scalars.resize(3);
  if (scratch.h_scalars.size() < 3) scratch.h_scalars.resize(3);
  float *d_out = scratch.d_scalars.data();
  auto read_back = [&](size_t slots = 2) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(scratch.h_scalars.data(), d_out, slots * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    return std::make_pair(scratch.h_scalars[0], static_cast<size_t>(scratch.h_scalars[1]));
  };
  // ComputeCostAsync leaves the per-factor costs at the start of the buffer.
  auto factor_costs = [&]() { return reinterpret_cast<const float *>(scratch.buffer.data()); };
  auto evaluate = [&](const MinimizerState &state, float *costs) {
    ComputeCostAsync(stream, problem, state, scratch, nullptr);
    partition.SumFactorCosts(stream, factor_costs(), costs);
  };

  evaluate(current, partition.Cost());
  partition.InitStepControl(stream, options.cost_tolerance, inner.problem_frozen, d_out);
  float cost = std::numeric_limits<float>::quiet_NaN();
  size_t active = partition.NumProblems();
  if (!fixed) {
    std::tie(cost, active) = read_back();
    LogMessage("Initial cost = {}, {} of {} subproblems active", cost, active,
               partition.NumProblems());
  }
  summary.initial_cost = cost;
  summary.final_cost = cost;
  if (active == 0) {
    // Nothing to iterate; the problem still gets the states projected onto
    // their bounds.
    if (m.state_ops_.HasBounds()) {
      Copy(stream, current, problem);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    }
    structure.problem = &problem;
    structure.signature = std::move(signature);
    return summary;
  }

  auto build = [&](size_t iteration) {
    auto build_range = m.profiler_domain_.CreateDomainRange("BuildSystem");
    BuildSystem(stream, problem, current, options, m.state_ops_, system, m.bounds_, scratch);
    m.UpdateSystem(stream, iteration, system.normal_equations, partition);
  };
  build(0);
  system.step.resize(system.rhs.size());
  if (!structure.solver_ready) {
    auto sa_range = m.profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    if (!system.normal_equations.InitializeSolver(stream, *system.solver, problem, system.rhs,
                                                  system.step)) {
      std::string str = "Failed to initialize linear solver";
      LogError(str);
      throw std::runtime_error(str);
    }
    structure.solver_ready = true;
  }
  structure.problem = &problem;
  structure.signature = std::move(signature);

  StepControlParams params;
  params.max_consecutive_rejected_steps = static_cast<int>(options.max_consecutive_rejected_steps);
  params.line_search = inner.max_line_search_steps > 0;
  for (summary.num_iterations = 0; summary.num_iterations < inner.max_num_iterations;) {
    auto it_range = m.profiler_domain_.CreateDomainRange("Iteration");
    summary.iteration_costs.push_back(summary.final_cost);
    {
      auto solve_range = m.profiler_domain_.CreateDomainRange("LinearSolve");
      SolveStep(stream, options, fixed, m.state_ops_, current, system, m.bounds_);
    }
    UpdateStates(stream, m.state_ops_, current, system.step, updated);
    if (params.line_search) partition.ResetLineSearch(stream);
    evaluate(updated, partition.NewCost());
    // Line search per subproblem: halve the steps that do not decrease their
    // subproblem's cost, re-evaluate, until none is left or the budget is
    // spent. Real time: every step of the budget, predicated on the device (a
    // subproblem that needs no shortening gets a scale of 1).
    for (size_t k = 0; k < inner.max_line_search_steps; ++k) {
      partition.MarkLineSearch(stream, d_out);
      if (!fixed) {
        float num_shortened = 0.f;
        THROW_ON_CUDA_ERROR(
            cudaMemcpyAsync(&num_shortened, d_out, sizeof(float), cudaMemcpyDeviceToHost, stream));
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (num_shortened == 0.f) break;
      }
      partition.ScaleRows(stream, partition.StepScale(), system.step.data(), system.step.data());
      UpdateStates(stream, m.state_ops_, current, system.step, updated);
      evaluate(updated, partition.NewCost());
    }
    partition.SumRows(stream, system.step.data(), nullptr, nullptr, partition.StepSquared());
    m.ClassifySteps(stream, system.normal_equations, system.step, partition);
    partition.StepControl(stream, params, d_out);
    // One subproblem, with a read-back anyway: a taken step swaps the trial
    // states in instead of copying them.
    const bool swap_taken = !fixed && partition.NumProblems() == 1;
    if (!swap_taken) partition.CopyAccepted(stream, problem, updated, current);
    summary.num_iterations++;
    if (!fixed) {
      std::tie(cost, active) = read_back(swap_taken ? 3 : 2);
      if (swap_taken && scratch.h_scalars[2] != 0.f) std::swap(current, updated);
      summary.final_cost = cost;
      LogMessage("Iteration #{}: cost = {}, {} subproblems active", summary.num_iterations, cost,
                 active);
      if (active == 0) break;
    }
    if (summary.num_iterations < inner.max_num_iterations) build(summary.num_iterations);
  }

  if (inner.problem_at_cap != nullptr) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(inner.problem_at_cap, partition.Active(),
                                        partition.NumProblems() * sizeof(int),
                                        cudaMemcpyDeviceToDevice, stream));
  }
  Copy(stream, current, problem);
  if (!fixed) THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return summary;
}

}  // namespace cunls
