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
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/minimizer/schur_system.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {
namespace {

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

/** dst[t * B + b] = src[order[t] * B + b]: the state pointers of the items of `order`. */
__global__ void GatherStatePointersKernel(size_t num_items, int num_slots, const int *order,
                                          float *const *src, float **dst) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= num_items * num_slots) return;
  const size_t t = i / num_slots;
  dst[i] = src[static_cast<size_t>(order[t]) * num_slots + (i - t * num_slots)];
}

/**
 * The state pointers of residual batch items evaluated in `order` (item t
 * evaluates factor order[t]), gathered into `buffer`.
 */
float const *const *GatherItemStatePointers(cudaStream_t stream, const int *order,
                                            float *const *state_pointers, size_t num_factors,
                                            int num_slots, dvector<float *> &buffer) {
  constexpr int kThreads = 256;
  const size_t count = num_factors * num_slots;
  if (buffer.size() < count) buffer.resize(count);
  GatherStatePointersKernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads, 0,
                              stream>>>(num_factors, num_slots, order, state_pointers,
                                        buffer.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return buffer.data();
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
      // Items in the linear system's order (residuals and Jacobians alike);
      // the state pointers are per item (FactorBatch::Evaluate).
      const int *order = system.linear_system->EvaluationOrder(i);
      const size_t num_factors = factor_batch->NumActiveFactors();
      float const *const *item_ptrs = ptrs;
      if (order != nullptr && num_factors > 0) {
        item_ptrs = GatherItemStatePointers(stream, order, ptrs, num_factors,
                                            static_cast<int>(factor_batch->StateSizes().size()),
                                            scratch.item_state_pointers);
      }
      CheckEvaluate(rb.Evaluate(stream, workspace_ptr, residuals_ptr, item_ptrs, nullptr,
                                jacobian_ptr, order, order != nullptr ? num_factors : 0),
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
 * Solves the system into system.step (again after each further restriction
 * the caller asks for, see InnerSolve::restrict_system), mapped back from the
 * column scaling (dx = S z).
 */
void SolveStep(cudaStream_t stream, const MinimizerOptions &options, const MinimizerState &state,
               const internal::InnerSolve &inner, bool reject_failed_solve,
               MinimizerSystem &system) {
  // A failed solve either throws or, when the minimizer can recover (by more
  // damping), becomes a NaN step: its trial cost is non-finite, so step
  // control rejects it like any other bad step.
  auto solve = [&]() {
    if (system.linear_system->Solve(stream, system.rhs, system.step)) return true;
    if (!reject_failed_solve) {
      std::string str =
          "GaussNewtonMinimizer: failed to solve the linear system (the normal equations are "
          "singular or not positive definite). Usual causes: a gauge freedom no constant state "
          "fixes, or a state no factor constrains. Fix: make a state constant or add a prior "
          "(in Python, pycunls.diagnose(problem) finds both), or use "
          "LevenbergMarquardtMinimizer, whose damping handles singular systems.";
      LogError(str);
      throw std::runtime_error(str);
    }
    LogWarning(
        "Failed to solve the linear system (singular or indefinite); the step is rejected and "
        "the damping grows");
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(system.step.data(), 0xff, system.step.size() * sizeof(float), stream));
    return false;
  };
  if (!solve()) return;
  while (inner.restrict_system &&
         inner.restrict_system(stream, state, *system.linear_system, system.rhs, &system.step)) {
    if (!solve()) return;
  }
  if (options.column_scaling != ColumnScaling::None && !system.step.empty()) {
    ElementwiseMultiplyInPlace(stream, system.step.data(), system.column_scale.data(),
                               system.step.size());
  }
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
  std::vector<bool> analytic(problem.GetResidualBatches().size());
  for (size_t i = 0; i < analytic.size(); i++) {
    analytic[i] = problem.JacobianModeFor(i, options.jacobian_mode) == JacobianMode::kAnalytic;
  }
  system.linear_system->Initialize(stream, problem, static_cast<int>(state_ops.NumReducedStates()),
                                   analytic);
  const size_t jacobian_floats = system.linear_system->JacobianValuesSize();
  if (system.factor_jacobians.size() != jacobian_floats) {
    system.factor_jacobians.resize(jacobian_floats);
  }
}

std::unique_ptr<LinearSystem> CreateLinearSystem(const MinimizerOptions &options) {
  const auto create_solver = [&](SparseLinearSolverType type) {
    SparseLinearSolverPtr solver =
        CreateSparseLinearSolver(type, options.sparse_linear_solver_config);
    if (options.disable_safety_checks) solver->DisableSafetyChecks();
    return solver;
  };
  switch (options.linear_system) {
    case LinearSystemType::Assembled:
      return std::make_unique<NormalEquations>(create_solver(options.sparse_linear_solver_type));
    case LinearSystemType::SchurComplement:
      // The reduced system has a sparse block pattern (dense covisibility
      // blocks, banded for visual-inertial chains) and must be solved exactly.
      if (options.sparse_linear_solver_type != SparseLinearSolverType::BlockSparseCholesky) {
        LogWarning(
            "LinearSystemType::SchurComplement solves the reduced system with "
            "BlockSparseCholesky; MinimizerOptions::sparse_linear_solver_type is not used");
      }
      return std::make_unique<SchurSystem>(
          create_solver(SparseLinearSolverType::BlockSparseCholesky),
          !options.disable_safety_checks);
  }
  throw std::invalid_argument("MinimizerOptions::linear_system: unknown LinearSystemType");
}

void BuildSystem(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
                 const MinimizerOptions &options, MinimizerSystem &system,
                 MinimizerScratch &scratch) {
  ComputeResidualAndJacobian(stream, problem, state, options, system, scratch);
  LinearSystem &linear_system = *system.linear_system;
  linear_system.Assemble(stream, problem, system.factor_jacobians.data(), system.residuals.data(),
                         system.rhs);
  linear_system.ClearModifications();
  if (options.column_scaling != ColumnScaling::None) {
    // H_jj = ||J_{:,j}||^2: scaling by the Hessian diagonal is scaling by the
    // Jacobian column norms.
    linear_system.ExtractHessianDiagonal(stream, system.column_scale);
    InvertSqrtWithFloorInPlace(stream, system.column_scale);
    linear_system.SetColumnScale(&system.column_scale);
    ElementwiseMultiplyInPlace(stream, system.rhs.data(), system.column_scale.data(),
                               system.rhs.size());
  }
}

}  // namespace internal

namespace {

/** Relative cost decrease of the last iteration, or NaN without one. */
float LastRelativeDecrease(const MinimizerSummary &summary) {
  if (summary.iteration_costs.empty()) return std::numeric_limits<float>::quiet_NaN();
  const float before = summary.iteration_costs.back();
  return (before - summary.final_cost) / std::max(std::abs(before), 1e-30f);
}

/**
 * The sentence of MinimizerSummary::message for one reason. `iteration_cap` is
 * the call's cap: options.max_num_iterations for Minimize(), the inner cap of
 * an AugmentedLagrangianMinimizer inner solve.
 */
std::string DescribeReason(TerminationReason reason, const MinimizerSummary &summary,
                           const MinimizerOptions &options, size_t iteration_cap) {
  std::ostringstream out;
  out.precision(4);
  const auto costs = [&]() {
    std::ostringstream c;
    c.precision(4);
    c << "cost " << summary.initial_cost << " -> " << summary.final_cost << " in "
      << summary.num_iterations << " iterations";
    return c.str();
  };
  switch (reason) {
    case TerminationReason::NotDetermined:
      out << "Real-time solve (fixed iterations): convergence was not checked.";
      break;
    case TerminationReason::NoFreeStates:
      out << "Nothing to optimize: every state is constant. Fix: make fewer states constant.";
      break;
    case TerminationReason::CostTolerance:
      out << "Converged: the cost fell below cost_tolerance (" << options.cost_tolerance << "); "
          << costs() << ".";
      break;
    case TerminationReason::StateTolerance:
      out << "Converged: the squared step norm fell below state_tolerance ("
          << options.state_tolerance << "); " << costs() << ".";
      break;
    case TerminationReason::RelativeReductionTolerance:
      out << "Converged: the predicted relative cost reduction fell below "
             "relative_reduction_tolerance; "
          << costs() << ".";
      break;
    case TerminationReason::NoCostDecrease:
      out << "Converged: a Gauss-Newton step no longer lowered the cost; " << costs()
          << ". Normal at a minimum; if the final cost is still high, check the Jacobian and "
             "the initial guess, or use LevenbergMarquardtMinimizer.";
      break;
    case TerminationReason::MaxIterations:
      if (iteration_cap == options.max_num_iterations) {
        out << "Not converged: reached max_num_iterations (" << iteration_cap << "); " << costs()
            << ", last relative decrease " << LastRelativeDecrease(summary)
            << ". Fix: raise max_num_iterations, or improve the initial guess.";
      } else {
        out << "Not converged: reached this inner solve's iteration cap (" << iteration_cap << "); "
            << costs() << ", last relative decrease " << LastRelativeDecrease(summary) << ".";
      }
      break;
    case TerminationReason::MaxRejectedSteps:
      out << "Stopped: " << options.max_consecutive_rejected_steps
          << " steps in a row were rejected; " << costs()
          << ". At a minimum this is float32 round-off (the cost cannot decrease further); if "
             "the final cost is still high, check the Jacobian (pycunls.check_jacobians), fix "
             "gauge freedoms (make a state constant), or improve the initial guess.";
      break;
    case TerminationReason::NonFiniteStep:
      out << "Stopped: the last trial steps were not finite; " << costs()
          << ". The linear solve failed (a singular system: a gauge freedom no constant state "
             "fixes, or a state no factor constrains) or a factor returned NaN at the trial "
             "states. Fix: check Problem.validate() and the factor's domain.";
      break;
    case TerminationReason::NonFiniteCost:
      out << "Not started: the initial cost is not finite (NaN or Inf). A factor returns NaN "
             "or Inf at the initial states. Fix: check that measurements and states are "
             "finite float32 in the layout the factor expects (pycunls.evaluate shows the "
             "cost per factor batch).";
      break;
  }
  return out.str();
}

/** MinimizerSummary::message from the reason counts. */
std::string DescribeTermination(const MinimizerSummary &summary, const MinimizerOptions &options,
                                size_t iteration_cap) {
  size_t total = 0;
  for (size_t c : summary.num_problems_by_reason) total += c;
  std::string text = DescribeReason(summary.termination_reason, summary, options, iteration_cap);
  if (total <= 1) return text;
  std::ostringstream out;
  out.precision(4);
  std::ostringstream counts;
  const char *separator = "";
  size_t converged = 0;
  for (int r = 0; r < kNumTerminationReasons; ++r) {
    const size_t count = summary.num_problems_by_reason[r];
    if (count == 0) continue;
    const auto reason = static_cast<TerminationReason>(r);
    if (IsConvergence(reason) || reason == TerminationReason::NoFreeStates) converged += count;
    counts << separator << count << " " << ToString(reason);
    separator = ", ";
  }
  if (converged == total) {
    out << "All " << total << " subproblems converged (" << counts.str() << "); total cost "
        << summary.initial_cost << " -> " << summary.final_cost << " in " << summary.num_iterations
        << " iterations.";
  } else {
    out << (total - converged) << " of " << total << " subproblems did not converge ("
        << counts.str() << "). Most severe: " << text;
  }
  return out.str();
}

}  // namespace

const char *ToString(TerminationReason reason) {
  switch (reason) {
    case TerminationReason::NotDetermined:
      return "NotDetermined";
    case TerminationReason::NoFreeStates:
      return "NoFreeStates";
    case TerminationReason::CostTolerance:
      return "CostTolerance";
    case TerminationReason::StateTolerance:
      return "StateTolerance";
    case TerminationReason::RelativeReductionTolerance:
      return "RelativeReductionTolerance";
    case TerminationReason::NoCostDecrease:
      return "NoCostDecrease";
    case TerminationReason::MaxIterations:
      return "MaxIterations";
    case TerminationReason::MaxRejectedSteps:
      return "MaxRejectedSteps";
    case TerminationReason::NonFiniteStep:
      return "NonFiniteStep";
    case TerminationReason::NonFiniteCost:
      return "NonFiniteCost";
  }
  return "Unknown";
}

bool IsConvergence(TerminationReason reason) {
  return reason >= TerminationReason::CostTolerance && reason <= TerminationReason::NoCostDecrease;
}

bool MinimizerSummary::Converged() const {
  return termination_reason == TerminationReason::NoFreeStates || IsConvergence(termination_reason);
}

Minimizer::Minimizer(const MinimizerOptions &options) : options_(options) {
  system_.linear_system = internal::CreateLinearSystem(options_);
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
    if (problem.HasBoxBounds()) {
      throw std::invalid_argument(
          "GaussNewtonMinimizer / LevenbergMarquardtMinimizer: the problem has box-bounded "
          "states (VectorStateBatch::SetBounds); solve it with AugmentedLagrangianMinimizer");
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
  const auto start = std::chrono::steady_clock::now();
  MinimizerSummary summary;
  // Every return: the message and the wall time.
  auto done = [&]() {
    summary.message = DescribeTermination(summary, options, inner.max_num_iterations);
    summary.solve_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!fixed) LogMessage("{}", summary.message);
    return summary;
  };
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

  summary.num_free_parameters = m.state_ops_.NumReducedStates();
  // No optimizable states (all constant): nothing to solve.
  if (m.state_ops_.NumReducedStates() == 0) {
    summary.initial_cost = ComputeCost(stream, problem, current, scratch);
    summary.final_cost = summary.initial_cost;
    summary.termination_reason = TerminationReason::NoFreeStates;
    summary.num_problems_by_reason.assign(kNumTerminationReasons, 0);
    summary.num_problems_by_reason[static_cast<int>(TerminationReason::NoFreeStates)] =
        problem.NumProblems();  // every subproblem
    return done();
  }

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
  // Why each subproblem stopped: counts per reason, read back with the final
  // synchronization (none in real time).
  if (scratch.d_termination_counts.size() < kNumTerminationReasons) {
    scratch.d_termination_counts.resize(kNumTerminationReasons);
    scratch.h_termination_counts.resize(kNumTerminationReasons);
  }
  auto finish_termination = [&]() {
    if (fixed) return;
    partition.FinishTermination(stream, scratch.d_termination_counts.data());
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
        scratch.h_termination_counts.data(), scratch.d_termination_counts.data(),
        kNumTerminationReasons * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  };
  auto read_termination = [&]() {
    if (fixed) return;
    summary.num_problems_by_reason.assign(kNumTerminationReasons, 0);
    for (int r = 0; r < kNumTerminationReasons; ++r) {
      summary.num_problems_by_reason[r] = scratch.h_termination_counts[r];
      if (scratch.h_termination_counts[r] != 0) {
        summary.termination_reason = static_cast<TerminationReason>(r);
      }
    }
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
  if (active == 0 && !fixed) {
    structure.problem = &problem;
    structure.signature = std::move(signature);
    finish_termination();
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    read_termination();
    return done();
  }

  // The subclass's update (damping) and the caller's restriction, on the
  // system built at the current states.
  auto update = [&](size_t iteration) {
    m.UpdateSystem(stream, iteration, *system.linear_system, partition);
    if (inner.restrict_system) {
      inner.restrict_system(stream, current, *system.linear_system, system.rhs, nullptr);
    }
  };
  auto build = [&](size_t iteration) {
    auto build_range = m.profiler_domain_.CreateDomainRange("BuildSystem");
    BuildSystem(stream, problem, current, options, system, scratch);
    update(iteration);
  };
  build(0);
  system.step.resize(system.rhs.size());
  if (!structure.solver_ready) {
    auto sa_range = m.profiler_domain_.CreateDomainRange("PerformSymbolicAnalysis");
    if (!system.linear_system->PrepareSolver(stream, problem, system.rhs, system.step)) {
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
  params.iteration_cap = fixed ? nullptr : inner.problem_iteration_cap;
  for (summary.num_iterations = 0; summary.num_iterations < inner.max_num_iterations;) {
    auto it_range = m.profiler_domain_.CreateDomainRange("Iteration");
    summary.iteration_costs.push_back(summary.final_cost);
    // The rows of subproblems that stopped iterating need no step: the
    // solver may skip them (their steps are never taken).
    if (partition.NumProblems() > 1) {
      if (system.row_active.size() != partition.NumRows()) {
        system.row_active.resize(partition.NumRows());
      }
      partition.ActiveRows(stream, system.row_active.data());
      system.linear_system->SetRowActivity(&system.row_active);
    } else {
      system.linear_system->SetRowActivity(nullptr);
    }
    {
      auto solve_range = m.profiler_domain_.CreateDomainRange("LinearSolve");
      SolveStep(stream, options, current, inner, m.RejectsFailedSolves(), system);
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
    m.ClassifySteps(stream, *system.linear_system, system.step, partition);
    params.iterations_done = static_cast<int>(summary.num_iterations) + 1;
    partition.StepControl(stream, params, d_out);
    // One subproblem, with a read-back anyway: a taken step swaps the trial
    // states in instead of copying them.
    const bool swap_taken = !fixed && partition.NumProblems() == 1;
    if (!swap_taken) partition.CopyAccepted(stream, problem, updated, current);
    summary.num_iterations++;
    // A rejected step leaves the states, so the linearization stands: only
    // the update (new damping) is redone. Not with restrict_system, which
    // zeroed right-hand side entries in place: the system is rebuilt.
    bool relinearize = true;
    if (!fixed) {
      std::tie(cost, active) = read_back(swap_taken ? 3 : 2);
      const bool step_taken = swap_taken && scratch.h_scalars[2] != 0.f;
      if (step_taken) std::swap(current, updated);
      relinearize = !swap_taken || step_taken || inner.restrict_system;
      summary.final_cost = cost;
      LogMessage("Iteration {}: cost {}, relative decrease {}, {} of {} subproblems active",
                 summary.num_iterations, cost, LastRelativeDecrease(summary), active,
                 partition.NumProblems());
      if (active == 0) break;
    }
    if (summary.num_iterations < inner.max_num_iterations) {
      if (relinearize) {
        build(summary.num_iterations);
      } else {
        update(summary.num_iterations);
      }
    }
  }

  if (inner.problem_at_cap != nullptr) partition.AtCap(stream, inner.problem_at_cap);
  Copy(stream, current, problem);
  finish_termination();
  if (!fixed) THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  read_termination();
  return done();
}

}  // namespace cunls
