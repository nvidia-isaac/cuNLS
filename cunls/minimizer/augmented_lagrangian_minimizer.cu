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
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/minimizer/sparse_matrix.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/state/state_batch.h"
#include "cunls/state/state_batch_ops.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

enum Status : int { kRunning = 0, kDone = 1, kStalled = 2 };

/// A violation above this fraction of the previous one is stagnating.
constexpr float kStagnation = 0.9f;

/// Read-back slots of the control kernel.
enum Out : int {
  kMaxViolation = 0,         ///< Over all subproblems.
  kNumRunning = 1,           ///< Subproblems still running.
  kMaxRunningViolation = 2,  ///< Over the running subproblems.
  kNumDone = 3,
  kNumStalled = 4,
  kNumOut = 5,
};

/** Max of non-negative floats through their integer bit patterns (monotone for v >= 0). */
__device__ void AtomicMaxNonNegative(float *address, float v) {
  atomicMax(reinterpret_cast<int *>(address), __float_as_int(v));
}

/** One thread per constraint row of a batch: the row's violation goes into violation[problem]. */
__global__ void violation_kernel(bool inequality, const float *values, size_t num_rows_total,
                                 int rows, const int *factor_problem, float *violation) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= num_rows_total) return;
  const int f = static_cast<int>(idx / rows);
  const int p = factor_problem != nullptr ? factor_problem[f] : 0;
  const float c = values[idx];
  float v = inequality ? fmaxf(c, 0.f) : fabsf(c);
  if (isnan(c)) v = INFINITY;
  AtomicMaxNonNegative(violation + p, v);
}

/**
 * One thread per constraint row of a batch: first-order multiplier update with
 * the factor's current penalty, for subproblems still running after the
 * control step (a finished subproblem keeps the multipliers it finished with).
 */
__global__ void multiplier_kernel(bool inequality, const float *values, size_t num_rows_total,
                                  int rows, const int *factor_problem, const int *status,
                                  const float *penalties, float *multipliers) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= num_rows_total) return;
  const int f = static_cast<int>(idx / rows);
  const int p = factor_problem != nullptr ? factor_problem[f] : 0;
  if (status[p] != kRunning) return;
  const float updated = multipliers[idx] + penalties[f] * values[idx];
  multipliers[idx] = inequality ? fmaxf(updated, 0.f) : updated;
}

/**
 * One thread per subproblem: penalty growth per constraint batch, stopping,
 * and the read-back scalars.
 */
__global__ void control_kernel(size_t num_problems, int num_constraints, float tolerance,
                               float violation_decrease, float penalty_increase, float max_penalty,
                               bool full_solve, const int *at_cap, const float *violation,
                               float *prev_violation, float *penalty, int *status, float *out) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  float vp = 0.f;
  for (int c = 0; c < num_constraints; ++c) vp = fmaxf(vp, violation[c * num_problems + p]);
  AtomicMaxNonNegative(out + kMaxViolation, vp);
  if (status[p] == kRunning) {
    bool stalled = false;
    for (int c = 0; c < num_constraints; ++c) {
      const size_t k = c * num_problems + p;
      const float v = violation[k];
      // Insufficient decrease grows the penalty. After an inner solve cut off by
      // its cap, only a stagnating violation does: a violation still falling is
      // limited by the solve, not the penalty, and a larger penalty would only
      // worsen the conditioning.
      const bool insufficient = !(v <= violation_decrease * prev_violation[k]);
      const bool stagnating = !(v <= kStagnation * prev_violation[k]);
      if (v > tolerance && insufficient && (at_cap[p] == 0 || stagnating)) {
        if (penalty[k] >= max_penalty) {
          // At the cap the multiplier updates alone still shrink the
          // violation linearly; only a stagnating one gives up.
          stalled = stalled || stagnating;
        } else {
          penalty[k] = fminf(penalty[k] * penalty_increase, max_penalty);
        }
      }
      prev_violation[k] = v;
    }
    // Done: feasible after an inner solve that converged on its own
    // (stationary), not one cut off by its iteration cap.
    if (vp <= tolerance && full_solve && at_cap[p] == 0) {
      status[p] = kDone;
    } else if (stalled) {
      status[p] = kStalled;
    }
  }
  if (status[p] == kRunning) {
    atomicAdd(out + kNumRunning, 1.f);
    AtomicMaxNonNegative(out + kMaxRunningViolation, vp);
  } else if (status[p] == kDone) {
    atomicAdd(out + kNumDone, 1.f);
  } else {
    atomicAdd(out + kNumStalled, 1.f);
  }
}

__global__ void scatter_penalty_kernel(const float *penalty, const int *factor_problem,
                                       size_t num_factors, float *factor_penalties) {
  const size_t f = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (f >= num_factors) return;
  factor_penalties[f] = penalty[factor_problem != nullptr ? factor_problem[f] : 0];
}

__global__ void decay_penalty_kernel(float *penalty, size_t n, float factor, float floor,
                                     float ceiling) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) penalty[i] = fminf(fmaxf(penalty[i] / factor, floor), ceiling);
}

__global__ void fill_kernel(float *data, size_t n, float value) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) data[i] = value;
}

void Fill(cudaStream_t stream, dvector<float> &v, float value) {
  if (v.size() == 0) return;
  fill_kernel<<<Blocks(v.size()), kThreads, 0, stream>>>(v.data(), v.size(), value);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

using internal::ConstraintBatchState;

/** Clamps the components of the non-constant states (column >= 0) into [lower, upper]. */
__global__ void project_kernel(float *x, const int *column_offsets, const float *lower,
                               const float *upper, size_t n, int dim) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n || column_offsets[i / dim] < 0) return;
  x[i] = fminf(fmaxf(x[i], lower[i]), upper[i]);
}

/**
 * mask[column] = 0 for the components of non-constant states that sit at a
 * bound with direction (the right-hand side, minus the gradient) pointing
 * outward. A component within a relative 1e-6 of a bound counts as at it
 * (projection puts it exactly there; this catches float round-off).
 */
__global__ void hold_kernel(const float *x, const int *column_offsets, const float *lower,
                            const float *upper, const float *direction, size_t n, int dim,
                            float *mask) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const int column = column_offsets[i / dim];
  if (column < 0) return;
  const int row = column + static_cast<int>(i % dim);
  const float v = x[i], d = direction[row], lo = lower[i], hi = upper[i];
  const bool at_lower = v <= lo + 1e-6f * (1.f + fabsf(lo));
  const bool at_upper = v >= hi - 1e-6f * (1.f + fabsf(hi));
  if ((at_lower && d <= 0.f) || (at_upper && d >= 0.f)) mask[row] = 0.f;
}

/**
 * Active-set refinement: components still free in `held` (1) but marked 0 in
 * `step_mask` (at a bound, the step pushes outward) become held. Writes
 * extra[i] = 0 for them (1 elsewhere), clears them in `held`, and counts them
 * in *count.
 */
__global__ void hold_outward_kernel(const float *step_mask, float *held, float *extra, int *count,
                                    size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const bool newly_held = held[i] != 0.f && step_mask[i] == 0.f;
  extra[i] = newly_held ? 0.f : 1.f;
  if (newly_held) {
    held[i] = 0.f;
    atomicAdd(count, 1);
  }
}

/** The bounds of a box-bounded state batch (see BoxBoundedStates). */
const BoxBoundedStates &Bounds(const StateBatch *batch) {
  return *dynamic_cast<const BoxBoundedStates *>(batch);
}

/**
 * Cost of the non-constraint residual batches at the problem's states, read
 * back (synchronizes the stream). `scratch`, `partials` and `d_out` are
 * evaluation buffers.
 */
float ComputeObjective(cudaStream_t stream, const Problem &problem, dvector<float> &scratch,
                       dvector<float> &partials, dvector<float> &d_out) {
  const auto &residual_batches = problem.GetResidualBatches();
  size_t num_costs = 0, max_residuals = 0, max_workspace = 0;
  for (const auto &rb : residual_batches) {
    const FactorBatch *fb = rb.GetFactorBatch();
    if (dynamic_cast<const ConstraintFactorBatchBase *>(fb) != nullptr) continue;
    const size_t n = fb->NumActiveFactors();
    num_costs += n;
    max_residuals = std::max(max_residuals, n * fb->ResidualsSize());
    max_workspace = std::max(max_workspace, ResidualBatchWorkspaceNumFloats(n));
  }
  if (d_out.size() < kNumOut) d_out.resize(kNumOut);
  if (num_costs == 0) return 0.f;
  // Keep the workspace aligned for its float3 part: it goes first.
  const size_t floats = max_workspace + max_residuals + num_costs;
  if (scratch.size() < floats) scratch.resize(floats);
  float *workspace = scratch.data();
  float *residuals = workspace + max_workspace;
  float *costs = residuals + max_residuals;
  float *cost = costs;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    const auto &rb = residual_batches[i];
    const FactorBatch *fb = rb.GetFactorBatch();
    if (dynamic_cast<const ConstraintFactorBatchBase *>(fb) != nullptr) continue;
    if (fb->NumActiveFactors() == 0) continue;
    CheckEvaluate(
        rb.Evaluate(stream, workspace, residuals, problem.DeviceStatePointers(i), cost, nullptr),
        i);
    cost += fb->NumActiveFactors();
  }
  const size_t num_partials = ReducePartialCount(num_costs);
  if (partials.size() < num_partials) partials.resize(num_partials);
  ReduceSumToDevice(stream, costs, num_costs, d_out.data(), partials.data());
  float result = 0.f;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&result, d_out.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return result;
}

/** penalties[f] = penalty[constraint, subproblem(f)] for every constraint batch. */
void ScatterPenalties(cudaStream_t stream, const std::vector<ConstraintBatchState> &constraints,
                      const dvector<float> &penalty, size_t num_problems) {
  for (size_t c = 0; c < constraints.size(); ++c) {
    const ConstraintBatchState &con = constraints[c];
    if (con.num_factors == 0) continue;
    scatter_penalty_kernel<<<Blocks(con.num_factors), kThreads, 0, stream>>>(
        penalty.data() + c * num_problems,
        con.factor_problem.size() > 0 ? con.factor_problem.data() : nullptr, con.num_factors,
        con.batch->Penalties());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

/** Fills the summary's violation and status from the control kernel's read-back slots. */
void FillStatus(const float *out, AugmentedLagrangianMinimizerSummary &summary) {
  summary.max_violation = out[kMaxViolation];
  summary.num_converged = static_cast<size_t>(out[kNumDone]);
  summary.num_max_penalty = static_cast<size_t>(out[kNumStalled]);
  if (summary.num_converged == summary.num_problems) {
    summary.status = AugmentedLagrangianMinimizerStatus::kConverged;
  } else if (out[kNumRunning] > 0.f) {
    summary.status = AugmentedLagrangianMinimizerStatus::kMaxOuterIterations;
  } else {
    summary.status = AugmentedLagrangianMinimizerStatus::kMaxPenalty;
  }
}

}  // namespace

AugmentedLagrangianMinimizer::AugmentedLagrangianMinimizer(
    Minimizer &minimizer, const AugmentedLagrangianMinimizerOptions &options)
    : minimizer_(minimizer) {
  SetOptions(options);
}

void AugmentedLagrangianMinimizer::SetOptions(const AugmentedLagrangianMinimizerOptions &options) {
  if (!(options.initial_penalty > 0.f) || !(options.max_penalty >= options.initial_penalty) ||
      !(options.penalty_increase >= 1.f) || !(options.constraint_tolerance >= 0.f) ||
      options.inner_iterations == 0 || options.max_outer_iterations == 0) {
    const std::string msg =
        "AugmentedLagrangianMinimizerOptions: need initial_penalty > 0, max_penalty >= "
        "initial_penalty, penalty_increase >= 1, constraint_tolerance >= 0, "
        "inner_iterations > 0 and max_outer_iterations > 0";
    LogError(msg);
    throw std::invalid_argument(msg);
  }
  options_ = options;
}

bool AugmentedLagrangianMinimizer::HoldActiveBounds(cudaStream_t stream,
                                                    const MinimizerState &state,
                                                    NormalEquations &system, dvector<float> &rhs,
                                                    const dvector<float> *step) {
  if (rhs.empty()) return false;
  // mask[column] = 0 for the components at a bound that `direction` pushes outward.
  auto mark = [&](const float *direction, dvector<float> &mask) {
    mask.resize(rhs.size());
    Fill(stream, mask, 1.f);
    const auto &state_batches = problem_->GetStateBatches();
    for (size_t b : bounded_) {
      const StateBatch *batch = state_batches[b];
      const size_t n = batch->NumActiveStates() * batch->TangentSize();
      if (n == 0) continue;
      hold_kernel<<<Blocks(n), kThreads, 0, stream>>>(
          state.GetStates()[b].data(), column_offsets_[b].data(), Bounds(batch).LowerBounds(),
          Bounds(batch).UpperBounds(), direction, n, static_cast<int>(batch->TangentSize()),
          mask.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
  };
  if (step == nullptr) {
    // Before the solve: hold where the steepest-descent direction points outward.
    refinements_ = 0;
    mark(rhs.data(), bound_mask_);
    system.ZeroMaskedLhsRowsColumns(stream, bound_mask_);
    ElementwiseMultiplyInPlace(stream, rhs.data(), bound_mask_.data(), rhs.size());
    return false;
  }
  // After a solve: a free component at a bound (its gradient points inward)
  // can still get an outward step through the coupling with the others;
  // projecting that step away would spoil the descent direction. Hold such
  // components too and solve again. In real time exactly
  // max_bound_refinements passes, without the read-back of the count.
  if (refinements_ >= options_.max_bound_refinements) return false;
  ++refinements_;
  mark(step->data(), bound_step_mask_);
  bound_extra_.resize(rhs.size());
  if (bound_count_.size() < 1) bound_count_.resize(1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(bound_count_.data(), 0, sizeof(int), stream));
  hold_outward_kernel<<<Blocks(rhs.size()), kThreads, 0, stream>>>(
      bound_step_mask_.data(), bound_mask_.data(), bound_extra_.data(), bound_count_.data(),
      rhs.size());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (!options_.real_time) {
    int count = 0;
    THROW_ON_CUDA_ERROR(
        cudaMemcpyAsync(&count, bound_count_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    if (count == 0) return false;
    LogMessage("Bounds: {} more components held", count);
  }
  system.ZeroMaskedLhsRowsColumns(stream, bound_extra_);
  ElementwiseMultiplyInPlace(stream, rhs.data(), bound_extra_.data(), rhs.size());
  return true;
}

AugmentedLagrangianMinimizerSummary AugmentedLagrangianMinimizer::Minimize(cudaStream_t stream,
                                                                           Problem &problem) {
  // A robust loss on the objective is fine (the inner solver applies it); on a
  // constraint it would down-weight large violations of a hard constraint and
  // corrupt the multiplier update, so it is rejected.
  for (const auto &rb : problem.GetResidualBatches()) {
    const LossFunctionBatch *loss = rb.GetLossFunction();
    if (loss != nullptr && dynamic_cast<const TrivialLossFunctionBatch *>(loss) == nullptr &&
        dynamic_cast<const ConstraintFactorBatchBase *>(rb.GetFactorBatch()) != nullptr) {
      throw std::invalid_argument(
          "AugmentedLagrangianMinimizer: a constraint factor batch has a robust loss; "
          "constraints are hard and take no loss (robust losses belong on objective factors)");
    }
  }
  AugmentedLagrangianMinimizerSummary summary;
  summary.num_problems = problem.NumProblems();

  // Constraint batches, and whether they are the ones of the last solve.
  const auto &residual_batches = problem.GetResidualBatches();
  std::vector<std::pair<const void *, size_t>> signature;
  std::vector<ConstraintBatchState> constraints;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    auto *batch = dynamic_cast<ConstraintFactorBatchBase *>(residual_batches[i].GetFactorBatch());
    if (batch == nullptr) continue;
    constraints.push_back({i, batch, batch->NumActiveFactors(), {}});
    signature.emplace_back(batch, batch->NumActiveFactors());
  }
  signature.emplace_back(nullptr, problem.NumProblems());
  const bool same_constraints = signature == signature_ && problem_ == &problem;
  const bool warm = options_.warm_start && same_constraints;
  // Structure reuse across calls (the caller vouches for an unchanged
  // structure; the inner minimizer checks the sizes again).
  const bool reuse = options_.reuse_structure && same_constraints;

  // Box-bounded states: the reduced-system column of every state (unless the
  // structure is reused), then the initial values of the free states clamped
  // into the box. The inner solves hold the active bounds (HoldActiveBounds)
  // and the bounded Plus keeps their iterates in the box.
  const auto &state_batches = problem.GetStateBatches();
  bounded_.clear();
  for (size_t b = 0; b < state_batches.size(); ++b) {
    if (HasBoxBounds(state_batches[b])) bounded_.push_back(b);
  }
  if (!bounded_.empty()) {
    if (!(options_.reuse_structure && problem_ == &problem) ||
        column_offsets_.size() != state_batches.size()) {
      problem.CheckSizes();
      column_offsets_.resize(state_batches.size());
      int first_column = 0;
      for (size_t b = 0; b < state_batches.size(); ++b) {
        const StateBatch *batch = state_batches[b];
        ComputeStateColumnOffsets(stream, first_column, batch, column_offsets_[b]);
        first_column += static_cast<int>((batch->NumActiveStates() - batch->NumConstStates()) *
                                         batch->TangentSize());
      }
    }
    for (size_t b : bounded_) {
      StateBatch *batch = state_batches[b];
      const size_t n = batch->NumActiveStates() * batch->TangentSize();
      if (n == 0) continue;
      project_kernel<<<Blocks(n), kThreads, 0, stream>>>(
          batch->StateDevicePtr(0), column_offsets_[b].data(), Bounds(batch).LowerBounds(),
          Bounds(batch).UpperBounds(), n, static_cast<int>(batch->TangentSize()));
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
  }
  auto restrict_system = [this](cudaStream_t s, const MinimizerState &state,
                                NormalEquations &system, dvector<float> &rhs,
                                const dvector<float> *step) {
    return HoldActiveBounds(s, state, system, rhs, step);
  };
  signature_ = signature;
  problem_ = &problem;

  if (constraints.empty()) {
    internal::InnerSolve inner_solve;
    inner_solve.max_num_iterations =
        options_.final_inner_iterations > 0
            ? options_.final_inner_iterations
            : (options_.real_time ? options_.inner_iterations
                                  : minimizer_.Options().max_num_iterations);
    inner_solve.max_line_search_steps = options_.inner_line_search_steps;
    inner_solve.reuse_structure = options_.reuse_structure;
    inner_solve.fixed_iterations = options_.real_time;
    inner_solve.constraints_managed = true;
    if (!bounded_.empty()) inner_solve.restrict_system = restrict_system;
    const MinimizerSummary inner =
        internal::InnerMinimize(minimizer_, stream, problem, inner_solve);
    summary.outer_iterations = 1;
    summary.inner_iterations = inner.num_iterations;
    summary.initial_cost = inner.initial_cost;
    summary.final_cost = inner.final_cost;
    summary.num_converged = summary.num_problems;
    return summary;
  }

  num_problems_ = problem.NumProblems();
  size_t total_rows = 0;
  if (reuse) {
    // Same constraint batches: keep their factor -> subproblem maps.
    for (const ConstraintBatchState &con : constraints_)
      total_rows += con.num_factors * con.batch->ResidualsSize();
  } else {
    problem.CheckSizes();
    problem.PrepareStatePointers(stream);
    constraints_ = std::move(constraints);
    for (ConstraintBatchState &con : constraints_) {
      total_rows += con.num_factors * con.batch->ResidualsSize();
      if (num_problems_ > 1 && con.num_factors > 0) {
        con.factor_problem.resize(con.num_factors);
        ComputeFactorProblemIds(stream, problem, con.residual_batch, con.factor_problem.data());
      }
    }
  }
  if (values_.size() < total_rows) values_.resize(total_rows);

  const size_t num_slots = constraints_.size() * num_problems_;
  // A warm start keeps the multipliers and the penalties they were computed
  // with. A converging call lowers the penalties by one increase step (the
  // penalties of a receding horizon would otherwise only ever grow) and
  // restarts the violation history (the old one would make the stall rule
  // fire on the first outer iteration). A real-time call runs a few outer
  // iterations only: its penalty schedule continues across calls, so both are
  // kept as they are.
  // Either way the kept penalties are clamped to [initial_penalty,
  // max_penalty] of the current options (which SetOptions may have changed).
  const bool keep_state = warm && penalty_.size() == num_slots;
  if (keep_state) {
    decay_penalty_kernel<<<Blocks(num_slots), kThreads, 0, stream>>>(
        penalty_.data(), num_slots, options_.real_time ? 1.f : options_.penalty_increase,
        options_.initial_penalty, options_.max_penalty);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  } else {
    penalty_.resize(num_slots);
    Fill(stream, penalty_, options_.initial_penalty);
    for (ConstraintBatchState &con : constraints_) con.batch->ResetMultipliers(stream);
  }
  if (!(keep_state && options_.real_time) || prev_violation_.size() != num_slots) {
    prev_violation_.resize(num_slots);
    Fill(stream, prev_violation_, INFINITY);
  }
  violation_.resize(num_slots);
  problem_status_.resize(num_problems_);
  at_cap_.resize(num_problems_);
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(problem_status_.data(), 0, num_problems_ * sizeof(int), stream));
  if (d_out_.size() < kNumOut) d_out_.resize(kNumOut);
  ScatterPenalties(stream, constraints_, penalty_, num_problems_);

  const bool real_time = options_.real_time;
  summary.initial_cost = real_time ? std::numeric_limits<float>::quiet_NaN()
                                   : ComputeObjective(stream, problem, scratch_, partials_, d_out_);
  summary.final_cost = summary.initial_cost;

  const size_t final_cap =
      options_.final_inner_iterations > 0
          ? options_.final_inner_iterations
          : (real_time ? options_.inner_iterations : minimizer_.Options().max_num_iterations);
  std::array<float, kNumOut> out{};
  // Real time: every outer iteration may finish a subproblem (there is no
  // read-back to decide on a final full solve).
  bool full_solve = real_time;
  for (size_t k = 0; k < options_.max_outer_iterations; ++k) {
    const bool last = k + 1 == options_.max_outer_iterations;
    if (last) full_solve = true;
    const size_t cap = last || (full_solve && !real_time) ? final_cap : options_.inner_iterations;
    internal::InnerSolve inner_solve;
    inner_solve.max_num_iterations = cap;
    inner_solve.max_line_search_steps = options_.inner_line_search_steps;
    inner_solve.fixed_iterations = real_time;
    inner_solve.problem_at_cap = at_cap_.data();
    // Finished subproblems (done or stalled) keep their states: re-solving one
    // for the others' sake could move it off the point that was accepted.
    inner_solve.problem_frozen = problem_status_.data();
    inner_solve.constraints_managed = true;
    if (!bounded_.empty()) inner_solve.restrict_system = restrict_system;
    // Later outer iterations solve the same structure.
    inner_solve.reuse_structure = k > 0 || reuse;
    const MinimizerSummary inner =
        internal::InnerMinimize(minimizer_, stream, problem, inner_solve);
    summary.inner_iterations += inner.num_iterations;
    summary.outer_iterations = k + 1;
    LogMessage("Outer iteration #{}: {} inner iterations (cap {}), cost {}", k,
               inner.num_iterations, cap, inner.final_cost);

    // Constraint values and violations at the new states.
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(violation_.data(), 0, num_slots * sizeof(float), stream));
    size_t offset = 0;
    for (size_t c = 0; c < constraints_.size(); ++c) {
      const ConstraintBatchState &con = constraints_[c];
      const size_t rows = con.num_factors * con.batch->ResidualsSize();
      if (rows == 0) continue;
      if (!con.batch->EvaluateConstraint(values_.data() + offset, nullptr,
                                         problem.DeviceStatePointers(con.residual_batch), stream)) {
        throw std::runtime_error("AugmentedLagrangianMinimizer: constraint evaluation failed");
      }
      violation_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
          con.batch->Kind() == ConstraintKind::kInequality, values_.data() + offset, rows,
          static_cast<int>(con.batch->ResidualsSize()),
          con.factor_problem.size() > 0 ? con.factor_problem.data() : nullptr,
          violation_.data() + c * num_problems_);
      THROW_ON_CUDA_ERROR(cudaGetLastError());
      offset += rows;
    }
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_out_.data(), 0, kNumOut * sizeof(float), stream));
    control_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
        num_problems_, static_cast<int>(constraints_.size()), options_.constraint_tolerance,
        options_.violation_decrease, options_.penalty_increase, options_.max_penalty, full_solve,
        at_cap_.data(), violation_.data(), prev_violation_.data(), penalty_.data(),
        problem_status_.data(), d_out_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    // Multiplier updates of the subproblems still running.
    offset = 0;
    for (const ConstraintBatchState &con : constraints_) {
      const size_t rows = con.num_factors * con.batch->ResidualsSize();
      if (rows == 0) continue;
      multiplier_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
          con.batch->Kind() == ConstraintKind::kInequality, values_.data() + offset, rows,
          static_cast<int>(con.batch->ResidualsSize()),
          con.factor_problem.size() > 0 ? con.factor_problem.data() : nullptr,
          problem_status_.data(), con.batch->Penalties(), con.batch->Multipliers());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
      offset += rows;
    }
    if (real_time) {
      // No read-back: the budget is fixed; penalties go back to the batches on the device.
      if (!last) ScatterPenalties(stream, constraints_, penalty_, num_problems_);
      continue;
    }
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(out.data(), d_out_.data(), kNumOut * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    LogMessage("Outer iteration #{}: max violation {}, {} of {} subproblems running", k,
               out[kMaxViolation], out[kNumRunning], num_problems_);
    if (out[kNumRunning] == 0.f) break;
    // Once every running subproblem is feasible, the next inner solve runs to
    // convergence: feasibility after it means stationarity too.
    full_solve = out[kMaxRunningViolation] <= options_.constraint_tolerance;
    ScatterPenalties(stream, constraints_, penalty_, num_problems_);
  }
  if (real_time) {
    // The one read-back of a real-time call: the last outer iteration's status.
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(out.data(), d_out_.data(), kNumOut * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
  FillStatus(out.data(), summary);
  if (!real_time) {
    summary.final_cost = ComputeObjective(stream, problem, scratch_, partials_, d_out_);
  }
  return summary;
}

}  // namespace cunls
