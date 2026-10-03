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

}  // namespace

AugmentedLagrangianMinimizer::AugmentedLagrangianMinimizer(
    GaussNewtonMinimizer &minimizer, const AugmentedLagrangianMinimizerOptions &options)
    : minimizer_(minimizer) {
  SetOptions(options);
}

void AugmentedLagrangianMinimizer::SetOptions(const AugmentedLagrangianMinimizerOptions &options) {
  if (!(options.initial_penalty > 0.f) || !(options.max_penalty >= options.initial_penalty) ||
      !(options.penalty_increase >= 1.f) || !(options.constraint_tolerance >= 0.f) ||
      options.inner_iterations == 0) {
    const std::string msg =
        "AugmentedLagrangianMinimizerOptions: need initial_penalty > 0, max_penalty >= "
        "initial_penalty, "
        "penalty_increase >= 1, constraint_tolerance >= 0 and inner_iterations > 0";
    LogError(msg);
    throw std::invalid_argument(msg);
  }
  options_ = options;
  ResetGraph();
  graph_failed_ = false;
}

float AugmentedLagrangianMinimizer::ComputeObjective(cudaStream_t stream, const Problem &problem) {
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
  if (d_out_.size() < kNumOut) d_out_.resize(kNumOut);
  if (num_costs == 0) return 0.f;
  // Keep the workspace aligned for its float3 part: it goes first.
  const size_t floats = max_workspace + max_residuals + num_costs;
  if (scratch_.size() < floats) scratch_.resize(floats);
  float *workspace = scratch_.data();
  float *residuals = workspace + max_workspace;
  float *costs = residuals + max_residuals;
  float *cost = costs;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    const auto &rb = residual_batches[i];
    const FactorBatch *fb = rb.GetFactorBatch();
    if (dynamic_cast<const ConstraintFactorBatchBase *>(fb) != nullptr) continue;
    if (fb->NumActiveFactors() == 0) continue;
    rb.Evaluate(stream, workspace, residuals, problem.DeviceStatePointers(i), cost, nullptr);
    cost += fb->NumActiveFactors();
  }
  const size_t partials = ReducePartialCount(num_costs);
  if (partials_.size() < partials) partials_.resize(partials);
  ReduceSumToDevice(stream, costs, num_costs, d_out_.data(), partials_.data());
  float result = 0.f;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&result, d_out_.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return result;
}

void AugmentedLagrangianMinimizer::ScatterPenalties(cudaStream_t stream) {
  for (size_t c = 0; c < constraints_.size(); ++c) {
    const Constraint &con = constraints_[c];
    if (con.num_factors == 0) continue;
    scatter_penalty_kernel<<<Blocks(con.num_factors), kThreads, 0, stream>>>(
        penalty_.data() + c * num_problems_,
        con.factor_problem.size() > 0 ? con.factor_problem.data() : nullptr, con.num_factors,
        con.batch->Penalties());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

std::vector<std::pair<const void *, size_t>> AugmentedLagrangianMinimizer::ConstraintSignature(
    const Problem &problem) {
  std::vector<std::pair<const void *, size_t>> signature;
  for (const auto &rb : problem.GetResidualBatches()) {
    auto *batch = dynamic_cast<ConstraintFactorBatchBase *>(rb.GetFactorBatch());
    if (batch != nullptr) signature.emplace_back(batch, batch->NumActiveFactors());
  }
  signature.emplace_back(nullptr, problem.NumProblems());
  return signature;
}

void AugmentedLagrangianMinimizer::ResetGraph() {
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
    graph_exec_ = nullptr;
  }
  graph_warm_ = false;
}

AugmentedLagrangianMinimizer::~AugmentedLagrangianMinimizer() { ResetGraph(); }

void AugmentedLagrangianMinimizer::ReadStatus(cudaStream_t stream,
                                              AugmentedLagrangianMinimizerSummary &summary) {
  std::array<float, kNumOut> out{};
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(out.data(), d_out_.data(), kNumOut * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  FillStatus(out.data(), summary);
}

void AugmentedLagrangianMinimizer::FillStatus(const float *out,
                                              AugmentedLagrangianMinimizerSummary &summary) const {
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

AugmentedLagrangianMinimizerSummary AugmentedLagrangianMinimizer::Minimize(cudaStream_t stream,
                                                                           Problem &problem) {
  // CUDA graph: a real-time call with an unchanged structure is a fixed
  // sequence of device work. The first such call runs eagerly (it sizes every
  // buffer), the second is captured, later calls replay the graph; the final
  // status read-back stays outside the graph.
  const bool eligible = options_.use_cuda_graph && options_.real_time && options_.warm_start &&
                        options_.reuse_structure && !graph_failed_ && problem_ == &problem &&
                        ConstraintSignature(problem) == signature_;
  if (!eligible) {
    ResetGraph();
    return MinimizeEager(stream, problem, true);
  }
  if (graph_exec_ != nullptr) {
    THROW_ON_CUDA_ERROR(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_), stream));
    AugmentedLagrangianMinimizerSummary summary = graph_summary_;
    if (has_constraints_) ReadStatus(stream, summary);
    return summary;
  }
  if (!graph_warm_) {
    graph_warm_ = true;
    return MinimizeEager(stream, problem, true);
  }
  THROW_ON_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  bool captured = false;
  try {
    graph_summary_ = MinimizeEager(stream, problem, false);
    captured = true;
  } catch (const std::exception &e) {
    LogWarning("AugmentedLagrangianMinimizer: CUDA graph capture failed ({}); running eagerly",
               e.what());
  }
  cudaGraph_t graph = nullptr;
  const cudaError_t end = cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t exec = nullptr;
  if (captured && end == cudaSuccess && graph != nullptr &&
      cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess) {
    graph_exec_ = exec;
  } else {
    if (captured) {
      LogWarning("AugmentedLagrangianMinimizer: CUDA graph capture failed ({}); running eagerly",
                 cudaGetErrorString(end));
    }
    graph_failed_ = true;
    cudaGetLastError();  // clear a sticky capture error
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (graph_exec_ == nullptr) return MinimizeEager(stream, problem, true);
  THROW_ON_CUDA_ERROR(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_), stream));
  AugmentedLagrangianMinimizerSummary summary = graph_summary_;
  if (has_constraints_) ReadStatus(stream, summary);
  return summary;
}

AugmentedLagrangianMinimizerSummary AugmentedLagrangianMinimizer::MinimizeEager(cudaStream_t stream,
                                                                                Problem &problem,
                                                                                bool read_back) {
  AugmentedLagrangianMinimizerSummary summary;
  summary.num_problems = problem.NumProblems();

  // Constraint batches, and whether they are the ones of the last solve.
  const auto &residual_batches = problem.GetResidualBatches();
  std::vector<std::pair<const void *, size_t>> signature;
  std::vector<Constraint> constraints;
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
  signature_ = signature;
  problem_ = &problem;

  if (constraints.empty()) {
    MinimizeCallOptions call;
    call.max_num_iterations = options_.final_inner_iterations > 0
                                  ? options_.final_inner_iterations
                                  : minimizer_.Options().max_num_iterations;
    call.max_line_search_steps = options_.inner_line_search_steps;
    call.reuse_structure = options_.reuse_structure;
    call.fixed_iterations = options_.real_time;
    if (options_.real_time) {
      call.max_num_iterations = options_.final_inner_iterations > 0
                                    ? options_.final_inner_iterations
                                    : options_.inner_iterations;
    }
    const MinimizerSummary inner = minimizer_.Minimize(stream, problem, call);
    summary.outer_iterations = 1;
    summary.inner_iterations = inner.num_iterations;
    summary.initial_cost = inner.initial_cost;
    summary.final_cost = inner.final_cost;
    summary.num_converged = summary.num_problems;
    has_constraints_ = false;
    return summary;
  }

  has_constraints_ = true;
  num_problems_ = problem.NumProblems();
  size_t total_rows = 0;
  if (reuse) {
    // Same constraint batches: keep their factor -> subproblem maps.
    for (const Constraint &con : constraints_)
      total_rows += con.num_factors * con.batch->ResidualsSize();
  } else {
    problem.CheckSizes();
    problem.PrepareStatePointers(stream);
    constraints_ = std::move(constraints);
    for (Constraint &con : constraints_) {
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
    for (Constraint &con : constraints_) con.batch->ResetMultipliers(stream);
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
  ScatterPenalties(stream);

  const bool real_time = options_.real_time;
  summary.initial_cost =
      real_time ? std::numeric_limits<float>::quiet_NaN() : ComputeObjective(stream, problem);
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
    MinimizeCallOptions call;
    call.max_num_iterations = cap;
    call.max_line_search_steps = options_.inner_line_search_steps;
    call.problem_at_cap = at_cap_.data();
    call.fixed_iterations = real_time;
    // Later outer iterations solve the same structure.
    call.reuse_structure = k > 0 || reuse;
    const MinimizerSummary inner = minimizer_.Minimize(stream, problem, call);
    summary.inner_iterations += inner.num_iterations;
    summary.outer_iterations = k + 1;
    LogMessage("Outer iteration #{}: {} inner iterations (cap {}), cost {}", k,
               inner.num_iterations, cap, inner.final_cost);

    // Constraint values and violations at the new states.
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(violation_.data(), 0, num_slots * sizeof(float), stream));
    size_t offset = 0;
    for (size_t c = 0; c < constraints_.size(); ++c) {
      const Constraint &con = constraints_[c];
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
    for (const Constraint &con : constraints_) {
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
      if (!last) ScatterPenalties(stream);
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
    ScatterPenalties(stream);
  }
  if (real_time) {
    // The one read-back of a real-time call (the last outer iteration's
    // status); left to the caller when the call is being captured.
    if (read_back) ReadStatus(stream, summary);
    return summary;
  }
  FillStatus(out.data(), summary);
  summary.final_cost = ComputeObjective(stream, problem);
  return summary;
}

}  // namespace cunls
