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

#include <stdexcept>
#include <string>

#include "cunls/common/helper.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/** A state batch as seen by the factor map: its storage and subproblem ids. */
struct BatchView {
  const float *base;
  size_t ambient;
  size_t count;
  const int *ids;
};

enum ErrorBits : int { kMixedFactor = 1, kIdOutOfRange = 2, kUnknownPointer = 4 };

/**
 * Whether the problem has a partition into more than one subproblem; throws if
 * it has one but not one id array per state batch (e.g. a state batch added
 * after Problem::SetProblemPartition).
 */
bool HasPartition(const Problem &problem) {
  if (problem.NumProblems() <= 1) return false;
  const size_t expected = problem.GetStateBatches().size();
  const size_t actual = problem.StateProblemIds().size();
  if (actual != expected) {
    throw std::invalid_argument("Problem partition: " + std::to_string(problem.NumProblems()) +
                                " subproblems need one id array per state batch (expected " +
                                std::to_string(expected) + ", got " + std::to_string(actual) +
                                "); call SetProblemPartition after adding every state batch");
  }
  return true;
}

/**
 * Adds v to out[key]. When every lane of the warp has the same key (the usual
 * case: subproblems own contiguous ranges), the warp reduces first and issues
 * one atomic. Must be called by all lanes of the warp; key < 0 adds nothing.
 */
__device__ void SegmentedAdd(float *out, int key, float v) {
  constexpr unsigned kFull = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int key0 = __shfl_sync(kFull, key, 0);
  if (__all_sync(kFull, key == key0) && key0 >= 0) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(kFull, v, offset);
    if (lane == 0) atomicAdd(out + key0, v);
  } else if (key >= 0) {
    atomicAdd(out + key, v);
  }
}

__global__ void factor_problem_kernel(float *const *pointers, size_t num_items, size_t slots,
                                      const BatchView *batches, int num_batches, int num_problems,
                                      int *factor_problem, int *error) {
  const size_t f = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (f >= num_items) return;
  int problem = -1;
  for (size_t k = 0; k < slots; ++k) {
    const float *p = pointers[f * slots + k];
    int id = -2;
    for (int b = 0; b < num_batches; ++b) {
      const BatchView v = batches[b];
      if (p >= v.base && p < v.base + v.count * v.ambient) {
        id = v.ids != nullptr ? v.ids[(p - v.base) / v.ambient] : 0;
        break;
      }
    }
    if (id == -2) {
      atomicOr(error, kUnknownPointer);
      continue;
    }
    if (id < 0 || id >= num_problems) atomicOr(error, kIdOutOfRange);
    if (problem == -1) {
      problem = id;
    } else if (id != problem) {
      atomicOr(error, kMixedFactor);
    }
  }
  factor_problem[f] = problem;
}

__global__ void row_problem_kernel(const int *column_offsets, const int *ids, size_t num_states,
                                   int tangent, int num_problems, int *row_problem, int *error) {
  const size_t s = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (s >= num_states) return;
  const int column = column_offsets[s];
  if (column < 0) return;  // constant state: no rows
  const int id = ids != nullptr ? ids[s] : 0;
  if (id < 0 || id >= num_problems) atomicOr(error, kIdOutOfRange);
  for (int t = 0; t < tangent; ++t) row_problem[column + t] = id;
}

__global__ void accumulate_by_key_kernel(const int *keys, const float *x, const float *y,
                                         const float *w, size_t n, float *out) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int key = -1;
  float v = 0.f;
  if (i < n) {
    key = keys[i];
    v = x[i] * (y != nullptr ? y[i] : x[i]);
    if (w != nullptr) v *= w[i];
  }
  SegmentedAdd(out, key, v);
}

__global__ void accumulate_costs_kernel(const int *keys, const float *costs, size_t n, float *out) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int key = -1;
  float v = 0.f;
  if (i < n) {
    key = keys[i];
    v = costs[i];
  }
  SegmentedAdd(out, key, v);
}

__global__ void scale_rows_single_kernel(const float *per_problem, const float *in, size_t n,
                                         float *out) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = per_problem[0] * in[i];
}

__global__ void scale_rows_kernel(const int *keys, const float *per_problem, const float *in,
                                  size_t n, float *out) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = per_problem[keys[i]] * in[i];
}

__global__ void copy_accepted_kernel(const float *from, float *to, const int *ids,
                                     const int *accept, size_t ambient, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n && accept[ids != nullptr ? ids[i / ambient] : 0]) to[i] = from[i];
}

/**
 * out[0] += cost, out[1] += active over the subproblems. One subproblem
 * writes them directly (out need not be zeroed); otherwise atomics into a
 * zeroed out.
 */
__device__ void AddToTotals(size_t num_problems, float cost, int active, float *out) {
  if (num_problems == 1) {
    out[0] = cost;
    out[1] = active ? 1.f : 0.f;
    return;
  }
  atomicAdd(out, cost);
  if (active) atomicAdd(out + 1, 1.f);
}

__global__ void init_control_kernel(const float *cost, size_t num_problems, float cost_tolerance,
                                    const int *frozen, int *active, int *rejected, int *accept,
                                    int *outcome, float *out) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  // A frozen subproblem (InnerSolve::problem_frozen) is never active: it takes
  // no step and keeps its states.
  const int is_active = cost[p] >= cost_tolerance && (frozen == nullptr || frozen[p] == 0);
  active[p] = is_active;
  rejected[p] = 0;
  accept[p] = 0;
  outcome[p] = kStepInactive;
  AddToTotals(num_problems, cost[p], is_active, out);
}

__global__ void line_search_kernel(size_t num_problems, const float *cost, const float *new_cost,
                                   const int *active, float *step_scale, int *shortened,
                                   float *count) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  // Written so that a NaN new cost is shortened too.
  if (active[p] && !(new_cost[p] < cost[p])) {
    step_scale[p] = 0.5f;
    shortened[p] = 1;
    atomicAdd(count, 1.f);
  } else {
    step_scale[p] = 1.f;
  }
}

/**
 * The decision shared by every minimizer, per subproblem, from the
 * minimizer's classification (reject, converged); see ProblemPartition::StepControl.
 */
__global__ void step_control_kernel(StepControlParams params, size_t num_problems, float *cost,
                                    const float *new_cost, const int *reject, const int *converged,
                                    const int *shortened, int *active, int *rejected, int *accept,
                                    int *outcome, float *out) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  int take = 0;
  int result = kStepInactive;
  if (active[p]) {
    const float current = cost[p], updated = new_cost[p];
    // A step to a non-finite cost is rejected, never taken.
    const bool finite = isfinite(updated);
    const bool is_rejected = reject[p] != 0 || !finite;
    if (params.line_search && updated < current && (shortened[p] || is_rejected)) {
      // Line search: a (possibly shortened) step that decreases the cost is
      // taken (see MinimizerOptions::max_line_search_steps).
      take = 1;
      rejected[p] = 0;
      result = kStepLineSearched;
    } else if (converged[p] != 0 && finite) {
      take = updated <= current;
      active[p] = 0;
      result = kStepConverged;
    } else if (is_rejected) {
      rejected[p] += 1;
      if (params.max_consecutive_rejected_steps > 0 &&
          rejected[p] >= params.max_consecutive_rejected_steps) {
        active[p] = 0;
      }
      result = kStepRejected;
    } else {
      take = 1;
      rejected[p] = 0;
      result = kStepTaken;
    }
    if (take) cost[p] = updated;
  }
  accept[p] = take;
  outcome[p] = result;
  AddToTotals(num_problems, cost[p], active[p], out);
  if (num_problems == 1) out[2] = static_cast<float>(take);
}

}  // namespace

void ComputeFactorProblemIds(cudaStream_t stream, const Problem &problem,
                             size_t residual_batch_index, int *factor_problem) {
  const auto &state_batches = problem.GetStateBatches();
  const auto &ids = problem.StateProblemIds();
  const size_t n =
      problem.GetResidualBatches()[residual_batch_index].GetFactorBatch()->NumActiveFactors();
  if (n == 0) return;
  if (!HasPartition(problem)) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(factor_problem, 0, n * sizeof(int), stream));
    return;
  }
  std::vector<BatchView> views;
  for (size_t b = 0; b < state_batches.size(); ++b) {
    const StateBatch *batch = state_batches[b];
    const size_t count = batch->NumActiveStates();
    if (count == 0) continue;
    const float *base = batch->StateDevicePtr(0);
    if (batch->StateDevicePtr(count - 1) != base + (count - 1) * batch->AmbientSize()) {
      throw std::invalid_argument(
          "Problem partition: a state batch does not store its states contiguously");
    }
    views.push_back({base, batch->AmbientSize(), count, ids[b]});
  }
  dvector<BatchView> d_views(views, stream);
  dvector<int> error(0, 1, stream);
  const size_t slots = problem.NumStatePointers(residual_batch_index) / n;
  factor_problem_kernel<<<Blocks(n), kThreads, 0, stream>>>(
      problem.DeviceStatePointers(residual_batch_index), n, slots, d_views.data(),
      static_cast<int>(views.size()), static_cast<int>(problem.NumProblems()), factor_problem,
      error.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  // d_views and error are freed at scope exit: wait for the kernel.
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void ProblemPartition::Build(cudaStream_t stream, const Problem &problem,
                             const MinimizerState &state, size_t num_rows) {
  num_problems_ = problem.NumProblems();
  const auto &state_batches = problem.GetStateBatches();
  const auto &ids = problem.StateProblemIds();
  const auto &states = state.GetStates();
  const int num_problems = static_cast<int>(num_problems_);

  const auto &residual_batches = problem.GetResidualBatches();
  num_factor_items_ = 0;
  for (const auto &rb : residual_batches)
    num_factor_items_ += rb.GetFactorBatch()->NumActiveFactors();
  num_rows_ = num_rows;
  for (auto *v : {&cost_, &new_cost_, &step_squared_, &step_scale_}) v->resize(num_problems_);
  for (auto *v : {&active_, &rejected_, &reject_, &converged_, &outcome_, &accept_, &shortened_}) {
    v->resize(num_problems_);
  }
  // Without a partition (one subproblem) every state belongs to subproblem 0,
  // and every per-subproblem operation has a direct path: no maps to build or
  // validate. A partition must have one id array per state batch.
  if (!HasPartition(problem)) return;

  error_.resize(1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(error_.data(), 0, sizeof(int), stream));
  std::vector<BatchView> views;
  for (size_t b = 0; b < state_batches.size(); ++b) {
    views.push_back({states[b].data(), state_batches[b]->AmbientSize(),
                     state_batches[b]->NumActiveStates(), ids[b]});
  }
  // Reused across solves (no allocation once sized); the copy is ordered on
  // the stream before the kernels that read it.
  views_.resize(views.size() * sizeof(BatchView));
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(views_.data(), views.data(), views_.size(), cudaMemcpyHostToDevice, stream));
  const auto *d_views = reinterpret_cast<const BatchView *>(views_.data());

  factor_problem_.resize(num_factor_items_);
  size_t offset = 0;
  for (size_t i = 0; i < residual_batches.size(); ++i) {
    const size_t n = residual_batches[i].GetFactorBatch()->NumActiveFactors();
    if (n == 0) continue;
    const size_t slots = problem.NumStatePointers(i) / n;
    factor_problem_kernel<<<Blocks(n), kThreads, 0, stream>>>(
        state.GetStatePointers()[i].data(), n, slots, d_views, static_cast<int>(views.size()),
        num_problems, factor_problem_.data() + offset, error_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    offset += n;
  }

  row_problem_.resize(num_rows);
  int first_column = 0;
  for (size_t b = 0; b < state_batches.size(); ++b) {
    const StateBatch *batch = state_batches[b];
    const size_t count = batch->NumActiveStates();
    if (count == 0) continue;
    ComputeStateColumnOffsets(stream, first_column, batch, column_offsets_);
    row_problem_kernel<<<Blocks(count), kThreads, 0, stream>>>(
        column_offsets_.data(), ids[b], count, static_cast<int>(batch->TangentSize()), num_problems,
        row_problem_.data(), error_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    first_column += static_cast<int>((count - batch->NumConstStates()) * batch->TangentSize());
  }

  int error = 0;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&error, error_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  if (error & kMixedFactor) {
    throw std::invalid_argument(
        "Problem partition: a factor connects states of different subproblems");
  }
  if (error & kIdOutOfRange) {
    throw std::invalid_argument("Problem partition: a subproblem id is outside [0, " +
                                std::to_string(num_problems_) + ")");
  }
  if (error & kUnknownPointer) {
    throw std::invalid_argument("Problem partition: a factor references an unknown state");
  }
}

void ProblemPartition::ResetLineSearch(cudaStream_t stream) {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(shortened_.data(), 0, num_problems_ * sizeof(int), stream));
}

void ProblemPartition::MarkLineSearch(cudaStream_t stream, float *d_count) {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_count, 0, sizeof(float), stream));
  line_search_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
      num_problems_, cost_.data(), new_cost_.data(), active_.data(), step_scale_.data(),
      shortened_.data(), d_count);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::SumFactorCosts(cudaStream_t stream, const float *factor_costs,
                                      float *costs) {
  if (num_problems_ == 1) {
    // One subproblem: the deterministic reduction of a plain sum.
    if (num_factor_items_ == 0) {
      THROW_ON_CUDA_ERROR(cudaMemsetAsync(costs, 0, sizeof(float), stream));
      return;
    }
    const size_t partials = ReducePartialCount(num_factor_items_);
    if (partials_.size() < partials) partials_.resize(partials);
    ReduceSumToDevice(stream, factor_costs, num_factor_items_, costs, partials_.data());
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(costs, 0, num_problems_ * sizeof(float), stream));
  if (num_factor_items_ == 0) return;
  accumulate_costs_kernel<<<Blocks(num_factor_items_), kThreads, 0, stream>>>(
      factor_problem_.data(), factor_costs, num_factor_items_, costs);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::SumRows(cudaStream_t stream, const float *x, const float *y, const float *w,
                               float *out) {
  if (num_rows_ == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(out, 0, num_problems_ * sizeof(float), stream));
    return;
  }
  if (num_problems_ == 1) {
    // One subproblem: the deterministic reductions of a plain dot product.
    const size_t partials = ReducePartialCount(num_rows_);
    if (partials_.size() < partials) partials_.resize(partials);
    if (w != nullptr) {
      WeightedDotProductToDevice(stream, x, w, y != nullptr ? y : x, num_rows_, out,
                                 partials_.data());
    } else {
      DotProductToDevice(stream, x, y != nullptr ? y : x, num_rows_, out, partials_.data());
    }
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(out, 0, num_problems_ * sizeof(float), stream));
  accumulate_by_key_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(row_problem_.data(), x, y, w,
                                                                       num_rows_, out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::ScaleRows(cudaStream_t stream, const float *per_problem, const float *in,
                                 float *out) const {
  if (num_rows_ == 0) return;
  if (num_problems_ == 1) {
    scale_rows_single_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(per_problem, in, num_rows_,
                                                                         out);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
  scale_rows_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(row_problem_.data(), per_problem,
                                                                in, num_rows_, out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::CopyAccepted(cudaStream_t stream, const Problem &problem,
                                    const MinimizerState &from, MinimizerState &to) const {
  const auto &state_batches = problem.GetStateBatches();
  const auto &ids = problem.StateProblemIds();
  for (size_t b = 0; b < state_batches.size(); ++b) {
    const size_t n = from.GetStates()[b].size();
    if (n == 0) continue;
    copy_accepted_kernel<<<Blocks(n), kThreads, 0, stream>>>(
        from.GetStates()[b].data(), to.GetStates()[b].data(),
        ids.size() == state_batches.size() ? ids[b] : nullptr, accept_.data(),
        state_batches[b]->AmbientSize(), n);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

void ProblemPartition::InitStepControl(cudaStream_t stream, float cost_tolerance, const int *frozen,
                                       float *d_out) {
  if (num_problems_ > 1) THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_out, 0, 2 * sizeof(float), stream));
  init_control_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
      cost_.data(), num_problems_, cost_tolerance, frozen, active_.data(), rejected_.data(),
      accept_.data(), outcome_.data(), d_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::StepControl(cudaStream_t stream, const StepControlParams &params,
                                   float *d_out) {
  if (num_problems_ > 1) THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_out, 0, 2 * sizeof(float), stream));
  step_control_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
      params, num_problems_, cost_.data(), new_cost_.data(), reject_.data(), converged_.data(),
      shortened_.data(), active_.data(), rejected_.data(), accept_.data(), outcome_.data(), d_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
