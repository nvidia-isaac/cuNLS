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
        id = v.ids[(p - v.base) / v.ambient];
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
  const int id = ids[s];
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

__global__ void scale_rows_kernel(const int *keys, const float *per_problem, const float *in,
                                  size_t n, float *out) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = per_problem[keys[i]] * in[i];
}

__global__ void copy_accepted_kernel(const float *from, float *to, const int *ids,
                                     const int *accept, size_t ambient, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n && accept[ids[i / ambient]]) to[i] = from[i];
}

__global__ void init_control_kernel(const float *cost, size_t num_problems, float cost_tolerance,
                                    float initial_lambda, float *lambda, int *active, int *rejected,
                                    int *accept, float *out) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  const int is_active = cost[p] >= cost_tolerance;
  active[p] = is_active;
  rejected[p] = 0;
  accept[p] = 0;
  lambda[p] = initial_lambda;
  atomicAdd(out, cost[p]);
  if (is_active) atomicAdd(out + 1, 1.f);
}

/**
 * The single-problem rules of GaussNewtonMinimizer / LevenbergMarquardtMinimizer
 * (EvaluateAndCheckConvergence, AcceptStep, RejectStep and the Minimize loop),
 * applied to each subproblem.
 */
__global__ void step_control_kernel(BatchedStepControlParams params, size_t num_problems,
                                    float *cost, const float *new_cost, const float *step_squared,
                                    const float *diag_weight, const float *matrix_weight,
                                    float *lambda, int *active, int *rejected, int *accept,
                                    float *out) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems) return;
  int take = 0;
  if (active[p]) {
    const float current = cost[p], updated = new_cost[p];
    bool converged;
    float quality;
    if (params.levenberg_marquardt) {
      const float predicted = (matrix_weight[p] + 2.f * lambda[p] * diag_weight[p]) / current;
      quality = (1.f - updated / current) / predicted;  // rho
      converged = step_squared[p] < params.state_tolerance ||
                  predicted < params.relative_reduction_tolerance ||
                  updated < params.cost_tolerance;
    } else {
      quality = updated / current;
      converged = step_squared[p] < params.state_tolerance || updated < params.cost_tolerance ||
                  quality >= 1.f;
    }
    if (!isfinite(updated)) {
      converged = false;  // a step to a non-finite cost is rejected, never taken
    }
    if (converged) {
      take = updated <= current;
      active[p] = 0;
    } else {
      // Written so that NaN quality (non-finite cost) rejects.
      const bool reject = params.levenberg_marquardt ? !(quality >= params.step_accept_threshold)
                                                     : !(quality < 1.f);
      if (reject) {
        if (params.levenberg_marquardt) lambda[p] *= params.lambda_upscale;
        rejected[p] += 1;
        if (params.max_consecutive_rejected_steps > 0 &&
            rejected[p] >= params.max_consecutive_rejected_steps) {
          active[p] = 0;
        }
      } else {
        take = 1;
        rejected[p] = 0;
        if (params.levenberg_marquardt && quality > params.lambda_downscale_threshold) {
          lambda[p] = fminf(fmaxf(lambda[p] * params.lambda_downscale, params.lambda_min),
                            params.lambda_max);
        }
      }
    }
    if (take) cost[p] = updated;
  }
  accept[p] = take;
  atomicAdd(out, cost[p]);
  if (active[p]) atomicAdd(out + 1, 1.f);
}

}  // namespace

void ProblemPartition::Build(cudaStream_t stream, const Problem &problem,
                             const MinimizerState &state, size_t num_rows) {
  num_problems_ = problem.NumProblems();
  const auto &state_batches = problem.GetStateBatches();
  const auto &ids = problem.StateProblemIds();
  const auto &states = state.GetStates();
  const int num_problems = static_cast<int>(num_problems_);

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

  const auto &residual_batches = problem.GetResidualBatches();
  num_factor_items_ = 0;
  for (const auto &rb : residual_batches)
    num_factor_items_ += rb.GetFactorBatch()->NumActiveFactors();
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

  num_rows_ = num_rows;
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

  for (auto *v : {&cost_, &new_cost_, &step_squared_, &diag_weight_, &matrix_weight_, &lambda_}) {
    v->resize(num_problems_);
  }
  for (auto *v : {&active_, &rejected_, &accept_}) v->resize(num_problems_);

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

void ProblemPartition::ResetAccumulators(cudaStream_t stream) {
  const size_t bytes = num_problems_ * sizeof(float);
  for (auto *v : {&new_cost_, &step_squared_, &diag_weight_, &matrix_weight_}) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(v->data(), 0, bytes, stream));
  }
}

void ProblemPartition::AccumulateFactorCosts(cudaStream_t stream, const float *factor_costs,
                                             float *costs) const {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(costs, 0, num_problems_ * sizeof(float), stream));
  if (num_factor_items_ == 0) return;
  accumulate_costs_kernel<<<Blocks(num_factor_items_), kThreads, 0, stream>>>(
      factor_problem_.data(), factor_costs, num_factor_items_, costs);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::AccumulateRows(cudaStream_t stream, const float *x, const float *y,
                                      const float *w, float *out) const {
  if (num_rows_ == 0) return;
  accumulate_by_key_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(row_problem_.data(), x, y, w,
                                                                       num_rows_, out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::ScaleRows(cudaStream_t stream, const float *per_problem, const float *in,
                                 float *out) const {
  if (num_rows_ == 0) return;
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
        from.GetStates()[b].data(), to.GetStates()[b].data(), ids[b], accept_.data(),
        state_batches[b]->AmbientSize(), n);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

void ProblemPartition::InitStepControl(cudaStream_t stream, float cost_tolerance,
                                       float initial_lambda, float *d_out) {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_out, 0, 2 * sizeof(float), stream));
  init_control_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
      cost_.data(), num_problems_, cost_tolerance, initial_lambda, lambda_.data(), active_.data(),
      rejected_.data(), accept_.data(), d_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ProblemPartition::StepControl(cudaStream_t stream, const BatchedStepControlParams &params,
                                   float *d_out) {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_out, 0, 2 * sizeof(float), stream));
  step_control_kernel<<<Blocks(num_problems_), kThreads, 0, stream>>>(
      params, num_problems_, cost_.data(), new_cost_.data(), step_squared_.data(),
      diag_weight_.data(), matrix_weight_.data(), lambda_.data(), active_.data(), rejected_.data(),
      accept_.data(), d_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
