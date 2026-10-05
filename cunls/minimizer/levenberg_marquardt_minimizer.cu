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
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/**
 * Rejections in a row that take the damping from lambda_min to lambda_max
 * under the escalation of the class comment (the k-th multiplies by
 * lambda_upscale * 2^(k-1)).
 */
size_t RejectionsToMaxDamping(const LevenbergMarquardtMinimizerOptions &o) {
  float lambda = o.lambda_min;
  size_t k = 0;
  while (lambda < o.lambda_max && k < 64) {
    lambda *= o.lambda_upscale * std::ldexp(1.f, std::min(static_cast<int>(k), 30));
    ++k;
  }
  return k;
}

/**
 * The base options with room in max_consecutive_rejected_steps for the damping
 * to escalate first: the cap then counts the rejections at full damping (a
 * small lambda says little about whether progress is still possible).
 */
MinimizerOptions BaseOptions(const LevenbergMarquardtMinimizerOptions &o) {
  if (!(o.lambda_min > 0.f)) {
    throw std::invalid_argument("LevenbergMarquardtMinimizer: lambda_min must be positive");
  }
  if (!(o.lambda_max >= o.lambda_min)) {
    throw std::invalid_argument("LevenbergMarquardtMinimizer: lambda_max must be >= lambda_min");
  }
  if (!(o.initial_lambda <= o.lambda_max)) {
    throw std::invalid_argument(
        "LevenbergMarquardtMinimizer: initial_lambda must be <= lambda_max");
  }
  if (!(o.lambda_upscale > 1.f)) {
    throw std::invalid_argument("LevenbergMarquardtMinimizer: lambda_upscale must be > 1");
  }
  MinimizerOptions base = o.base_options;
  if (base.max_consecutive_rejected_steps > 0) {
    base.max_consecutive_rejected_steps += RejectionsToMaxDamping(o);
  }
  return base;
}

/** The damping options the λ update reads (kernel arguments must be trivially copyable). */
struct DampingRule {
  float initial_lambda, lambda_min, lambda_max, lambda_upscale, lambda_downscale,
      lambda_downscale_threshold;
};

/** λ of a subproblem after its previous step's outcome (see the class comment). */
__device__ float UpdatedLambda(const DampingRule &o, bool first, float lambda, int outcome,
                               int rejected, float quality) {
  if (first) return o.initial_lambda;
  if (outcome == kStepRejected) {
    // Nielsen's escalation: the k-th consecutive rejection multiplies by
    // lambda_upscale * 2^(k-1). Floored at lambda_min first, so an initial
    // lambda of 0 can grow.
    return fminf(
        fmaxf(lambda, o.lambda_min) * o.lambda_upscale * ldexpf(1.f, min(rejected - 1, 30)),
        o.lambda_max);
  }
  if (outcome == kStepTaken && quality > o.lambda_downscale_threshold) {
    return fminf(fmaxf(lambda * o.lambda_downscale, o.lambda_min), o.lambda_max);
  }
  // A step taken by the line search keeps the damping: the step length is the
  // line search's job.
  return lambda;
}

/**
 * One thread per row and per subproblem (whichever is more): thread p < P
 * writes subproblem p's updated λ to lambda_out; thread r < rows writes
 * λ'_{problem(r)} diag(H)_r to damping (recomputing λ', so no thread reads
 * what another writes).
 */
__global__ void damping_kernel(size_t num_rows, size_t num_problems, const int *row_problem,
                               DampingRule o, bool first, const int *outcome, const int *rejected,
                               const float *quality, const float *lambda_in, float *lambda_out,
                               const float *diagonal, float *damping) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < num_problems) {
    lambda_out[i] = UpdatedLambda(o, first, first ? 0.f : lambda_in[i], first ? 0 : outcome[i],
                                  first ? 0 : rejected[i], first ? 0.f : quality[i]);
  }
  if (i < num_rows) {
    const int p = row_problem != nullptr ? row_problem[i] : 0;
    const float lambda = UpdatedLambda(o, first, first ? 0.f : lambda_in[p], first ? 0 : outcome[p],
                                       first ? 0 : rejected[p], first ? 0.f : quality[p]);
    damping[i] = lambda * diagonal[i];
  }
}

/** ρ and the rules of the class comment, per active subproblem. */
__global__ void classify_kernel(size_t num_problems, const float *cost, const float *new_cost,
                                const float *step_squared, const float *diag_weight,
                                const float *matrix_weight, const float *lambda, const int *active,
                                float state_tolerance, float cost_tolerance,
                                float relative_reduction_tolerance, float step_accept_threshold,
                                float *quality, int *reject, int *converged) {
  const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (p >= num_problems || !active[p]) return;
  const float current = cost[p], updated = new_cost[p];
  // The damped step solves (H + λD) δ = -g, so the model decrease
  // -gᵀδ - ½ δᵀHδ is ½ δᵀHδ + λ δᵀDδ (H undamped; cost = ½ Σ ρ). Relative:
  const float predicted = (0.5f * matrix_weight[p] + lambda[p] * diag_weight[p]) / current;
  const float rho = (1.f - updated / current) / predicted;
  quality[p] = rho;
  // Written so that a NaN rho (non-finite cost) rejects.
  reject[p] = !(rho >= step_accept_threshold);
  converged[p] = step_squared[p] < state_tolerance || predicted < relative_reduction_tolerance ||
                 updated < cost_tolerance;
}

}  // namespace

LevenbergMarquardtMinimizer::LevenbergMarquardtMinimizer(
    const LevenbergMarquardtMinimizerOptions &options)
    : Minimizer(BaseOptions(options)), options_(options) {}

void LevenbergMarquardtMinimizer::UpdateSystem(cudaStream_t stream, size_t iteration,
                                               NormalEquations &system,
                                               const ProblemPartition &partition) {
  const size_t num_problems = partition.NumProblems();
  const bool first = iteration == 0;
  if (first) {
    lambda_.resize(num_problems);
    next_lambda_.resize(num_problems);
    quality_.resize(num_problems);
  }
  // (H + λ_p diag(H)) on the rows of subproblem p, with λ updated in the same pass.
  system.ExtractLhsDiagonal(stream, diagonal_);
  damping_.resize(diagonal_.size());
  const size_t n = std::max(diagonal_.size(), num_problems);
  damping_kernel<<<Blocks(n), kThreads, 0, stream>>>(
      diagonal_.size(), num_problems, partition.RowProblem(),
      DampingRule{options_.initial_lambda, options_.lambda_min, options_.lambda_max,
                  options_.lambda_upscale, options_.lambda_downscale,
                  options_.lambda_downscale_threshold},
      first, partition.Outcome(), partition.Rejected(), quality_.data(), lambda_.data(),
      next_lambda_.data(), diagonal_.data(), damping_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  std::swap(lambda_, next_lambda_);
  system.AddScaledDiagonalToLhs(stream, 1.f, damping_);
}

void LevenbergMarquardtMinimizer::ClassifySteps(cudaStream_t stream, const NormalEquations &system,
                                                const dvector<float> &step,
                                                ProblemPartition &partition) {
  const size_t n = partition.NumProblems();
  diag_weight_.resize(n);
  matrix_weight_.resize(n);
  // The predicted reduction's terms δᵀDδ and δᵀHδ, per subproblem. The step
  // is in physical coordinates (dx = S z, see ColumnScaling), so the damping
  // term λ zᵀ diag(S H S) z is λ dxᵀ diag(H) dx with the unscaled Hessian;
  // without column scaling that is the diagonal UpdateSystem extracted.
  const dvector<float> *hessian_diagonal = &diagonal_;
  if (Options().column_scaling != ColumnScaling::None) {
    system.ExtractHessianDiagonal(stream, hessian_diagonal_);
    hessian_diagonal = &hessian_diagonal_;
  }
  partition.SumRows(stream, step.data(), nullptr, hessian_diagonal->data(), diag_weight_.data());
  hessian_step_.resize(step.size());
  system.MultiplyHessian(stream, cusparse_handle_.GetHandle(stream), step, hessian_step_, buffer_);
  partition.SumRows(stream, step.data(), hessian_step_.data(), nullptr, matrix_weight_.data());
  const MinimizerOptions &base = Options();
  classify_kernel<<<Blocks(n), kThreads, 0, stream>>>(
      n, partition.Cost(), partition.NewCost(), partition.StepSquared(), diag_weight_.data(),
      matrix_weight_.data(), lambda_.data(), partition.Active(), base.state_tolerance,
      base.cost_tolerance, options_.relative_reduction_tolerance, options_.step_accept_threshold,
      quality_.data(), partition.Reject(), partition.Converged());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
