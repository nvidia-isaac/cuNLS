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

#include "cunls/minimizer/levenberg_marquardt_minimizer.h"

#include <algorithm>
#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

/**
 * @brief Builds the Levenberg-Marquardt linear system.
 *
 * First builds the Gauss-Newton system (H = J^T J), then extracts the diagonal
 * and adds lambda * diag(H) to create the damped system:
 *   (H + lambda * diag(H)) dx = -J^T r
 *
 * This damping makes the system more positive definite and improves convergence
 * robustness, especially when J^T J is ill-conditioned.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem.
 * @param minimizer_state Current minimizer state.
 */
void LevenbergMarquardtMinimizer::BuildSystem(cudaStream_t stream, const Problem &problem,
                                              const MinimizerState &minimizer_state) {
  GaussNewtonMinimizer::BuildSystem(stream, problem, minimizer_state);
  normal_equations_.ExtractLhsDiagonal(stream, diagonal_);
  if (batched_) {
    // One lambda per subproblem: add lambda_p * diag(H) row by row.
    damping_.resize(diagonal_.size());
    partition_.ScaleRows(stream, partition_.Lambda(), diagonal_.data(), damping_.data());
    normal_equations_.AddScaledDiagonalToLhs(stream, 1.f, damping_);
    return;
  }
  normal_equations_.AddScaledDiagonalToLhs(stream, lambda_, diagonal_);
}

BatchedStepControlParams LevenbergMarquardtMinimizer::BatchedParams() const {
  BatchedStepControlParams params = GaussNewtonMinimizer::BatchedParams();
  params.levenberg_marquardt = true;
  params.relative_reduction_tolerance = options_.relative_reduction_tolerance;
  params.step_accept_threshold = options_.step_accept_threshold;
  params.lambda_upscale = options_.lambda_upscale;
  params.lambda_downscale = options_.lambda_downscale;
  params.lambda_downscale_threshold = options_.lambda_downscale_threshold;
  params.lambda_min = options_.lambda_min;
  params.lambda_max = options_.lambda_max;
  return params;
}

void LevenbergMarquardtMinimizer::AccumulatePredictedReduction(cudaStream_t stream) {
  // The single-problem terms of EvaluateAndCheckConvergence, per subproblem.
  partition_.AccumulateRows(stream, step_.data(), nullptr, diagonal_.data(),
                            partition_.DiagWeight());
  hessian_step_.resize(step_.size());
  normal_equations_.MultiplyHessian(stream, cusparse_handle_.GetHandle(stream), step_,
                                    hessian_step_, buffer_);
  partition_.AccumulateRows(stream, step_.data(), hessian_step_.data(), nullptr,
                            partition_.MatrixWeight());
}

bool LevenbergMarquardtMinimizer::EvaluateAndCheckConvergence(
    cudaStream_t stream, const Problem &problem, const MinimizerState &updated_state,
    float current_cost, const dvector<float> &step, float &updated_cost, float &step_quality) {
  // 4 slots: [0]=cost, [1]=step_sq_norm, [2]=diag_weight, [3]=matrix_weight
  constexpr size_t kSlots = 4;
  if (d_scalars_.size() < kSlots) d_scalars_.resize(kSlots);
  if (h_scalars_.size() < kSlots) h_scalars_.resize(kSlots);

  // Enqueue cost reduction
  ComputeCostAsync(stream, problem, updated_state, d_scalars_.data());

  size_t partials_needed = ReducePartialCount(step_.size());
  if (d_reduce_partials_.size() < partials_needed) {
    d_reduce_partials_.resize(partials_needed);
  }

  // Enqueue squared step norm
  ComputeSquaredStepAsync(stream, step_, d_scalars_.data() + 1, d_reduce_partials_.data());

  // Enqueue diag-weighted step norm
  ComputeWeightedSquaredStepAsync(stream, diagonal_, step_, d_scalars_.data() + 2,
                                  d_reduce_partials_.data());

  // Enqueue sparse-weighted step norm (SpMV + dot, all on stream)
  normal_equations_.WeightedSquaredStepAsync(stream, cusparse_handle_.GetHandle(stream), step_,
                                             d_scalars_.data() + 3, d_reduce_partials_.data(),
                                             buffer_);

  // Single D2H + single sync for all 4 scalars
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(h_scalars_.data(), d_scalars_.data(), kSlots * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  updated_cost = h_scalars_[0];
  float step_sq_norm = h_scalars_[1];
  float diag_weight = h_scalars_[2];
  float matrix_weight = h_scalars_[3];

  // The damped step solves (H + λD) δ = -g, so the model decrease -gᵀδ - ½ δᵀHδ is
  // ½ δᵀHδ + λ δᵀDδ (H undamped; cost = ½ Σ ρ). Relative to the current cost:
  float predicted_relative_reduction =
      (0.5f * matrix_weight + lambda_ * diag_weight) / current_cost;

  LogMessage("Predicted relative reduction = {}", predicted_relative_reduction);
  LogMessage("Step squared norm = {}", step_sq_norm);

  float rho = (1.f - updated_cost / current_cost) / predicted_relative_reduction;

  step_quality = rho;

  return (step_sq_norm < options_.base_options.state_tolerance ||
          predicted_relative_reduction < options_.relative_reduction_tolerance ||
          updated_cost < options_.base_options.cost_tolerance);
}

/**
 * @brief Determines if a step should be rejected (LM version).
 *
 * Rejects step if rho < step_accept_threshold. When rejecting, increases
 * lambda by lambda_upscale * 2^(k-1) at the k-th consecutive rejection, so the
 * damping escalates until the model is trusted (see lambda_upscale).
 *
 * @param step_quality Rho metric (actual/predicted cost reduction).
 * @return True if step should be rejected, false otherwise.
 */
bool LevenbergMarquardtMinimizer::RejectStep(float step_quality) {
  // !(rho >= threshold): a NaN rho (a step to a non-finite cost) is rejected.
  if (!(step_quality >= options_.step_accept_threshold)) {
    LogMessage("Reject step");
    // Increase lambda to make next step more conservative, faster at each
    // consecutive rejection.
    // Floored at lambda_min first: an initial lambda of 0 is valid (pure
    // Gauss-Newton steps) but would never grow.
    lambda_ = std::max(lambda_, options_.lambda_min) * options_.lambda_upscale *
              std::ldexp(1.f, std::min(consecutive_rejects_, 30));
    lambda_ = std::min(lambda_, options_.lambda_max);
    ++consecutive_rejects_;
    return true;
  }
  return false;
}

/**
 * @brief Determines if a step should be accepted (LM version).
 *
 * Accepts step if rho >= step_accept_threshold. If the step is very successful
 * (rho > lambda_downscale_threshold), decreases lambda by lambda_downscale to
 * make the next step more aggressive (closer to Gauss-Newton). Lambda is
 * clamped between lambda_min and lambda_max.
 *
 * @param step_quality Rho metric (actual/predicted cost reduction).
 * @return True if step should be accepted, false otherwise.
 */
bool LevenbergMarquardtMinimizer::AcceptStep(float step_quality) {
  if (step_quality >= options_.step_accept_threshold) {
    LogMessage("Accept step");
    consecutive_rejects_ = 0;

    // If step is very successful, decrease lambda to be more aggressive
    if (step_quality > options_.lambda_downscale_threshold) {
      LogMessage("Adjust lambda");
      lambda_ = lambda_ * options_.lambda_downscale;
      // Clamp lambda to valid range
      lambda_ = std::max(lambda_, options_.lambda_min);
      lambda_ = std::min(lambda_, options_.lambda_max);
    }
    return true;
  }
  return false;
}

/**
 * @brief Initializes LM-specific data structures.
 *
 * Calls the base class initialization to set up residuals, Jacobian, and
 * state operations, then initializes the damping factor lambda.
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The optimization problem to initialize for.
 */
void LevenbergMarquardtMinimizer::Initialize(cudaStream_t stream, Problem &problem) {
  GaussNewtonMinimizer::Initialize(stream, problem);
}

void LevenbergMarquardtMinimizer::BeginCall() {
  lambda_ = options_.initial_lambda;
  consecutive_rejects_ = 0;
}
}  // namespace cunls
