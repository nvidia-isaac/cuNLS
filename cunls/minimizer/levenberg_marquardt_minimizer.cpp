/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
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
 * @param[out] lhs Output left-hand side matrix (H + lambda * diag(H)).
 * @param[out] rhs Output right-hand side vector (-J^T r).
 */
void LevenbergMarquardtMinimizer::BuildSystem(
    cudaStream_t stream, const Problem& problem,
    const MinimizerState& minimizer_state, CSRSparseMatrix& lhs,
    dvector<float>& rhs) {
  GaussNewtonMinimizer::BuildSystem(stream, problem, minimizer_state, hessian_,
                                    rhs);
  ExtractDiagonal(stream, hessian_, diagonal_);
  AddScaledDiagonal(stream, lambda_, diagonal_, hessian_, lhs);
}

/**
 * @brief Checks convergence using LM-specific criteria.
 *
 * Computes the rho metric which measures the ratio of actual cost reduction to
 * predicted cost reduction:
 *   rho = (actual_reduction) / (predicted_reduction)
 *
 * Convergence is determined by:
 * 1. Step size: squared step norm < state_tolerance
 * 2. Cost: updated_cost < cost_tolerance
 * 3. Predicted reduction: predicted_relative_reduction <
 * relative_reduction_tolerance
 *
 * @param stream CUDA stream for GPU operations.
 * @param updated_cost Cost after applying the step.
 * @param current_cost Cost before applying the step.
 * @param step State update step vector.
 * @param[out] step_quality Output rho metric (actual/predicted reduction).
 * @return True if converged, false otherwise.
 */
bool LevenbergMarquardtMinimizer::CheckConvergence(cudaStream_t stream,
                                                   float updated_cost,
                                                   float current_cost,
                                                   const dvector<float>& step,
                                                   float& step_quality) {
  float step_sq_norm = ComputeSquaredStep(stream, step_);

  auto handle = cusparse_handle_.GetHandle(stream);

  float diag_weight =
      ComputeWeightedSquaredStep(stream, diagonal_, step_, buffer_);
  float matrix_weight =
      ComputeWeightedSquaredStep(stream, handle, hessian_, step_, buffer_);

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Predicted reduction = (step^T H step + 2 * lambda * step^T diag(H) step) /
  // cost
  float predicted_relative_reduction =
      (matrix_weight + 2.f * lambda_ * diag_weight) / current_cost;

  LogMessage("Predicted relative reduction = {}", predicted_relative_reduction);
  LogMessage("Step squared norm = {}", step_sq_norm);

  // rho = actual_reduction / predicted_reduction
  float rho =
      (1.f - updated_cost / current_cost) / predicted_relative_reduction;

  step_quality = rho;

  if (step_sq_norm < options_.base_options.state_tolerance ||
      predicted_relative_reduction < options_.relative_reduction_tolerance ||
      updated_cost < options_.base_options.cost_tolerance) {
    return true;
  }

  return false;
}

/**
 * @brief Determines if a step should be rejected (LM version).
 *
 * Rejects step if rho < step_accept_threshold. When rejecting, increases
 * lambda by lambda_upscale to make the next step more conservative.
 *
 * @param step_quality Rho metric (actual/predicted cost reduction).
 * @return True if step should be rejected, false otherwise.
 */
bool LevenbergMarquardtMinimizer::RejectStep(float step_quality) {
  if (step_quality < options_.step_accept_threshold) {
    LogMessage("Reject step");
    // Increase lambda to make next step more conservative
    lambda_ *= options_.lambda_upscale;
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
void LevenbergMarquardtMinimizer::Initialize(cudaStream_t stream,
                                             const Problem& problem) {
  GaussNewtonMinimizer::Initialize(stream, problem);
  lambda_ = options_.initial_lambda;
}
}  // namespace cunls