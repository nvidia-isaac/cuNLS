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

#pragma once

#include <cuda_runtime.h>

#include "gauss_newton_minimizer.h"

namespace cunls {

/**
 * @brief Configuration options for the Levenberg-Marquardt optimizer.
 *
 * Extends MinimizerOptions with Levenberg-Marquardt specific parameters for
 * controlling the damping factor lambda and step acceptance criteria.
 */
struct LevenbergMarquardtMinimizerOptions {
  /** @brief Base Gauss-Newton options. */
  MinimizerOptions base_options;

  /**
   * @brief Initial value for the damping factor lambda.
   *
   * Controls the initial regularization strength. Higher values make the
   * algorithm more conservative (closer to gradient descent).
   * Default: 1e-3
   */
  float initial_lambda = 1e-3;

  /**
   * @brief Convergence threshold for predicted relative cost reduction.
   *
   * Optimizer terminates when the predicted relative reduction falls below
   * this threshold.
   * Default: 1e-6
   */
  float relative_reduction_tolerance = 1e-6;

  /**
   * @brief Factor by which lambda is increased when a step is rejected.
   *
   * When a step increases the cost, lambda is multiplied by this factor to
   * make the next step more conservative.
   * Default: 2.0
   */
  float lambda_upscale = 2.0f;

  /**
   * @brief Factor by which lambda is decreased when a step is accepted.
   *
   * When a step is very successful, lambda is multiplied by this factor to
   * make the next step more aggressive (closer to Gauss-Newton).
   * Default: 0.5
   */
  float lambda_downscale = 0.5f;

  /**
   * @brief Maximum allowed value for lambda.
   *
   * Prevents lambda from growing too large, which would make the algorithm
   * too conservative.
   * Default: 1e+6
   */
  float lambda_max = 1e+6;

  /**
   * @brief Minimum allowed value for lambda.
   *
   * Prevents lambda from becoming too small, which could cause numerical
   * instability.
   * Default: 1e-6
   */
  float lambda_min = 1e-6;

  /**
   * @brief Threshold for step acceptance based on rho (step quality).
   *
   * A step is accepted if rho >= step_accept_threshold. The rho value
   * measures the ratio of actual to predicted cost reduction.
   * Default: 0.25
   */
  float step_accept_threshold = 0.25f;

  /**
   * @brief Threshold for decreasing lambda based on rho.
   *
   * If rho > lambda_downscale_threshold, lambda is decreased to make the
   * algorithm more aggressive.
   * Default: 0.75
   */
  float lambda_downscale_threshold = 0.75f;
};

/**
 * @brief Levenberg-Marquardt optimizer for nonlinear least squares problems.
 *
 * Extends GaussNewtonMinimizer with adaptive damping to improve robustness.
 * The Levenberg-Marquardt algorithm solves:
 *   (J^T J + lambda * diag(J^T J)) dx = -J^T r
 *
 * where lambda is a damping factor that:
 * - Increases when steps are rejected (making the algorithm more conservative)
 * - Decreases when steps are very successful (making it more aggressive)
 *
 * This provides a smooth interpolation between gradient descent (large lambda)
 * and Gauss-Newton (lambda = 0), making it more robust than pure Gauss-Newton
 * while often converging faster than gradient descent.
 */
class LevenbergMarquardtMinimizer : public GaussNewtonMinimizer {
 public:
  /**
   * @brief Constructs a Levenberg-Marquardt optimizer.
   *
   * @param options Configuration options. Defaults to standard LM options.
   */
  LevenbergMarquardtMinimizer(
      const LevenbergMarquardtMinimizerOptions& options =
          LevenbergMarquardtMinimizerOptions())
      : GaussNewtonMinimizer(options.base_options), options_(options) {}

 private:
  /**
   * @brief Initializes LM-specific data structures.
   *
   * Calls base class initialization and sets initial lambda value.
   */
  void Initialize(cudaStream_t stream, const Problem& problem) override;

  /**
   * @brief Builds the Levenberg-Marquardt linear system.
   *
   * Extends the Gauss-Newton system by adding lambda * diag(J^T J) to the
   * diagonal of the Hessian: (J^T J + lambda * diag(J^T J)) dx = -J^T r
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param minimizer_state Current minimizer state.
   * @param[out] lhs Output left-hand side matrix (H + lambda * diag(H)).
   * @param[out] rhs Output right-hand side vector (-J^T r).
   */
  void BuildSystem(cudaStream_t stream, const Problem& problem,
                   const MinimizerState& minimizer_state, CSRSparseMatrix& lhs,
                   dvector<float>& rhs) override;

  /**
   * @brief Checks convergence using LM-specific criteria.
   *
   * Computes the rho metric (ratio of actual to predicted cost reduction) and
   * checks convergence based on step size, cost, and predicted relative
   * reduction.
   *
   * @param stream CUDA stream for GPU operations.
   * @param updated_cost Cost after applying the step.
   * @param current_cost Cost before applying the step.
   * @param step State update step vector.
   * @param[out] step_quality Output rho metric (actual/predicted reduction).
   * @return True if converged, false otherwise.
   */
  bool CheckConvergence(cudaStream_t stream, float updated_cost,
                        float current_cost, const dvector<float>& step,
                        float& step_quality) override;

  /**
   * @brief Determines if a step should be accepted (LM version).
   *
   * Accepts step if rho >= step_accept_threshold. If step is very successful
   * (rho > lambda_downscale_threshold), decreases lambda.
   *
   * @param step_quality Rho metric (actual/predicted cost reduction).
   * @return True if step should be accepted, false otherwise.
   */
  bool AcceptStep(float step_quality) override;

  /**
   * @brief Determines if a step should be rejected (LM version).
   *
   * Rejects step if rho < step_accept_threshold. Increases lambda when
   * rejecting a step.
   *
   * @param step_quality Rho metric (actual/predicted cost reduction).
   * @return True if step should be rejected, false otherwise.
   */
  bool RejectStep(float step_quality) override;

  const LevenbergMarquardtMinimizerOptions options_;  ///< LM-specific options.

  dvector<float> diagonal_;  ///< Diagonal of the Hessian matrix (J^T J).

  float lambda_;  ///< Current damping factor.
};

}  // namespace cunls
