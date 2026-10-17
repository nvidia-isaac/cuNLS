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

#pragma once

#include <cuda_runtime.h>

#include "cunls/minimizer/minimizer.h"

namespace cunls {

/**
 * @brief Options of LevenbergMarquardtMinimizer: the common MinimizerOptions
 * plus the damping λ and the step acceptance rule.
 */
struct LevenbergMarquardtMinimizerOptions {
  /** @brief Options common to all minimizers. */
  MinimizerOptions base_options;

  /**
   * @brief Initial value for the damping factor lambda.
   *
   * Controls the initial regularization strength. Higher values make the
   * algorithm more conservative (closer to gradient descent). Must be
   * <= lambda_max; 0 (Gauss-Newton steps) is allowed.
   * Default: 1e-3
   */
  float initial_lambda = 1e-3;

  /**
   * @brief Convergence threshold for predicted relative cost reduction.
   *
   * A subproblem has converged when the cost reduction its linear model
   * predicts, relative to its cost, falls below this threshold.
   * Default: 1e-6
   */
  float relative_reduction_tolerance = 1e-6;

  /**
   * @brief Factor by which lambda is increased when a step is rejected; must
   * be > 1.
   *
   * The k-th consecutive rejection multiplies lambda by
   * lambda_upscale * 2^(k-1) (Nielsen's rule, as in Ceres and g2o), so the
   * damping escalates quickly when the model is poor: with the default,
   * 2, 4, 8, 16, 32, i.e. 2^15 after five rejections. An accepted step resets
   * the escalation. Lambda stays within [lambda_min, lambda_max].
   * Default: 2.0
   */
  float lambda_upscale = 2.0f;

  /**
   * @brief Factor by which lambda is decreased when a step is accepted.
   *
   * When a step is very successful (rho > lambda_downscale_threshold), lambda
   * is multiplied by this factor to make the next step more aggressive
   * (closer to Gauss-Newton).
   * Default: 0.5
   */
  float lambda_downscale = 0.5f;

  /**
   * @brief Maximum allowed value for lambda; must be >= lambda_min.
   *
   * Prevents lambda from growing too large, which would make the algorithm
   * too conservative.
   * Default: 1e+6
   */
  float lambda_max = 1e+6;

  /**
   * @brief Minimum allowed value for lambda; must be positive.
   *
   * Prevents lambda from becoming too small, which could cause numerical
   * instability. A rejected step escalates from at least this value, so an
   * initial_lambda of 0 (pure Gauss-Newton steps) still recovers.
   * Default: 1e-6
   */
  float lambda_min = 1e-6;

  /**
   * @brief Floor of the diagonal the damping scales: λ max(diag(H)_i,
   * min_diagonal) (the diagonal of the column-scaled system when column
   * scaling is on).
   *
   * An unobservable unknown (a zero Jacobian column, e.g. a pose whose
   * observations are all invalid) has a zero diagonal, which no λ diag(H)
   * damping fixes: the system stays singular and every step fails. With the
   * floor it gets a positive pivot and, its right-hand side being zero, a
   * zero step. Ceres' min_lm_diagonal. Default: 1e-6
   */
  float min_diagonal = 1e-6f;

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
 * @brief Levenberg-Marquardt: Gauss-Newton with adaptive damping.
 *
 * Solves (H + λ D) δ = -g with one λ per subproblem, D = max(diag(H),
 * min_diagonal) (of the column-scaled system with column scaling). A large λ
 * gives short, gradient-like steps; a small λ gives Gauss-Newton steps. More
 * robust than GaussNewtonMinimizer far from the solution and when H is close
 * to singular.
 *
 * Per subproblem, ρ = actual cost reduction / reduction predicted by the
 * linear model, ½ δᵀHδ + λ δᵀ diag(H) δ:
 *   - ρ ≥ step_accept_threshold: the step is good; if also
 *     ρ > lambda_downscale_threshold, λ shrinks by lambda_downscale;
 *   - otherwise the step is rejected and λ grows: the k-th consecutive
 *     rejection multiplies it by lambda_upscale · 2^(k-1);
 *   - converged if ‖δ‖² < state_tolerance, the predicted relative reduction
 *     < relative_reduction_tolerance, or the trial cost < cost_tolerance.
 * λ always stays in [lambda_min, lambda_max] and starts at initial_lambda on
 * every Minimize() call.
 *
 * See Minimizer for the iteration and LevenbergMarquardtMinimizerOptions for
 * the settings.
 */
class LevenbergMarquardtMinimizer : public Minimizer {
 public:
  /**
   * @brief Constructs the minimizer; see LevenbergMarquardtMinimizerOptions.
   *
   * Options() returns options.base_options with max_consecutive_rejected_steps
   * widened by the rejections the damping needs to escalate from lambda_min to
   * lambda_max.
   *
   * @throws std::invalid_argument unless 0 < lambda_min <= lambda_max,
   *         initial_lambda <= lambda_max, lambda_upscale > 1 and
   *         min_diagonal >= 0.
   */
  explicit LevenbergMarquardtMinimizer(
      const LevenbergMarquardtMinimizerOptions &options = LevenbergMarquardtMinimizerOptions());

 private:
  /**
   * @brief Sets every subproblem's λ to initial_lambda on iteration 0, and
   * otherwise updates it from the outcome of the previous step (see the class
   * comment); then declares the damping λ_p max(diag(S H S), min_diagonal)
   * of the rows of subproblem p (LinearSystem::SetDamping; S the column
   * scale).
   */
  void UpdateSystem(cudaStream_t stream, size_t iteration, LinearSystem &system,
                    const ProblemPartition &partition) override;

  /** @brief A failed solve is a rejected step: λ grows until the system factors. */
  bool RejectsFailedSolves() const override { return true; }

  /** @brief Computes ρ per subproblem and applies the rules of the class comment. */
  void ClassifySteps(cudaStream_t stream, const LinearSystem &system, const dvector<float> &step,
                     ProblemPartition &partition) override;

  const LevenbergMarquardtMinimizerOptions options_;  ///< As constructed (not widened).

  dvector<float> lambda_;         ///< Per subproblem: current damping λ.
  dvector<float> next_lambda_;    ///< Per subproblem: λ being updated (swapped with lambda_).
  dvector<float> quality_;        ///< Per subproblem: ρ of the last step (for the λ update).
  dvector<float> diag_weight_;    ///< Per subproblem: δᵀ diag(H) δ.
  dvector<float> matrix_weight_;  ///< Per subproblem: δᵀ H δ.
  /// Per row: diag(S H S) of the current linearization (UpdateSystem()).
  dvector<float> scaled_diagonal_;
  /// Per row: diag(H), unscaled; extracted by ClassifySteps() with column scaling only.
  dvector<float> unscaled_diagonal_;
  /// Per row: λ max(diag(S H S), min_diagonal), declared to the system by pointer.
  dvector<float> damping_;
};

}  // namespace cunls
