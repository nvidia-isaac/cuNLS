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

#include <cstdint>
#include <vector>

#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac/ransac_context.h"

namespace cunls {

/**
 * @brief Largest total free tangent dimension a RANSAC minimizer accepts.
 *
 * The sum of TangentSize() over every non-constant state of every state
 * batch in the problem must not exceed this value. Constant states do not
 * count, whatever their number.
 */
constexpr int kMaxRansacTangentDim = 64;

/** @brief Role of a residual batch inside a RANSAC minimizer. */
enum class RansacRole {
  /**
   * Data factors that may be outliers. Minimal samples are drawn from them
   * and every factor is classified as inlier or outlier.
   */
  kSampled = 0,
  /**
   * Trusted factors (priors, motion priors, extrinsic constraints). Included
   * in every hypothesis solve and the final refinement; never classified.
   */
  kAlwaysOn = 1,
};

/** @brief Hypothesis scoring rule. */
enum class RansacScoring {
  /** Truncated quadratic: sum over sampled factors of min(|r|^2, tau^2). */
  kMSAC = 0,
  /** Number of inliers (ties broken by the lower MSAC score). */
  kInlierCount = 1,
};

/** @brief Dense solver used for the per-hypothesis normal equations. */
enum class RansacLinearSolverType {
  /** In-kernel Cholesky. A non-positive pivot marks the hypothesis invalid. */
  kCholesky = 0,
  /**
   * In-kernel LDL^T with symmetric diagonal pivoting. Rank-deficient
   * directions get a zero step instead of failing, which suits the
   * near-singular systems minimal samples produce.
   */
  kLDLT = 1,
};

/** @brief Per residual batch RANSAC configuration. */
struct RansacFactorBatchOptions {
  /** @brief Role of the residual batch. */
  RansacRole role = RansacRole::kSampled;

  /**
   * @brief Inlier threshold tau on the raw residual norm, in the batch's
   * residual units. A factor is an inlier iff |r|^2 <= tau^2. Ignored for
   * kAlwaysOn batches.
   */
  float inlier_threshold = 1.0f;
};

/** @brief Options common to RansacGaussNewtonMinimizer and RansacLevenbergMarquardtMinimizer. */
struct RansacMinimizerOptions {
  /** @brief Hypotheses generated and scored together in one round. */
  size_t hypotheses_per_round = 256;

  /** @brief Upper bound on the number of rounds (> 0). */
  size_t max_rounds = 8;

  /**
   * @brief Number of sampled factors per minimal sample. 0 selects
   * ceil(D / m_min), where D is the free tangent dimension and m_min the
   * smallest residual dimension among sampled batches.
   */
  size_t sample_size = 0;

  /**
   * @brief Target probability of drawing at least one all-inlier sample;
   * drives adaptive stopping between rounds.
   */
  float confidence = 0.999f;

  /** @brief Stop once the best inlier ratio reaches this value. 1 disables. */
  float early_stop_inlier_ratio = 1.0f;

  /** @brief Seed of the counter-based sampler. Same seed, same result. */
  uint64_t seed = 0;

  /**
   * @brief Per residual batch configuration, indexed like
   * Problem::GetResidualBatches(). Empty means every batch is kSampled with
   * @ref default_inlier_threshold.
   */
  std::vector<RansacFactorBatchOptions> factor_batches;

  /** @brief Inlier threshold used when @ref factor_batches is empty. */
  float default_inlier_threshold = 1.0f;

  /** @brief Hypothesis scoring rule. */
  RansacScoring scoring = RansacScoring::kMSAC;

  /** @brief Add 2 * (cost of the kAlwaysOn factors) to each hypothesis score. */
  bool score_always_on = true;

  /**
   * @brief Count a factor as an inlier only if its Jacobian has a non-zero
   * entry on a free state.
   *
   * Some factors report a zero residual (and zero Jacobian) for configurations
   * they cannot evaluate, e.g. PnPFactorBatch for points behind the camera.
   * Without this check a hypothesis that pushes every point behind the camera
   * would look like a perfect fit. Costs one Jacobian evaluation per scored
   * factor; disable only for factors that never do this.
   */
  bool require_informative_inliers = true;

  /**
   * @brief Device memory budget for scoring buffers. Bounds how many
   * hypotheses are scored per chunk.
   */
  size_t scoring_memory_budget_bytes = 64ull << 20;

  /**
   * @brief Two-stage scoring for large problems. When there are more than
   * 2 * scoring_subset_size kSampled factors, every hypothesis is first scored
   * on the same random subset of scoring_subset_size factors (drawn anew each
   * round), and only the best @ref scoring_finalists are scored on all
   * factors. 0 disables.
   */
  size_t scoring_subset_size = 16384;

  /**
   * @brief Hypotheses scored on all factors in two-stage scoring (at most 64).
   * Guards against a wrong pick from the noisier subset scores.
   */
  size_t scoring_finalists = 4;

  /** @brief Gauss-Newton / LM iterations that turn a sample into a hypothesis. */
  size_t hypothesis_iterations = 5;

  /** @brief Iterations of the final refinement on the best inlier set. */
  size_t final_iterations = 20;

  /** @brief Per-hypothesis convergence: squared step norm threshold. */
  float state_tolerance = 1e-10f;

  /** @brief Per-hypothesis convergence: relative cost decrease threshold. */
  float cost_tolerance = 1e-7f;

  /** @brief Per-hypothesis dense solver. */
  RansacLinearSolverType linear_solver = RansacLinearSolverType::kLDLT;
};

/** @brief Options of RansacLevenbergMarquardtMinimizer. */
struct RansacLevenbergMarquardtMinimizerOptions {
  /** @brief Options common to all RANSAC minimizers. */
  RansacMinimizerOptions base_options;

  /** @brief Damping each hypothesis starts from. */
  float initial_lambda = 1e-3f;

  /** @brief Damping multiplier on a rejected step. */
  float lambda_upscale = 2.0f;

  /** @brief Damping multiplier on a very successful step. */
  float lambda_downscale = 0.5f;

  /** @brief Upper damping bound; a hypothesis that exceeds it stops. */
  float lambda_max = 1e6f;

  /** @brief Lower damping bound. */
  float lambda_min = 1e-6f;

  /** @brief Accept a step if actual / predicted cost reduction >= this value. */
  float step_accept_threshold = 0.25f;

  /** @brief Decrease damping if actual / predicted reduction > this value. */
  float lambda_downscale_threshold = 0.75f;
};

/** @brief Result of a RANSAC minimization. */
struct RansacSummary : MinimizerSummary {
  /** @brief Rounds executed. */
  size_t num_rounds = 0;

  /** @brief Hypotheses generated across all rounds. */
  size_t num_hypotheses = 0;

  /** @brief Hypotheses whose first linear solve succeeded. */
  size_t num_valid_hypotheses = 0;

  /** @brief Inliers of the final estimate, over all kSampled factors. */
  size_t num_inliers = 0;

  /** @brief num_inliers divided by the number of kSampled factors. */
  float inlier_ratio = 0.f;

  /** @brief Score (lower is better) of the final estimate. */
  float best_score = 0.f;

  /**
   * @brief True if the final refinement worsened the score and the best
   * hypothesis (before refinement) was returned instead.
   */
  bool refinement_reverted = false;
};

/**
 * @brief Common base of RansacGaussNewtonMinimizer and
 * RansacLevenbergMarquardtMinimizer: RANSAC over an ordinary Problem.
 *
 * The problem is built exactly as for Minimizer. Every factor and state batch
 * must honor the item parameters of FactorBatch::Evaluate (factor_ids,
 * num_factor_ids) and the num_replicas parameter of StateBatch::Plus; all
 * built-in batches do. The one restriction is the total free tangent
 * dimension (see kMaxRansacTangentDim); any number of state batches of any
 * supported types, and any number of factor batches and factors, are allowed.
 *
 * Minimize() samples minimal sets of kSampled factors, turns each into a
 * hypothesis by a few iterations (hypothesis_iterations) from the current
 * state values, scores every hypothesis against all kSampled factors, refines
 * the best one on its inliers (final_iterations) and writes it back into the
 * problem's state batches. InlierMask() then exposes the classification. The
 * subclass decides how a hypothesis iterates: Gauss-Newton or
 * Levenberg-Marquardt steps. See docs/sphinx/ransac.rst.
 *
 * Not instantiable on its own: construct a RansacGaussNewtonMinimizer or a
 * RansacLevenbergMarquardtMinimizer, and use a `RansacMinimizer &` to accept
 * either.
 */
class RansacMinimizer {
 public:
  virtual ~RansacMinimizer() = default;
  RansacMinimizer(const RansacMinimizer &) = delete;
  RansacMinimizer &operator=(const RansacMinimizer &) = delete;
  RansacMinimizer(RansacMinimizer &&) = delete;
  RansacMinimizer &operator=(RansacMinimizer &&) = delete;

  /**
   * @brief Runs RANSAC and writes the refined estimate into the problem's
   * state batches.
   *
   * Enqueues work on `stream` and synchronizes it before returning.
   *
   * @param stream CUDA stream for all work.
   * @param problem The problem; its current state values are the initial guess.
   * @return Rounds, hypotheses, inliers, score and the refinement's costs.
   * @throws std::invalid_argument for invalid configurations (free tangent
   *         dimension 0 or above kMaxRansacTangentDim, no kSampled factors,
   *         fewer sampled factors than the sample size, numeric Jacobians, a
   *         mismatched factor_batches vector), including active sizes that are
   *         not set properly (see Problem::CheckSizes).
   */
  RansacSummary Minimize(cudaStream_t stream, Problem &problem);

  /**
   * @brief Device inlier mask (1 = inlier) of a kSampled residual batch, one
   * byte per factor, for the estimate of the last Minimize(). Valid until the
   * next Minimize() or destruction.
   *
   * @param residual_batch_index Index into Problem::GetResidualBatches().
   * @return Device pointer, or nullptr for kAlwaysOn batches, an out-of-range
   *         index, or before any run.
   */
  const uint8_t *InlierMask(size_t residual_batch_index) const;

  /**
   * @brief Number of bytes of InlierMask(residual_batch_index): the factor
   * count the batch had in the last Minimize(). 0 whenever InlierMask()
   * returns nullptr.
   */
  size_t InlierMaskSize(size_t residual_batch_index) const;

  /** @brief Options common to all RANSAC minimizers, as constructed. */
  const RansacMinimizerOptions &Options() const { return options_; }

 protected:
  /**
   * @brief For subclasses: stores the options and how hypotheses iterate.
   * @param options Options common to all RANSAC minimizers.
   * @param settings The subclass's per-hypothesis step rule and dense solver
   *        (internal; see ransac/slot_set.h).
   */
  RansacMinimizer(const RansacMinimizerOptions &options,
                  const ransac_internal::SolverSettings &settings);

 private:
  // Implementation state; the behaviour lives in ransac/ransac_context.cpp.
  const RansacMinimizerOptions options_;
  const ransac_internal::SolverSettings settings_;  ///< Step rule and dense solver.
  ransac_internal::RansacContext context_;          ///< Layout, slots, scorer, statistics.
};

/**
 * @brief RANSAC with Gauss-Newton hypotheses and refinement.
 *
 * Per hypothesis: a step is taken if it lowers the cost; the hypothesis stops
 * at the first step that does not, or when the relative cost decrease is at
 * most cost_tolerance or ‖δ‖² at most state_tolerance. See RansacMinimizer.
 */
class RansacGaussNewtonMinimizer : public RansacMinimizer {
 public:
  /** @brief Constructs the minimizer; see RansacMinimizerOptions. */
  explicit RansacGaussNewtonMinimizer(
      const RansacMinimizerOptions &options = RansacMinimizerOptions());
};

/**
 * @brief RANSAC with Levenberg-Marquardt hypotheses and refinement; each
 * hypothesis carries its own damping λ.
 *
 * Per hypothesis, ρ = actual / predicted cost reduction: ρ ≥
 * step_accept_threshold takes the step (and multiplies λ by lambda_downscale
 * when ρ > lambda_downscale_threshold, floored at lambda_min); otherwise λ is
 * multiplied by lambda_upscale and the hypothesis stops once λ exceeds
 * lambda_max. Convergence as in RansacGaussNewtonMinimizer. See
 * RansacMinimizer.
 */
class RansacLevenbergMarquardtMinimizer : public RansacMinimizer {
 public:
  /** @brief Constructs the minimizer; see RansacLevenbergMarquardtMinimizerOptions. */
  explicit RansacLevenbergMarquardtMinimizer(
      const RansacLevenbergMarquardtMinimizerOptions &options =
          RansacLevenbergMarquardtMinimizerOptions());
};

}  // namespace cunls
