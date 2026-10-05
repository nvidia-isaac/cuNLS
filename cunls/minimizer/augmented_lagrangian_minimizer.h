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

#include <vector>

#include "cunls/common/types.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

/** @brief Options of the augmented Lagrangian outer loop. */
struct AugmentedLagrangianMinimizerOptions {
  /**
   * @brief A subproblem is feasible when every scaled constraint row satisfies
   * |c| <= tol (equalities) or c <= tol (inequalities). Default: 1e-4
   */
  float constraint_tolerance = 1e-4f;

  /** @brief Maximum number of outer (multiplier update) iterations. Default: 20 */
  size_t max_outer_iterations = 20;

  /**
   * @brief Iteration cap of the inner solves while the constraints are not
   * yet satisfied (inexact augmented Lagrangian). Default: 5
   */
  size_t inner_iterations = 5;

  /**
   * @brief Iteration cap of the inner solves once every running subproblem is
   * feasible (and of the last outer iteration): the solve that establishes
   * stationarity. 0 means the wrapped minimizer's
   * MinimizerOptions::max_num_iterations. Default: 0
   */
  size_t final_inner_iterations = 0;

  /**
   * @brief Line-search steps of the inner solves (see
   * MinimizerOptions::max_line_search_steps): the inequality rows make the
   * cost piecewise quadratic, and a Gauss-Newton step that activates rows the
   * model did not see can overshoot. Default: 10
   */
  size_t inner_line_search_steps = 10;

  /** @brief Initial penalty ρ0 of every constraint batch and subproblem. Default: 10 */
  float initial_penalty = 10.f;

  /** @brief Factor β by which a penalty grows. Default: 10 */
  float penalty_increase = 10.f;

  /**
   * @brief Penalty cap ρ_max: keeps the float32 normal equations usable. A
   * subproblem whose violation stops decreasing at the cap ends with
   * AugmentedLagrangianMinimizerStatus::kMaxPenalty. Default: 1e8
   */
  float max_penalty = 1e8f;

  /**
   * @brief Required violation decrease η: the penalty of a constraint batch
   * grows when its violation is above η times the previous one. Default: 0.25
   */
  float violation_decrease = 0.25f;

  /**
   * @brief Start from the multipliers of the previous Minimize call
   * (receding-horizon warm start) instead of λ = 0, and from its penalties
   * divided by penalty_increase (at least initial_penalty); with real_time,
   * from its penalties and violation history as they are (the penalty
   * schedule continues across calls). Only effective when the constraint
   * batches and their sizes are unchanged. Default: false
   */
  bool warm_start = false;

  /**
   * @brief The problem's structure is unchanged since the previous Minimize
   * call (see MinimizerOptions::reuse_structure): skip the structure setup
   * of the inner minimizer and of the constraint maps. Within one call the
   * outer iterations always reuse it. Default: false
   */
  bool reuse_structure = false;

  /**
   * @brief Real-time mode: a fixed budget and no host synchronization but one
   * read-back at the end. Exactly max_outer_iterations outer iterations; each
   * runs exactly inner_iterations inner iterations (the last one
   * final_inner_iterations if > 0), each with exactly inner_line_search_steps
   * line-search evaluations and MinimizerOptions::max_bound_refinements
   * bound refinements.
   * Penalties, multipliers and per-subproblem status are updated on the
   * device; the summary reports the last outer iteration's violation and
   * status, and NaN costs. Typical use (receding horizon): a converged first
   * solve, then real time with warm_start and reuse_structure (SetOptions).
   * Default: false
   */
  bool real_time = false;
};

/** @brief Why the outer loop stopped. */
enum class AugmentedLagrangianMinimizerStatus {
  kConverged = 0,           ///< Every subproblem is feasible and stationary.
  kMaxOuterIterations = 1,  ///< Some subproblem ran out of outer iterations.
  kMaxPenalty = 2,          ///< Some subproblem cannot reduce its violation at ρ_max.
};

/** @brief Result of AugmentedLagrangianMinimizer::Minimize. */
struct AugmentedLagrangianMinimizerSummary {
  AugmentedLagrangianMinimizerStatus status = AugmentedLagrangianMinimizerStatus::kConverged;

  /** @brief Outer iterations performed. */
  size_t outer_iterations = 0;

  /** @brief Inner iterations, summed over the outer iterations. */
  size_t inner_iterations = 0;

  /** @brief Largest violation (see constraint_tolerance) over all rows and subproblems. */
  float max_violation = 0.f;

  /** @brief Objective (the cost of the non-constraint factors) before and after. */
  float initial_cost = 0.f;
  float final_cost = 0.f;

  /** @brief Subproblems in total, converged and stopped at the penalty cap. */
  size_t num_problems = 1;
  size_t num_converged = 0;
  size_t num_max_penalty = 0;
};

namespace internal {

/** @brief A constraint batch of the problem, as found by AugmentedLagrangianMinimizer. */
struct ConstraintBatchState {
  size_t residual_batch;             ///< Index into Problem::GetResidualBatches().
  ConstraintFactorBatchBase *batch;  ///< The constraint batch.
  size_t num_factors;                ///< Active factors at the start of the solve.
  dvector<int> factor_problem;       ///< Per factor: subproblem (empty: all 0).
};

}  // namespace internal

/**
 * @brief Solves problems with constraint factor batches
 * (ConstraintFactorBatchBase: ConstraintFactorBatch, BoundFactorBatch) by the
 * augmented Lagrangian method around a Gauss-Newton or Levenberg-Marquardt
 * minimizer.
 *
 * @verbatim
 *   λ = 0, μ = 0, ρ = ρ0
 *   repeat:
 *     x ← inner_minimize(x)                      warm-started, few iterations
 *     v ← max violation                           per constraint batch and subproblem
 *     λ ← λ + ρ c_E,   μ ← max(0, μ + ρ c_I)
 *     ρ ← min(β ρ, ρ_max) where v > η v_prev
 *   until v <= tol after a full inner solve that converged (not cut off)
 * @endverbatim
 *
 * Constraint batches are found among the problem's residual batches by type.
 * With a problem partition (Problem::SetProblemPartition) every subproblem has
 * its own penalties per constraint batch and stops on its own; multiplier and
 * penalty updates run on the device, with one small read-back per outer
 * iteration. Without constraint batches the call is the wrapped minimizer's
 * Minimize.
 *
 * The multipliers and penalties live in the constraint batches, so the batches
 * may be solved again with warm_start.
 */
class AugmentedLagrangianMinimizer {
 public:
  /**
   * @brief Wraps an inner minimizer.
   * @param minimizer Inner minimizer, GaussNewtonMinimizer or
   *        LevenbergMarquardtMinimizer (not owned; must outlive this object).
   *        Its options apply to the inner solves, except the iteration cap and
   *        line search, which this class sets per outer iteration.
   * @param options Outer-loop options.
   * @throws std::invalid_argument on invalid options (see SetOptions).
   */
  explicit AugmentedLagrangianMinimizer(
      Minimizer &minimizer,
      const AugmentedLagrangianMinimizerOptions &options = AugmentedLagrangianMinimizerOptions());

  AugmentedLagrangianMinimizer(const AugmentedLagrangianMinimizer &) = delete;
  AugmentedLagrangianMinimizer &operator=(const AugmentedLagrangianMinimizer &) = delete;

  /**
   * @brief Minimizes the problem's objective subject to its constraint
   * batches, starting from its current states.
   *
   * On return the problem's states hold the result and the constraint batches
   * hold the final multipliers and penalties (the warm start of a following
   * call). Enqueues work on `stream` and synchronizes it before returning.
   *
   * @param stream CUDA stream for all device work.
   * @param problem Problem to solve; its states are updated in place.
   * @return Outer and inner iteration counts, violation, costs and status.
   * @throws std::invalid_argument if a constraint batch has a robust loss, or
   *         as Minimizer::Minimize for an invalid problem.
   */
  AugmentedLagrangianMinimizerSummary Minimize(cudaStream_t stream, Problem &problem);

  /** @brief Options of the following Minimize() calls. */
  const AugmentedLagrangianMinimizerOptions &Options() const { return options_; }

  /**
   * @brief Replaces the options for the following calls; the warm-start state
   * (multipliers, penalties, structure) is kept. Typical use: a converged
   * first solve, then a fixed real-time budget.
   * @throws std::invalid_argument unless initial_penalty > 0, max_penalty >=
   *         initial_penalty, penalty_increase >= 1, constraint_tolerance >= 0,
   *         inner_iterations > 0 and max_outer_iterations > 0.
   */
  void SetOptions(const AugmentedLagrangianMinimizerOptions &options);

 private:
  // Implementation state; the behaviour lives in augmented_lagrangian_minimizer.cu.
  Minimizer &minimizer_;  ///< Inner minimizer.
  AugmentedLagrangianMinimizerOptions options_;

  std::vector<internal::ConstraintBatchState>
      constraints_;          ///< Constraint batches of the last solve.
  size_t num_problems_ = 0;  ///< Subproblems of the last solve.
  /// Constraint batches with their sizes, and the number of subproblems, of the
  /// last solve (warm_start and reuse_structure apply only when unchanged).
  std::vector<std::pair<const void *, size_t>> signature_;
  const Problem *problem_ = nullptr;  ///< Problem of the last solve.

  dvector<float> penalty_;         ///< [constraint * P + p]
  dvector<float> violation_;       ///< [constraint * P + p], this outer iteration.
  dvector<float> prev_violation_;  ///< [constraint * P + p], previous outer iteration.
  dvector<int> problem_status_;    ///< [p]: 0 running, 1 converged, 2 max penalty.
  dvector<int> at_cap_;            ///< [p]: last inner solve stopped by its cap.
  dvector<float> values_;          ///< Constraint values of every batch, concatenated.
  dvector<float> d_out_;           ///< Read-back scalars.
  dvector<float> scratch_;         ///< Objective evaluation buffers.
  dvector<float> partials_;        ///< Reduction partials.
};

}  // namespace cunls
