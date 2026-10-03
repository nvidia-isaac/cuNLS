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
#include "cunls/minimizer/gauss_newton_minimizer.h"
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
   * call (see MinimizeCallOptions::reuse_structure): skip the structure setup
   * of the inner minimizer and of the constraint maps. Within one call the
   * outer iterations always reuse it. Default: false
   */
  bool reuse_structure = false;

  /**
   * @brief Real-time mode: a fixed budget and no host synchronization but one
   * read-back at the end. Exactly max_outer_iterations outer iterations; each
   * runs inner_iterations inner iterations (the last one
   * final_inner_iterations if > 0) with MinimizeCallOptions::fixed_iterations.
   * Penalties, multipliers and per-subproblem status are updated on the
   * device; the summary reports the last outer iteration's violation and
   * status, and NaN costs. Typical use (receding horizon): a converged first
   * solve, then real time with warm_start and reuse_structure (SetOptions).
   * Default: false
   */
  bool real_time = false;

  /**
   * @brief With real_time, warm_start and reuse_structure: replay the device
   * work of a call as a CUDA graph. The first such call runs eagerly, the
   * second is captured, later calls launch the graph (removing the kernel
   * launch overhead; the one status read-back stays). The graph is dropped
   * when the options or the constraint batches change; if the capture fails
   * (e.g. a factor batch synchronizes or uses other streams), the calls run
   * eagerly. Default: false
   */
  bool use_cuda_graph = false;
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
   * @param minimizer Inner minimizer (not owned; must outlive this object).
   * @param options Outer-loop options.
   */
  explicit AugmentedLagrangianMinimizer(
      GaussNewtonMinimizer &minimizer,
      const AugmentedLagrangianMinimizerOptions &options = AugmentedLagrangianMinimizerOptions());

  ~AugmentedLagrangianMinimizer();

  AugmentedLagrangianMinimizer(const AugmentedLagrangianMinimizer &) = delete;
  AugmentedLagrangianMinimizer &operator=(const AugmentedLagrangianMinimizer &) = delete;

  /**
   * @brief Minimizes the problem's objective subject to its constraint
   * batches. State values are updated in place.
   */
  AugmentedLagrangianMinimizerSummary Minimize(cudaStream_t stream, Problem &problem);

  const AugmentedLagrangianMinimizerOptions &Options() const { return options_; }

  /**
   * @brief Replaces the options for the following calls; the warm-start state
   * (multipliers, penalties, structure) is kept. Typical use: a converged
   * first solve, then a fixed real-time budget.
   * @throws std::invalid_argument on invalid options (as the constructor).
   */
  void SetOptions(const AugmentedLagrangianMinimizerOptions &options);

  /** @brief Whether the last call replayed (or captured) a CUDA graph (Options::use_cuda_graph). */
  bool UsesCudaGraph() const { return graph_exec_ != nullptr; }

 private:
  struct Constraint {
    size_t residual_batch;             ///< Index into Problem::GetResidualBatches().
    ConstraintFactorBatchBase *batch;  ///< The constraint batch.
    size_t num_factors;                ///< Active factors at the start of the solve.
    dvector<int> factor_problem;       ///< Per factor: subproblem (empty: all 0).
  };

  /** @brief The solve itself; `read_back` false leaves out the final status read-back. */
  AugmentedLagrangianMinimizerSummary MinimizeEager(cudaStream_t stream, Problem &problem,
                                                    bool read_back);

  /** @brief Constraint batches with their active sizes, and the number of subproblems. */
  static std::vector<std::pair<const void *, size_t>> ConstraintSignature(const Problem &problem);

  /** @brief Reads the control kernel's scalars into the summary (one synchronization). */
  void ReadStatus(cudaStream_t stream, AugmentedLagrangianMinimizerSummary &summary);
  void FillStatus(const float *out, AugmentedLagrangianMinimizerSummary &summary) const;

  /** @brief Drops the captured graph. */
  void ResetGraph();

  /** @brief Cost of the non-constraint residual batches at the problem's states. */
  float ComputeObjective(cudaStream_t stream, const Problem &problem);

  /** @brief penalties[f] = ρ[constraint, subproblem(f)] for every constraint batch. */
  void ScatterPenalties(cudaStream_t stream);

  GaussNewtonMinimizer &minimizer_;
  AugmentedLagrangianMinimizerOptions options_;

  std::vector<Constraint> constraints_;
  size_t num_problems_ = 0;
  /// Signature (batch pointers and sizes) of the last solve, for warm_start.
  std::vector<std::pair<const void *, size_t>> signature_;
  const Problem *problem_ = nullptr;  ///< Problem of the last solve.
  void *graph_exec_ = nullptr;        ///< cudaGraphExec_t of the captured real-time call.
  bool graph_warm_ = false;           ///< An eligible call ran eagerly: the next one is captured.
  bool graph_failed_ = false;         ///< Capture failed: run eagerly until the options change.
  AugmentedLagrangianMinimizerSummary graph_summary_;  ///< Host part of the captured call.
  bool has_constraints_ = true;  ///< The last call had constraint batches (a status to read).

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
