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

#include <string>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

/**
 * @brief Options of AugmentedLagrangianMinimizer: those of its inner
 * Levenberg-Marquardt solves (the base struct) and of the outer loop.
 *
 * The inner solver is private to AugmentedLagrangianMinimizer and built from
 * the LevenbergMarquardtMinimizerOptions part: damping, linear solver,
 * tolerances, Jacobian mode. Four of its fields set the budget of the call,
 * with or without constraint batches:
 *
 * - base_options.max_num_iterations: iteration cap of a subproblem's full
 *   inner solve, its first after it is feasible (and every subproblem's in the
 *   last outer iteration), the solve that establishes stationarity. Without
 *   constraint batches every subproblem is feasible from the start, so this
 *   is the iteration cap of a bounds-only solve. Must be > 0. Default: 50
 * - base_options.max_line_search_steps: line-search steps of the inner
 *   solves (see MinimizerOptions::max_line_search_steps): the inequality rows
 *   make the cost piecewise quadratic, and a step that activates rows the
 *   linear model did not see can overshoot. Default here: 10
 * - base_options.real_time: a fixed budget without host synchronization but
 *   one read-back at the end. Exactly max_outer_iterations outer iterations;
 *   each runs exactly inner_iterations inner iterations (the last one
 *   base_options.max_num_iterations), each with exactly max_line_search_steps
 *   line-search evaluations and max_bound_refinements bound refinements.
 *   Penalties, multipliers and per-subproblem status are updated on the
 *   device; the summary reports the last outer iteration's violation and
 *   status, and NaN costs. Without constraint batches there is no violation
 *   to report: nothing is read back (status kNotDetermined), so the call can
 *   be captured in a CUDA graph once the structure is set up (Initialize()
 *   before the capture, with reuse_structure). Typical use
 *   (receding horizon): a converged first solve, then real time with
 *   warm_start and reuse_structure (SetOptions). Default: false
 * - base_options.reuse_structure: the problem's structure is unchanged since
 *   the previous Minimize() or Initialize() call (see
 *   MinimizerOptions::reuse_structure): skip the structure setup of the inner
 *   solves and of the constraint maps.
 *   Within one call the outer iterations always reuse it. Default: false
 */
struct AugmentedLagrangianMinimizerOptions : LevenbergMarquardtMinimizerOptions {
  /** @brief The defaults, with a 10-step line search in the inner solves. */
  AugmentedLagrangianMinimizerOptions() { base_options.max_line_search_steps = 10; }

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
   * divided by penalty_increase (at least initial_penalty); in real time,
   * from its penalties and violation history as they are (the penalty
   * schedule continues across calls). Only effective when the constraint
   * batches and their sizes are unchanged. Default: false
   */
  bool warm_start = false;

  /**
   * @brief Box-bounded states (VectorStateBatch::SetBounds): at most this many
   * extra linear solves per inner iteration that also hold the free
   * components the step would push through their bound (active-set
   * refinement; rarely more than one is needed). In real time exactly this
   * many (there is no read-back to stop early), so real-time callers may set
   * 0 or 1; Levenberg-Marquardt's damping usually does without them.
   * Default: 3
   */
  size_t max_bound_refinements = 3;
};

/** @brief Why the outer loop stopped. */
enum class AugmentedLagrangianMinimizerStatus {
  kConverged = 0,           ///< Every subproblem is feasible and stationary.
  kMaxOuterIterations = 1,  ///< Some subproblem ran out of outer iterations.
  kMaxPenalty = 2,          ///< Some subproblem cannot reduce its violation at ρ_max.
  /// Convergence was not checked: a real-time (fixed-iteration) call without
  /// constraint batches reads nothing back, so nothing is known about it.
  kNotDetermined = 3,
  /// The inner solve of some subproblem stopped on a non-finite cost or step
  /// (TerminationReason NonFiniteCost / NonFiniteStep), feasible or not.
  kInnerSolveFailed = 4,
};

/** @brief Result of AugmentedLagrangianMinimizer::Minimize. */
struct AugmentedLagrangianMinimizerSummary {
  /** @brief Outcome of the call; see AugmentedLagrangianMinimizerStatus and #message. */
  AugmentedLagrangianMinimizerStatus status = AugmentedLagrangianMinimizerStatus::kNotDetermined;

  /** @brief Outer iterations performed. */
  size_t outer_iterations = 0;

  /** @brief Inner iterations, summed over the outer iterations. */
  size_t inner_iterations = 0;

  /** @brief Largest violation (see constraint_tolerance) over all rows and subproblems. */
  float max_violation = 0.f;

  /** @brief Objective (the cost of the non-constraint factors) before the call. */
  float initial_cost = 0.f;
  /** @brief Objective after the call. */
  float final_cost = 0.f;

  /** @brief Subproblems in total (Problem::NumProblems()). */
  size_t num_problems = 1;
  /** @brief Subproblems with established convergence (not kNotDetermined ones). */
  size_t num_converged = 0;
  /** @brief Subproblems stopped at the penalty cap (kMaxPenalty). */
  size_t num_max_penalty = 0;
  /** @brief Subproblems whose inner solve failed (kInnerSolveFailed). */
  size_t num_inner_failed = 0;

  /**
   * @brief One or two sentences for a person or an agent: the outcome and,
   * when not converged, the likely causes and what to try.
   */
  std::string message;

  /** @brief Wall time of the Minimize() call in milliseconds. */
  double solve_time_ms = 0.0;

  /** @brief status == kConverged. */
  bool Converged() const { return status == AugmentedLagrangianMinimizerStatus::kConverged; }
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
 * (ConstraintFactorBatchBase: ConstraintFactorBatch, BoundFactorBatch) or
 * box-bounded states (VectorStateBatch::SetBounds) by the augmented Lagrangian
 * method around a Levenberg-Marquardt minimizer of its own.
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
 * its own penalties per constraint batch and stops on its own: a subproblem
 * that becomes feasible gets its own full inner solve in the next outer
 * iteration (base_options.max_num_iterations) and finishes there, while the others
 * continue with capped solves; it never waits for another subproblem.
 * Multiplier and penalty updates run on the device, with one small read-back
 * per outer iteration. Box bounds are held in every inner solve
 * (HoldActiveBounds). Without constraint batches every subproblem is feasible
 * from the start: its first inner solve is its full solve, and a bounds-only
 * call takes one outer iteration when that solve converges. The budget of a
 * call is in AugmentedLagrangianMinimizerOptions.
 *
 * The multipliers and penalties live in the constraint batches, so the batches
 * may be solved again with warm_start.
 */
class AugmentedLagrangianMinimizer {
 public:
  /**
   * @brief Builds the minimizer and its private inner Levenberg-Marquardt
   * solver from `options`.
   * @throws std::invalid_argument on invalid options (see SetOptions).
   */
  explicit AugmentedLagrangianMinimizer(
      const AugmentedLagrangianMinimizerOptions &options = AugmentedLagrangianMinimizerOptions());

  AugmentedLagrangianMinimizer(const AugmentedLagrangianMinimizer &) = delete;
  AugmentedLagrangianMinimizer &operator=(const AugmentedLagrangianMinimizer &) = delete;

  /**
   * @brief Minimizes the problem's objective subject to its constraint
   * batches, starting from its current states.
   *
   * On return the problem's states hold the result and the constraint batches
   * hold the final multipliers and penalties (the warm start of a following
   * call). Enqueues work on `stream` and synchronizes it before returning,
   * except a real-time call without constraint batches (see real_time).
   *
   * @param stream CUDA stream for all device work.
   * @param problem Problem to solve; its states are updated in place.
   * @return Outer and inner iteration counts, violation, costs and status.
   * @throws std::invalid_argument if a constraint batch has a robust loss, or
   *         as Minimizer::Minimize for an invalid problem, or if `stream` is
   *         being captured into a CUDA graph and the call would synchronize:
   *         a converging solve, a real-time solve with constraint batches
   *         (its status is read back at the end), or a call that sets up the
   *         structure (a first call on the problem that Initialize() did not
   *         set up, or any call without base_options.reuse_structure). It
   *         throws before enqueuing any work, so the capture stays valid.
   */
  AugmentedLagrangianMinimizerSummary Minimize(cudaStream_t stream, Problem &problem);

  /**
   * @brief Sets the problem's structure up without solving: the work a
   * Minimize() call does first when the structure is not set up (the
   * constraint batches' subproblem maps, the bounded states' columns, and the
   * inner solver's setup, see Minimizer::Initialize), and the only part of it
   * that synchronizes. Optional: Minimize() calls it itself on a first call,
   * on a changed structure and on every call without
   * base_options.reuse_structure. Call it before capturing Minimize() in a
   * CUDA graph (base_options.real_time and reuse_structure set, no constraint
   * batches), or to keep the setup out of the first solve.
   *
   * Leaves the states unchanged; the multipliers and penalties of a new
   * constraint layout start as in a first call (a warm start keeps those of an
   * unchanged one). Enqueues work on `stream` and synchronizes it before
   * returning.
   *
   * @param stream CUDA stream for all device work; not one being captured.
   * @param problem Problem the following Minimize() calls solve.
   * @throws std::invalid_argument if `stream` is being captured into a CUDA
   *         graph, or for a problem Minimize() rejects.
   * @throws std::runtime_error if the linear solver rejects the system.
   */
  void Initialize(cudaStream_t stream, const Problem &problem);

  /** @brief Options of the following Minimize() calls. */
  const AugmentedLagrangianMinimizerOptions &Options() const { return options_; }

  /**
   * @brief Replaces the options of the following calls: the outer loop and
   * the budget (base_options.max_num_iterations, max_line_search_steps,
   * real_time, reuse_structure); the warm-start state (multipliers, penalties,
   * structure) is kept. The inner solver keeps the damping, linear solver and
   * tolerances it was built with (a new minimizer changes those). Typical use:
   * a converged first solve, then a fixed real-time budget.
   * @throws std::invalid_argument unless initial_penalty > 0, max_penalty >=
   *         initial_penalty, penalty_increase >= 1, constraint_tolerance >= 0,
   *         inner_iterations > 0, base_options.max_num_iterations > 0 and
   *         max_outer_iterations > 0.
   */
  void SetOptions(const AugmentedLagrangianMinimizerOptions &options);

  /**
   * @brief Device array of the last Minimize() call's subproblem count of ints
   * (Problem::NumProblems()): the TerminationReason value of each
   * subproblem's last inner solve (see Minimizer::SubproblemTerminationReasons).
   */
  const int *SubproblemTerminationReasons() const {
    return minimizer_.SubproblemTerminationReasons();
  }

  /** @brief Number of entries of SubproblemTerminationReasons(). */
  size_t NumSubproblems() const { return minimizer_.NumSubproblems(); }

 private:
  /**
   * @brief Initialize() without its checks and final synchronization: the
   * constraint batches and signatures of `problem`, its bounded states
   * (UpdateBoundedStates), and the inner solver's setup.
   */
  void SetUp(cudaStream_t stream, const Problem &problem);

  /**
   * @brief Finds the box-bounded state batches (bounded_) and the
   * reduced-system column of every state (column_offsets_): always with
   * `recompute` (the setup, also without bounds), otherwise only with bounds
   * when the number of state batches changed.
   */
  void UpdateBoundedStates(cudaStream_t stream, const Problem &problem, bool recompute);

  /**
   * @brief Bound handling of the box-bounded states, run by every inner solve
   * as its InnerSolve::restrict_system, at the current `state`.
   *
   * Before the solve (`step` null), once the linear system is built and
   * damped: the components that sit at a bound
   * with the steepest-descent direction (the right-hand side) pointing
   * outward are masked out of the system (LinearSystem::SetMask with
   * bound_mask_; their rhs entries zeroed here), so their step is exactly 0.
   * After a solve (`step` given): a free component the step still pushes
   * through its bound is held too and true asks for another solve (active-set
   * refinement), at most
   * max_bound_refinements times; false keeps the step. Together with
   * VectorStateBatch's bounded Plus, which clamps the moved components into
   * the box, the iterates stay feasible.
   */
  bool HoldActiveBounds(cudaStream_t stream, const MinimizerState &state, LinearSystem &system,
                        dvector<float> &rhs, const dvector<float> *step);

  // Implementation state; the behaviour lives in augmented_lagrangian_minimizer.cu.
  LevenbergMarquardtMinimizer minimizer_;  ///< The inner solver, built from the options.
  AugmentedLagrangianMinimizerOptions options_;

  std::vector<internal::ConstraintBatchState>
      constraints_;          ///< Constraint batches of the last solve.
  size_t num_problems_ = 0;  ///< Subproblems of the last solve.
  /// Constraint batches with their sizes, and the number of subproblems, of the
  /// last solve (warm_start and reuse_structure apply only when unchanged).
  std::vector<std::pair<const void *, size_t>> signature_;
  /// State and factor counts of the last solve's problem (reuse_structure
  /// applies only when they are unchanged).
  std::vector<size_t> structure_signature_;
  const Problem *problem_ = nullptr;  ///< Problem of the last solve.

  dvector<float> penalty_;         ///< [constraint * P + p]
  dvector<float> violation_;       ///< [constraint * P + p], this outer iteration.
  dvector<float> prev_violation_;  ///< [constraint * P + p], previous outer iteration.
  /// [p]: -1 polishing (next inner solve is its full solve), 0 running,
  /// 1 converged, 2 max penalty, 3 failed; positive entries are frozen.
  dvector<int> problem_status_;
  dvector<int> at_cap_;         ///< [p]: last inner solve stopped by its cap.
  dvector<int> iteration_cap_;  ///< [p]: inner iteration cap of the next solve (not real time).
  dvector<float> values_;       ///< Constraint values of every batch, concatenated.
  dvector<float> d_out_;        ///< Read-back scalars.
  dvector<float> scratch_;      ///< Objective evaluation buffers.
  dvector<float> partials_;     ///< Reduction partials.

  std::vector<size_t> bounded_;  ///< Box-bounded state batches of the problem.
  std::vector<dvector<int>>
      column_offsets_;  ///< Per state batch: column of each state (-1: constant).
  /// Per column: 0 where held at a bound; the LinearSystem's declared mask (by pointer).
  dvector<float> bound_mask_;
  dvector<float> bound_step_mask_;  ///< Per column: 0 where the step leaves the box.
  dvector<float> bound_extra_;      ///< Per column: 0 where newly held by a refinement.
  dvector<int> bound_count_;        ///< Components newly held by the last refinement.
  size_t refinements_ = 0;          ///< Refinements of the current inner iteration.
  int device_ = -1;  ///< Device of the buffers above (that of the first problem solved).
};

}  // namespace cunls
