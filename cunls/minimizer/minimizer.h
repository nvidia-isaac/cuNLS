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

#include <functional>
#include <vector>

#include "cunls/common/pinned_vector.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/numeric_diff_jacobian.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {

/**
 * @brief Diagonal column scaling S for the normal equations.
 *
 * When not ``None``, the linear solve uses \f$S H S \, z = S b\f$ with
 * \f$b = -J^\top r\f$, then applies \f$\Delta x = S z\f$. ``None`` leaves
 * the standard system \f$H \Delta x = b\f$ unchanged.
 */
enum class ColumnScaling {
  /** No scaling (identity S). */
  None = 0,
  /**
   * \f$S_{ii} = 1 / \sqrt{H_{ii}}\f$ with a floor on the diagonal.
   *
   * Equivalently \f$1 / \|J_{:,j}\|_2\f$, since \f$H_{jj} = \|J_{:,j}\|_2^2\f$.
   */
  HessianDiagonal = 1,
};

/**
 * @brief Result of one Minimizer::Minimize() call.
 *
 * Costs are totals over all subproblems (see Problem::SetProblemPartition).
 */
struct MinimizerSummary {
  /** @brief Iterations performed (each builds and solves one linear system). */
  size_t num_iterations = 0;

  /** @brief Total cost of the states passed in. */
  float initial_cost = 0.0f;

  /** @brief Total cost of the states written back to the problem. */
  float final_cost = 0.0f;

  /** @brief Total cost at the start of each iteration. */
  std::vector<float> iteration_costs;
};

/**
 * @brief Options common to GaussNewtonMinimizer and LevenbergMarquardtMinimizer.
 *
 * Convergence criteria, iteration limits, the linear solver, line search and
 * structure reuse. Every criterion is applied per subproblem (see
 * Problem::SetProblemPartition); a problem without a partition is one
 * subproblem.
 */
struct MinimizerOptions {
  /** @brief Maximum number of iterations of one Minimize() call. Default: 50 */
  size_t max_num_iterations = 50;

  /**
   * @brief Convergence threshold for state updates.
   *
   * A subproblem has converged when its squared step norm falls below this
   * threshold.
   * Default: 1e-6
   */
  float state_tolerance = 1e-6f;

  /**
   * @brief Convergence threshold for the cost.
   *
   * A subproblem has converged when its cost falls below this threshold.
   * Default: 1e-6
   */
  float cost_tolerance = 1e-6f;

  /**
   * @brief Maximum number of consecutive rejected steps before declaring
   *        convergence.
   *
   * When a subproblem rejects this many steps in a row (every trial step is
   * not good enough by the minimizer's rule), it is treated as converged
   * because it can no longer make progress. Levenberg-Marquardt adds the
   * rejections its damping needs to escalate from lambda_min to lambda_max,
   * so the cap counts the rejections at full damping. Set to 0 to disable this
   * criterion.
   * Default: 5
   */
  size_t max_consecutive_rejected_steps = 5;

  /**
   * @brief Type of sparse linear solver to use.
   *
   * Supported options:
   *  - BlockSparsePCG (default): block-Jacobi preconditioned CG; derives
   *    the per-state-batch block layout automatically.  Recommended for
   *    most SBA and PGO workloads — see `profile/PCG_RESULTS.md` and
   *    `profile/benchmark_a6000.png`.
   *  - cuDSS: NVIDIA cuDSS sparse direct solver.  Preferred when many
   *    near-identical small Hessians are solved back-to-back and the
   *    per-iteration kernel-launch overhead of PCG dominates.
   *  - DenseLDLT: converts CSR to dense, solves with a custom pivoted
   *    LDLT kernel.  Suitable for small dense Hessians.
   *  - DenseCholesky / DenseQR: cuSOLVER-backed dense solves.
   *  - BlockTridiagonal: block-tridiagonal Cholesky for stage-ordered
   *    problems (Problem::SetStateStages).
   *
   * Default: BlockSparsePCG.
   */
  SparseLinearSolverType sparse_linear_solver_type = SparseLinearSolverType::BlockSparsePCG;

  /**
   * @brief Configuration for the sparse linear solver.
   *
   * Contains backend-specific options.  Only the field corresponding to
   * @ref sparse_linear_solver_type is read:
   *  - `block_sparse_pcg_options` for `BlockSparsePCG`,
   *  - `cudss_solver_options` for `cuDSS`,
   *  - Dense backends have no extra options.
   *
   * Defaults: `BlockSparsePCG` with the layout auto-derived from the
   * problem's state batches when the structure is set up (the first
   * Minimize() call on a problem).
   *
   * To enable multi-threaded cuDSS, set
   * `cudss_solver_options.threading_lib_path` to the full path of
   * `libcudss_mtlayer_gomp.so` (or equivalent).
   */
  SparseLinearSolverConfig sparse_linear_solver_config = {};

  /**
   * @brief Optional diagonal scaling of the normal equations.
   *
   * See ColumnScaling. Default: None.
   */
  ColumnScaling column_scaling = ColumnScaling::None;

  /**
   * @brief Disable runtime safety checks inside the minimizer.
   *
   * When false, the minimizer enables all optional runtime validation
   * that can catch numerical problems early.  Currently this covers
   * post-factorization checks in the linear solver:
   *  - Cholesky: cuSOLVER devInfo after potrf and potrs (non-SPD or
   *    invalid-parameter detection).
   *  - QR: diagonal-of-R inspection for rank deficiency.
   *  - LDLT: in-kernel pivot and diagonal checks during factorization
   *    and solve.
   *
   * On failure the solver's Solve() returns false and a diagnostic is
   * emitted via LogError().  The minimizer treats a false return as a
   * fatal error and throws std::runtime_error.
   *
   * When true, every check listed above is skipped: no device-to-host
   * memcpy, no stream synchronization, and no in-kernel validation.
   * This can noticeably reduce per-iteration latency for small systems
   * but may produce silently incorrect results if the matrix is singular
   * or ill-conditioned.
   *
   * Default: true (safety checks disabled).
   */
  bool disable_safety_checks = true;

  /**
   * @brief Global default Jacobian strategy for every residual batch.
   *
   * `kAnalytic` (default) uses each FactorBatch's own hand-derived Jacobian.
   * `kNumeric` derives Jacobians via finite differences instead (see
   * `NumericDiffJacobianBuilder`), requiring only residual-only evaluation
   * support from the factor batch. Individual residual batches can override
   * this default via `Problem::AddFactorBatch`'s `jacobian_mode_override`
   * parameter.
   */
  JacobianMode jacobian_mode = JacobianMode::kAnalytic;

  /**
   * @brief Backtracking line search: when a step does not decrease the cost,
   * it is halved along the same direction up to this many times, and the first
   * shorter step that decreases the cost is taken. Any step that decreases the
   * cost is then taken, also one the step-quality rule would reject
   * (Levenberg-Marquardt keeps its damping: the step length is the line
   * search's job). Useful when the cost has kinks or curvature the
   * Gauss-Newton model does not see, such as the inequality rows and curved
   * constraints of constraint batches. 0 disables it. Default: 0
   */
  size_t max_line_search_steps = 0;

  /**
   * @brief The problem's structure is unchanged since this minimizer's
   * previous Minimize() call on it: same batches, connectivity (also the
   * contents of device index tables), active and constant counts, constant
   * ids and subproblem partition; only state values, factor data and state
   * bounds may differ. Calls after the first then skip the structure setup (index
   * expansion, Hessian pattern, symbolic analysis of the linear solver,
   * subproblem maps). A cheap host-side check of the sizes falls back to the
   * full setup when they differ; changes it cannot see (rewritten index
   * tables or constant ids at the same sizes) must not be combined with this
   * option. Typical use: a real-time loop that re-solves the same problem
   * with new measurements. Default: false
   */
  bool reuse_structure = false;

  /**
   * @brief Tuning for numeric-diff Jacobians. Ignored when `jacobian_mode`
   * (and every per-group override) is `kAnalytic`.
   */
  NumericDiffOptions numeric_diff_options = {};
};

class Minimizer;

namespace internal {

/**
 * @brief Settings of one inner solve of AugmentedLagrangianMinimizer (internal).
 *
 * Minimizer::Minimize() runs with the values of MinimizerOptions and no
 * device arrays.
 */
struct InnerSolve {
  /// Iteration cap of the call.
  size_t max_num_iterations = 0;
  /// Line-search steps of the call (see MinimizerOptions::max_line_search_steps).
  size_t max_line_search_steps = 0;
  /// See MinimizerOptions::reuse_structure.
  bool reuse_structure = false;
  /// Real time: exactly max_num_iterations iterations, each with exactly
  /// max_line_search_steps line-search evaluations, and no host
  /// synchronization. The returned summary has the cap as its iteration count
  /// and NaN costs.
  bool fixed_iterations = false;
  /// The caller manages the problem's constraints: constraint batches
  /// evaluate augmented-Lagrangian rows with its multipliers and penalties,
  /// and box-bounded states are kept in their bounds by its restrict_system
  /// (otherwise a problem with either is rejected).
  bool constraints_managed = false;
  /// Removes unknowns from the linear system: zeroes their rows, columns and
  /// right-hand side entries (keeping the diagonal), so their step is exactly
  /// 0. Called every iteration once the normal equations are built (and
  /// updated by the subclass) with `step` null, before the solve; then after
  /// each solve with the solved `step`: returning true asks for another solve
  /// (it removed more unknowns), false keeps the step. `state` holds the
  /// current states. Empty: nothing is removed.
  std::function<bool(cudaStream_t stream, const MinimizerState &state, NormalEquations &system,
                     dvector<float> &rhs, const dvector<float> *step)>
      restrict_system;
  /// Device array of Problem::NumProblems() ints, written: nonzero where a
  /// subproblem was still iterating at the iteration cap.
  int *problem_at_cap = nullptr;
  /// Device array of Problem::NumProblems() ints, read: subproblems with a
  /// nonzero entry are not iterated (keep their states).
  const int *problem_frozen = nullptr;
};

/** @brief Minimize() with the settings of `inner` (AugmentedLagrangianMinimizer, tests). */
MinimizerSummary InnerMinimize(Minimizer &minimizer, cudaStream_t stream, Problem &problem,
                               const InnerSolve &inner);

/** @brief What was set up for the last problem; skipped on reuse_structure. */
struct MinimizerStructure {
  const Problem *problem = nullptr;  ///< Problem of the last setup.
  std::vector<size_t> signature;     ///< Its size signature.
  bool solver_ready = false;         ///< The solver's symbolic analysis has run.
  bool partition_ready = false;      ///< The subproblem maps are built.
};

/** @brief The linearized problem and its solution. */
struct MinimizerSystem {
  SparseLinearSolverPtr solver;             ///< Linear solver of the normal equations.
  NumericDiffJacobianBuilder numeric_diff;  ///< Jacobians of kNumeric residual batches.
  dvector<float> residuals;                 ///< Residuals of every active factor.
  PerFactorJacobians factor_jacobians;      ///< Dense Jacobian block of every factor.
  NormalEquations normal_equations;         ///< H = JᵀJ and the working left-hand side.
  dvector<float> rhs;                       ///< -Jᵀr (scaled and masked like the lhs).
  dvector<float> column_scale;              ///< S of ColumnScaling::HessianDiagonal.
  dvector<float> step;                      ///< The step δ, one entry per reduced row.
};

/** @brief Scratch buffers and read-back slots. */
struct MinimizerScratch {
  dvector<uint8_t> buffer;        ///< Per-factor costs, residuals and workspaces.
  dvector<float> d_scalars;       ///< Device scalars: totals and read-back slots.
  PinnedVector<float> h_scalars;  ///< Host mirror of d_scalars.
  dvector<float> d_partials;      ///< Reduction partials.
};

/**
 * @brief Structure setup of Minimize() for `problem`: checks the sizes,
 * expands the index tables, re-plans numeric-diff batches, sizes the
 * residuals and Jacobians, lays out the tangent space (`state_ops`) and builds
 * the Hessian pattern. Internal; tests use it to assemble systems in isolation.
 */
void SetUpStructure(cudaStream_t stream, Problem &problem, const MinimizerOptions &options,
                    StateBatchOps &state_ops, MinimizerSystem &system);

/**
 * @brief Step 1 of Minimize(): the Gauss-Newton system at `state`, H = JᵀJ and
 * rhs = -Jᵀr, then S H S z = S b with column scaling. Requires SetUpStructure
 * for the same problem. Internal; tests use it to assemble systems in
 * isolation.
 */
void BuildSystem(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
                 const MinimizerOptions &options, MinimizerSystem &system,
                 MinimizerScratch &scratch);

}  // namespace internal

/**
 * @brief Common base of GaussNewtonMinimizer and LevenbergMarquardtMinimizer.
 *
 * Minimizes the problem's total cost, ½ Σ ρ(‖r‖²), by repeating:
 *   1. build the normal equations H δ = -g at the current states (with
 *      column scaling),
 *   2. let the subclass update them (UpdateSystem),
 *   3. solve for the step δ,
 *   4. evaluate the cost at the trial states, shortening δ by line search
 *      if enabled (MinimizerOptions::max_line_search_steps),
 *   5. let the subclass classify each step (ClassifySteps),
 *   6. take or reject each step, and stop each subproblem that converged or
 *      hit the rejection cap (MinimizerOptions::max_consecutive_rejected_steps).
 *
 * A problem made of independent subproblems (Problem::SetProblemPartition)
 * is solved in one linear system, but every decision in steps 4 to 6 is
 * taken per subproblem. A problem without a partition is one subproblem.
 * Each iteration makes one small read-back to the host (total cost and number
 * of running subproblems), plus one per line-search step.
 *
 * Not instantiable on its own: construct a GaussNewtonMinimizer or a
 * LevenbergMarquardtMinimizer, and use a `Minimizer &` to accept either.
 * A minimizer is bound to no problem; it caches the structure of the last
 * problem it solved (see MinimizerOptions::reuse_structure).
 */
class Minimizer {
 public:
  virtual ~Minimizer() = default;
  Minimizer(const Minimizer &) = delete;
  Minimizer &operator=(const Minimizer &) = delete;
  Minimizer(Minimizer &&) = delete;
  Minimizer &operator=(Minimizer &&) = delete;

  /**
   * @brief Minimizes the problem's cost, starting from its current states.
   *
   * On return the problem's states hold the result (also when the iteration
   * limit was hit).
   * Enqueues work on `stream` and synchronizes it before returning.
   *
   * @param stream CUDA stream for all device work.
   * @param problem Problem to solve; its states are updated in place.
   * @return Iteration count and costs.
   * @throws std::invalid_argument if the problem has constraint factor
   *         batches or box-bounded states (solve those with
   *         AugmentedLagrangianMinimizer), or its
   *         sizes, connectivity or partition are invalid (see
   *         Problem::CheckSizes).
   * @throws std::runtime_error if the linear solver fails.
   */
  MinimizerSummary Minimize(cudaStream_t stream, Problem &problem);

  /**
   * @brief Options the minimizer runs with (for Levenberg-Marquardt the base
   * options, with max_consecutive_rejected_steps widened by the damping's
   * escalation room).
   */
  const MinimizerOptions &Options() const { return options_; }

 protected:
  /**
   * @brief For subclasses: stores the options and creates the linear solver.
   * @param options Options common to all minimizers.
   */
  explicit Minimizer(const MinimizerOptions &options);

 private:
  /**
   * @brief Subclass step 2: modifies the normal equations before the solve.
   *
   * Called once per iteration, after the Gauss-Newton system for the current
   * states has been assembled into `system`. Gauss-Newton leaves it as is;
   * Levenberg-Marquardt adds its damping.
   *
   * @param stream CUDA stream; enqueue only, do not synchronize.
   * @param iteration 0 on the first iteration of this Minimize() call.
   * @param system Normal equations; the working left-hand side may be modified.
   * @param partition Subproblems, with the outcome of the previous
   *        iteration's steps (ProblemPartition::Outcome); read only.
   */
  virtual void UpdateSystem(cudaStream_t stream, size_t iteration, NormalEquations &system,
                            const ProblemPartition &partition) = 0;

  /**
   * @brief Subclass step 5: classifies each subproblem's trial step.
   *
   * Called once per iteration, after the trial cost of every subproblem is
   * known. For each active subproblem p, writes two flags:
   *   - ProblemPartition::Reject()[p]: the step is not good enough by this
   *     method's rule;
   *   - ProblemPartition::Converged()[p]: the subproblem meets this method's
   *     convergence test.
   * Only classifies; the base then decides. A rejected step is still taken
   * if the line search shortened it to a lower cost; a converged subproblem
   * stops, keeping its step only if the cost did not increase; a step to a
   * non-finite cost is always rejected.
   *
   * @param stream CUDA stream; enqueue only, do not synchronize.
   * @param system Normal equations the step was solved from (the assembled
   *        Hessian and the working left-hand side).
   * @param step The (possibly shortened) step δ, one entry per reduced row.
   * @param partition Subproblems with current cost (Cost), trial cost
   *        (NewCost) and ‖δ‖² (StepSquared); write Reject() and Converged()
   *        only.
   */
  virtual void ClassifySteps(cudaStream_t stream, const NormalEquations &system,
                             const dvector<float> &step, ProblemPartition &partition) = 0;

  friend MinimizerSummary internal::InnerMinimize(Minimizer &minimizer, cudaStream_t stream,
                                                  Problem &problem,
                                                  const internal::InnerSolve &inner);

  // Implementation state; the behaviour lives in minimizer.cu.
  const MinimizerOptions options_;
  internal::MinimizerStructure structure_;  ///< Cached setup of the last problem.
  internal::MinimizerSystem system_;        ///< Linearization, normal equations, step.
  internal::MinimizerScratch scratch_;      ///< Buffers and read-back slots.
  StateBatchOps state_ops_;                 ///< Manifold plus and tangent layout.
  ProblemPartition partition_;              ///< Subproblems and their step bookkeeping.
  MinimizerState current_state_;            ///< States before the step.
  MinimizerState updated_state_;            ///< States at the trial point.
  profiler::Domain profiler_domain_{"Minimizer"};
};

}  // namespace cunls
