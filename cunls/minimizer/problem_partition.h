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

// Subproblems of a problem (Problem::SetProblemPartition; a problem without a
// partition is one subproblem): maps from factors and from rows of the reduced
// system to subproblems, per-subproblem sums over them, and the step
// bookkeeping every minimizer shares (take or reject, line search, rejection
// cap, convergence). The method-specific rule that classifies a step lives in
// the minimizers (Minimizer::ClassifySteps).

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

/**
 * @brief Subproblem id of every active factor of residual batch
 * `residual_batch_index`, read through the problem's own state storage
 * (Problem::DeviceStatePointers: call Problem::PrepareStatePointers first).
 * Writes NumActiveFactors() ints to `factor_problem`; asynchronous on
 * `stream`. The partition itself is validated by the minimizers.
 */
void ComputeFactorProblemIds(cudaStream_t stream, const Problem &problem,
                             size_t residual_batch_index, int *factor_problem);

/** @brief Parameters of ProblemPartition::StepControl. */
struct StepControlParams {
  /// Stop a subproblem after this many consecutive rejections (0: never).
  int max_consecutive_rejected_steps = 0;
  /// Line search on: any step (full or shortened) that decreases the cost is taken.
  bool line_search = false;
};

/** @brief What StepControl decided for a subproblem's step (ProblemPartition::Outcome). */
enum StepOutcome : int {
  kStepInactive = 0,      ///< The subproblem was not iterating (converged, frozen or capped).
  kStepTaken = 1,         ///< Accepted by the minimizer's rule.
  kStepLineSearched = 2,  ///< Taken because the line search lowered the cost.
  kStepRejected = 3,      ///< Rejected; the states are kept.
  kStepConverged = 4,     ///< The subproblem converged (step taken if the cost did not rise).
};

/**
 * @brief Subproblem maps and the step bookkeeping shared by all minimizers.
 *
 * Build() once per structure; then, per Minimize() call, InitStepControl(),
 * and per iteration: the trial costs (SumFactorCosts into NewCost()), the
 * line search (MarkLineSearch, ScaleRows), the squared steps (SumRows into
 * StepSquared()), the minimizer's classification (Reject(), Converged()) and
 * StepControl(), which leaves the decisions on the device and the total cost
 * and number of still-active subproblems in a 2-float device array (one
 * read-back per iteration).
 */
class ProblemPartition {
 public:
  /**
   * @brief Sizes the per-subproblem arrays and, with a partition, builds the
   * factor and row maps for `problem` and the states of `state`; throws if a
   * factor connects states of different subproblems or an id is out of range
   * (synchronizes `stream` once). Without a partition there is one subproblem
   * and no device work.
   */
  void Build(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
             size_t num_rows);

  /** @brief Number of subproblems (1 without a partition). */
  size_t NumProblems() const { return num_problems_; }

  /**
   * @brief costs[p] = sum of the per-factor costs (cost-buffer order) of
   * subproblem p. Deterministic with one subproblem.
   */
  void SumFactorCosts(cudaStream_t stream, const float *factor_costs, float *costs);

  /**
   * @brief out[p] = sum over rows r of subproblem p of x[r] * y[r] (y = x when
   * null), times w[r] when w is given. Deterministic with one subproblem.
   */
  void SumRows(cudaStream_t stream, const float *x, const float *y, const float *w, float *out);

  /** @brief out[r] = per_problem[problem(r)] * in[r]. */
  void ScaleRows(cudaStream_t stream, const float *per_problem, const float *in, float *out) const;

  /** @brief Copies the states of subproblems whose step was taken from `from` into `to`. */
  void CopyAccepted(cudaStream_t stream, const Problem &problem, const MinimizerState &from,
                    MinimizerState &to) const;

  /**
   * @brief Starts a Minimize() call: the costs in Cost() become the current
   * costs; subproblems below `cost_tolerance` or with a nonzero `frozen` entry
   * (may be null) are inactive. Writes [total cost, active count] to `d_out`.
   */
  void InitStepControl(cudaStream_t stream, float cost_tolerance, const int *frozen, float *d_out);

  /**
   * @brief Line search: every active subproblem whose NewCost() is not below
   * its Cost() gets StepScale() = 0.5 and is marked shortened (others 1).
   * Writes the number of such subproblems to `d_count` (one float).
   */
  void MarkLineSearch(cudaStream_t stream, float *d_count);

  /**
   * @brief Takes or rejects the step of every active subproblem from
   * NewCost(), Reject() and Converged() (see StepOutcome): a step to a
   * non-finite cost is rejected; with line search, a shortened or rejected
   * step that lowers the cost is taken; a converged subproblem stops, taking
   * its step if the cost did not rise; a rejection counts towards the cap.
   * Writes Outcome(), the accepted costs into Cost(), and [total cost, active
   * count] into `d_out`; with one subproblem also d_out[2] = 1 if its step was
   * taken (`d_out` must then hold 3 floats).
   */
  void StepControl(cudaStream_t stream, const StepControlParams &params, float *d_out);

  /** @brief Clears the shortened flags of the line search (start of an iteration). */
  void ResetLineSearch(cudaStream_t stream);

  /**
   * @name Per-subproblem device arrays (NumProblems() entries).
   * @{
   */
  float *Cost() { return cost_.data(); }  ///< Cost at the current states.
  const float *Cost() const { return cost_.data(); }
  float *NewCost() { return new_cost_.data(); }  ///< Cost at the trial states.
  const float *NewCost() const { return new_cost_.data(); }
  float *StepSquared() { return step_squared_.data(); }  ///< ‖δ‖² of the (shortened) step.
  const float *StepSquared() const { return step_squared_.data(); }
  const float *StepScale() const { return step_scale_.data(); }  ///< Line-search scale.
  int *Reject() { return reject_.data(); }        ///< Written by Minimizer::ClassifySteps.
  int *Converged() { return converged_.data(); }  ///< Written by Minimizer::ClassifySteps.
  /** @brief Nonzero for subproblems still iterating. */
  const int *Active() const { return active_.data(); }
  /** @brief StepOutcome of the last StepControl. */
  const int *Outcome() const { return outcome_.data(); }
  /** @brief Consecutive rejections, including the last StepControl's. */
  const int *Rejected() const { return rejected_.data(); }
  /** @} */

  /**
   * @brief Subproblem of every row of the reduced system (NumRows() entries),
   * or nullptr without a partition (every row belongs to subproblem 0).
   */
  const int *RowProblem() const { return num_problems_ > 1 ? row_problem_.data() : nullptr; }

  /** @brief Rows of the reduced system. */
  size_t NumRows() const { return num_rows_; }

 private:
  size_t num_problems_ = 0;
  size_t num_factor_items_ = 0;
  size_t num_rows_ = 0;
  dvector<int> factor_problem_;  ///< Per factor, in cost-buffer order.
  dvector<int> row_problem_;     ///< Per row of the reduced system.
  dvector<int> error_;           ///< Build() error flags.
  dvector<int> column_offsets_;  ///< Build() scratch.
  dvector<uint8_t> views_;       ///< Build(): per state batch, storage and ids.
  dvector<float> partials_;      ///< Reduction partials (one subproblem).
  dvector<float> cost_, new_cost_, step_squared_, step_scale_;
  dvector<int> active_, rejected_, reject_, converged_, outcome_, accept_, shortened_;
};

}  // namespace cunls
