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

// Per-subproblem step control for problems that are batches of independent
// subproblems (Problem::SetProblemPartition): maps from factors and from rows
// of the reduced system to subproblems, segmented reductions over them, and the
// device-side accept / reject / damping / convergence decision per subproblem.

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

/** @brief Step-control parameters shared by every subproblem. */
struct BatchedStepControlParams {
  bool levenberg_marquardt = false;
  float state_tolerance = 0.f;
  float cost_tolerance = 0.f;
  int max_consecutive_rejected_steps = 0;  ///< 0: never stop on rejections.
  /// Line search on: any step (full or shortened) that decreases the cost is taken.
  bool line_search = false;
  // Levenberg-Marquardt only.
  float relative_reduction_tolerance = 0.f;
  float step_accept_threshold = 0.f;
  float lambda_upscale = 1.f;
  float lambda_downscale = 1.f;
  float lambda_downscale_threshold = 1.f;
  float lambda_min = 0.f;
  float lambda_max = 0.f;
};

/**
 * @brief Subproblem maps and per-subproblem step control.
 *
 * Build() once per solve; then, per iteration, accumulate the per-subproblem
 * costs, squared steps (and, for Levenberg-Marquardt, the predicted-reduction
 * terms) and call StepControl(), which leaves the accept flags on the device
 * and the total cost and number of still-active subproblems in a 2-float
 * device array (one read-back per iteration).
 */
class ProblemPartition {
 public:
  /**
   * @brief Builds the factor and row maps for `problem` and the states of
   * `state`; throws if a factor connects states of different subproblems or
   * an id is out of range. Synchronizes `stream` once.
   */
  void Build(cudaStream_t stream, const Problem &problem, const MinimizerState &state,
             size_t num_rows);

  size_t NumProblems() const { return num_problems_; }

  /** @brief costs[p] = sum of the per-factor costs (cost-buffer order) of subproblem p. */
  void AccumulateFactorCosts(cudaStream_t stream, const float *factor_costs, float *costs) const;

  /**
   * @brief out[p] = sum over rows r of subproblem p of x[r] * y[r] (y = x when
   * null), times w[r] when w is given.
   */
  void AccumulateRows(cudaStream_t stream, const float *x, const float *y, const float *w,
                      float *out) const;

  /** @brief out[r] = per_problem[problem(r)] * in[r]. */
  void ScaleRows(cudaStream_t stream, const float *per_problem, const float *in, float *out) const;

  /** @brief Copies the states of subproblems with accept[p] != 0 from `from` into `to`. */
  void CopyAccepted(cudaStream_t stream, const Problem &problem, const MinimizerState &from,
                    MinimizerState &to) const;

  /**
   * @brief Starts step control: per-subproblem costs (already accumulated in
   * Cost()) become the current costs; subproblems below the cost tolerance are
   * done; damping starts at `initial_lambda`. Writes [total cost, active count]
   * to `d_out`.
   */
  void InitStepControl(cudaStream_t stream, float cost_tolerance, float initial_lambda,
                       const int *frozen, float *d_out);

  /**
   * @brief Decides, for every active subproblem, from NewCost(), StepSquared()
   * and (Levenberg-Marquardt) DiagWeight(), MatrixWeight(): accept or reject
   * the step, update its damping and its convergence. Writes [total cost after
   * the decision, active count] to `d_out`.
   */
  void StepControl(cudaStream_t stream, const BatchedStepControlParams &params, float *d_out);

  /**
   * @brief Line search: every active subproblem whose NewCost() is not below
   * its cost gets StepScale() = 0.5 and is marked shortened (others 1). Writes
   * the number of such subproblems to `d_count` (one float).
   */
  void MarkLineSearch(cudaStream_t stream, float *d_count);

  /**
   * @brief Per-subproblem device arrays (NumProblems() entries). StepScale() is
   * written by MarkLineSearch; Shortened() marks subproblems whose step was
   * shortened since ResetAccumulators (StepControl takes such a step when it
   * decreases the cost).
   */
  const float *StepScale() const { return step_scale_.data(); }
  float *Cost() { return cost_.data(); }
  float *NewCost() { return new_cost_.data(); }
  float *StepSquared() { return step_squared_.data(); }
  float *DiagWeight() { return diag_weight_.data(); }
  float *MatrixWeight() { return matrix_weight_.data(); }
  const float *Lambda() const { return lambda_.data(); }
  const int *Accept() const { return accept_.data(); }
  /** @brief Nonzero for subproblems still iterating (not converged). */
  const int *Active() const { return active_.data(); }

  /** @brief Zeroes the per-iteration accumulators (new cost, step, weights, shortened flags). */
  void ResetAccumulators(cudaStream_t stream);

 private:
  size_t num_problems_ = 0;
  size_t num_factor_items_ = 0;
  size_t num_rows_ = 0;
  dvector<int> factor_problem_;  ///< Per factor, in cost-buffer order.
  dvector<int> row_problem_;     ///< Per row of the reduced system.
  dvector<int> error_;           ///< Build() error flags.
  dvector<int> column_offsets_;  ///< Build() scratch.
  dvector<uint8_t> views_;       ///< Build(): per state batch, storage and ids.
  dvector<float> cost_, new_cost_, step_squared_, diag_weight_, matrix_weight_, lambda_;
  dvector<float> step_scale_;
  dvector<int> active_, rejected_, accept_, shortened_;
};

}  // namespace cunls
