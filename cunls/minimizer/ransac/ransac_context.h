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

/**
 * @file ransac_context.h
 * @brief Orchestrates one RANSAC minimization: rounds of hypothesis
 * generation and scoring, final refinement.
 */

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/pinned_vector.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/hypothesis_scorer.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_set.h"

namespace cunls {

struct RansacMinimizerOptions;
struct RansacSummary;

namespace ransac_internal {

/**
 * @brief State of a RANSAC minimizer between and during Minimize() calls
 * (data only; RunRansac() is the behaviour).
 *
 * RunRansac():
 *  1. Prepare: validate, derive the layout, size every buffer.
 *  2. Rounds: sample K minimal sets, iterate them into hypotheses, score them
 *     and keep the best so far. Stop adaptively.
 *  3. Refine the best estimate on its inliers, write it back, fill the summary.
 *
 * The device keeps the best-so-far estimate and statistics; the host reads
 * them once per round and once at the end.
 */
struct RansacContext {
  RansacLayout layout;                      ///< Batches, tangent layout, sample size.
  HypothesisScorer scorer;                  ///< Scores hypotheses against the sampled factors.
  SlotSet hypotheses;                       ///< One slot per hypothesis of a round.
  SlotSet refinement;                       ///< Final refinement (one slot).
  SlotSet best;                             ///< Best-so-far estimate (one slot, states only).
  dvector<DeviceStats> stats;               ///< Round statistics on the device.
  PinnedVector<DeviceStats> host_stats{1};  ///< Host mirror of stats.
  dvector<int> selected;                    ///< Best hypothesis of the last round (device).
  dvector<float> cost_history;              ///< Refinement cost per iteration.
  dvector<float> initial_res;               ///< Scratch of the initial cost.
  dvector<float> initial_cost;              ///< Scratch of the initial cost.
  dvector<float> initial_workspace;         ///< Scratch of the initial cost.
  dvector<float> initial_sums;              ///< Per-batch sums, then reduction partials.
  bool has_run = false;                     ///< A run completed (the inlier masks are valid).
};

/** @brief One RANSAC minimization (see RansacMinimizer::Minimize). */
RansacSummary RunRansac(cudaStream_t stream, Problem &problem,
                        const RansacMinimizerOptions &options, const SolverSettings &settings,
                        RansacContext &context);

/** @brief See RansacMinimizer::InlierMask. */
const uint8_t *InlierMask(const RansacContext &context, size_t residual_batch_index);

/** @brief See RansacMinimizer::InlierMaskSize. */
size_t InlierMaskSize(const RansacContext &context, size_t residual_batch_index);

}  // namespace ransac_internal
}  // namespace cunls
