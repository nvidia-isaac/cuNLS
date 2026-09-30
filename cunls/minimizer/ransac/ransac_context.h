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
 * generation and scoring, local optimization, final refinement.
 */

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/hypothesis_sampler.h"
#include "cunls/minimizer/ransac/hypothesis_scorer.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_set.h"
#include "cunls/minimizer/ransac_minimizer.h"

namespace cunls {
namespace ransac_internal {

/**
 * @brief State of a RANSAC minimizer between and during Minimize() calls.
 *
 * Minimize():
 *  1. Prepare: validate, derive the layout, size every buffer.
 *  2. Rounds: sample K minimal sets, iterate them into hypotheses, score and
 *     select; when a round improves the best score, re-solve its top-M
 *     hypotheses on their inliers (local optimization). Stop adaptively.
 *  3. Refine the best estimate on its inliers, write it back, fill the summary.
 *
 * The device keeps the best-so-far estimate and statistics; the host reads
 * them once per round and once at the end.
 */
class RansacContext {
 public:
  RansacContext(const RansacMinimizerOptions &options, const SolverSettings &settings);
  ~RansacContext();
  RansacContext(const RansacContext &) = delete;
  RansacContext &operator=(const RansacContext &) = delete;

  RansacSummary Minimize(cudaStream_t stream, Problem &problem);
  const uint8_t *InlierMask(size_t residual_batch_index) const;

 private:
  void Prepare(cudaStream_t stream, const Problem &problem);
  float InitialCost(cudaStream_t stream);

  void RunRounds(cudaStream_t stream, RansacSummary &summary);
  void GenerateHypotheses(cudaStream_t stream, uint64_t round);
  void SelectHypotheses(cudaStream_t stream, uint64_t round);
  void LocalOptimization(cudaStream_t stream);
  bool EnoughHypotheses(size_t drawn) const;
  /** Whether per-iteration convergence checks (one sync each) pay off for this size. */
  bool WorthCheckingConvergence(int num_slots) const;
  /** Reads the candidates' inlier counts; true if equal to `previous` (then updated). */
  bool InlierCountsUnchanged(cudaStream_t stream, std::vector<int> &previous);

  void Refine(cudaStream_t stream, RansacSummary &summary);
  void RevertRefinementIfWorse(cudaStream_t stream, RansacSummary &summary);
  void ReadRefinement(cudaStream_t stream, RansacSummary &summary);
  void ReadStats(cudaStream_t stream);

  const RansacMinimizerOptions options_;
  const SolverSettings settings_;
  RansacLayout layout_;
  HypothesisSampler sampler_;
  HypothesisScorer scorer_;
  SlotSet hypotheses_;
  SlotSet candidates_;  ///< Local-optimization candidates (top-M).
  SlotSet refinement_;  ///< Final refinement (one slot).
  SlotReplicas best_;   ///< Best-so-far estimate (one slot).
  dvector<DeviceStats> stats_;
  DeviceStats *host_stats_ = nullptr;
  dvector<int> top_indices_;
  dvector<int> candidate_top_;
  dvector<float> cost_history_;
  dvector<float> initial_res_;        ///< Scratch of InitialCost().
  dvector<float> initial_cost_;       ///< Scratch of InitialCost().
  dvector<float> initial_workspace_;  ///< Scratch of InitialCost().
  dvector<float> initial_sums_;       ///< Per-batch sums, then reduction partials.
  bool has_run_ = false;
};

}  // namespace ransac_internal
}  // namespace cunls
