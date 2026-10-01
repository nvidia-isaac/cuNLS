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
 * @file hypothesis_scorer.h
 * @brief Scores every hypothesis of a round against the sampled factors.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_set.h"

namespace cunls {
namespace ransac_internal {

/**
 * @brief Computes each hypothesis's score and inlier count.
 *
 * Exhaustive scoring evaluates all kSampled factors for every hypothesis, in
 * chunks that fit the scoring memory budget, with one FactorBatch::Evaluate
 * call per batch and chunk.
 *
 * Two-stage scoring (large problems, see RansacMinimizerOptions::
 * scoring_subset_size) first scores every hypothesis on a random factor
 * subset, then scores only the best finalists on all factors; every other
 * hypothesis gets +inf. Both stages apply the same inlier rules.
 */
class HypothesisScorer {
 public:
  void Allocate(const RansacLayout &layout, const RansacMinimizerOptions &options,
                int num_hypotheses);

  /** @brief Writes hypotheses.score() / inliers(); invalid hypotheses score +inf. */
  void Score(cudaStream_t stream, const RansacLayout &layout, SlotSet &hypotheses, uint64_t round);

  int chunk() const { return chunk_; }
  bool two_stage() const { return two_stage_; }

 private:
  /** Buffers of one kSampled residual batch. */
  struct Batch {
    dvector<float> res;
    dvector<float> jac;
    dvector<float *> table;
    dvector<int> item_ids;      ///< Factor per item (subset stage).
    dvector<int> subset;        ///< Subset factor ids of the current round.
    dvector<int> subset_local;  ///< Local columns of the subset factors.
    int subset_size = 0;
  };

  void AllocateBatches(const RansacLayout &layout);
  void AllocateViews(const RansacLayout &layout);
  BatchView MakeView(const ResidualLayout &r, const Batch &buf, bool subset) const;

  void ScoreAll(cudaStream_t stream, const RansacLayout &layout, SlotSet &hypotheses);
  void ScoreSubset(cudaStream_t stream, const RansacLayout &layout, SlotSet &hypotheses,
                   uint64_t round);
  void ScoreFinalists(cudaStream_t stream, const RansacLayout &layout, SlotSet &hypotheses);
  void DrawSubsets(cudaStream_t stream, const RansacLayout &layout, uint64_t round);

  /** Evaluates all factors of every batch for `count` hypotheses. */
  void EvaluateChunk(cudaStream_t stream, const RansacLayout &layout, const SlotSet &hypotheses,
                     int first, int count, const int *slot_index);
  /** Evaluates the subset factors for hypotheses [first, first + count). */
  void EvaluateSubsetChunk(cudaStream_t stream, const RansacLayout &layout,
                           const SlotSet &hypotheses, int first, int count);
  ScoreInputs Inputs(const SlotSet &hypotheses, bool subset) const;

  int num_hypotheses_ = 0;
  int chunk_ = 1;         ///< Hypotheses per exhaustive chunk.
  int subset_chunk_ = 1;  ///< Hypotheses per subset chunk.
  int finalists_count_ = 0;
  bool jacobians_ = true;
  bool two_stage_ = false;
  uint64_t seed_ = 0;
  std::vector<Batch> batches_;
  dvector<BatchView> views_;
  dvector<BatchView> subset_views_;
  int subset_total_ = 0;
  dvector<float> scratch_;
  dvector<float> subset_score_;
  dvector<int> subset_inliers_;
  dvector<int> finalists_;
  dvector<DeviceStats> selection_stats_;  ///< Unused output of the finalist selection.
};

}  // namespace ransac_internal
}  // namespace cunls
