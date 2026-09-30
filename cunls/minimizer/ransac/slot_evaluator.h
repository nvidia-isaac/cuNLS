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
 * @file slot_evaluator.h
 * @brief Evaluates the problem's residual batches for a set of slots: one
 * FactorBatch::EvaluateIndexed call per batch when supported, otherwise one
 * Evaluate call per copy.
 */

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_replicas.h"

namespace cunls {
namespace ransac_internal {

/** @brief Where a slot's kSampled rows come from. */
enum class SampledRows {
  /**
   * Minimal samples (hypothesis generation): one Evaluate per sampled batch
   * and wave serves every slot of the wave, because the samples of a wave are
   * disjoint and each factor's pointers lead to its owning slot.
   */
  kWaves,
  /** Every sampled factor per slot, outliers skipped by the slot's mask (LO, refinement). */
  kMaskedPerSlot,
};

/**
 * @brief Output buffers and state-pointer tables of every residual batch for
 * one slot set, and the Evaluate calls that fill them.
 *
 * kAlwaysOn batches are always evaluated once per slot. Buffers are laid out
 * copy-major (copy = wave or slot) so that a slot's rows are found through
 * BatchView strides.
 */
class SlotEvaluator {
 public:
  void Allocate(cudaStream_t stream, const RansacLayout &layout, const SlotReplicas &replicas,
                const RansacMinimizerOptions &options, SampledRows sampled_rows, int num_waves);

  /**
   * @brief Points wave w's sampled factors at their owning slots.
   * @param owner num_waves x total_sampled owning slot per factor (-1 = none).
   */
  void BuildWaveTables(cudaStream_t stream, const RansacLayout &layout,
                       const SlotReplicas &replicas, const int *owner);

  /** @brief Residuals, costs (with loss) and optionally Jacobians at the current states. */
  void EvaluateCurrent(cudaStream_t stream, const RansacLayout &layout, bool jacobians);

  /** @brief Residuals and costs (with loss) at the candidate states. */
  void EvaluateCandidate(cudaStream_t stream, const RansacLayout &layout);

  /**
   * @brief Raw (loss-free) residuals of the kSampled batches at the current
   * states, plus Jacobians when inliers must be informative. Per-slot sampled
   * rows only (kMaskedPerSlot).
   */
  void EvaluateRawSampled(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Costs of the kAlwaysOn batches at the current states. */
  void EvaluateAlwaysOnCost(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Item description for the normal-equation / cost kernels. */
  SlotItems Items(const int *samples, int hyp_per_wave, const uint8_t *mask) const;

  /** @brief Score-kernel inputs over this set's per-slot buffers. */
  const ScoreInputs &score_inputs() const { return score_inputs_; }

 private:
  struct Buffers {
    dvector<float> res;
    dvector<float> jac;
    dvector<float> cost;
    dvector<float *> table_cur;
    dvector<float *> table_cand;
    int copies = 0;  ///< Waves or slots.
  };

  enum class Target { kCurrent, kCandidate };
  enum class Batches { kAll, kSampled, kAlwaysOn };

  /** Evaluates every copy of the selected batches. */
  void Run(cudaStream_t stream, const RansacLayout &layout, Target target, Batches which,
           bool with_loss, bool jacobians);
  /** One EvaluateIndexed call for all copies; false if the batch does not support it. */
  bool EvaluateAllCopies(cudaStream_t stream, const ResidualLayout &r, Buffers &buf,
                         float *const *table, bool with_loss, bool jacobians);
  /** Fallback: one Evaluate call per copy. */
  void EvaluateEachCopy(cudaStream_t stream, const ResidualLayout &r, Buffers &buf,
                        float *const *table, bool with_loss, bool jacobians);
  BatchView MakeView(const ResidualLayout &r, const Buffers &buf, int kind) const;
  void PublishViews(const RansacLayout &layout);

  SampledRows sampled_rows_ = SampledRows::kMaskedPerSlot;
  int num_slots_ = 0;
  int sample_size_ = 0;
  int total_sampled_ = 0;
  int m_max_ = 0;
  std::vector<Buffers> buffers_;
  dvector<float> workspace_;
  dvector<BatchView> views_;
  dvector<BatchView> score_sampled_;
  dvector<BatchView> score_always_on_;
  int num_views_ = 0;
  int per_slot_items_ = 0;
  ScoreInputs score_inputs_;
};

}  // namespace ransac_internal
}  // namespace cunls
