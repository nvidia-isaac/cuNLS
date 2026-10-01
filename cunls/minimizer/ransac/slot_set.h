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
 * @file slot_set.h
 * @brief A set of independent small problems ("slots") solved together with
 * Gauss-Newton or Levenberg-Marquardt.
 */

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"
#include "cunls/minimizer/ransac/ransac_layout.h"

namespace cunls {
namespace ransac_internal {

/** @brief Step policy and dense solver shared by every slot set of a minimizer. */
struct SolverSettings {
  DevicePolicy policy;
  float initial_lambda = 0.f;
  SolverKind solver = kSolveLDLT;
};

/** @brief Which factors of the kSampled batches a slot is solved on. */
enum class SlotRows {
  kNone,            ///< States only (no evaluation), e.g. the best-so-far estimate.
  kMinimalSamples,  ///< The slot's minimal sample (hypothesis generation).
  kAllMasked,       ///< Every sampled factor, skipping the slot's outliers (refinement).
};

/**
 * @brief Slots, each with its own copy of the free states, solved in lockstep.
 *
 * A slot's state lives in per-slot replicas of every state batch that has a
 * free block (fully constant batches are read from user storage). Every
 * residual batch is evaluated for all slots with one FactorBatch::Evaluate
 * call: the items are (factor, slot) pairs, addressed through state-pointer
 * tables and, for minimal samples, factor ids. kAlwaysOn batches always
 * contribute all of their factors.
 *
 * Iterate() runs one GN / LM step per active slot; every reduction is
 * deterministic.
 */
class SlotSet {
 public:
  /** @brief Sizes every buffer for `num_slots` slots, keeping capacity. */
  void Allocate(cudaStream_t stream, const RansacLayout &layout,
                const RansacMinimizerOptions &options, SlotRows rows, int num_slots);

  // --- States -------------------------------------------------------------

  /** @brief Every slot's state = the problem's current state values. */
  void LoadInitialGuess(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Every slot's state = src's slot 0. */
  void CopyFrom(cudaStream_t stream, const RansacLayout &layout, const SlotSet &src);

  /**
   * @brief Slot 0's state = src's slot *src_slot, if *only_if != 0. Both are
   * device pointers, so the decision needs no host synchronization.
   */
  void CopySlotIf(cudaStream_t stream, const RansacLayout &layout, const SlotSet &src,
                  const int *src_slot, const int *only_if);

  /** @brief Copies `slot`'s state into the problem's state batches. */
  void WriteBack(cudaStream_t stream, const RansacLayout &layout, int slot) const;

  // --- Solving (not for kNone) ----------------------------------------------

  /**
   * @brief kMinimalSamples: draws round `round`'s minimal samples, slot p's
   * from the keyed permutation PermutationKey(seed, round, p).
   */
  void DrawSamples(cudaStream_t stream, const RansacLayout &layout, uint64_t seed, uint64_t round);

  /** @brief Starts a solve: every slot active, lambda reset, counters and validity reset. */
  void ResetSolver(cudaStream_t stream, float initial_lambda);

  /**
   * @brief One GN / LM iteration of every active slot: evaluate, build the
   * normal equations, solve, apply the step, evaluate the candidates, accept
   * or reject.
   */
  void Iterate(cudaStream_t stream, const RansacLayout &layout, const SolverSettings &settings);

  /**
   * @brief True while some slot is still iterating. Synchronizes the stream
   * (one scalar readback), so call it only when an iteration is costly.
   */
  bool AnyActive(cudaStream_t stream);

  /**
   * @brief kAllMasked: classifies every sampled factor at the current states,
   * scores the slots and updates their inlier masks.
   */
  void Classify(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Cost of each slot's current state over its (masked) items. */
  void EvaluateCost(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Costs of the kAlwaysOn batches at the current states (for scoring). */
  void EvaluateAlwaysOnCost(cudaStream_t stream, const RansacLayout &layout);

  // --- Accessors ------------------------------------------------------------

  int num_slots() const { return num_slots_; }
  /** @brief Device array of per-state-batch views, for building pointer tables. */
  const StateView *state_views() const { return state_views_.data(); }
  /** @brief Scoring rules and kAlwaysOn cost views (sampled views only for kAllMasked). */
  const ScoreInputs &score_inputs() const { return score_inputs_; }
  float *score() { return score_.data(); }
  const float *score() const { return score_.data(); }
  int *inliers() { return inliers_.data(); }
  const int *inliers() const { return inliers_.data(); }
  const int *valid() const { return valid_.data(); }
  const uint8_t *mask() const { return mask_.data(); }
  const float *cost() const { return cost_cur_.data(); }
  const int *iterations() const { return iterations_.data(); }

 private:
  /** Per-slot copies of one state batch (empty if it has no free block). */
  struct Replicas {
    dvector<float> cur, cand, delta;
  };
  /** Evaluation buffers of one residual batch, item-major. */
  struct Buffers {
    dvector<float> res, jac, cost;
    dvector<float *> table_cur, table_cand;  ///< Item-major state pointers.
    dvector<int> factor_ids;                 ///< Minimal samples only.
    size_t items = 0;
  };
  enum class Target { kCurrent, kCandidate };
  enum class Batches { kAll, kSampled, kAlwaysOn };

  bool SampleRows(const ResidualLayout &r) const {
    return r.sampled && rows_ == SlotRows::kMinimalSamples;
  }
  void AllocateReplicas(const RansacLayout &layout);
  void AllocateBuffers(cudaStream_t stream, const RansacLayout &layout);
  void AllocateSolver(const RansacLayout &layout);
  void PublishViews(const RansacLayout &layout, const RansacMinimizerOptions &options);
  BatchView MakeView(const ResidualLayout &r, size_t b, int kind) const;
  SlotItems Items() const;

  /** Evaluates the selected batches for every slot, one Evaluate call per batch. */
  void Evaluate(cudaStream_t stream, const RansacLayout &layout, Target target, Batches which,
                bool with_loss, bool jacobians);
  /** candidate = current (+) delta, one Plus call per state batch for all slots. */
  void ApplyStep(cudaStream_t stream, const RansacLayout &layout);
  /** current = candidate for the slots whose step was accepted. */
  void AcceptCandidates(cudaStream_t stream, const RansacLayout &layout);

  SlotRows rows_ = SlotRows::kNone;
  int num_slots_ = 0;
  int dim_ = 0;
  int sample_size_ = 0;
  int total_sampled_ = 0;
  int m_max_ = 1;

  std::vector<Replicas> replicas_;
  dvector<StateView> state_views_;

  std::vector<Buffers> buffers_;
  dvector<float> workspace_;  ///< ResidualBatch loss workspace.
  dvector<int> samples_;      ///< num_slots x sample_size (kMinimalSamples).
  dvector<uint8_t> mask_;     ///< num_slots x total_sampled (kAllMasked).
  dvector<BatchView> views_, score_sampled_, score_always_on_;
  int num_views_ = 0;
  int per_slot_items_ = 0;
  ScoreInputs score_inputs_;

  dvector<float> hessian_, gradient_, delta_;
  dvector<float> normal_scratch_, cost_scratch_, score_scratch_;  ///< Split-reduction partials.
  dvector<float> cost_cur_, cost_cand_, predicted_, step_sq_, lambda_, score_;
  dvector<int> solve_ok_, active_, accept_, valid_, iterations_, num_accepted_, inliers_;
  dvector<int> active_count_;
};

}  // namespace ransac_internal
}  // namespace cunls
