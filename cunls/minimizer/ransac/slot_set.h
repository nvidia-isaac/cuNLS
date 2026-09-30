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
 * @brief A set of independent small problems ("slots") iterated together with
 * Gauss-Newton or Levenberg-Marquardt.
 */

#include <cuda_runtime.h>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_evaluator.h"
#include "cunls/minimizer/ransac/slot_replicas.h"

namespace cunls {
namespace ransac_internal {

/** @brief Step policy and dense solver shared by every slot set of a minimizer. */
struct SolverSettings {
  DevicePolicy policy;
  float initial_lambda = 0.f;
  SolverKind solver = kSolveLDLT;
};

/**
 * @brief Slots with their state replicas, evaluation buffers and per-slot
 * solver state.
 *
 * Used three ways: hypotheses (minimal samples, SampledRows::kWaves), local
 * optimization candidates and the final refinement (every sampled factor with
 * a per-slot inlier mask, SampledRows::kMaskedPerSlot).
 */
class SlotSet {
 public:
  void Allocate(cudaStream_t stream, const RansacLayout &layout,
                const RansacMinimizerOptions &options, SampledRows sampled_rows, int num_slots,
                int num_waves = 1);

  int num_slots() const { return replicas_.num_slots(); }
  SlotReplicas &replicas() { return replicas_; }
  const SlotReplicas &replicas() const { return replicas_; }
  SlotEvaluator &evaluator() { return evaluator_; }

  /** @brief Minimal samples of a kWaves set: num_slots x sample_size, and slots per wave. */
  void SetSamples(const int *samples, int hyp_per_wave);

  /** @brief Starts a new solve: every slot active, lambda reset, counters and validity reset. */
  void ResetSolver(cudaStream_t stream, float initial_lambda);

  /**
   * @brief One GN / LM iteration of every active slot: evaluate, build the
   * normal equations, solve, apply the step, evaluate the candidates, accept
   * or reject.
   */
  void Iterate(cudaStream_t stream, const RansacLayout &layout, const SolverSettings &settings);

  /**
   * @brief Classifies every sampled factor at the current states and scores
   * the slots; the per-slot inlier mask of a kMaskedPerSlot set is updated.
   */
  void Classify(cudaStream_t stream, const RansacLayout &layout);

  /**
   * @brief True while some slot is still iterating. Synchronizes the stream
   * (one scalar readback), so callers use it only when an iteration is costly.
   */
  bool AnyActive(cudaStream_t stream);

  /** @brief Cost of each slot's current state over its (masked) items. */
  void EvaluateCost(cudaStream_t stream, const RansacLayout &layout);

  const float *score() const { return score_.data(); }
  float *score() { return score_.data(); }
  const int *inliers() const { return inliers_.data(); }
  int *inliers() { return inliers_.data(); }
  const int *valid() const { return valid_.data(); }
  const uint8_t *mask() const { return mask_.data(); }
  const float *cost() const { return cost_cur_.data(); }
  const int *iterations() const { return iterations_.data(); }

 private:
  SlotItems Items() const { return evaluator_.Items(samples_, hyp_per_wave_, mask_.data()); }

  SlotReplicas replicas_;
  SlotEvaluator evaluator_;
  int dim_ = 0;
  const int *samples_ = nullptr;
  int hyp_per_wave_ = 1;
  bool track_validity_ = false;

  dvector<uint8_t> mask_;
  dvector<float> hessian_, gradient_, delta_;
  dvector<float> normal_scratch_;  ///< Split partials of the normal equations.
  dvector<float> cost_scratch_;    ///< Split partials of the slot costs.
  dvector<float> score_scratch_;   ///< Split partials of the scores.
  dvector<float> cost_cur_, cost_cand_, predicted_, step_sq_, lambda_, score_;
  dvector<int> solve_ok_, active_, accept_, valid_, iterations_, num_accepted_, inliers_;
  dvector<int> active_count_;
};

}  // namespace ransac_internal
}  // namespace cunls
