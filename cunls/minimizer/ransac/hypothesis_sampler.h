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
 * @file hypothesis_sampler.h
 * @brief Draws disjoint minimal samples ("waves") for a round of hypotheses.
 */

#include <cuda_runtime.h>

#include <cstdint>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_layout.h"
#include "cunls/minimizer/ransac/slot_set.h"

namespace cunls {
namespace ransac_internal {

/**
 * @brief Samples minimal sets for the hypothesis slots.
 *
 * A wave is a keyed random permutation of the concatenated sampled factors
 * cut into floor(N / s) disjoint samples, so each factor belongs to at most
 * one hypothesis of the wave and a single Evaluate per batch serves all of
 * them. ceil(K / floor(N / s)) waves cover K hypotheses. No sorting is done.
 */
class HypothesisSampler {
 public:
  /** @brief Sizes the owner / sample arrays for `num_hypotheses` per round. */
  void Configure(const RansacLayout &layout, int num_hypotheses);

  int num_waves() const { return num_waves_; }
  int hypotheses_per_wave() const { return per_wave_; }

  /**
   * @brief Draws round `round`'s samples and points the hypothesis slots'
   * wave tables at them.
   */
  void Sample(cudaStream_t stream, const RansacLayout &layout, uint64_t seed, uint64_t round,
              SlotSet &hypotheses);

 private:
  int num_hypotheses_ = 0;
  int per_wave_ = 0;
  int num_waves_ = 0;
  dvector<int> owner_;    ///< num_waves x total_sampled owning slot (-1 = none).
  dvector<int> samples_;  ///< num_hypotheses x sample_size factor indices.
};

}  // namespace ransac_internal
}  // namespace cunls
