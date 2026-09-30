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

#include "cunls/minimizer/ransac/hypothesis_sampler.h"

#include <algorithm>

namespace cunls {
namespace ransac_internal {

void HypothesisSampler::Configure(const RansacLayout &layout, int num_hypotheses) {
  num_hypotheses_ = num_hypotheses;
  per_wave_ = layout.total_sampled() / layout.sample_size();
  num_waves_ = (num_hypotheses + per_wave_ - 1) / per_wave_;
  owner_.resize(static_cast<size_t>(num_waves_) * layout.total_sampled());
  samples_.resize(static_cast<size_t>(num_hypotheses) * layout.sample_size());
}

void HypothesisSampler::Sample(cudaStream_t stream, const RansacLayout &layout, uint64_t seed,
                               uint64_t round, SlotSet &hypotheses) {
  for (int w = 0; w < num_waves_; ++w) {
    const int first = w * per_wave_;
    LaunchWaveAssignment(stream, layout.total_sampled(), layout.sample_size(), first,
                         std::min(per_wave_, num_hypotheses_ - first), num_hypotheses_,
                         PermutationKey(seed, round, static_cast<uint64_t>(w)),
                         owner_.data() + static_cast<size_t>(w) * layout.total_sampled(),
                         samples_.data());
  }
  hypotheses.evaluator().BuildWaveTables(stream, layout, hypotheses.replicas(), owner_.data());
  hypotheses.SetSamples(samples_.data(), per_wave_);
}

}  // namespace ransac_internal
}  // namespace cunls
