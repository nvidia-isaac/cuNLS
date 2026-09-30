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

#include "cunls/minimizer/ransac/slot_set.h"

namespace cunls {
namespace ransac_internal {

void SlotSet::Allocate(cudaStream_t stream, const RansacLayout &layout,
                       const RansacMinimizerOptions &options, SampledRows sampled_rows,
                       int num_slots, int num_waves) {
  dim_ = layout.dim();
  track_validity_ = sampled_rows == SampledRows::kWaves;
  replicas_.Allocate(layout, num_slots);
  evaluator_.Allocate(stream, layout, replicas_, options, sampled_rows, num_waves);
  const size_t slots = static_cast<size_t>(num_slots);
  const size_t d = static_cast<size_t>(dim_);
  mask_.resize(sampled_rows == SampledRows::kMaskedPerSlot ? slots * layout.total_sampled() : 0);
  hessian_.resize(slots * d * d);
  gradient_.resize(slots * d);
  delta_.resize(slots * d);
  for (auto *v : {&cost_cur_, &cost_cand_, &predicted_, &step_sq_, &lambda_, &score_}) {
    v->resize(slots);
  }
  for (auto *v : {&solve_ok_, &active_, &accept_, &valid_, &iterations_, &num_accepted_,
                  &inliers_}) {
    v->resize(slots);
  }
  active_count_.resize(1);
  normal_scratch_.resize(NormalEquationsScratchFloats(Items(), num_slots, dim_));
  cost_scratch_.resize(SlotCostScratchFloats(Items(), num_slots));
  score_scratch_.resize(ScoreScratchFloats(layout.total_sampled(), num_slots));
}

void SlotSet::SetSamples(const int *samples, int hyp_per_wave) {
  samples_ = samples;
  hyp_per_wave_ = hyp_per_wave;
}

void SlotSet::ResetSolver(cudaStream_t stream, float initial_lambda) {
  LaunchResetSlots(stream, num_slots(), initial_lambda, active_.data(), lambda_.data(),
                   valid_.data(), iterations_.data(), num_accepted_.data());
}

void SlotSet::Iterate(cudaStream_t stream, const RansacLayout &layout,
                      const SolverSettings &settings) {
  const SlotItems items = Items();
  const bool lm = settings.policy.levenberg_marquardt != 0;
  evaluator_.EvaluateCurrent(stream, layout, true);
  LaunchNormalEquations(stream, items, num_slots(), dim_, hessian_.data(), gradient_.data(),
                        cost_cur_.data(), normal_scratch_.data());
  LaunchSolve(stream, num_slots(), dim_, settings.solver, hessian_.data(), gradient_.data(),
              lm ? lambda_.data() : nullptr, active_.data(), delta_.data(), predicted_.data(),
              step_sq_.data(), solve_ok_.data());
  replicas_.ApplyStep(stream, layout, delta_.data());
  evaluator_.EvaluateCandidate(stream, layout);
  LaunchSlotCost(stream, items, num_slots(), cost_cand_.data(), cost_scratch_.data());
  LaunchAccept(stream, num_slots(), settings.policy, cost_cur_.data(), cost_cand_.data(),
               predicted_.data(), step_sq_.data(), solve_ok_.data(), active_.data(),
               lambda_.data(), accept_.data(), track_validity_ ? valid_.data() : nullptr,
               iterations_.data(), num_accepted_.data());
  replicas_.AcceptCandidates(stream, layout, accept_.data());
}

void SlotSet::Classify(cudaStream_t stream, const RansacLayout &layout) {
  evaluator_.EvaluateRawSampled(stream, layout);
  evaluator_.EvaluateAlwaysOnCost(stream, layout);
  LaunchScore(stream, evaluator_.score_inputs(), num_slots(), 0, nullptr, score_.data(),
              inliers_.data(), mask_.empty() ? nullptr : mask_.data(), score_scratch_.data());
}

bool SlotSet::AnyActive(cudaStream_t stream) {
  LaunchCountActive(stream, num_slots(), active_.data(), active_count_.data());
  int count = 0;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&count, active_count_.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return count > 0;
}

void SlotSet::EvaluateCost(cudaStream_t stream, const RansacLayout &layout) {
  evaluator_.EvaluateCurrent(stream, layout, false);
  LaunchSlotCost(stream, Items(), num_slots(), cost_cur_.data(), cost_scratch_.data());
}

}  // namespace ransac_internal
}  // namespace cunls
