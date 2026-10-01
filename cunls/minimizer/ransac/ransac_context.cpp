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

#include "cunls/minimizer/ransac/ransac_context.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/residual_batch.h"

namespace cunls {
namespace ransac_internal {

namespace {

void ValidateOptions(const RansacMinimizerOptions &o) {
  if (o.hypotheses_per_round == 0) {
    FailConfiguration("RANSAC: hypotheses_per_round must be > 0");
  }
  if (o.max_rounds == 0) {
    FailConfiguration("RANSAC: max_rounds must be > 0");
  }
  if (o.hypothesis_iterations == 0) {
    FailConfiguration(
        "RANSAC: hypothesis_iterations must be > 0 (there is no hypothesis "
        "generator)");
  }
  if (!(o.confidence > 0.f && o.confidence < 1.f)) {
    FailConfiguration("RANSAC: confidence must be in (0, 1)");
  }
}

template <typename T>
T CopyScalarToHost(cudaStream_t stream, const T *device) {
  T value{};
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&value, device, sizeof(T), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return value;
}

}  // namespace

RansacContext::RansacContext(const RansacMinimizerOptions &options, const SolverSettings &settings)
    : options_(options), settings_(settings) {
  THROW_ON_CUDA_ERROR(cudaMallocHost(&host_stats_, sizeof(DeviceStats)));
  stats_.resize(1);
}

RansacContext::~RansacContext() { cudaFreeHost(host_stats_); }

RansacSummary RansacContext::Minimize(cudaStream_t stream, Problem &problem) {
  Prepare(stream, problem);
  RansacSummary summary;
  summary.initial_cost = InitialCost(stream);
  RunRounds(stream, summary);
  Refine(stream, summary);
  has_run_ = true;
  return summary;
}

void RansacContext::Prepare(cudaStream_t stream, const Problem &problem) {
  problem.CheckSizes();  // cheap host-only guard; see Problem::CheckSizes
  ValidateOptions(options_);
  has_run_ = false;
  layout_.Build(problem, options_);
  const int k = static_cast<int>(options_.hypotheses_per_round);
  hypotheses_.Allocate(stream, layout_, options_, SlotRows::kMinimalSamples, k);
  refinement_.Allocate(stream, layout_, options_, SlotRows::kAllMasked, 1);
  best_.Allocate(stream, layout_, options_, SlotRows::kNone, 1);
  // Fallback when no hypothesis is ever selected (e.g. every sample degenerate):
  // the refinement then starts from the initial guess.
  best_.LoadInitialGuess(stream, layout_);
  scorer_.Allocate(layout_, options_, k);
  selected_.resize(1);
  cost_history_.resize(std::max<size_t>(options_.final_iterations, 1));
}

float RansacContext::InitialCost(cudaStream_t stream) {
  const auto &residuals = layout_.residuals();
  const size_t max_factors = static_cast<size_t>(layout_.max_factors());
  initial_res_.resize(max_factors * layout_.m_max());
  initial_cost_.resize(max_factors);
  initial_workspace_.resize(ResidualBatchWorkspaceNumFloats(max_factors));
  initial_sums_.resize(residuals.size() + ReducePartialCount(max_factors));
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(initial_sums_.data(), 0, residuals.size() * sizeof(float), stream));
  float *partials = initial_sums_.data() + residuals.size();
  for (size_t b = 0; b < residuals.size(); ++b) {
    const ResidualLayout &r = residuals[b];
    if (r.num_factors > 0) {
      r.residual_batch->Evaluate(stream, initial_workspace_.data(), initial_res_.data(),
                                 r.x0_table.data(), initial_cost_.data(), nullptr);
      ReduceSumToDevice(stream, initial_cost_.data(), r.num_factors, initial_sums_.data() + b,
                        partials);
    }
  }
  std::vector<float> sums(residuals.size());
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(sums.data(), initial_sums_.data(),
                                      sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  double total = 0.0;
  for (float v : sums) {
    total += v;
  }
  return static_cast<float>(total);
}

void RansacContext::RunRounds(cudaStream_t stream, RansacSummary &summary) {
  LaunchInitStats(stream, stats_.data());
  size_t drawn = 0;
  for (size_t round = 0; round < options_.max_rounds; ++round) {
    GenerateHypotheses(stream, round);
    SelectHypotheses(stream, round);
    ReadStats(stream);
    drawn += options_.hypotheses_per_round;
    summary.num_rounds = round + 1;
    if (EnoughHypotheses(drawn)) {
      break;
    }
  }
  summary.num_hypotheses = drawn;
}

void RansacContext::GenerateHypotheses(cudaStream_t stream, uint64_t round) {
  hypotheses_.LoadInitialGuess(stream, layout_);
  hypotheses_.ResetSolver(stream, settings_.initial_lambda);
  hypotheses_.DrawSamples(stream, layout_, options_.seed, round);
  for (size_t it = 0; it < options_.hypothesis_iterations; ++it) {
    hypotheses_.Iterate(stream, layout_, settings_);
  }
}

void RansacContext::SelectHypotheses(cudaStream_t stream, uint64_t round) {
  scorer_.Score(stream, layout_, hypotheses_, round);
  LaunchSelect(stream, hypotheses_.score(), hypotheses_.inliers(), hypotheses_.valid(),
               hypotheses_.num_slots(), 1, selected_.data(), stats_.data());
  best_.CopySlotIf(stream, layout_, hypotheses_, &stats_.data()->best_slot,
                   &stats_.data()->improved);
}

bool RansacContext::WorthCheckingConvergence() const {
  // A convergence check costs one stream synchronization (~10 us); it pays off
  // once an iteration touches this many factor evaluations.
  constexpr size_t kMinItemsForChecks = size_t{1} << 17;
  return static_cast<size_t>(layout_.total_sampled()) >= kMinItemsForChecks;
}

bool RansacContext::EnoughHypotheses(size_t drawn) const {
  const double w = static_cast<double>(host_stats_->best_inliers) / layout_.total_sampled();
  if (w >= options_.early_stop_inlier_ratio) {
    return true;
  }
  const double p_good = std::pow(w, layout_.sample_size());
  if (p_good <= 0.0) {
    return false;
  }
  if (p_good >= 1.0) {
    return true;
  }
  const double needed = std::log(1.0 - options_.confidence) / std::log(1.0 - p_good);
  return static_cast<double>(drawn) >= needed;
}

void RansacContext::Refine(cudaStream_t stream, RansacSummary &summary) {
  refinement_.CopyFrom(stream, layout_, best_);
  refinement_.Classify(stream, layout_);
  refinement_.ResetSolver(stream, settings_.initial_lambda);
  const bool check = WorthCheckingConvergence();
  for (size_t it = 0; it < options_.final_iterations; ++it) {
    refinement_.Iterate(stream, layout_, settings_);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cost_history_.data() + it, refinement_.cost(),
                                        sizeof(float), cudaMemcpyDeviceToDevice, stream));
    if (check && !refinement_.AnyActive(stream)) {
      break;
    }
  }
  refinement_.Classify(stream, layout_);
  refinement_.EvaluateCost(stream, layout_);
  RevertRefinementIfWorse(stream, summary);
  refinement_.WriteBack(stream, layout_, 0);
  ReadRefinement(stream, summary);
}

void RansacContext::RevertRefinementIfWorse(cudaStream_t stream, RansacSummary &summary) {
  const float refined = CopyScalarToHost(stream, refinement_.score());
  ReadStats(stream);
  const float best = host_stats_->best_score;
  if (!std::isfinite(best) || refined <= best + 1e-6f * std::fabs(best) + 1e-12f) {
    return;
  }
  LogMessage("RANSAC: refinement worsened the score ({} > {}); keeping the best hypothesis",
             refined, best);
  refinement_.CopyFrom(stream, layout_, best_);
  refinement_.Classify(stream, layout_);
  refinement_.EvaluateCost(stream, layout_);
  summary.refinement_reverted = true;
}

void RansacContext::ReadRefinement(cudaStream_t stream, RansacSummary &summary) {
  const int iterations = CopyScalarToHost(stream, refinement_.iterations());
  const int inliers = CopyScalarToHost(stream, refinement_.inliers());
  summary.final_cost = CopyScalarToHost(stream, refinement_.cost());
  summary.best_score = CopyScalarToHost(stream, refinement_.score());
  summary.num_inliers = static_cast<size_t>(inliers);
  summary.inlier_ratio = static_cast<float>(inliers) / layout_.total_sampled();
  summary.num_iterations = static_cast<size_t>(iterations);
  summary.iteration_costs.resize(std::min<size_t>(iterations, options_.final_iterations));
  if (!summary.iteration_costs.empty()) {
    cost_history_.CopyToHost(summary.iteration_costs.data(), summary.iteration_costs.size());
  }
  ReadStats(stream);
  summary.num_valid_hypotheses = static_cast<size_t>(host_stats_->valid_total);
}

void RansacContext::ReadStats(cudaStream_t stream) {
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(host_stats_, stats_.data(), sizeof(DeviceStats),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

const uint8_t *RansacContext::InlierMask(size_t residual_batch_index) const {
  const auto &residuals = layout_.residuals();
  if (!has_run_ || residual_batch_index >= residuals.size() ||
      !residuals[residual_batch_index].sampled) {
    return nullptr;
  }
  return refinement_.mask() + residuals[residual_batch_index].u_offset;
}

size_t RansacContext::InlierMaskSize(size_t residual_batch_index) const {
  // layout_ was built by the last Prepare(), so this is the batch size the mask
  // was computed for, whatever the problem looks like now.
  return InlierMask(residual_batch_index) == nullptr
             ? 0
             : static_cast<size_t>(layout_.residuals()[residual_batch_index].num_factors);
}

}  // namespace ransac_internal
}  // namespace cunls
