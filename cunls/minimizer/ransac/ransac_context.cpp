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
#include "cunls/minimizer/ransac_minimizer.h"
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

void Prepare(cudaStream_t stream, const Problem &problem, const RansacMinimizerOptions &options,
             RansacContext &c) {
  problem.CheckSizes();  // cheap host-only guard; see Problem::CheckSizes
  ValidateOptions(options);
  c.has_run = false;
  c.layout.Build(problem, options);
  const int k = static_cast<int>(options.hypotheses_per_round);
  c.hypotheses.Allocate(stream, c.layout, options, SlotRows::kMinimalSamples, k);
  c.refinement.Allocate(stream, c.layout, options, SlotRows::kAllMasked, 1);
  c.best.Allocate(stream, c.layout, options, SlotRows::kNone, 1);
  // Fallback when no hypothesis is ever selected (e.g. every sample degenerate):
  // the refinement then starts from the initial guess.
  c.best.LoadInitialGuess(stream, c.layout);
  c.scorer.Allocate(c.layout, options, k);
  c.selected.resize(1);
  c.cost_history.resize(std::max<size_t>(options.final_iterations, 1));
}

float InitialCost(cudaStream_t stream, RansacContext &c) {
  const auto &residuals = c.layout.residuals();
  const size_t max_factors = static_cast<size_t>(c.layout.max_factors());
  c.initial_res.resize(max_factors * c.layout.m_max());
  c.initial_cost.resize(max_factors);
  c.initial_workspace.resize(ResidualBatchWorkspaceNumFloats(max_factors));
  c.initial_sums.resize(residuals.size() + ReducePartialCount(max_factors));
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(c.initial_sums.data(), 0, residuals.size() * sizeof(float), stream));
  float *partials = c.initial_sums.data() + residuals.size();
  for (size_t b = 0; b < residuals.size(); ++b) {
    const ResidualLayout &r = residuals[b];
    if (r.num_factors > 0) {
      CheckEvaluate(
          r.residual_batch->Evaluate(stream, c.initial_workspace.data(), c.initial_res.data(),
                                     r.x0_table.data(), c.initial_cost.data(), nullptr),
          b);
      ReduceSumToDevice(stream, c.initial_cost.data(), r.num_factors, c.initial_sums.data() + b,
                        partials);
    }
  }
  std::vector<float> sums(residuals.size());
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(sums.data(), c.initial_sums.data(),
                                      sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  double total = 0.0;
  for (float v : sums) {
    total += v;
  }
  return static_cast<float>(total);
}

void ReadStats(cudaStream_t stream, RansacContext &c) {
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(c.host_stats.data(), c.stats.data(), sizeof(DeviceStats),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void GenerateHypotheses(cudaStream_t stream, uint64_t round, const RansacMinimizerOptions &options,
                        const SolverSettings &settings, RansacContext &c) {
  c.hypotheses.LoadInitialGuess(stream, c.layout);
  c.hypotheses.ResetSolver(stream, settings.initial_lambda);
  c.hypotheses.DrawSamples(stream, c.layout, options.seed, round);
  for (size_t it = 0; it < options.hypothesis_iterations; ++it) {
    c.hypotheses.Iterate(stream, c.layout, settings);
  }
}

void SelectHypotheses(cudaStream_t stream, uint64_t round, RansacContext &c) {
  c.scorer.Score(stream, c.layout, c.hypotheses, round);
  LaunchSelect(stream, c.hypotheses.score(), c.hypotheses.inliers(), c.hypotheses.valid(),
               c.hypotheses.num_slots(), 1, c.selected.data(), c.stats.data());
  c.best.CopySlotIf(stream, c.layout, c.hypotheses, &c.stats.data()->best_slot,
                    &c.stats.data()->improved);
}

/** Whether per-iteration convergence checks (one sync each) pay off for this size. */
bool WorthCheckingConvergence(const RansacContext &c) {
  // A convergence check costs one stream synchronization (~10 us); it pays off
  // once an iteration touches this many factor evaluations.
  constexpr size_t kMinItemsForChecks = size_t{1} << 17;
  return static_cast<size_t>(c.layout.total_sampled()) >= kMinItemsForChecks;
}

bool EnoughHypotheses(size_t drawn, const RansacMinimizerOptions &options, const RansacContext &c) {
  const double w =
      static_cast<double>(c.host_stats.data()->best_inliers) / c.layout.total_sampled();
  if (w >= options.early_stop_inlier_ratio) {
    return true;
  }
  const double p_good = std::pow(w, c.layout.sample_size());
  if (p_good <= 0.0) {
    return false;
  }
  if (p_good >= 1.0) {
    return true;
  }
  const double needed = std::log(1.0 - options.confidence) / std::log(1.0 - p_good);
  return static_cast<double>(drawn) >= needed;
}

void RunRounds(cudaStream_t stream, const RansacMinimizerOptions &options,
               const SolverSettings &settings, RansacContext &c, RansacSummary &summary) {
  LaunchInitStats(stream, c.stats.data());
  size_t drawn = 0;
  for (size_t round = 0; round < options.max_rounds; ++round) {
    GenerateHypotheses(stream, round, options, settings, c);
    SelectHypotheses(stream, round, c);
    ReadStats(stream, c);
    drawn += options.hypotheses_per_round;
    summary.num_rounds = round + 1;
    if (EnoughHypotheses(drawn, options, c)) {
      break;
    }
  }
  summary.num_hypotheses = drawn;
}

void RevertRefinementIfWorse(cudaStream_t stream, RansacContext &c, RansacSummary &summary) {
  const float refined = CopyScalarToHost(stream, c.refinement.score());
  ReadStats(stream, c);
  const float best = c.host_stats.data()->best_score;
  if (!std::isfinite(best) || refined <= best + 1e-6f * std::fabs(best) + 1e-12f) {
    return;
  }
  LogMessage("RANSAC: refinement worsened the score ({} > {}); keeping the best hypothesis",
             refined, best);
  c.refinement.CopyFrom(stream, c.layout, c.best);
  c.refinement.Classify(stream, c.layout);
  c.refinement.EvaluateCost(stream, c.layout);
  summary.refinement_reverted = true;
}

void ReadRefinement(cudaStream_t stream, const RansacMinimizerOptions &options, RansacContext &c,
                    RansacSummary &summary) {
  const int iterations = CopyScalarToHost(stream, c.refinement.iterations());
  const int inliers = CopyScalarToHost(stream, c.refinement.inliers());
  summary.final_cost = CopyScalarToHost(stream, c.refinement.cost());
  summary.best_score = CopyScalarToHost(stream, c.refinement.score());
  summary.num_inliers = static_cast<size_t>(inliers);
  summary.inlier_ratio = static_cast<float>(inliers) / c.layout.total_sampled();
  summary.num_iterations = static_cast<size_t>(iterations);
  summary.iteration_costs.resize(std::min<size_t>(iterations, options.final_iterations));
  if (!summary.iteration_costs.empty()) {
    c.cost_history.CopyToHost(summary.iteration_costs.data(), summary.iteration_costs.size());
  }
  ReadStats(stream, c);
  summary.num_valid_hypotheses = static_cast<size_t>(c.host_stats.data()->valid_total);
}

void Refine(cudaStream_t stream, const RansacMinimizerOptions &options,
            const SolverSettings &settings, RansacContext &c, RansacSummary &summary) {
  c.refinement.CopyFrom(stream, c.layout, c.best);
  c.refinement.Classify(stream, c.layout);
  c.refinement.ResetSolver(stream, settings.initial_lambda);
  const bool check = WorthCheckingConvergence(c);
  for (size_t it = 0; it < options.final_iterations; ++it) {
    c.refinement.Iterate(stream, c.layout, settings);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(c.cost_history.data() + it, c.refinement.cost(),
                                        sizeof(float), cudaMemcpyDeviceToDevice, stream));
    if (check && !c.refinement.AnyActive(stream)) {
      break;
    }
  }
  c.refinement.Classify(stream, c.layout);
  c.refinement.EvaluateCost(stream, c.layout);
  RevertRefinementIfWorse(stream, c, summary);
  c.refinement.WriteBack(stream, c.layout, 0);
  ReadRefinement(stream, options, c, summary);
}

}  // namespace

RansacSummary RunRansac(cudaStream_t stream, Problem &problem,
                        const RansacMinimizerOptions &options, const SolverSettings &settings,
                        RansacContext &c) {
  if (c.stats.size() < 1) c.stats.resize(1);
  Prepare(stream, problem, options, c);
  RansacSummary summary;
  summary.initial_cost = InitialCost(stream, c);
  RunRounds(stream, options, settings, c, summary);
  Refine(stream, options, settings, c, summary);
  c.has_run = true;
  return summary;
}

const uint8_t *InlierMask(const RansacContext &context, size_t residual_batch_index) {
  const auto &residuals = context.layout.residuals();
  if (!context.has_run || residual_batch_index >= residuals.size() ||
      !residuals[residual_batch_index].sampled) {
    return nullptr;
  }
  return context.refinement.mask() + residuals[residual_batch_index].u_offset;
}

size_t InlierMaskSize(const RansacContext &context, size_t residual_batch_index) {
  // context.layout was built by the last Prepare(), so this is the batch size the mask
  // was computed for, whatever the problem looks like now.
  return InlierMask(context, residual_batch_index) == nullptr
             ? 0
             : static_cast<size_t>(context.layout.residuals()[residual_batch_index].num_factors);
}

}  // namespace ransac_internal
}  // namespace cunls
