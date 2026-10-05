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

#include "cunls/minimizer/ransac/hypothesis_scorer.h"

#include <algorithm>
#include <cmath>

#include "cunls/minimizer/ransac_minimizer.h"

namespace cunls {
namespace ransac_internal {

namespace {

/** Device bytes one hypothesis needs to be scored on all factors. */
size_t BytesPerHypothesis(const RansacLayout &layout, bool jacobians) {
  size_t bytes = 0;
  for (const ResidualLayout &r : layout.residuals()) {
    if (r.sampled) {
      const size_t rows = static_cast<size_t>(r.num_factors) * r.m;
      bytes += rows * sizeof(float) +
               static_cast<size_t>(r.num_factors) * (r.nb * sizeof(float *) + sizeof(int)) +
               (jacobians ? rows * r.n * sizeof(float) : 0);
    }
  }
  return std::max<size_t>(bytes, 1);
}

}  // namespace

void HypothesisScorer::Allocate(const RansacLayout &layout, const RansacMinimizerOptions &options,
                                int num_hypotheses) {
  num_hypotheses_ = num_hypotheses;
  jacobians_ = options.require_informative_inliers;
  seed_ = options.seed;
  chunk_ = static_cast<int>(std::max<size_t>(
      1, std::min<size_t>(num_hypotheses, options.scoring_memory_budget_bytes /
                                              BytesPerHypothesis(layout, jacobians_))));
  const size_t n = static_cast<size_t>(layout.total_sampled());
  const size_t subset = options.scoring_subset_size;
  finalists_count_ = static_cast<int>(std::min<size_t>(
      {options.scoring_finalists, size_t{64}, static_cast<size_t>(num_hypotheses)}));
  finalists_count_ = std::max(finalists_count_, 1);
  two_stage_ = subset > 0 && n > 2 * subset && finalists_count_ < num_hypotheses;

  batches_.resize(layout.residuals().size());
  subset_total_ = 0;
  subset_chunk_ = num_hypotheses;
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Batch &buf = batches_[b];
    buf.subset_size = 0;
    if (!r.sampled || !two_stage_ || r.num_factors == 0) {
      continue;
    }
    const size_t share = (subset * r.num_factors + n - 1) / n;
    buf.subset_size = static_cast<int>(std::min<size_t>(r.num_factors, std::max<size_t>(share, 1)));
    subset_total_ += buf.subset_size;
    const size_t capacity = static_cast<size_t>(chunk_) * r.num_factors;
    subset_chunk_ = std::min(subset_chunk_, static_cast<int>(capacity / buf.subset_size));
  }
  subset_chunk_ = std::max(1, subset_chunk_);
  AllocateBatches(layout);
  AllocateViews(layout);
  scratch_.resize(std::max(ScoreScratchFloats(layout.total_sampled(), chunk_),
                           ScoreScratchFloats(std::max(subset_total_, 1), subset_chunk_)));
  subset_score_.resize(two_stage_ ? num_hypotheses : 0);
  subset_inliers_.resize(two_stage_ ? num_hypotheses : 0);
  finalists_.resize(finalists_count_);
  selection_stats_.resize(1);
}

void HypothesisScorer::AllocateBatches(const RansacLayout &layout) {
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Batch &buf = batches_[b];
    if (!r.sampled) {
      continue;
    }
    const size_t items = static_cast<size_t>(chunk_) * r.num_factors;
    buf.res.resize(items * r.m);
    buf.jac.resize(jacobians_ ? items * r.m * r.n : 0);
    buf.table.resize(items * r.nb);
    buf.item_ids.resize(two_stage_ ? items : 0);
    buf.subset.resize(buf.subset_size);
    buf.subset_local.resize(static_cast<size_t>(buf.subset_size) * r.nb);
  }
}

BatchView HypothesisScorer::MakeView(const ResidualLayout &r, const Batch &buf, bool subset) const {
  const int count = subset ? buf.subset_size : r.num_factors;
  BatchView v;
  v.m = r.m;
  v.n = r.n;
  v.nb = r.nb;
  v.num_factors = count;
  v.u_offset = r.u_offset;
  std::copy(r.block_off.begin(), r.block_off.end(), v.block_col_off);
  std::copy(r.block_size.begin(), r.block_size.end(), v.block_size);
  v.local_col = subset ? buf.subset_local.data() : r.local_col.data();
  v.res = buf.res.data();
  v.jac = jacobians_ ? buf.jac.data() : nullptr;
  v.stride_res = static_cast<size_t>(count) * r.m;
  v.stride_jac = v.stride_res * r.n;
  v.tau_sq = r.tau * r.tau;
  return v;
}

void HypothesisScorer::AllocateViews(const RansacLayout &layout) {
  std::vector<BatchView> full, subset;
  int subset_offset = 0;
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    if (!r.sampled) {
      continue;
    }
    full.push_back(MakeView(r, batches_[b], false));
    if (batches_[b].subset_size > 0) {
      subset.push_back(MakeView(r, batches_[b], true));
      subset.back().u_offset = subset_offset;
      subset_offset += batches_[b].subset_size;
    }
  }
  views_.resize(full.size());
  views_.CopyFromHost(full.data(), full.size());
  subset_views_.resize(subset.size());
  subset_views_.CopyFromHost(subset.data(), subset.size());
}

ScoreInputs HypothesisScorer::Inputs(const SlotSet &hypotheses, bool subset) const {
  ScoreInputs in = hypotheses.score_inputs();  // always-on views and rules
  in.sampled = subset ? subset_views_.data() : views_.data();
  in.num_sampled = static_cast<int>(subset ? subset_views_.size() : views_.size());
  in.total_sampled = subset ? subset_total_ : in.total_sampled;
  return in;
}

void HypothesisScorer::Score(cudaStream_t stream, const RansacLayout &layout, SlotSet &hypotheses,
                             uint64_t round) {
  hypotheses.EvaluateAlwaysOnCost(stream, layout);
  if (two_stage_) {
    ScoreSubset(stream, layout, hypotheses, round);
    ScoreFinalists(stream, layout, hypotheses);
  } else {
    ScoreAll(stream, layout, hypotheses);
  }
}

void HypothesisScorer::ScoreAll(cudaStream_t stream, const RansacLayout &layout,
                                SlotSet &hypotheses) {
  const ScoreInputs inputs = Inputs(hypotheses, false);
  for (int first = 0; first < num_hypotheses_; first += chunk_) {
    const int count = std::min(chunk_, num_hypotheses_ - first);
    EvaluateChunk(stream, layout, hypotheses, first, count, nullptr);
    LaunchScore(stream, inputs, count, first, hypotheses.valid(), hypotheses.score(),
                hypotheses.inliers(), nullptr, scratch_.data());
  }
}

void HypothesisScorer::DrawSubsets(cudaStream_t stream, const RansacLayout &layout,
                                   uint64_t round) {
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Batch &buf = batches_[b];
    if (buf.subset_size == 0) {
      continue;
    }
    const uint64_t key = PermutationKey(seed_ ^ 0x5C0E5C0E5C0E5C0Eull, round, b);
    LaunchPermutationPrefix(stream, r.num_factors, buf.subset_size, key, buf.subset.data());
    LaunchGatherLocalColumns(stream, r.local_col.data(), buf.subset.data(), buf.subset_size, r.nb,
                             buf.subset_local.data());
  }
}

void HypothesisScorer::ScoreSubset(cudaStream_t stream, const RansacLayout &layout,
                                   SlotSet &hypotheses, uint64_t round) {
  DrawSubsets(stream, layout, round);
  const ScoreInputs inputs = Inputs(hypotheses, true);
  for (int first = 0; first < num_hypotheses_; first += subset_chunk_) {
    const int count = std::min(subset_chunk_, num_hypotheses_ - first);
    EvaluateSubsetChunk(stream, layout, hypotheses, first, count);
    LaunchScore(stream, inputs, count, first, hypotheses.valid(), subset_score_.data(),
                subset_inliers_.data(), nullptr, scratch_.data());
  }
  LaunchInitStats(stream, selection_stats_.data());
  LaunchSelect(stream, subset_score_.data(), subset_inliers_.data(), nullptr, num_hypotheses_,
               finalists_count_, finalists_.data(), selection_stats_.data());
}

void HypothesisScorer::ScoreFinalists(cudaStream_t stream, const RansacLayout &layout,
                                      SlotSet &hypotheses) {
  LaunchFill(stream, hypotheses.score(), num_hypotheses_, INFINITY);
  const ScoreInputs inputs = Inputs(hypotheses, false);
  for (int first = 0; first < finalists_count_; first += chunk_) {
    const int count = std::min(chunk_, finalists_count_ - first);
    EvaluateChunk(stream, layout, hypotheses, first, count, finalists_.data() + first);
    LaunchScore(stream, inputs, count, 0, hypotheses.valid(), hypotheses.score(),
                hypotheses.inliers(), nullptr, scratch_.data(), finalists_.data() + first);
  }
}

void HypothesisScorer::EvaluateChunk(cudaStream_t stream, const RansacLayout &layout,
                                     const SlotSet &hypotheses, int first, int count,
                                     const int *slot_index) {
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    if (!r.sampled) {
      continue;
    }
    Batch &buf = batches_[b];
    const size_t nf = static_cast<size_t>(r.num_factors);
    if (slot_index != nullptr) {
      LaunchIndexedSlotTables(stream, hypotheses.state_views(), r.blocks.data(), r.num_factors,
                              r.nb, slot_index, count, buf.table.data());
    } else {
      LaunchSlotTables(stream, hypotheses.state_views(), r.blocks.data(), r.num_factors, r.nb,
                       count, first, false, buf.table.data());
    }
    // Hypothesis q's rows are items q * N .. q * N + N - 1 (factor t % N).
    CheckEvaluate(r.factor->Evaluate(buf.res.data(), jacobians_ ? buf.jac.data() : nullptr,
                                     buf.table.data(), stream, nullptr, count * nf),
                  b);
  }
}

void HypothesisScorer::EvaluateSubsetChunk(cudaStream_t stream, const RansacLayout &layout,
                                           const SlotSet &hypotheses, int first, int count) {
  for (size_t b = 0; b < batches_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Batch &buf = batches_[b];
    if (buf.subset_size == 0) {
      continue;
    }
    LaunchSubsetTables(stream, hypotheses.state_views(), r.blocks.data(), r.nb, buf.subset.data(),
                       buf.subset_size, count, first, buf.table.data(), buf.item_ids.data());
    const size_t items = static_cast<size_t>(count) * buf.subset_size;
    CheckEvaluate(r.factor->Evaluate(buf.res.data(), jacobians_ ? buf.jac.data() : nullptr,
                                     buf.table.data(), stream, buf.item_ids.data(), items),
                  b);
  }
}

}  // namespace ransac_internal
}  // namespace cunls
