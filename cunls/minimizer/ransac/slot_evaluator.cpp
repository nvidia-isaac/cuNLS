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

#include "cunls/minimizer/ransac/slot_evaluator.h"

#include <algorithm>

#include "cunls/minimizer/residual_batch.h"

namespace cunls {
namespace ransac_internal {

void SlotEvaluator::Allocate(cudaStream_t stream, const RansacLayout &layout,
                             const SlotReplicas &replicas, const RansacMinimizerOptions &options,
                             SampledRows sampled_rows, int num_waves) {
  sampled_rows_ = sampled_rows;
  num_slots_ = replicas.num_slots();
  sample_size_ = layout.sample_size();
  total_sampled_ = layout.total_sampled();
  m_max_ = layout.m_max();
  buffers_.resize(layout.residuals().size());
  size_t max_items = 1;  // one indexed call evaluates every copy of a batch
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Buffers &buf = buffers_[b];
    const bool per_wave = r.sampled && sampled_rows == SampledRows::kWaves;
    buf.copies = per_wave ? num_waves : num_slots_;
    const size_t nf = static_cast<size_t>(r.num_factors);
    max_items = std::max(max_items, buf.copies * nf);
    buf.res.resize(buf.copies * nf * r.m);
    buf.jac.resize(buf.copies * nf * r.m * r.n);
    buf.cost.resize(buf.copies * nf);
    buf.table_cur.resize(buf.copies * nf * r.nb);
    buf.table_cand.resize(buf.copies * nf * r.nb);
    if (!per_wave) {  // per-slot tables depend only on replica addresses
      LaunchSlotTables(stream, replicas.views(), r.blocks.data(), r.num_factors, r.nb, num_slots_,
                       0, false, buf.table_cur.data());
      LaunchSlotTables(stream, replicas.views(), r.blocks.data(), r.num_factors, r.nb, num_slots_,
                       0, true, buf.table_cand.data());
    }
  }
  workspace_.resize(ResidualBatchWorkspaceNumFloats(max_items));
  PublishViews(layout);
  score_inputs_.total_sampled = total_sampled_;
  score_inputs_.rule =
      options.scoring == RansacScoring::kInlierCount ? kScoreInlierCount : kScoreMSAC;
  score_inputs_.add_always_on = options.score_always_on ? 1 : 0;
  score_inputs_.require_informative = options.require_informative_inliers ? 1 : 0;
}

BatchView SlotEvaluator::MakeView(const ResidualLayout &r, const Buffers &buf, int kind) const {
  BatchView v;
  v.kind = kind;
  v.m = r.m;
  v.n = r.n;
  v.nb = r.nb;
  v.num_factors = r.num_factors;
  v.u_offset = r.u_offset;
  std::copy(r.block_off.begin(), r.block_off.end(), v.block_col_off);
  std::copy(r.block_size.begin(), r.block_size.end(), v.block_size);
  v.local_col = r.local_col.data();
  v.res = buf.res.data();
  v.jac = buf.jac.data();
  v.cost = buf.cost.data();
  v.stride_res = static_cast<size_t>(r.num_factors) * r.m;
  v.stride_jac = v.stride_res * r.n;
  v.stride_cost = static_cast<size_t>(r.num_factors);
  v.tau_sq = r.tau * r.tau;
  return v;
}

void SlotEvaluator::PublishViews(const RansacLayout &layout) {
  std::vector<BatchView> views, sampled, always_on;
  per_slot_items_ = 0;
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    int kind = kViewPerSlot;
    if (r.sampled) {
      kind = sampled_rows_ == SampledRows::kWaves ? kViewWave : kViewPerSlotMasked;
      sampled.push_back(MakeView(r, buffers_[b], kViewPerSlot));
    } else {
      always_on.push_back(MakeView(r, buffers_[b], kViewPerSlot));
    }
    views.push_back(MakeView(r, buffers_[b], kind));
    per_slot_items_ += kind == kViewWave ? 0 : r.num_factors;
  }
  auto upload = [](dvector<BatchView> &d, const std::vector<BatchView> &h) {
    d.resize(h.size());
    d.CopyFromHost(h.data(), h.size());
  };
  upload(views_, views);
  upload(score_sampled_, sampled);
  upload(score_always_on_, always_on);
  num_views_ = static_cast<int>(views.size());
  score_inputs_.sampled = score_sampled_.data();
  score_inputs_.num_sampled = static_cast<int>(sampled.size());
  score_inputs_.always_on = score_always_on_.data();
  score_inputs_.num_always_on = static_cast<int>(always_on.size());
}

void SlotEvaluator::BuildWaveTables(cudaStream_t stream, const RansacLayout &layout,
                                    const SlotReplicas &replicas, const int *owner) {
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    if (!r.sampled) {
      continue;
    }
    Buffers &buf = buffers_[b];
    for (int w = 0; w < buf.copies; ++w) {
      const int *wave_owner = owner + static_cast<size_t>(w) * total_sampled_ + r.u_offset;
      const size_t offset = static_cast<size_t>(w) * r.num_factors * r.nb;
      LaunchWaveTable(stream, replicas.views(), r.blocks.data(), r.num_factors, r.nb, wave_owner,
                      false, buf.table_cur.data() + offset);
      LaunchWaveTable(stream, replicas.views(), r.blocks.data(), r.num_factors, r.nb, wave_owner,
                      true, buf.table_cand.data() + offset);
    }
  }
}

void SlotEvaluator::Run(cudaStream_t stream, const RansacLayout &layout, Target target,
                        Batches which, bool with_loss, bool jacobians) {
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    if ((which == Batches::kSampled && !r.sampled) || (which == Batches::kAlwaysOn && r.sampled)) {
      continue;
    }
    Buffers &buf = buffers_[b];
    float *const *table =
        target == Target::kCurrent ? buf.table_cur.data() : buf.table_cand.data();
    if (!EvaluateAllCopies(stream, r, buf, table, with_loss, jacobians)) {
      EvaluateEachCopy(stream, r, buf, table, with_loss, jacobians);
    }
  }
}

bool SlotEvaluator::EvaluateAllCopies(cudaStream_t stream, const ResidualLayout &r, Buffers &buf,
                                      float *const *table, bool with_loss, bool jacobians) {
  // Copies are contiguous and item t = copy * N + factor, so a single indexed
  // call with factor_ids == nullptr (factor = t % N) covers all of them.
  const size_t items = static_cast<size_t>(buf.copies) * r.num_factors;
  float *jac = jacobians ? buf.jac.data() : nullptr;
  if (with_loss) {
    return r.residual_batch->EvaluateIndexed(stream, workspace_.data(), buf.res.data(), table,
                                             nullptr, items, buf.cost.data(), jac);
  }
  return r.factor->EvaluateIndexed(buf.res.data(), jac, table, nullptr, items, stream);
}

void SlotEvaluator::EvaluateEachCopy(cudaStream_t stream, const ResidualLayout &r, Buffers &buf,
                                     float *const *table, bool with_loss, bool jacobians) {
  const size_t nf = static_cast<size_t>(r.num_factors);
  for (int c = 0; c < buf.copies; ++c) {
    float *res = buf.res.data() + c * nf * r.m;
    float *jac = jacobians ? buf.jac.data() + c * nf * r.m * r.n : nullptr;
    float const *const *ptrs = table + c * nf * r.nb;
    if (with_loss) {
      r.residual_batch->Evaluate(stream, workspace_.data(), res, ptrs, buf.cost.data() + c * nf,
                                 jac);
    } else {
      r.factor->Evaluate(res, jac, ptrs, stream);
    }
  }
}

void SlotEvaluator::EvaluateCurrent(cudaStream_t stream, const RansacLayout &layout,
                                    bool jacobians) {
  Run(stream, layout, Target::kCurrent, Batches::kAll, true, jacobians);
}

void SlotEvaluator::EvaluateCandidate(cudaStream_t stream, const RansacLayout &layout) {
  Run(stream, layout, Target::kCandidate, Batches::kAll, true, false);
}

void SlotEvaluator::EvaluateRawSampled(cudaStream_t stream, const RansacLayout &layout) {
  Run(stream, layout, Target::kCurrent, Batches::kSampled, false,
      score_inputs_.require_informative != 0);
}

void SlotEvaluator::EvaluateAlwaysOnCost(cudaStream_t stream, const RansacLayout &layout) {
  Run(stream, layout, Target::kCurrent, Batches::kAlwaysOn, true, false);
}

SlotItems SlotEvaluator::Items(const int *samples, int hyp_per_wave, const uint8_t *mask) const {
  SlotItems items;
  items.views = views_.data();
  items.num_views = num_views_;
  const bool waves = sampled_rows_ == SampledRows::kWaves;
  items.samples = waves ? samples : nullptr;
  items.sample_size = waves ? sample_size_ : 0;
  items.hyp_per_wave = waves ? hyp_per_wave : 1;
  items.mask = waves ? nullptr : mask;
  items.mask_stride = total_sampled_;
  items.items_per_slot = items.sample_size + per_slot_items_;
  items.m_max = m_max_;
  return items;
}

}  // namespace ransac_internal
}  // namespace cunls
