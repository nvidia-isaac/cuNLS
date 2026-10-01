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

#include <algorithm>

#include "cunls/common/helper.h"
#include "cunls/minimizer/residual_batch.h"

namespace cunls {
namespace ransac_internal {

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

void SlotSet::Allocate(cudaStream_t stream, const RansacLayout &layout,
                       const RansacMinimizerOptions &options, SlotRows rows, int num_slots) {
  rows_ = rows;
  num_slots_ = num_slots;
  dim_ = layout.dim();
  sample_size_ = layout.sample_size();
  total_sampled_ = layout.total_sampled();
  m_max_ = layout.m_max();
  AllocateReplicas(layout);
  if (rows_ == SlotRows::kNone) {
    return;
  }
  AllocateBuffers(stream, layout);
  PublishViews(layout, options);
  AllocateSolver(layout);
}

void SlotSet::AllocateReplicas(const RansacLayout &layout) {
  const auto &states = layout.states();
  replicas_.resize(states.size());
  std::vector<StateView> views(states.size());
  for (size_t j = 0; j < states.size(); ++j) {
    const StateLayout &s = states[j];
    StateView &v = views[j];
    v.base = s.batch->StateBlockDevicePtr(0);
    v.num_blocks = s.num_blocks;
    v.ambient = s.ambient;
    if (s.replicated) {
      Replicas &rep = replicas_[j];
      rep.cur.resize(s.slot_floats * num_slots_);
      rep.cand.resize(s.slot_floats * num_slots_);
      rep.delta.resize(s.slot_tangent * num_slots_);
      v.rep_cur = rep.cur.data();
      v.rep_cand = rep.cand.data();
    }
  }
  state_views_.resize(views.size());
  state_views_.CopyFromHost(views.data(), views.size());
}

void SlotSet::AllocateBuffers(cudaStream_t stream, const RansacLayout &layout) {
  buffers_.resize(layout.residuals().size());
  size_t max_items = 1;
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Buffers &buf = buffers_[b];
    // Item p * k + t is row t of slot p, where k is the sample size (minimal
    // samples) or the batch size (every factor).
    const size_t per_slot = SampleRows(r) ? sample_size_ : r.num_factors;
    buf.items = r.num_factors == 0 ? 0 : per_slot * num_slots_;
    max_items = std::max(max_items, buf.items);
    buf.res.resize(buf.items * r.m);
    buf.jac.resize(buf.items * r.m * r.n);
    buf.cost.resize(buf.items);
    buf.table_cur.resize(buf.items * r.nb);
    buf.table_cand.resize(buf.items * r.nb);
    buf.factor_ids.resize(SampleRows(r) ? buf.items : 0);
    if (buf.items > 0 && !SampleRows(r)) {  // fixed: depends only on replica addresses
      LaunchSlotTables(stream, state_views(), r.blocks.data(), r.num_factors, r.nb, num_slots_, 0,
                       false, buf.table_cur.data());
      LaunchSlotTables(stream, state_views(), r.blocks.data(), r.num_factors, r.nb, num_slots_, 0,
                       true, buf.table_cand.data());
    }
  }
  workspace_.resize(ResidualBatchWorkspaceNumFloats(max_items));
  samples_.resize(
      rows_ == SlotRows::kMinimalSamples ? static_cast<size_t>(num_slots_) * sample_size_ : 0);
  mask_.resize(rows_ == SlotRows::kAllMasked ? static_cast<size_t>(num_slots_) * total_sampled_
                                             : 0);
}

BatchView SlotSet::MakeView(const ResidualLayout &r, size_t b, int kind) const {
  const Buffers &buf = buffers_[b];
  const size_t rows = kind == kViewSamples ? sample_size_ : r.num_factors;
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
  v.stride_res = rows * r.m;
  v.stride_jac = v.stride_res * r.n;
  v.stride_cost = rows;
  v.tau_sq = r.tau * r.tau;
  return v;
}

void SlotSet::PublishViews(const RansacLayout &layout, const RansacMinimizerOptions &options) {
  std::vector<BatchView> views, sampled, always_on;
  per_slot_items_ = 0;
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    int kind = kViewPerSlot;
    if (SampleRows(r)) {
      kind = kViewSamples;
    } else if (r.sampled) {
      kind = kViewPerSlotMasked;
      sampled.push_back(MakeView(r, b, kViewPerSlot));
    } else {
      always_on.push_back(MakeView(r, b, kViewPerSlot));
    }
    views.push_back(MakeView(r, b, kind));
    per_slot_items_ += kind == kViewSamples ? 0 : r.num_factors;
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
  score_inputs_.total_sampled = total_sampled_;
  score_inputs_.rule =
      options.scoring == RansacScoring::kInlierCount ? kScoreInlierCount : kScoreMSAC;
  score_inputs_.add_always_on = options.score_always_on ? 1 : 0;
  score_inputs_.require_informative = options.require_informative_inliers ? 1 : 0;
}

void SlotSet::AllocateSolver(const RansacLayout &layout) {
  const size_t slots = static_cast<size_t>(num_slots_);
  const size_t d = static_cast<size_t>(dim_);
  hessian_.resize(slots * d * d);
  gradient_.resize(slots * d);
  delta_.resize(slots * d);
  for (auto *v : {&cost_cur_, &cost_cand_, &predicted_, &step_sq_, &lambda_, &score_}) {
    v->resize(slots);
  }
  for (auto *v :
       {&solve_ok_, &active_, &accept_, &valid_, &iterations_, &num_accepted_, &inliers_}) {
    v->resize(slots);
  }
  active_count_.resize(1);
  normal_scratch_.resize(NormalEquationsScratchFloats(Items(), num_slots_, dim_));
  cost_scratch_.resize(SlotCostScratchFloats(Items(), num_slots_));
  score_scratch_.resize(ScoreScratchFloats(layout.total_sampled(), num_slots_));
}

SlotItems SlotSet::Items() const {
  const bool samples = rows_ == SlotRows::kMinimalSamples;
  SlotItems items;
  items.views = views_.data();
  items.num_views = num_views_;
  items.samples = samples ? samples_.data() : nullptr;
  items.sample_size = samples ? sample_size_ : 0;
  items.mask = samples ? nullptr : mask_.data();
  items.mask_stride = total_sampled_;
  items.items_per_slot = items.sample_size + per_slot_items_;
  items.m_max = m_max_;
  return items;
}

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

void SlotSet::LoadInitialGuess(cudaStream_t stream, const RansacLayout &layout) {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, num_slots_, s.slot_floats, s.batch->StateBlockDevicePtr(0),
                         nullptr, nullptr, replicas_[j].cur.data());
    }
  }
}

void SlotSet::CopyFrom(cudaStream_t stream, const RansacLayout &layout, const SlotSet &src) {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, num_slots_, s.slot_floats, src.replicas_[j].cur.data(), nullptr,
                         nullptr, replicas_[j].cur.data());
    }
  }
}

void SlotSet::CopySlotIf(cudaStream_t stream, const RansacLayout &layout, const SlotSet &src,
                         const int *src_slot, const int *only_if) {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, 1, s.slot_floats, src.replicas_[j].cur.data(), src_slot, only_if,
                         replicas_[j].cur.data());
    }
  }
}

void SlotSet::WriteBack(cudaStream_t stream, const RansacLayout &layout, int slot) const {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
          s.batch->StateBlockDevicePtr(0), replicas_[j].cur.data() + slot * s.slot_floats,
          s.slot_floats * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
  }
}

void SlotSet::ApplyStep(cudaStream_t stream, const RansacLayout &layout) {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (!s.replicated) {
      continue;
    }
    Replicas &rep = replicas_[j];
    LaunchScatterDelta(stream, num_slots_, dim_, delta_.data(), s.num_blocks, s.tangent,
                       s.block_col.data(), rep.delta.data());
    // The slots' replicas are contiguous copies of the batch.
    s.batch->Plus(rep.cur.data(), rep.delta.data(), rep.cand.data(), stream, num_slots_);
  }
}

void SlotSet::AcceptCandidates(cudaStream_t stream, const RansacLayout &layout) {
  for (size_t j = 0; j < replicas_.size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyAccepted(stream, num_slots_, s.slot_floats, accept_.data(),
                         replicas_[j].cand.data(), replicas_[j].cur.data());
    }
  }
}

// ---------------------------------------------------------------------------
// Evaluation and solving
// ---------------------------------------------------------------------------

void SlotSet::DrawSamples(cudaStream_t stream, const RansacLayout &layout, uint64_t seed,
                          uint64_t round) {
  LaunchDrawSamples(stream, total_sampled_, sample_size_, num_slots_, seed, round, samples_.data());
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Buffers &buf = buffers_[b];
    if (!SampleRows(r) || buf.items == 0) {
      continue;
    }
    for (bool candidate : {false, true}) {
      LaunchSampleTables(stream, state_views(), r.blocks.data(), r.nb, r.num_factors, r.u_offset,
                         samples_.data(), sample_size_, num_slots_, candidate,
                         candidate ? buf.table_cand.data() : buf.table_cur.data(),
                         buf.factor_ids.data());
    }
  }
}

void SlotSet::Evaluate(cudaStream_t stream, const RansacLayout &layout, Target target,
                       Batches which, bool with_loss, bool jacobians) {
  for (size_t b = 0; b < buffers_.size(); ++b) {
    const ResidualLayout &r = layout.residuals()[b];
    Buffers &buf = buffers_[b];
    if (buf.items == 0 || (which == Batches::kSampled && !r.sampled) ||
        (which == Batches::kAlwaysOn && r.sampled)) {
      continue;
    }
    float *const *table = target == Target::kCurrent ? buf.table_cur.data() : buf.table_cand.data();
    float *jac = jacobians ? buf.jac.data() : nullptr;
    // Without factor ids, item t evaluates factor t % N: exactly the per-slot layout.
    const int *ids = buf.factor_ids.empty() ? nullptr : buf.factor_ids.data();
    if (with_loss) {
      r.residual_batch->Evaluate(stream, workspace_.data(), buf.res.data(), table, buf.cost.data(),
                                 jac, ids, buf.items);
    } else {
      r.factor->Evaluate(buf.res.data(), jac, table, stream, ids, buf.items);
    }
  }
}

void SlotSet::ResetSolver(cudaStream_t stream, float initial_lambda) {
  LaunchResetSlots(stream, num_slots_, initial_lambda, active_.data(), lambda_.data(),
                   valid_.data(), iterations_.data(), num_accepted_.data());
}

void SlotSet::Iterate(cudaStream_t stream, const RansacLayout &layout,
                      const SolverSettings &settings) {
  const SlotItems items = Items();
  const bool lm = settings.policy.levenberg_marquardt != 0;
  // A hypothesis whose first solve fails is invalid (degenerate sample).
  int *valid = rows_ == SlotRows::kMinimalSamples ? valid_.data() : nullptr;
  Evaluate(stream, layout, Target::kCurrent, Batches::kAll, true, true);
  LaunchNormalEquations(stream, items, num_slots_, dim_, hessian_.data(), gradient_.data(),
                        cost_cur_.data(), normal_scratch_.data());
  LaunchSolve(stream, num_slots_, dim_, settings.solver, hessian_.data(), gradient_.data(),
              lm ? lambda_.data() : nullptr, active_.data(), delta_.data(), predicted_.data(),
              step_sq_.data(), solve_ok_.data());
  ApplyStep(stream, layout);
  Evaluate(stream, layout, Target::kCandidate, Batches::kAll, true, false);
  LaunchSlotCost(stream, items, num_slots_, cost_cand_.data(), cost_scratch_.data());
  LaunchAccept(stream, num_slots_, settings.policy, cost_cur_.data(), cost_cand_.data(),
               predicted_.data(), step_sq_.data(), solve_ok_.data(), active_.data(), lambda_.data(),
               accept_.data(), valid, iterations_.data(), num_accepted_.data());
  AcceptCandidates(stream, layout);
}

bool SlotSet::AnyActive(cudaStream_t stream) {
  LaunchCountActive(stream, num_slots_, active_.data(), active_count_.data());
  int count = 0;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&count, active_count_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return count > 0;
}

void SlotSet::Classify(cudaStream_t stream, const RansacLayout &layout) {
  // Raw (loss-free) residuals decide inliers; Jacobians only for the
  // informative-inlier check.
  Evaluate(stream, layout, Target::kCurrent, Batches::kSampled, false,
           score_inputs_.require_informative != 0);
  EvaluateAlwaysOnCost(stream, layout);
  LaunchScore(stream, score_inputs_, num_slots_, 0, nullptr, score_.data(), inliers_.data(),
              mask_.data(), score_scratch_.data());
}

void SlotSet::EvaluateCost(cudaStream_t stream, const RansacLayout &layout) {
  Evaluate(stream, layout, Target::kCurrent, Batches::kAll, true, false);
  LaunchSlotCost(stream, Items(), num_slots_, cost_cur_.data(), cost_scratch_.data());
}

void SlotSet::EvaluateAlwaysOnCost(cudaStream_t stream, const RansacLayout &layout) {
  Evaluate(stream, layout, Target::kCurrent, Batches::kAlwaysOn, true, false);
}

}  // namespace ransac_internal
}  // namespace cunls
