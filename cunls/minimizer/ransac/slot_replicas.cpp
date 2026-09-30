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

#include "cunls/minimizer/ransac/slot_replicas.h"

#include "cunls/common/helper.h"

namespace cunls {
namespace ransac_internal {

void SlotReplicas::Allocate(const RansacLayout &layout, int num_slots) {
  num_slots_ = num_slots;
  const auto &states = layout.states();
  cur_.resize(states.size());
  cand_.resize(states.size());
  delta_.resize(states.size());
  std::vector<StateView> views(states.size());
  for (size_t j = 0; j < states.size(); ++j) {
    const StateLayout &s = states[j];
    StateView &v = views[j];
    v.base = s.batch->StateBlockDevicePtr(0);
    v.num_blocks = s.num_blocks;
    v.ambient = s.ambient;
    if (s.replicated) {
      cur_[j].resize(s.slot_floats * num_slots);
      cand_[j].resize(s.slot_floats * num_slots);
      delta_[j].resize(s.slot_tangent * num_slots);
      v.rep_cur = cur_[j].data();
      v.rep_cand = cand_[j].data();
    }
  }
  views_.resize(views.size());
  views_.CopyFromHost(views.data(), views.size());
}

void SlotReplicas::LoadInitialGuess(cudaStream_t stream, const RansacLayout &layout) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, num_slots_, s.slot_floats, s.batch->StateBlockDevicePtr(0), 0,
                         nullptr, nullptr, nullptr, cur_[j].data());
    }
  }
}

void SlotReplicas::Gather(cudaStream_t stream, const RansacLayout &layout,
                          const SlotReplicas &src, const int *src_index) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, num_slots_, s.slot_floats, src.cur_[j].data(), s.slot_floats,
                         src_index, nullptr, nullptr, cur_[j].data());
    }
  }
}

void SlotReplicas::CopySlotIf(cudaStream_t stream, const RansacLayout &layout,
                              const SlotReplicas &src, const int *src_slot, const int *only_if) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, 1, s.slot_floats, src.cur_[j].data(), s.slot_floats, nullptr,
                         src_slot, only_if, cur_[j].data());
    }
  }
}

void SlotReplicas::Broadcast(cudaStream_t stream, const RansacLayout &layout,
                             const SlotReplicas &src) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyReplicas(stream, num_slots_, s.slot_floats, src.cur_[j].data(), 0, nullptr,
                         nullptr, nullptr, cur_[j].data());
    }
  }
}

void SlotReplicas::ApplyStep(cudaStream_t stream, const RansacLayout &layout, const float *delta) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (!s.replicated) {
      continue;
    }
    LaunchScatterDelta(stream, num_slots_, layout.dim(), delta, s.num_blocks, s.tangent,
                       s.block_col.data(), delta_[j].data());
    // Slot copies are contiguous: one call (built-ins launch once; the
    // StateBatch default loops over Plus).
    s.batch->PlusReplicated(cur_[j].data(), delta_[j].data(), cand_[j].data(), num_slots_,
                            stream);
  }
}

void SlotReplicas::AcceptCandidates(cudaStream_t stream, const RansacLayout &layout,
                                    const int *accept) {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      LaunchCopyAccepted(stream, num_slots_, s.slot_floats, accept, cand_[j].data(),
                         cur_[j].data());
    }
  }
}

void SlotReplicas::WriteBack(cudaStream_t stream, const RansacLayout &layout, int slot) const {
  for (size_t j = 0; j < layout.states().size(); ++j) {
    const StateLayout &s = layout.states()[j];
    if (s.replicated) {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(s.batch->StateBlockDevicePtr(0),
                                          cur_[j].data() + slot * s.slot_floats,
                                          s.slot_floats * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream));
    }
  }
}

}  // namespace ransac_internal
}  // namespace cunls
