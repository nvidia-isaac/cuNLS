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
 * @file slot_replicas.h
 * @brief Per-slot copies of the replicated state batches.
 */

#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/types.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"
#include "cunls/minimizer/ransac/ransac_layout.h"

namespace cunls {
namespace ransac_internal {

/**
 * @brief Current and candidate copies of every replicated state batch, one per
 * slot. A slot's current state is what it is evaluated at; a step writes the
 * candidate, and accepted candidates are copied back to current.
 *
 * Fully constant state batches are never copied: every slot reads them from
 * user storage.
 */
class SlotReplicas {
 public:
  /** @brief Sizes the buffers (keeping capacity) and publishes the state views. */
  void Allocate(const RansacLayout &layout, int num_slots);

  int num_slots() const { return num_slots_; }
  const StateView *views() const { return views_.data(); }
  const float *current(size_t batch) const { return cur_[batch].data(); }

  /** @brief Every slot's current state = the problem's state values. */
  void LoadInitialGuess(cudaStream_t stream, const RansacLayout &layout);

  /** @brief Slot p's current state = src's current state of slot src_index[p]. */
  void Gather(cudaStream_t stream, const RansacLayout &layout, const SlotReplicas &src,
              const int *src_index);

  /**
   * @brief Slot 0's current state = src's slot *src_slot, if *only_if != 0.
   * Both pointers are device pointers, so no host synchronization is needed.
   */
  void CopySlotIf(cudaStream_t stream, const RansacLayout &layout, const SlotReplicas &src,
                  const int *src_slot, const int *only_if);

  /** @brief Every slot's current state = src's slot 0. */
  void Broadcast(cudaStream_t stream, const RansacLayout &layout, const SlotReplicas &src);

  /** @brief candidate = current (+) delta, through each batch's own Plus. */
  void ApplyStep(cudaStream_t stream, const RansacLayout &layout, const float *delta);

  /** @brief current = candidate for the slots with accept[p] != 0. */
  void AcceptCandidates(cudaStream_t stream, const RansacLayout &layout, const int *accept);

  /** @brief Copies `slot`'s current state into the problem's state batches. */
  void WriteBack(cudaStream_t stream, const RansacLayout &layout, int slot) const;

 private:
  int num_slots_ = 0;
  std::vector<dvector<float>> cur_;
  std::vector<dvector<float>> cand_;
  std::vector<dvector<float>> delta_;
  dvector<StateView> views_;
};

}  // namespace ransac_internal
}  // namespace cunls
