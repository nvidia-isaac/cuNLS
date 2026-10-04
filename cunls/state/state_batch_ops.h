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
#include <cuda_runtime.h>

#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/state/state_batch.h"

namespace cunls {

/**
 * @brief Computes each state's starting column in the reduced system.
 *
 * The reduced system omits constant states, so a state's column offset is
 * the running sum of the tangent sizes of the non-constant states before it.
 * Constant states are marked with -1 so callers can distinguish "column 0" from
 * "no column at all".
 *
 * @param stream CUDA stream for GPU operations.
 * @param first_column Column index at which this batch's states start, i.e. the
 *        total tangent size of all preceding batches' non-constant states.
 * @param state_batch The state batch.
 * @param[out] column_offsets One entry per state; resized as needed.
 */
void ComputeStateColumnOffsets(cudaStream_t stream, int first_column, const StateBatch *state_batch,
                               DeviceVector<int> &column_offsets);

/**
 * @brief Orchestrates manifold Plus operations across multiple state batches.
 *
 * StateBatchOps manages the mapping between a reduced (optimizable)
 * state vector and the full set of states, automatically excluding
 * any states marked as constant. It handles scattering the reduced delta vector
 * into per-batch update segments and dispatching the Plus operation on each
 * batch.
 */
class StateBatchOps {
 public:
  /**
   * @brief Constructs and preprocesses the operator for the given state
   * batches.
   *
   * @param stream        CUDA stream for asynchronous GPU operations during
   * preprocessing.
   * @param state_batches Vector of pointers to state batches to manage.
   */
  StateBatchOps(cudaStream_t stream, const std::vector<StateBatch *> &state_batches);

  /** @brief Default constructor. Call Preprocess() before use. */
  StateBatchOps() = default;

  /**
   * @brief Preprocesses the state batches, building internal mapping
   * structures.
   *
   * Initializes the full-to-reduced state mapping and the per-batch update
   * buffer. Must be called before Plus() if the default constructor was used.
   *
   * @param stream        CUDA stream for asynchronous GPU operations.
   * @param state_batches Vector of pointers to state batches to manage.
   */
  void Preprocess(cudaStream_t stream, const std::vector<StateBatch *> &state_batches);

  /**
   * @brief Applies manifold Plus operations across all state batches.
   *
   * Scatters the reduced delta vector into per-batch segments (zero-filling
   * entries that correspond to constant states), then dispatches the Plus
   * operation on each state batch and projects the non-constant states of the
   * bounded batches onto their bounds.
   *
   * @param stream           CUDA stream for asynchronous execution.
   * @param x_ptrs           Vector of device pointers to current state values,
   *                         one per state batch.
   * @param delta            Reduced delta vector on the device containing
   * updates only for non-constant states.
   * @param x_plus_delta_ptrs Vector of device pointers to output state values,
   *                         one per state batch.
   */
  void Plus(cudaStream_t stream, const std::vector<const float *> &x_ptrs,
            const DeviceVector<float> &delta, std::vector<float *> &x_plus_delta_ptrs);

  /** @brief Whether any of the state batches has bounds (StateBatch::HasBounds). */
  bool HasBounds() const;

  /**
   * @brief Projects the non-constant states of the bounded batches onto their
   * bounds, in place (one pointer per state batch; Plus does it already).
   */
  void ProjectToBounds(cudaStream_t stream, const std::vector<float *> &x_ptrs);

  /**
   * @brief Free-component mask of a reduced vector: mask[i] = 0 where the
   * component sits at a bound and the reduced `direction` points outward,
   * 1 elsewhere (StateBatch::MaskActiveBounds).
   *
   * @param x_ptrs    Current state values, one pointer per state batch.
   * @param direction Reduced vector (e.g. the negative gradient).
   * @param mask      Output, resized to NumReducedStates().
   */
  void BoundMask(cudaStream_t stream, const std::vector<const float *> &x_ptrs,
                 const DeviceVector<float> &direction, DeviceVector<float> &mask);

  /**
   * @brief Returns the number of reduced (non-constant) states.
   * @return Total number of optimizable scalar state components across all
   * batches.
   */
  size_t NumReducedStates() const { return num_reduced_states_; }

  // Protected for testing
 protected:
  /** @brief Device vector storing the mapping from reduced state indices to
   *         full (including constant) state indices. */
  DeviceVector<int> map_;

 private:
  /**
   * @brief Allocates the full-size state updates buffer and computes per-batch
   * delta pointers.
   * @param state_batches Vector of state batches.
   */
  void InitUpdatesVector(const std::vector<StateBatch *> &state_batches);

  /**
   * @brief Builds the reduced-to-full state index mapping on the GPU.
   *
   * Creates a binary pattern marking non-constant states, then sorts
   * the mapping so that constant state indices are pushed to the end.
   *
   * @param stream        CUDA stream for asynchronous GPU operations.
   * @param state_batches Vector of state batches.
   */
  void InitMapping(cudaStream_t stream, const std::vector<StateBatch *> &state_batches);

  /** @brief Cached pointers to the user-supplied state batches. */
  std::vector<StateBatch *> user_state_batches_;

  /** @brief Pointers into state_updates_ for each state batch's segment. */
  std::vector<float *> delta_ptrs_;

  /** @brief Device buffer storing the full tangent-space updates for all
   * states. */
  DeviceVector<float> state_updates_;

  /** @brief Full-size flags: 1 for the components of non-constant states. */
  DeviceVector<float> free_;

  /** @brief Full-size scratch of BoundMask. */
  DeviceVector<float> bound_direction_;
  DeviceVector<float> bound_mask_;

  /** @brief Number of scalar state components remaining after excluding
   * constant states. */
  size_t num_reduced_states_ = 0;
};
}  // namespace cunls
