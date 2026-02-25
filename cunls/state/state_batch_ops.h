/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @brief Orchestrates manifold Plus operations across multiple state batches.
 *
 * StateBatchOps manages the mapping between a reduced (optimizable)
 * state vector and the full set of state blocks, automatically excluding
 * any blocks marked as constant. It handles scattering the reduced delta vector
 * into per-batch update segments and dispatching the Plus operation on each batch.
 */
class StateBatchOps {
 public:
  /**
   * @brief Constructs and preprocesses the operator for the given state batches.
   *
   * @param stream        CUDA stream for asynchronous GPU operations during preprocessing.
   * @param state_batches Vector of pointers to state batches to manage.
   */
  StateBatchOps(cudaStream_t stream,
                const std::vector<StateBatch*>& state_batches);

  /** @brief Default constructor. Call Preprocess() before use. */
  StateBatchOps() = default;

  /**
   * @brief Preprocesses the state batches, building internal mapping structures.
   *
   * Initializes the full-to-reduced state mapping and the per-batch update
   * buffer. Must be called before Plus() if the default constructor was used.
   *
   * @param stream        CUDA stream for asynchronous GPU operations.
   * @param state_batches Vector of pointers to state batches to manage.
   */
  void Preprocess(cudaStream_t stream,
                  const std::vector<StateBatch*>& state_batches);

  /**
   * @brief Applies manifold Plus operations across all state batches.
   *
   * Scatters the reduced delta vector into per-batch segments (zero-filling
   * entries that correspond to constant states), then dispatches the Plus
   * operation on each state batch.
   *
   * @param stream           CUDA stream for asynchronous execution.
   * @param x_ptrs           Vector of device pointers to current state values,
   *                         one per state batch.
   * @param delta            Reduced delta vector on the device containing updates
   *                         only for non-constant states.
   * @param x_plus_delta_ptrs Vector of device pointers to output state values,
   *                         one per state batch.
   */
  void Plus(cudaStream_t stream, const std::vector<const float*>& x_ptrs,
            const DeviceVector<float>& delta,
            std::vector<float*>& x_plus_delta_ptrs);

  /**
   * @brief Returns the number of reduced (non-constant) states.
   * @return Total number of optimizable scalar state components across all batches.
   */
  size_t NumReducedStates() const { return num_reduced_states_; }

  // Protected for testing
 protected:
  /** @brief Device vector storing the mapping from reduced state indices to
   *         full (including constant) state indices. */
  DeviceVector<int> map_;

 private:
  /**
   * @brief Allocates the full-size state updates buffer and computes per-batch delta pointers.
   * @param state_batches Vector of state batches.
   */
  void InitUpdatesVector(const std::vector<StateBatch*>& state_batches);

  /**
   * @brief Builds the reduced-to-full state index mapping on the GPU.
   *
   * Creates a binary pattern marking non-constant states, then sorts
   * the mapping so that constant state indices are pushed to the end.
   *
   * @param stream        CUDA stream for asynchronous GPU operations.
   * @param state_batches Vector of state batches.
   */
  void InitMapping(cudaStream_t stream,
                   const std::vector<StateBatch*>& state_batches);

  /** @brief Cached pointers to the user-supplied state batches. */
  std::vector<StateBatch*> user_state_batches_;

  /** @brief Pointers into state_updates_ for each state batch's segment. */
  std::vector<float*> delta_ptrs_;

  /** @brief Device buffer storing the full tangent-space updates for all state blocks. */
  DeviceVector<float> state_updates_;

  /** @brief Number of scalar state components remaining after excluding constant blocks. */
  size_t num_reduced_states_ = 0;
};
}  // namespace cunls
