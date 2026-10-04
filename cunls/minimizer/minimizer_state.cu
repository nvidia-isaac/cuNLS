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

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/minimizer/minimizer_state.h"

namespace cunls {

/** @brief Thread block size for CUDA kernels. */
constexpr size_t block_size = 256;

/**
 * @brief CUDA kernel to remap state pointers.
 *
 * Updates state pointers to point into new state storage instead of
 * the original problem storage. For each pointer in old_pointers, computes its
 * offset from old_start_ptr and creates a corresponding pointer in new_pointers
 * at the same offset from new_start_ptr.
 *
 * @param[out] new_pointers Output array of remapped pointers.
 * @param old_pointers Input array of original pointers.
 * @param old_start_ptr Base pointer for original storage.
 * @param new_start_ptr Base pointer for new storage.
 * @param num_states_in_batch Number of state elements in the batch.
 * @param num_new_pointers Number of pointers to remap.
 */
__global__ void set_state_pointers_kernel(float **new_pointers, float *const *old_pointers,
                                          float *old_start_ptr, float *new_start_ptr,
                                          size_t num_states_in_batch, size_t num_new_pointers) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_new_pointers) {
    return;
  }

  float *old_ptr = old_pointers[tid];
  int offset = static_cast<int>(old_ptr - old_start_ptr);
  if (offset < 0 || offset >= num_states_in_batch) {
    return;
  }

  new_pointers[tid] = new_start_ptr + offset;
}

/**
 * @brief Allocates state storage vectors.
 *
 * Creates one device vector per state batch, sized to hold all
 * states in that batch flattened into a single vector.
 *
 * @param problem The problem containing state batch information.
 */
void MinimizerState::CreateStates(const Problem &problem) {
  const auto &state_batches = problem.GetStateBatches();
  // Resize to the exact batch count so stale trailing batches from a previous,
  // larger problem are dropped when the topology shrinks. Surviving batches keep
  // their device buffers; only whole unused batch buffers are freed.
  if (states_.size() != state_batches.size()) {
    states_.resize(state_batches.size());
  }

  for (size_t i = 0; i < state_batches.size(); i++) {
    const auto &param_batch_ptr = state_batches[i];
    auto &state_vec = states_[i];

    size_t size = param_batch_ptr->NumActiveStates() * param_batch_ptr->AmbientSize();

    if (state_vec.size() != size) {
      state_vec.resize(size);
    }
  }
}

/**
 * @brief Allocates state pointer vectors.
 *
 * Creates one device vector per residual batch for storing state pointers.
 * The vectors are sized to match the number of state pointers needed
 * by each residual batch.
 *
 * @param problem The problem containing residual batch information.
 */
void MinimizerState::CreateStatePointers(const Problem &problem) {
  const size_t num_batches = problem.GetResidualBatches().size();
  if (state_pointers_.size() != num_batches) {
    state_pointers_.resize(num_batches);
  }
  for (size_t i = 0; i < num_batches; i++) {
    state_pointers_[i].resize(problem.NumStatePointers(i));
  }
}

/**
 * @brief Creates minimizer state from a problem.
 *
 * This method:
 * 1. Allocates storage for states and state pointers
 * 2. Copies state values from problem to local storage
 * 3. Remaps state pointers to point into local storage
 *
 * @param stream CUDA stream for GPU operations.
 * @param problem The problem to create a state snapshot from.
 */
void MinimizerState::Create(cudaStream_t stream, const Problem &problem) {
  CreateStates(problem);
  CreateStatePointers(problem);

  const auto &state_batches = problem.GetStateBatches();
  {
    // Copy state values from problem to local storage (plain async copies:
    // capturable in a CUDA graph).
    for (size_t i = 0; i < state_batches.size(); i++) {
      const auto &param_batch_ptr = state_batches[i];
      auto &state_vec = states_[i];

      const float *ptr = param_batch_ptr->StateDevicePtr(0);
      size_t size = param_batch_ptr->NumActiveStates() * param_batch_ptr->AmbientSize();
      if (size == 0) continue;
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(state_vec.data(), ptr, size * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream));
    }
  }

  {
    // Remap state pointers to point into local storage
    const auto &residual_batches = problem.GetResidualBatches();

    for (size_t i = 0; i < residual_batches.size(); i++) {
      // The problem's device table (Problem::PrepareStatePointers has already
      // run on `stream` for this solve).
      float *const *old_pointers = problem.DeviceStatePointers(i);
      const size_t num_pointers = problem.NumStatePointers(i);
      auto &new_ptrs = state_pointers_[i];
      assert(num_pointers == new_ptrs.size());

      // A factor batch may legitimately hold zero factors; a zero-size grid is
      // an invalid launch configuration.
      if (num_pointers == 0) {
        continue;
      }

      float **new_pointers = new_ptrs.data();

      for (size_t j = 0; j < state_batches.size(); j++) {
        const auto &param_batch_ptr = state_batches[j];
        auto &new_states = states_[j];

        size_t num_states_in_batch =
            param_batch_ptr->NumActiveStates() * param_batch_ptr->AmbientSize();

        assert(num_states_in_batch == new_states.size());

        float *new_param_ptr = new_states.data();

        float *state_batch_ptr = param_batch_ptr->StateDevicePtr(0);

        size_t num_blocks = (num_pointers + block_size - 1) / block_size;

        set_state_pointers_kernel<<<num_blocks, block_size, 0, stream>>>(
            new_pointers, old_pointers, state_batch_ptr, new_param_ptr, num_states_in_batch,
            num_pointers);
        THROW_ON_CUDA_ERROR(cudaGetLastError());
      }
    }
  }
}

/**
 * @brief Copies state values from another state.
 *
 * Updates this state's values with values from the provided state
 * vectors. Resizes storage if necessary to match the source.
 *
 * @param stream CUDA stream for GPU operations.
 * @param from Source state vectors to copy from.
 */
void MinimizerState::Copy(cudaStream_t stream, const std::vector<dvector<float>> &from) {
  if (states_.size() != from.size()) {
    states_.resize(from.size());
  }

  for (size_t i = 0; i < from.size(); i++) {
    const auto &from_dvec = from[i];
    auto &to_dvec = states_[i];

    if (to_dvec.size() != from_dvec.size()) {
      to_dvec.resize(from_dvec.size());
    }
    if (from_dvec.size() == 0) continue;
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(to_dvec.data(), from_dvec.data(),
                                        from_dvec.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                        stream));
  }
}

/**
 * @brief Copies minimizer state back to a problem.
 *
 * Updates the problem's state batches with values from the minimizer state.
 * This commits the optimized state values back to the original problem.
 *
 * @param stream CUDA stream for GPU operations.
 * @param state Source minimizer state.
 * @param[out] problem Destination problem to update.
 */
void Copy(cudaStream_t stream, const MinimizerState &state, Problem &problem) {
  auto &state_batches = problem.GetStateBatches();
  const auto &state_values = state.GetStates();

  for (size_t i = 0; i < state_batches.size(); i++) {
    auto &param_batch_ptr = state_batches[i];
    const auto &dvec = state_values[i];
    if (dvec.size() == 0) continue;
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(param_batch_ptr->StateDevicePtr(0), dvec.data(),
                                        dvec.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                        stream));
  }
}
}  // namespace cunls
