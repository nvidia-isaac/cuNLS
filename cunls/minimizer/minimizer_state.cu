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

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

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
__global__ void set_state_pointers_kernel(
    float** new_pointers, float* const* old_pointers, float* old_start_ptr,
    float* new_start_ptr, size_t num_states_in_batch, size_t num_new_pointers) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_new_pointers) {
    return;
  }

  float* old_ptr = old_pointers[tid];
  int offset = static_cast<int>(old_ptr - old_start_ptr);
  if (offset < 0 || offset >= num_states_in_batch) {
    return;
  }

  new_pointers[tid] = new_start_ptr + offset;
}

/**
 * @brief Allocates state storage vectors.
 *
 * Creates one device vector per state batch, sized to hold all state
 * blocks in that batch flattened into a single vector.
 *
 * @param problem The problem containing state batch information.
 */
void MinimizerState::CreateStates(const Problem& problem) {
  const auto& state_batches = problem.GetStateBatches();
  if (states_.size() != state_batches.size()) {
    states_.resize(state_batches.size());
  }

  for (size_t i = 0; i < state_batches.size(); i++) {
    const auto& param_batch_ptr = state_batches[i];
    auto& state_vec = states_[i];

    size_t size =
        param_batch_ptr->NumStateBlocks() * param_batch_ptr->AmbientSize();

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
void MinimizerState::CreateStatePointers(const Problem& problem) {
  const auto& problem_param_pointers = problem.GetStatePointers();
  state_pointers_.resize(problem_param_pointers.size());

  for (size_t i = 0; i < problem_param_pointers.size(); i++) {
    const auto& param_ptrs = problem_param_pointers[i];
    auto& new_ptrs = state_pointers_[i];

    new_ptrs.resize(param_ptrs.size());
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
void MinimizerState::Create(cudaStream_t stream, const Problem& problem) {
  CreateStates(problem);
  CreateStatePointers(problem);

  const auto& state_batches = problem.GetStateBatches();
  {
    // Copy state values from problem to local storage
    auto stream_policy = thrust::cuda::par_nosync.on(stream);

    for (size_t i = 0; i < state_batches.size(); i++) {
      const auto& param_batch_ptr = state_batches[i];
      auto& state_vec = states_[i];

      float* ptr = param_batch_ptr->StateBlockDevicePtr(0);
      size_t size = param_batch_ptr->NumStateBlocks() *
                    param_batch_ptr->AmbientSize();

      thrust::device_ptr<float> src_ptr(ptr);
      thrust::device_ptr<float> dst_ptr(state_vec.data());
      thrust::copy(stream_policy, src_ptr, src_ptr + size, dst_ptr);
    }
  }

  const auto& problem_param_pointers = problem.GetStatePointers();

  {
    // Remap state pointers to point into local storage
    const auto& residual_batches = problem.GetResidualBatches();

    for (size_t i = 0; i < residual_batches.size(); i++) {
      const auto& param_ptrs = problem_param_pointers[i];
      auto& new_ptrs = state_pointers_[i];

      assert(param_ptrs.size() == new_ptrs.size());

      float** new_pointers = new_ptrs.data();
      float* const* old_pointers = param_ptrs.data();

      for (size_t j = 0; j < state_batches.size(); j++) {
        const auto& param_batch_ptr = state_batches[j];
        auto& new_states = states_[j];

        size_t num_states_in_batch = param_batch_ptr->NumStateBlocks() *
                                     param_batch_ptr->AmbientSize();

        assert(num_states_in_batch == new_states.size());

        float* new_param_ptr = new_states.data();

        float* state_batch_ptr =
            param_batch_ptr->StateBlockDevicePtr(0);

        size_t num_blocks = (param_ptrs.size() + block_size - 1) / block_size;

        set_state_pointers_kernel<<<num_blocks, block_size, 0, stream>>>(
            new_pointers, old_pointers, state_batch_ptr, new_param_ptr,
            num_states_in_batch, param_ptrs.size());
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
void MinimizerState::Copy(cudaStream_t stream,
                          const std::vector<dvector<float>>& from) {
  if (states_.size() != from.size()) {
    states_.resize(from.size());
  }

  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  for (size_t i = 0; i < from.size(); i++) {
    const auto& from_dvec = from[i];
    auto& to_dvec = states_[i];

    if (to_dvec.size() != from_dvec.size()) {
      to_dvec.resize(from_dvec.size());
    }

    thrust::device_ptr<const float> src_ptr(from_dvec.data());
    thrust::device_ptr<float> dst_ptr(to_dvec.data());
    thrust::copy(stream_policy, src_ptr, src_ptr + from_dvec.size(),
                 dst_ptr);
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
void Copy(cudaStream_t stream, const MinimizerState& state, Problem& problem) {
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  auto& state_batches = problem.GetStateBatches();
  const auto& state_values = state.GetStates();

  for (size_t i = 0; i < state_batches.size(); i++) {
    auto& param_batch_ptr = state_batches[i];
    const auto& dvec = state_values[i];

    float* ptr = param_batch_ptr->StateBlockDevicePtr(0);
    thrust::device_ptr<const float> src_ptr(dvec.data());
    thrust::device_ptr<float> dst_ptr(ptr);
    thrust::copy(stream_policy, src_ptr, src_ptr + dvec.size(), dst_ptr);
  }
}
}  // namespace cunls
