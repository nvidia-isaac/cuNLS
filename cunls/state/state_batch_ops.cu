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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/state/state_batch_ops.h"

namespace cunls {

/**
 * @brief CUDA kernel that marks constant state entries as false in a binary pattern.
 *
 * For each constant state block index in const_ids, sets all tangent_dim
 * entries in binary_states corresponding to that block to false.
 *
 * @param binary_states Device array of booleans (one per scalar tangent state component).
 *                      Pre-filled with true; this kernel sets entries to false
 *                      for constant blocks.
 * @param tangent_dim   Tangent space dimension per state block.
 * @param const_ids     Device array of constant state block indices.
 * @param num_const_ids Number of constant state blocks.
 *
 * Grid/block: launched with ceil(num_const_ids / 32) blocks of 32 threads.
 */
__global__ void binary_pattern_kernel(bool* __restrict__ binary_states,
                                      size_t tangent_dim,
                                      const int* __restrict__ const_ids,
                                      size_t num_const_ids) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_const_ids) {
    return;
  }

  int id = const_ids[tid];

  for (int i = 0; i < tangent_dim; i++) {
    binary_states[id * tangent_dim + i] = false;
  }
}

/**
 * @brief Fills a binary pattern marking non-constant tangent state components as true.
 *
 * First fills the entire binary_pattern vector with true, then launches
 * binary_pattern_kernel to set entries for constant state blocks to false.
 *
 * @param stream                 CUDA stream for asynchronous execution.
 * @param tangent_dim            Tangent space dimension per state block.
 * @param const_state_block_ids  Device pointer to constant state block indices,
 *                               or nullptr if no blocks are constant.
 * @param num_const_state_blocks Number of constant state blocks.
 * @param binary_pattern         Output device vector of booleans (resized by caller).
 */
void fill_binary_pattern(cudaStream_t stream, size_t tangent_dim,
                         const int* const_state_block_ids,
                         size_t num_const_state_blocks,
                         thrust::device_vector<bool>& binary_pattern) {
  auto stream_policy = thrust::cuda::par.on(stream);
  // Fill pattern with trues
  thrust::fill(stream_policy, binary_pattern.begin(), binary_pattern.end(),
               true);

  if (const_state_block_ids == nullptr || num_const_state_blocks == 0) {
    return;
  }

  bool* input_ptr = thrust::raw_pointer_cast(binary_pattern.data());
  size_t block_size = 32;  // one WARP
  size_t num_blocks =
      (num_const_state_blocks + block_size - 1) / block_size;

  // Set the pattern to false for the constant state blocks
  binary_pattern_kernel<<<num_blocks, block_size, 0, stream>>>(
      input_ptr, tangent_dim, const_state_block_ids, num_const_state_blocks);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Functor that maps constant state indices to INT_MAX for sorting.
 *
 * Used with thrust::transform to replace map entries of constant states
 * with the maximum integer value so they sort to the end of the mapping array.
 */
struct CustomBinaryFunctor {
  /**
   * @brief Returns map_value if active, or INT_MAX if the state is constant.
   * @param map_value      The original mapping index.
   * @param binary_pattern True if the state is active (non-constant).
   * @return map_value for active states, INT_MAX for constant states.
   */
  __host__ __device__ int operator()(int map_value, bool binary_pattern) const {
    if (!binary_pattern) {
      return cuda::std::numeric_limits<int>::max();
    }
    return map_value;
  }
};

/** @copydoc StateBatchOps::StateBatchOps(cudaStream_t, const std::vector<StateBatch*>&) */
StateBatchOps::StateBatchOps(
    cudaStream_t stream, const std::vector<StateBatch*>& state_batches) {
  Preprocess(stream, state_batches);
}

/** @copydoc StateBatchOps::InitUpdatesVector */
void StateBatchOps::InitUpdatesVector(
    const std::vector<StateBatch*>& state_batches) {
  delta_ptrs_.clear();
  size_t updates_size = 0;

  // Calculate the total size of the state_updates_ buffer
  for (const auto batch : state_batches) {
    size_t tangent_dim = batch->TangentSize();
    size_t num_blocks = batch->NumStateBlocks();

    updates_size += num_blocks * tangent_dim;
  }

  state_updates_.resize(updates_size);
  float* delta_ptr = state_updates_.data();
  float* const delta_end = delta_ptr + state_updates_.size();

  // Fill the ptrs to the updates w.r.t individual state batches.
  for (const auto batch : state_batches) {
    delta_ptrs_.push_back(delta_ptr);
    delta_ptr += batch->NumStateBlocks() * batch->TangentSize();
    assert(delta_ptr <= delta_end && "State updates buffer overflow");
  }
}

/** @copydoc StateBatchOps::InitMapping */
void StateBatchOps::InitMapping(
    cudaStream_t stream, const std::vector<StateBatch*>& state_batches) {
  thrust::device_vector<bool> temp_buffer;
  map_.resize(state_updates_.size());
  thrust::device_ptr<int> map_ptr(map_.data());
  auto map_it = map_ptr;
  auto stream_policy = thrust::cuda::par.on(stream);

  num_reduced_states_ = 0;
  for (auto batch : state_batches) {
    size_t tangent_dim = batch->TangentSize();
    const int* const_state_ids = batch->ConstStateIds();
    size_t num_const_blocks = batch->NumConstStateBlocks();

    size_t num_elements = batch->NumStateBlocks() * tangent_dim;
    temp_buffer.resize(num_elements);
    fill_binary_pattern(stream, tangent_dim, const_state_ids,
                        num_const_blocks, temp_buffer);

    thrust::counting_iterator<int> iter(static_cast<int>(map_it - map_ptr));
    // Make map contain valid ids for non-const states, and INT_MAX for
    // constant states (so they sort to the end)
    thrust::transform(stream_policy, iter, iter + num_elements,
                      temp_buffer.begin(), map_it, CustomBinaryFunctor());

    num_reduced_states_ += num_elements - num_const_blocks * tangent_dim;
    map_it += num_elements;
  }

  // Sort so constant state indices are at the end
  thrust::sort(stream_policy, map_ptr, map_ptr + map_.size());
}

/** @copydoc StateBatchOps::Preprocess */
void StateBatchOps::Preprocess(
    cudaStream_t stream, const std::vector<StateBatch*>& state_batches) {
  user_state_batches_ = state_batches;

  InitUpdatesVector(state_batches);
  InitMapping(stream, state_batches);
}

/** @copydoc StateBatchOps::Plus */
void StateBatchOps::Plus(cudaStream_t stream,
                         const std::vector<const float*>& x_ptrs,
                         const DeviceVector<float>& delta,
                         std::vector<float*>& x_plus_delta_ptrs) {
  assert(delta.size() == num_reduced_states_);
  assert(x_ptrs.size() == x_plus_delta_ptrs.size());
  assert(x_ptrs.size() == delta_ptrs_.size());
  auto stream_policy = thrust::cuda::par.on(stream);

  // Zero out the updates
  thrust::device_ptr<float> updates_ptr(state_updates_.data());
  thrust::fill(stream_policy, updates_ptr,
               updates_ptr + state_updates_.size(), 0.0f);

  // Scatter the delta values across the update vector
  thrust::device_ptr<const float> delta_ptr(delta.data());
  thrust::device_ptr<const int> map_ptr(map_.data());
  thrust::scatter(stream_policy, delta_ptr, delta_ptr + delta.size(), map_ptr,
                  updates_ptr);

  for (size_t i = 0; i < x_ptrs.size(); i++) {
    auto state_batch = user_state_batches_[i];
    state_batch->Plus(x_ptrs[i], delta_ptrs_[i], x_plus_delta_ptrs[i], stream);
  }
}

}  // namespace cunls
