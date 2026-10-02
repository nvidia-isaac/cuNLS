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

#include "cunls/common/types.h"
#include "cunls/state/sized_state_batch.h"

namespace cunls {

/**
 * @brief Batch processing for SO(3) Lie group states.
 *
 * This class implements the Plus operation for the SO(3) Lie group,
 * which represents rotations in 3D.
 * The tangent space has dimension 3 (rotation vector),
 * while the ambient space has dimension 9 (3x3 rotation matrix).
 *
 * The class uses GPU-accelerated operations via fused CUDA kernels
 * for efficient batch processing of multiple rotations.
 */
class SO3StateBatch : public SizedStateBatch<9, 3> {
 public:
  using Base = SizedStateBatch<9, 3>;

  /**
   * @brief Constructs a batch of SO(3) states.
   *
   * @param device_ptr Pointer to GPU device memory containing the SO(3)
   * rotation matrices. Must point to at least capacity * 9 floats of
   * allocated memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   */
  SO3StateBatch(const float *device_ptr, size_t capacity);

  /**
   * @brief Constructs a batch of SO(3) states with constant state
   * constraints.
   *
   * @param device_ptr Pointer to GPU device memory containing the SO(3)
   * rotation matrices. Must point to at least capacity * 9 floats of
   * allocated memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of states that should remain constant.
   * @param const_capacity Number of ids the constant-id buffer holds.
   */
  SO3StateBatch(const float *device_ptr, size_t capacity, const int *device_constant_state_ids,
                size_t const_capacity);

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(skew(delta))
   *
   * Applies a right-multiplication update to the rotation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input rotation matrices (device pointer)
   * @param delta Tangent space updates (3D rotation vectors, device pointer)
   * @param x_plus_delta Output rotation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override;

 private:
  mutable dvector<Matrix<3>> delta_rotations_;
  mutable dvector<float> twists_;

  /**
   * @brief Applies an SO(3) update: result = x * Exp(skew(delta)) or
   * result = x * Exp(-skew(delta))
   *
   * Computes the right-multiplication update for SO(3) rotations.
   * First computes the update matrix Exp(skew(delta)) or Exp(-skew(delta))
   * using Exp, then performs batched matrix multiplication in a CUDA kernel.
   *
   * This is a helper function used by Plus (invert_delta=false).
   *
   * @param x Input rotation matrices (device pointer, row-major)
   * @param delta Tangent space updates (3D rotation vectors, device pointer)
   * @param result Output rotation matrices (device pointer, row-major)
   * @param invert_delta If true, compute Exp(-skew(delta)), otherwise
   * Exp(skew(delta))
   * @param stream CUDA stream for asynchronous execution
   */
  void ApplyUpdate(const float *x, const float *delta, float *result, bool invert_delta,
                   cudaStream_t stream);
};
}  // namespace cunls
