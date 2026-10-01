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
 * @brief Batch processing for Sim(2) Lie group states.
 *
 * This class implements the Plus operation for the Sim(2) Lie group,
 * which represents similarity transformations (rotation + translation + scale)
 * in 2D. The tangent space has dimension 4 (2 for translation, 1 for rotation,
 * 1 for log-scale), while the ambient space has dimension 9 (3x3 matrix).
 *
 * A Sim(2) transformation matrix is stored in row-major order as 9 floats:
 * [cos(theta), -sin(theta), tx, sin(theta), cos(theta), ty, 0, 0, 1/s]
 *
 * The tangent vector convention is [u_x, u_y, theta, lambda] where u_x, u_y are
 * translational components, theta is the rotation angle, and lambda = log(s)
 * is the log-scale.
 *
 * The exponential map for Sim(2) maps a tangent vector [u_x, u_y, theta,
 * lambda] to: Exp([u_x, u_y, theta, lambda]) = [[cos(theta), -sin(theta), tx],
 *                                      [sin(theta),  cos(theta), ty],
 *                                      [    0,           0,     e^{-lambda}]]
 * where [tx, ty] = V(theta, lambda) * [u_x, u_y] with V being the Sim(2)
 * V-matrix as described in Eade's "Lie Groups for 2D and 3D Transformations".
 *
 * The class uses GPU-accelerated operations via fused CUDA kernels
 * for efficient batch processing of multiple transformations.
 */
class Similarity2StateBatch : public SizedStateBatch<9, 4> {
 public:
  using Base = SizedStateBatch<9, 4>;

  /**
   * @brief Constructs a batch of Sim(2) states.
   *
   * @param device_ptr Pointer to GPU device memory containing the Sim(2)
   * transforms. Must point to at least capacity * 9 floats of allocated
   * memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   */
  Similarity2StateBatch(const float *device_ptr, size_t capacity);

  /**
   * @brief Constructs a batch of Sim(2) states with constant state
   * constraints.
   *
   * @param device_ptr Pointer to GPU device memory containing the Sim(2)
   * transforms. Must point to at least capacity * 9 floats of allocated
   * memory.
   * @param capacity Number of states the buffer holds. The active count
   *        starts at 0: call SetNumActiveStates(n) before solving.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of states that should remain constant.
   * @param const_capacity Number of ids the constant-id buffer holds.
   */
  Similarity2StateBatch(const float *device_ptr, size_t capacity,
                        const int *device_constant_state_ids, size_t const_capacity);

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
   *
   * Applies a right-multiplication update to the transformation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input transformation matrices (device pointer)
   * @param delta Tangent space updates (4D vectors [u_x, u_y, theta, lambda],
   * device pointer)
   * @param x_plus_delta Output transformation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
            size_t num_replicas = 1) override;

 private:
  mutable dvector<Matrix<3>> delta_transforms_;
  mutable dvector<float> tangents_;

  /**
   * @brief Applies a Sim(2) update: result = x * Exp(delta) or
   * result = x * Exp(-delta)
   *
   * Computes the right-multiplication update for Sim(2) transformations.
   * First computes the update matrix Exp(delta) or Exp(-delta)
   * using the Sim(2) exponential map, then performs batched matrix
   * multiplication in a CUDA kernel.
   *
   * This is a helper function used by Plus (invert_delta=false).
   *
   * @param x Input transformation matrices (device pointer, row-major)
   * @param delta Tangent space updates (4D vectors [u_x, u_y, theta, lambda],
   * device pointer)
   * @param result Output transformation matrices (device pointer, row-major)
   * @param invert_delta If true, compute Exp(-delta), otherwise Exp(delta)
   * @param stream CUDA stream for asynchronous execution
   */
  void ApplyUpdate(const float *x, const float *delta, float *result, bool invert_delta,
                   cudaStream_t stream, size_t num_blocks);
};
}  // namespace cunls
