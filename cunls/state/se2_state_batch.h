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

#include "cunls/common/cublas_helper.h"
#include "cunls/common/types.h"
#include "cunls/state/sized_state_batch.h"

namespace cunls {

/**
 * @brief Batch processing for SE(2) Lie group state blocks.
 *
 * This class implements the Plus operation for the SE(2) Lie group,
 * which represents rigid body transformations (rotation + translation) in 2D.
 * The tangent space has dimension 3 (2 for translation, 1 for rotation),
 * while the ambient space has dimension 9 (3x3 homogeneous matrix).
 *
 * An SE(2) transformation matrix is stored in row-major order as 9 floats:
 * [cos(theta), -sin(theta), tx, sin(theta), cos(theta), ty, 0, 0, 1]
 *
 * The tangent vector convention is [v_x, v_y, theta] where v_x, v_y are
 * translational components and theta is the rotation angle.
 *
 * The exponential map for SE(2) maps a tangent vector [v_x, v_y, theta] to:
 *   Exp([v_x, v_y, theta]) = [[cos(theta), -sin(theta), tx],
 *                              [sin(theta),  cos(theta), ty],
 *                              [    0,           0,       1]]
 * where [tx, ty] = V * [v_x, v_y] with V = [[sin(theta)/theta,
 * -(1-cos(theta))/theta],
 *                                             [(1-cos(theta))/theta,
 * sin(theta)/theta]]
 *
 * The class uses GPU-accelerated operations via CUDA kernels and cuBLAS
 * for efficient batch processing of multiple transformations.
 */
class SE2StateBatch : public SizedStateBatch<9, 3> {
public:
  using Base = SizedStateBatch<9, 3>;

  /**
   * @brief Constructs a batch of SE(2) state blocks.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the SE(2)
   * transforms. Must point to at least num_blocks * 9 floats of allocated
   * memory.
   * @param num_blocks The number of SE(2) state blocks in this batch.
   */
  SE2StateBatch(cuBLASHandle &cublas_handle, const float *device_ptr,
                size_t num_blocks);

  /**
   * @brief Constructs a batch of SE(2) state blocks with constant state
   * constraints.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param device_ptr Pointer to GPU device memory containing the SE(2)
   * transforms. Must point to at least num_blocks * 9 floats of allocated
   * memory.
   * @param num_blocks The number of SE(2) state blocks in this batch.
   * @param device_constant_state_ids Pointer to GPU device memory containing
   * the indices of state blocks that should remain constant.
   * @param num_const_state_blocks The number of constant state blocks.
   */
  SE2StateBatch(cuBLASHandle &cublas_handle, const float *device_ptr,
                size_t num_blocks, const int *device_constant_state_ids,
                size_t num_const_state_blocks);

  /**
   * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
   *
   * Applies a right-multiplication update to the transformation matrix x
   * using the exponential map of the Lie algebra element delta.
   *
   * @param x Input transformation matrices (device pointer)
   * @param delta Tangent space updates (3D vectors [v_x, v_y, theta], device
   * pointer)
   * @param x_plus_delta Output transformation matrices (device pointer)
   * @param stream CUDA stream for asynchronous execution
   */
  void Plus(const float *x, const float *delta, float *x_plus_delta,
            cudaStream_t stream) override;

private:
  cuBLASHandle &cublas_handle_; ///< cuBLAS handle for matrix operations

  mutable dvector<Matrix<3>> delta_transforms_;
  mutable dvector<float> tangents_;

  /**
   * @brief Applies an SE(2) update: result = x * Exp(delta) or
   * result = x * Exp(-delta)
   *
   * Computes the right-multiplication update for SE(2) transformations.
   * First computes the update matrix Exp(delta) or Exp(-delta)
   * using the SE(2) exponential map, then performs batched matrix
   * multiplication using cuBLAS.
   *
   * This is a helper function used by Plus (invert_delta=false).
   *
   * Note: cuBLAS uses column-major storage, but our matrices are row-major.
   * For right-multiplication (x * update), we use CUBLAS_OP_N for both
   * operands. cuBLAS interprets the matrices as column-major, so this computes
   * the equivalent of the desired row-major result.
   *
   * @param x Input transformation matrices (device pointer, row-major)
   * @param delta Tangent space updates (3D vectors [v_x, v_y, theta], device
   * pointer)
   * @param result Output transformation matrices (device pointer, row-major)
   * @param invert_delta If true, compute Exp(-delta), otherwise Exp(delta)
   * @param stream CUDA stream for asynchronous execution
   */
  void ApplyUpdate(const float *x, const float *delta, float *result,
                   bool invert_delta, cudaStream_t stream);
};
} // namespace cunls
