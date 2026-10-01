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

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for Sim(2) prior constraints.
 *
 * Computes the residual between a current Sim(2) transform and a target:
 *   residual = Log(T_target^{-1} * T_current)
 *
 * The Jacobian with respect to a right-perturbation delta is:
 *   J = J_r^{-1}(residual)
 * where J_r^{-1} is the inverse right Jacobian of Sim(2).
 *
 * The factor has:
 * - 4 residuals (4D tangent vector [u_x, u_y, theta, lambda])
 * - 1 state block with tangent dimension 4 (transform stored as 3x3 matrix)
 *
 * @note The observations_ptr must point to GPU device memory containing target
 *       transformation matrices and remain valid for the lifetime of this
 * object. Memory layout: [T0: 9 floats][T1: 9 floats]...[TN-1: 9 floats] Each
 * transform is stored row-major: [cos(theta), -sin(theta), tx, sin(theta),
 * cos(theta), ty, 0, 0, 1/s]
 */
class Similarity2PriorFactorBatch : public SizedFactorBatch<4, 4> {
  using Base = SizedFactorBatch<4, 4>;

 public:
  /**
   * @brief Constructs a batch of Sim(2) prior factors.
   *
   * Pre-computes T_target^{-1} for all targets during construction.
   *
   * @param observations_ptr Pointer to GPU device memory containing target
   * transforms. Must point to at least capacity * 9 floats.
   * @param capacity Number of factors the measurement buffers hold. The active
   *        count starts at 0: call SetNumFactors(n) before evaluating or solving.
   */
  Similarity2PriorFactorBatch(const Similarity2Transform *observations_ptr, size_t capacity);

  /**
   * @brief Evaluates Sim(2) prior residuals and optionally Jacobians.
   *
   * @param residuals Output residuals (4 floats per factor, device pointer).
   * @param jacobians Output Jacobians (4x4 floats per factor, device pointer).
   *                  Can be nullptr to skip Jacobian computation.
   * @param state_pointers Device pointer to state block pointers.
   * @param stream CUDA stream for asynchronous execution.
   * @param factor_ids Optional per-item factor indices (device pointer).
   * @param num_factor_ids Number of items (the length of factor_ids when it
   *        is given); 0 means NumFactors().
   * @return true on success.
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final;

 private:
  Similarity2PriorFactorBatch() = default;

  /// Pointer to user-managed device memory containing target transforms.
  const Matrix<3> *observations_ptr_;

  /// Preallocated memory for transform error T_target^{-1} * T_current.
  mutable DeviceVector<float> transforms_error_;  ///< T^{-1} * C, 9 floats per item.
};

}  // namespace cunls
