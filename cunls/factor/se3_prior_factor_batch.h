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
 * @brief Batch factor for SE(3) prior constraints.
 *
 * Computes the residual between a current SE(3) transform and a target:
 *   residual = Log(T_target^{-1} * T_current)
 *
 * The Jacobian with respect to a right-perturbation delta is:
 *   J = J_r^{-1}(residual)
 * where J_r^{-1} is the inverse right Jacobian of SE(3).
 *
 * The factor has:
 * - 6 residuals (6D twist vector)
 * - 1 state block with tangent dimension 6 (transform stored as 4x4 matrix)
 *
 * @note The observations_ptr must point to GPU device memory containing target
 *       transformation matrices and remain valid for the lifetime of this
 * object. Memory layout: [T0: 16 floats][T1: 16 floats]...[TN-1: 16 floats]
 */
class SE3PriorFactorBatch : public SizedFactorBatch<6, 6> {
  using Base = SizedFactorBatch<6, 6>;

public:
  /**
   * @brief Constructs a batch of SE(3) prior factors.
   *
   * Pre-computes T_target^{-1} for all targets during construction.
   *
   * @param observations_ptr Pointer to GPU device memory containing target
   * transforms. Must point to at least num_factors * 16 floats.
   * @param num_factors Number of factors in the batch.
   */
  SE3PriorFactorBatch(const SE3Transform *observations_ptr, size_t num_factors);

  /**
   * @brief Evaluates SE(3) prior residuals and optionally Jacobians.
   *
   * @param residuals Output residuals (6 floats per factor, device pointer).
   * @param jacobians Output Jacobians (6x6 floats per factor, device pointer).
   *                  Can be nullptr to skip Jacobian computation.
   * @param state_pointers Device pointer to state block pointers.
   * @param stream CUDA stream for asynchronous execution.
   * @return true on success.
   */
  bool Evaluate(float *residuals, float *jacobians,
                float const *const *state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Returns the number of factors in the batch.
   * @return Number of factors.
   */
  size_t NumFactors() const final { return num_factors_; }

private:
  SE3PriorFactorBatch() = default;

  /// Pointer to user-managed device memory containing target transforms.
  const SE3Transform *observations_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;

  /// Pre-computed inverse of target transforms T_target^{-1}.
  DeviceVector<SE3Transform> observations_inverse_;

  /// Preallocated memory for transform error T_target^{-1} * T_current.
  mutable DeviceVector<SE3Transform> transforms_error_;
};

} // namespace cunls
