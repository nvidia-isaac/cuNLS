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

#include <cuda/std/array>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for point-to-point constraints.
 *
 * Computes the residual between a target point p and a source point q
 * transformed by an SE(3) pose T:
 *   residual = p - T @ q = p - (R * q + t)
 *
 * where R is the 3x3 rotation matrix and t is the translation vector of T.
 *
 * The factor has:
 * - 3 residuals (3D vector difference)
 * - 1 state block with 6 state_pointers (SE(3) tangent space dimension)
 *
 * The Jacobian is computed with respect to the SE(3) tangent vector
 * delta = [omega; rho] (rotation, translation) using right perturbation
 * T' = T * Exp(delta):
 *   - d(r)/d(omega) = R * [q]_x   (columns 0-2)
 *   - d(r)/d(rho)   = -R          (columns 3-5)
 *
 * where [q]_x is the skew-symmetric matrix of q.
 *
 * @note The p_observations_ptr and q_observations_ptr must point to GPU device
 *       memory and remain valid for the lifetime of this object. The memory
 *       layout for each is:
 *       [pt0: 3 floats][pt1: 3 floats]...[ptN-1: 3 floats]
 */
class PointToPointFactorBatch : public SizedFactorBatch<3, 6> {
  using Base = SizedFactorBatch<3, 6>;
  using Vector3 = Vector<3>;

public:
  /**
   * @brief Constructs a batch of point-to-point factors.
   *
   * @param p_observations_ptr Pointer to GPU device memory containing target
   * points. Must point to at least num_factors * 3 floats of allocated memory.
   * @param q_observations_ptr Pointer to GPU device memory containing source
   * points. Must point to at least num_factors * 3 floats of allocated memory.
   * @param num_factors Number of factors in the batch.
   */
  PointToPointFactorBatch(const Vector3 *p_observations_ptr,
                          const Vector3 *q_observations_ptr, size_t num_factors)
      : p_observations_ptr_(p_observations_ptr),
        q_observations_ptr_(q_observations_ptr), num_factors_(num_factors) {}

  /**
   * @brief Evaluates point-to-point residuals and optionally Jacobians.
   *
   * Computes residual = p - T @ q for each point correspondence in the batch.
   * If jacobians is not nullptr, also computes the 3x6 Jacobian with respect
   * to the SE(3) tangent space.
   *
   * @param residuals Output device pointer for residuals (3 floats per
   *                  factor). Can be nullptr to skip residual computation.
   * @param jacobians Output device pointer for Jacobians (3 x 6 = 18 floats per
   *                  factor). Can be nullptr to skip Jacobian computation.
   * @param state_pointers Device pointer to state block pointers. Each entry
   *                   points to an SE(3) transform (16 floats) on the device.
   * @param stream CUDA stream for asynchronous execution.
   * @return true on success.
   */
  bool Evaluate(float *residuals, float *jacobians,
                float const *const *state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Returns the number of point-to-point factors in this batch.
   * @return Number of factors.
   */
  size_t NumFactors() const final { return num_factors_; }

private:
  PointToPointFactorBatch() = default;

  /// Pointer to user-managed device memory containing target points (p).
  const Vector3 *p_observations_ptr_;

  /// Pointer to user-managed device memory containing source points (q).
  const Vector3 *q_observations_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;
};

} // namespace cunls
