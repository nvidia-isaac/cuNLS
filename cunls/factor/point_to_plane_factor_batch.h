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
 * @brief Batch factor for point-to-plane constraints.
 *
 * Computes the scalar residual measuring the signed distance between a target
 * point p and the plane defined by a source point q and its normal Nq, after
 * applying an SE(3) transformation T to q:
 *   residual = Nq^T * (p - T @ q) = Nq . (p - (R * q + t))
 *
 * where R is the 3x3 rotation matrix and t is the translation vector of T.
 *
 * The factor has:
 * - 1 residual (scalar signed distance to plane)
 * - 1 state block with 6 state_pointers (SE(3) tangent space dimension)
 *
 * The Jacobian is computed with respect to the SE(3) tangent vector
 * delta = [omega; rho] (rotation, translation) using right perturbation
 * T' = T * Exp(delta). Since r = Nq^T * (p - T@q), the Jacobian is
 * Nq^T times the point-to-point Jacobian:
 *   - d(r)/d(omega) = Nq^T * R * [q]_x   (columns 0-2)
 *   - d(r)/d(rho)   = -Nq^T * R           (columns 3-5)
 *
 * where [q]_x is the skew-symmetric matrix of q.
 *
 * @note The p_observations_ptr, q_observations_ptr, and nq_observations_ptr
 *       must point to GPU device memory and remain valid for the lifetime of
 *       this object. The memory layout for each is:
 *       [vec0: 3 floats][vec1: 3 floats]...[vecN-1: 3 floats]
 */
class PointToPlaneFactorBatch : public SizedFactorBatch<1, 6> {
  using Base = SizedFactorBatch<1, 6>;
  using Vector3 = Vector<3>;

public:
  /**
   * @brief Constructs a batch of point-to-plane factors.
   *
   * @param p_observations_ptr Pointer to GPU device memory containing target
   * points. Must point to at least num_factors * 3 floats of allocated memory.
   * @param q_observations_ptr Pointer to GPU device memory containing source
   * points. Must point to at least num_factors * 3 floats of allocated memory.
   * @param nq_observations_ptr Pointer to GPU device memory containing normal
   * vectors at each source point (in the source/q frame). Must point to at
   * least num_factors * 3 floats of allocated memory.
   * @param num_factors Number of factors in the batch.
   */
  PointToPlaneFactorBatch(const Vector3 *p_observations_ptr,
                          const Vector3 *q_observations_ptr,
                          const Vector3 *nq_observations_ptr,
                          size_t num_factors)
      : p_observations_ptr_(p_observations_ptr),
        q_observations_ptr_(q_observations_ptr),
        nq_observations_ptr_(nq_observations_ptr), num_factors_(num_factors) {}

  /**
   * @brief Evaluates point-to-plane residuals and optionally Jacobians.
   *
   * Computes residual = Nq^T * (p - T @ q) for each correspondence in the
   * batch. If jacobians is not nullptr, also computes the 1x6 Jacobian with
   * respect to the SE(3) tangent space.
   *
   * @param residuals Output device pointer for residuals (1 float per
   *                  factor). Can be nullptr to skip residual computation.
   * @param jacobians Output device pointer for Jacobians (1 x 6 = 6 floats per
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
   * @brief Returns the number of point-to-plane factors in this batch.
   * @return Number of factors.
   */
  size_t NumFactors() const final { return num_factors_; }

private:
  PointToPlaneFactorBatch() = default;

  /// Pointer to user-managed device memory containing target points (p).
  const Vector3 *p_observations_ptr_;

  /// Pointer to user-managed device memory containing source points (q).
  const Vector3 *q_observations_ptr_;

  /// Pointer to user-managed device memory containing normal vectors (Nq).
  const Vector3 *nq_observations_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;
};

} // namespace cunls
