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

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Dynamics of a car with Ackermann steering (kinematic bicycle model,
 * rear-axle reference) driving on non-planar roads (slopes, ramps).
 *
 * States (in order): pose T_k (SE3StateBatch; vehicle frame x forward, z up),
 * z_k = (v, δ) [m/s, rad] (VectorStateBatch<2>), control u_k = (a, δ̇)
 * (VectorStateBatch<2>), pose T_{k+1}, z_{k+1}.
 *
 * Euler step with the body twist ξ(z) = [0, 0, v tan δ / L, v, 0, 0] (tangent
 * order [φ, ρ]); with e = Log((T_k Exp(dt_k ξ(z_k)))⁻¹ T_{k+1}) ∈ R⁶:
 *
 * @verbatim
 *   residual[0:4] = e[2:6]                       (yaw, forward, lateral, vertical)
 *   residual[4:6] = z_{k+1} - z_k - dt_k u_k
 * @endverbatim
 *
 * Roll and pitch changes are left to terrain factors, as in
 * SE3DifferentialDriveFactorBatch. Analytic Jacobians. The planar counterpart
 * is SE2KinematicBicycleFactorBatch.
 */
class SE3KinematicBicycleFactorBatch : public SizedFactorBatch<6, 6, 2, 2, 6, 2> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param wheelbase Distance L between the axles [m].
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null or L is not positive.
   */
  SE3KinematicBicycleFactorBatch(const float *time_steps, float wheelbase, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *time_steps_;
  float wheelbase_;
};

}  // namespace cunls
