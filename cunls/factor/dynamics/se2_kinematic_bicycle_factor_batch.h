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
 * @brief Planar dynamics of a car with Ackermann steering (kinematic bicycle model,
 * rear-axle reference) between consecutive trajectory steps.
 *
 * States (in order): pose T_k (SE2StateBatch), speed and steering angle
 * z_k = (v, δ) [m/s, rad] (VectorStateBatch<2>), control u_k = (a, δ̇)
 * [m/s², rad/s] (VectorStateBatch<2>), pose T_{k+1}, z_{k+1}.
 *
 * Euler step over dt_k with the body twist ξ(z) = [v, 0, v tan δ / L]
 * (wheelbase L):
 *
 * @verbatim
 *   residual[0:3] = Log((T_k Exp(dt_k ξ(z_k)))⁻¹ T_{k+1})
 *   residual[3:5] = z_{k+1} - z_k - dt_k u_k
 * @endverbatim
 *
 * Analytic Jacobians. The model holds at moderate lateral acceleration (no
 * tire slip). Use as a soft factor or wrapped in
 * ConstraintFactorBatch(&factor, ConstraintKind::kEquality). For roads with
 * slopes and ramps, an SE(3) variant (same body twist) is the planned
 * SE3KinematicBicycleFactorBatch.
 */
class SE2KinematicBicycleFactorBatch : public SizedFactorBatch<5, 3, 2, 2, 3, 2> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param wheelbase Distance L between the axles [m].
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null or L is not positive.
   */
  SE2KinematicBicycleFactorBatch(const float *time_steps, float wheelbase, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *time_steps_;
  float wheelbase_;
};

}  // namespace cunls
