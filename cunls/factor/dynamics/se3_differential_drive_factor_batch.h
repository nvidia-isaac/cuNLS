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
 * @brief Dynamics of a robot with two driven wheels and passive casters
 * (NVIDIA Carter) driving on non-planar terrain (slopes, ramps), kinematic.
 *
 * States (in order): pose T_k (SE3StateBatch; vehicle frame x forward, z up),
 * wheel speeds u_k = (ω_L, ω_R) [rad/s] (VectorStateBatch<2>), pose T_{k+1}
 * (SE3StateBatch).
 *
 * The wheels set the body twist ξ(u) = [0, 0, ω, v, 0, 0] (tangent order
 * [φ, ρ]: yaw rate ω = r (ω_R - ω_L) / b about body z, forward speed
 * v = r (ω_R + ω_L) / 2 along body x). With e = Log((T_k Exp(dt_k ξ))⁻¹ T_{k+1})
 * ∈ R⁶ the residual is
 *
 * @verbatim
 *   residual = [e_yaw, e_forward, e_lateral, e_vertical] = e[2:6]
 * @endverbatim
 *
 * Roll and pitch changes (e[0:2]) are not constrained: on a slope they follow
 * the terrain, which other factors provide (an attitude or terrain-normal
 * prior, a height map, ground contact). Without such factors roll and pitch
 * are unobserved. Analytic Jacobians (rows 2..5 of the SE(3) Jacobians).
 *
 * The planar counterpart is SE2DifferentialDriveFactorBatch.
 */
class SE3DifferentialDriveFactorBatch : public SizedFactorBatch<4, 6, 2, 6> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param wheel_radius Wheel radius r [m].
   * @param track_width Distance b between the wheels [m].
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null or r, b are not positive.
   */
  SE3DifferentialDriveFactorBatch(const float *time_steps, float wheel_radius, float track_width,
                                  size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *time_steps_;
  float wheel_radius_;
  float track_width_;
};

}  // namespace cunls
