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
 * @brief Planar dynamics of a robot with two driven wheels and passive casters
 * (NVIDIA Carter), kinematic, between consecutive trajectory steps.
 *
 * States (in order): pose T_k (SE2StateBatch), wheel speeds u_k = (ω_L, ω_R)
 * [rad/s] (VectorStateBatch<2>), pose T_{k+1} (SE2StateBatch).
 *
 * With the body twist ξ(u) = [r (ω_R + ω_L) / 2, 0, r (ω_R - ω_L) / b] (wheel
 * radius r, track width b) held over the step dt_k,
 *
 * @verbatim
 *   residual = Log((T_k Exp(dt_k ξ(u_k)))⁻¹ T_{k+1})        (3: [v_x, v_y, θ])
 * @endverbatim
 *
 * which is exact for wheel speeds that are constant over the step. Analytic
 * Jacobians: -J_l⁻¹(r) Ad(Exp(-dt ξ)) (T_k), -J_l⁻¹(r) J_r(dt ξ) dt ∂ξ/∂u
 * (u_k), J_r⁻¹(r) (T_{k+1}).
 *
 * Use as a soft factor (weighted) or as a hard constraint:
 * ConstraintFactorBatch(&factor, ConstraintKind::kEquality). For driving on
 * non-planar terrain, SE3DifferentialDriveFactorBatch is the SE(3) variant
 * (same twist in the body frame).
 */
class SE2DifferentialDriveFactorBatch : public SizedFactorBatch<3, 3, 2, 3> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param wheel_radius Wheel radius r [m].
   * @param track_width Distance b between the wheels [m].
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null or r, b are not positive.
   */
  SE2DifferentialDriveFactorBatch(const float *time_steps, float wheel_radius, float track_width,
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
