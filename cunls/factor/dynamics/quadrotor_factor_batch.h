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
 * @brief Physical parameters of a quadrotor in X configuration.
 *
 * Body frame: x forward, y left, z up. Rotor i sits at (x_i, y_i) =
 * a (s_x, s_y) with a = arm_length / √2 and (s_x, s_y) = (+1, -1), (-1, +1),
 * (+1, +1), (-1, -1) for i = 0..3 (front-right, back-left, front-left,
 * back-right). Rotors 0 and 1 spin counter-clockwise seen from above and exert
 * the yaw reaction torque -k_m f_i; rotors 2 and 3 spin clockwise, +k_m f_i.
 */
struct QuadrotorParameters {
  float mass = 1.f;                          ///< m [kg]
  float inertia[3] = {0.01f, 0.01f, 0.02f};  ///< diagonal body inertia (J_x, J_y, J_z) [kg m²]
  float arm_length = 0.17f;                  ///< rotor distance from the center [m]
  float torque_coefficient = 0.016f;         ///< k_m: yaw reaction torque per thrust [m]
  float linear_drag = 0.f;                   ///< D: v̇ gets -D v [1/s]
  float gravity = 9.81f;                     ///< g [m/s²], along -z of the world frame
};

/**
 * @brief Quadrotor rigid-body dynamics with rotor thrusts as controls, between
 * consecutive trajectory steps (Euler).
 *
 * States (in order): pose T_k = (R, p) (SE3StateBatch), world velocity v_k
 * (VectorStateBatch<3>), body rates ω_k (VectorStateBatch<3>), rotor thrusts
 * f_k ∈ R⁴ [N] (VectorStateBatch<4>), T_{k+1}, v_{k+1}, ω_{k+1}.
 *
 * Continuous model: ṗ = v, Ṙ = R ω^, v̇ = -g e₃ + R e₃ Σf / m - D v,
 * J ω̇ = τ(f) - ω × J ω, with τ from the rotor layout (QuadrotorParameters).
 * Euler step over dt_k:
 *
 * @verbatim
 *   residual[0:6]  = Log((T_k Exp(dt [ω_k; R_kᵀ v_k]))⁻¹ T_{k+1})
 *   residual[6:9]  = v_{k+1} - v_k - dt (-g e₃ + R_k e₃ Σf / m - D v_k)
 *   residual[9:12] = ω_{k+1} - ω_k - dt J⁻¹ (τ(f_k) - ω_k × J ω_k)
 * @endverbatim
 *
 * Analytic Jacobians. Use as soft factors, or wrapped in
 * ConstraintFactorBatch(&factor, ConstraintKind::kEquality) for MPC; thrust
 * limits 0 <= f_i <= f_max as BoundFactorBatch<4>. The model needs short
 * steps (about 10-25 ms).
 */
class QuadrotorFactorBatch : public SizedFactorBatch<12, 6, 3, 3, 4, 6, 3, 3> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param parameters Mass, inertia, rotor layout, drag, gravity.
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null or mass, inertia or
   *         arm length are not positive.
   */
  QuadrotorFactorBatch(const float *time_steps, const QuadrotorParameters &parameters,
                       size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

  const QuadrotorParameters &Parameters() const { return parameters_; }

 private:
  const float *time_steps_;
  QuadrotorParameters parameters_;
};

}  // namespace cunls
