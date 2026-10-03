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

/** @brief Physical parameters of the single-rigid-body quadruped model. */
struct QuadrupedParameters {
  float mass = 12.f;                         ///< m [kg]
  float inertia[3] = {0.07f, 0.26f, 0.24f};  ///< diagonal body inertia (J_x, J_y, J_z) [kg m²]
  float gravity = 9.81f;                     ///< g [m/s²], along -z of the world frame
};

/**
 * @brief Quadruped base dynamics as a single rigid body driven by the ground
 * reaction forces of its four feet (the standard locomotion MPC model, with
 * the full nonlinear rotation), between consecutive trajectory steps (Euler).
 *
 * States (in order): base pose T_k = (R, p) (SE3StateBatch), world velocity
 * v_k (VectorStateBatch<3>), body rates ω_k (VectorStateBatch<3>), foot forces
 * F_k = (f_0, ..., f_3) in the world frame [N] (VectorStateBatch<12>),
 * T_{k+1}, v_{k+1}, ω_{k+1}.
 *
 * Per-step inputs (device buffers, from the gait planner): contact flags s_i
 * (1: stance, 0: swing; 4 per factor) and world foot positions p_i (12 per
 * factor). Model: m v̇ = Σ s_i f_i - m g e₃, J ω̇ + ω × J ω =
 * Rᵀ Σ s_i (p_i - p) × f_i, Ṫ = T [ω; Rᵀ v]. Euler step over dt_k:
 *
 * @verbatim
 *   residual[0:6]  = Log((T_k Exp(dt [ω_k; R_kᵀ v_k]))⁻¹ T_{k+1})
 *   residual[6:9]  = v_{k+1} - v_k - dt (Σ s_i f_i / m - g e₃)
 *   residual[9:12] = ω_{k+1} - ω_k - dt J⁻¹ (R_kᵀ Σ s_i (p_i - p_k) × f_i - ω_k × J ω_k)
 * @endverbatim
 *
 * Analytic Jacobians. Leg masses and joint torque limits are outside this
 * model. Contact constraints (unilateral normal force, friction cone, zero
 * force in swing) are separate constraint factors.
 */
class QuadrupedFactorBatch : public SizedFactorBatch<12, 6, 3, 3, 12, 6, 3, 3> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param contacts Device array of 4 contact flags per factor.
   * @param foot_positions Device array of 12 floats per factor: the world
   *        positions of feet 0..3.
   * @param parameters Mass, inertia, gravity.
   * @param capacity Number of factors (steps) the buffers hold. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if a buffer is null or mass, inertia are not positive.
   */
  QuadrupedFactorBatch(const float *time_steps, const float *contacts, const float *foot_positions,
                       const QuadrupedParameters &parameters, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

  const QuadrupedParameters &Parameters() const { return parameters_; }

 private:
  const float *time_steps_;
  const float *contacts_;
  const float *foot_positions_;
  QuadrupedParameters parameters_;
};

}  // namespace cunls
