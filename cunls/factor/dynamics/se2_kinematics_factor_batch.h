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
 * @brief SE(2) kinematics with the body twist as control:
 * X_{k+1} = X_k Exp(dt_k ξ_k).
 *
 * States (in order): X_k (SE2StateBatch), body twist ξ_k
 * (VectorStateBatch<3>, [v_x, v_y, ω]), X_{k+1} (SE2StateBatch).
 *
 * @verbatim
 *   residual = Log((X_k Exp(dt_k ξ_k))⁻¹ X_{k+1})          (3)
 * @endverbatim
 *
 * exact for a twist held over the step. Analytic Jacobians: -J_l⁻¹(r)
 * Ad(Exp(-dt ξ)) (X_k), -dt J_l⁻¹(r) J_r(dt ξ) (ξ_k), J_r⁻¹(r) (X_{k+1}).
 * Planar camera / robot motion with commanded body velocities.
 */
class SE2KinematicsFactorBatch : public SizedFactorBatch<3, 3, 3, 3> {
 public:
  /**
   * @param time_steps Device array of `capacity` step durations dt_k [s].
   * @param capacity Number of factors (steps) the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if time_steps is null.
   */
  SE2KinematicsFactorBatch(const float *time_steps, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *time_steps_;
};

}  // namespace cunls
