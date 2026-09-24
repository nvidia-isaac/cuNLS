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
 * @brief Batch factor for a constant body-velocity motion prior on SE(3).
 *
 * Connects two consecutive poses and their body-frame velocities
 * (twists), `pose_k`, `pose_{k+1}`, `vel_k`, `vel_{k+1}`, each velocity in
 * `R^6` matching the SE(3) tangent layout `[angular; linear]`. Residual
 * (`twist := Log(pose_k^{-1} * pose_{k+1})`, `Jl_inv := J_l^{-1}(twist)`):
 *
 *     r_pose = twist - dt * vel_k
 *     r_vel  = Jl_inv * vel_{k+1} - vel_k
 *
 * i.e. the relative pose should equal `dt` times the velocity at `k`, and
 * the velocity at `k+1`, transported back into the local frame at `k`
 * through the inverse left Jacobian, should equal the velocity at `k`. See
 * `docs/design/motion_prior_factors.md` for the derivation and the
 * documented simplification used for the pose-block Jacobian of `r_vel`
 * (treated as zero; the residual itself is exact).
 *
 * Inherits from SizedFactorBatch<12, 6, 6, 6, 6>:
 *   - 12: residual dimension (6 pose + 6 velocity)
 *   - 6, 6: pose_k, pose_{k+1} tangent size (SE(3))
 *   - 6, 6: vel_k, vel_{k+1} size (R^6 body twist)
 */
class ConstantVelocitySE3FactorBatch : public SizedFactorBatch<12, 6, 6, 6, 6> {
 public:
  /**
   * @brief Constructs a batch of SE(3) constant-velocity factors.
   *
   * @param dt_ptr Device pointer to per-factor time deltas (t_{k+1} - t_k),
   *               at least num_factors floats. Not owned; must outlive this
   *               object.
   * @param num_factors Number of factors in the batch.
   */
  ConstantVelocitySE3FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantVelocitySE3FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  /// Scratch: T_k^{-1} * T_{k+1} per factor.
  mutable DeviceVector<SE3Transform> pose_rel_;
  /// Scratch: Log(pose_rel) per factor.
  mutable DeviceVector<Vector<6>> twist_;
  /// Scratch: J_l^{-1}(twist) per factor (row-major 6x6).
  mutable DeviceVector<Matrix<6>> jl_inv_;
  /// Scratch: J_r^{-1}(twist) per factor (row-major 6x6), only filled when
  /// Jacobians are requested.
  mutable DeviceVector<Matrix<6>> jr_inv_;
};

}  // namespace cunls
