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
 * @brief Batch factor for a constant body-acceleration motion prior on SE(3).
 *
 * Connects `pose_k`, `pose_{k+1}`, `vel_k`, `vel_{k+1}`, `accel_k`,
 * `accel_{k+1}`, each velocity/acceleration in `R^6` matching the SE(3)
 * tangent layout. Residual (`twist := Log(pose_k^{-1} * pose_{k+1})`,
 * `Jl_inv := J_l^{-1}(twist)`):
 *
 *     r_pose  = twist - dt * vel_k - 0.5 * dt^2 * accel_k
 *     r_vel   = Jl_inv * vel_{k+1} - vel_k - dt * accel_k
 *     r_accel = Jl_inv * accel_{k+1} - accel_k
 *
 * i.e. a second-order (constant-acceleration) Taylor prediction of the
 * relative pose, with velocity and acceleration transported back into the
 * local frame at `k` through the inverse left Jacobian, exactly as
 * ConstantVelocitySE3FactorBatch does for its single velocity block. The
 * pose-block Jacobians of `r_vel` and `r_accel` are treated as zero (a
 * documented simplification; the residuals themselves are exact).
 *
 * Inherits from SizedFactorBatch<18, 6, 6, 6, 6, 6, 6>.
 */
class ConstantAccelerationSE3FactorBatch : public SizedFactorBatch<18, 6, 6, 6, 6, 6, 6> {
 public:
  /**
   * @param dt_ptr Device pointer to per-factor time deltas, not owned.
   * @param num_factors Number of factors in the batch.
   */
  ConstantAccelerationSE3FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantAccelerationSE3FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  mutable DeviceVector<SE3Transform> pose_rel_;
  mutable DeviceVector<Vector<6>> twist_;
  mutable DeviceVector<Matrix<6>> jl_inv_;
  mutable DeviceVector<Matrix<6>> jr_inv_;
};

}  // namespace cunls
