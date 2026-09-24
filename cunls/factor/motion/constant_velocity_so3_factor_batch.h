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
 * @brief Batch factor for a constant angular-velocity motion prior on SO(3).
 *
 * See ConstantVelocitySE3FactorBatch for the full derivation; this is the
 * same construction specialized to SO(3), where "velocity" is angular
 * velocity in `R^3` (`twist := Log(pose_k^{-1} * pose_{k+1})`,
 * `Jl_inv := J_l^{-1}(twist)`):
 *
 *     r_pose = twist - dt * vel_k
 *     r_vel  = Jl_inv * vel_{k+1} - vel_k
 *
 * Inherits from SizedFactorBatch<6, 3, 3, 3, 3>.
 */
class ConstantVelocitySO3FactorBatch : public SizedFactorBatch<6, 3, 3, 3, 3> {
 public:
  /**
   * @param dt_ptr Device pointer to per-factor time deltas, not owned.
   * @param num_factors Number of factors in the batch.
   */
  ConstantVelocitySO3FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantVelocitySO3FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  mutable DeviceVector<Matrix<3>> pose_rel_;  ///< R_k^T * R_{k+1}
  mutable DeviceVector<Vector<3>> twist_;     ///< Log(pose_rel)
  mutable DeviceVector<Matrix<3>> jl_inv_;    ///< J_l^{-1}(twist)
  mutable DeviceVector<Matrix<3>> jr_inv_;    ///< J_r^{-1}(twist), jacobians only
};

}  // namespace cunls
