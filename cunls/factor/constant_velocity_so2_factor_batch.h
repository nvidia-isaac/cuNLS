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
 * @brief Batch factor for a constant angular-velocity motion prior on SO(2).
 *
 * See ConstantVelocitySE3FactorBatch for the full derivation. SO(2) is
 * abelian, so `J_l = J_r = 1` and the residual/Jacobian reduce to scalar
 * arithmetic (`theta := Log(pose_k^{-1} * pose_{k+1})`):
 *
 *     r_pose = theta - dt * vel_k
 *     r_vel  = vel_{k+1} - vel_k
 *
 * Inherits from SizedFactorBatch<2, 1, 1, 1, 1>.
 */
class ConstantVelocitySO2FactorBatch : public SizedFactorBatch<2, 1, 1, 1, 1> {
 public:
  ConstantVelocitySO2FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantVelocitySO2FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  mutable DeviceVector<Matrix<2>> pose_rel_;  ///< R_k^T * R_{k+1}
  mutable DeviceVector<float> twist_;         ///< Log(pose_rel)
};

}  // namespace cunls
