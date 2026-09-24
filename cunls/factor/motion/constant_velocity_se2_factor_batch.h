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
 * @brief Batch factor for a constant body-velocity motion prior on SE(2).
 *
 * See ConstantVelocitySE3FactorBatch for the full derivation; this is the
 * same construction specialized to SE(2). `J_l^{-1}` is obtained from the
 * identity `J_l^{-1}(x) = J_r^{-1}(-x)` (only `ComputeJacobianRightInverseSE2`
 * exists in the math library; this avoids adding a new primitive).
 *
 * Inherits from SizedFactorBatch<6, 3, 3, 3, 3>.
 */
class ConstantVelocitySE2FactorBatch : public SizedFactorBatch<6, 3, 3, 3, 3> {
 public:
  ConstantVelocitySE2FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantVelocitySE2FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  mutable DeviceVector<Matrix<3>> pose_rel_;   ///< T_k^{-1} * T_{k+1}
  mutable DeviceVector<Vector<3>> twist_;      ///< Log(pose_rel)
  mutable DeviceVector<Vector<3>> neg_twist_;  ///< -twist
  mutable DeviceVector<Matrix<3>> jl_inv_;     ///< J_r^{-1}(-twist) == J_l^{-1}(twist)
  mutable DeviceVector<Matrix<3>> jr_inv_;     ///< J_r^{-1}(twist), jacobians only
};

}  // namespace cunls
