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
 * @brief Batch factor for a constant angular-acceleration motion prior on
 * SO(3). See ConstantAccelerationSE3FactorBatch for the full derivation;
 * this is the same construction specialized to SO(3).
 *
 * Inherits from SizedFactorBatch<9, 3, 3, 3, 3, 3, 3>.
 */
class ConstantAccelerationSO3FactorBatch : public SizedFactorBatch<9, 3, 3, 3, 3, 3, 3> {
 public:
  ConstantAccelerationSO3FactorBatch(const float *dt_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  ConstantAccelerationSO3FactorBatch() = delete;

  const float *dt_ptr_;
  size_t num_factors_;

  mutable DeviceVector<Matrix<3>> pose_rel_;
  mutable DeviceVector<Vector<3>> twist_;
  mutable DeviceVector<Matrix<3>> jl_inv_;
  mutable DeviceVector<Matrix<3>> jr_inv_;
};

}  // namespace cunls
