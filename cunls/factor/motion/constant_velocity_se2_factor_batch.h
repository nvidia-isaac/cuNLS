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

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for a constant body-velocity motion prior on SE(2).
 *
 * See ConstantVelocitySE3FactorBatch for the full derivation; this is the
 * same construction specialized to SE(2) (`J_l^{-1}(x) = J_r^{-1}(-x)`).
 *
 * Inherits from SizedFactorBatch<6, 3, 3, 3, 3>.
 */
class ConstantVelocitySE2FactorBatch : public SizedFactorBatch<6, 3, 3, 3, 3> {
 public:
  ConstantVelocitySE2FactorBatch(const float *dt_ptr, size_t capacity);

  /**
   * @brief Evaluates residuals and (optionally) Jacobians. Follows
   * FactorBatch::Evaluate's item contract: item t reads the time delta of
   * factor `factor_ids[t]` (or `t % NumActiveFactors()`) and the states of item t.
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *dt_ptr_;  ///< Per-factor time deltas (user memory).
};

}  // namespace cunls
