/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "cunls/robustifier/loss_function_batch.h"

namespace cunls {

/**
 * @brief GPU-batched Tukey (biweight) loss function.
 *
 * For squared residual s and threshold a_squared = a^2:
 *   Inlier (s <= a_squared): value = 1 - s/a_squared,
 *     rho(s) = (a_squared/3) * (1 - value^3),
 *     rho'(s) = value^2,
 *     rho''(s) = -2 * value / a_squared
 *   Outlier (s > a_squared): rho(s) = a_squared/3, rho'(s) = 0, rho''(s) = 0
 *
 * Tukey completely rejects outliers (zero weight) beyond the threshold.
 */
class TukeyLossFunctionBatch : public LossFunctionBatch {
 public:
  /**
   * @brief Constructs a Tukey loss with the given threshold.
   * @param a Threshold parameter (a > 0). Residuals with s > a^2 are
   *          treated as outliers and given zero weight.
   */
  TukeyLossFunctionBatch(float a);

  /**
   * @brief Evaluates the Tukey loss for a batch of squared residuals.
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float* s, float3* out, int num_losses,
                cudaStream_t stream) const override;

 private:
  float a_squared_;  ///< a^2, threshold separating inlier and outlier regions.
};

}  // namespace cunls
