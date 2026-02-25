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
 * @brief GPU-batched Huber loss function.
 *
 * The Huber loss is a robust loss function that is quadratic for small residuals
 * (inlier region, s <= delta^2) and linear for large residuals (outlier region,
 * s > delta^2). This provides robustness against outliers while preserving
 * the efficiency of least-squares for inliers.
 *
 * For squared error s:
 *   - Inlier (s <= delta^2):  rho(s) = s,  rho'(s) = 1,  rho''(s) = 0
 *   - Outlier (s > delta^2):  rho(s) = 2*delta*sqrt(s) - delta^2,
 *                              rho'(s) = delta / sqrt(s),
 *                              rho''(s) = -delta / (2*s^(3/2))
 */
class HuberLossFunctionBatch : public LossFunctionBatch {
 public:
  /**
   * @brief Constructs a Huber loss function with the given threshold.
   * @param delta The Huber threshold. Residuals with sqrt(s) > delta are
   *              treated as outliers and down-weighted.
   */
  HuberLossFunctionBatch(float delta);

  /**
   * @brief Evaluates the Huber loss for a batch of squared residuals.
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float *s, float3 *out, int num_losses,
                cudaStream_t stream) const override;

 private:
  float delta_;  ///< Huber threshold separating inlier and outlier regions.
};
}  // namespace cunls
