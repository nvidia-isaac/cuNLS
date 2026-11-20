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

#include "cunls/robustifier/loss_function_batch.h"

namespace cunls {

/**
 * @brief Loss function that scales the output of another loss function.
 *
 * Given a loss function f(s) and a scalar a > 0, ScaledLossFunctionBatch
 * implements:
 *   rho(s)   = a * f(s)
 *   rho'(s)  = a * f'(s)
 *   rho''(s) = a * f''(s)
 *
 * This is useful for weighting different error terms differently (e.g.,
 * pixel reprojection errors vs. terrain errors).
 *
 * The scaled loss is referenced, not owned: it must outlive this object.
 */
class ScaledLossFunctionBatch : public LossFunctionBatch {
 public:
  /**
   * @brief Scales @p loss_function by @p a.
   *
   * @param loss_function The scaled loss (not owned; must outlive this object).
   * @param a Positive scale factor applied to all loss outputs.
   * @throws std::invalid_argument if a <= 0.
   */
  ScaledLossFunctionBatch(const LossFunctionBatch &loss_function, float a);

  /**
   * @brief Evaluates the scaled loss for a batch of squared residuals: the
   * wrapped loss, then every output component (rho, rho', rho'') times a.
   *
   * @copydetails LossFunctionBatch::Evaluate
   */
  void Evaluate(float *s, float3 *out, int num_losses, cudaStream_t stream) const override;

 private:
  const LossFunctionBatch &loss_function_;  ///< The scaled loss (not owned).
  float a_;                                 ///< Positive scale factor.
};

}  // namespace cunls
