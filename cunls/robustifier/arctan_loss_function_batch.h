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
 * @brief GPU-batched Arctan loss function.
 *
 * For squared residual s:
 *   sum = 1 + s^2 * b
 *   rho(s) = a * atan(s / a)
 *   rho'(s) = 1 / sum
 *   rho''(s) = -2 * s * b / sum^2
 *
 * The Arctan loss grows sublinearly for large residuals and is smooth.
 */
class ArctanLossFunctionBatch : public LossFunctionBatch {
 public:
  /**
   * @brief Constructs an Arctan loss with the given parameters.
   * @param a Scale parameter (a > 0).
   * @param b Scale parameter (b > 0).
   */
  ArctanLossFunctionBatch(float a, float b);

  /**
   * @brief Evaluates the Arctan loss for a batch of squared residuals.
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float* s, float3* out, int num_losses,
                cudaStream_t stream) const override;

 private:
  float a_;
  float b_;
};

}  // namespace cunls
