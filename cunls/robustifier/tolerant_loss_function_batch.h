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

#include "cunls/robustifier/loss_function_batch.h"
#include <cuda_runtime.h>

namespace cunls {

/**
 * @brief GPU-batched Tolerant loss function.
 *
 * For squared residual s, let x = (s - a) / b. Then:
 * For x > threshold: rho(s) = s - a - c, rho'(s) = 1, rho''(s) = 0.
 * Otherwise: rho(s) = b * log(1 + exp(x)) - c,
 *            rho'(s) = exp(x) / (1 + exp(x)),
 *            rho''(s) = 0.5 / (b * (1 + cosh(x))).
 *
 * c = b * log(1 + exp(-a/b)) is computed in the constructor so that
 * rho(0) = 0. Requires a >= 0 and b > 0.
 */
class TolerantLossFunctionBatch : public LossFunctionBatch {
public:
  /**
   * @brief Constructs a Tolerant loss with the given parameters.
   * @param a Offset parameter (a >= 0).
   * @param b Scale parameter (b > 0).
   */
  TolerantLossFunctionBatch(float a, float b);

  /**
   * @brief Evaluates the Tolerant loss for a batch of squared residuals.
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float *s, float3 *out, int num_losses,
                cudaStream_t stream) const override;

private:
  float a_;
  float b_;
  float c_;
};

} // namespace cunls
