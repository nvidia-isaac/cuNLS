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
 * @brief GPU-batched trivial (identity) loss function.
 *
 * The trivial loss applies no robustification: rho(s) = s, rho'(s) = 1,
 * rho''(s) = 0. This is equivalent to standard least-squares without any
 * outlier down-weighting.
 */
class TrivialLossFunctionBatch : public LossFunctionBatch {
 public:
  /**
   * @brief Evaluates the identity loss for a batch of squared residuals.
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float *s, float3 *out, int num_losses,
                cudaStream_t stream) const override;
};
}  // namespace cunls
