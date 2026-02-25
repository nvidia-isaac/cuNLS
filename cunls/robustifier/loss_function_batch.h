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

namespace cunls {

/**
 * @brief Abstract base class for GPU-batched robust loss (robustifier) functions.
 *
 * A loss function modifies the squared residual to reduce the influence of
 * outliers. Given squared error s, it computes a triplet rho = (rho(s), rho'(s), rho''(s))
 * stored as a float3, where:
 *   - rho.x = rho(s):   the robustified cost value
 *   - rho.y = rho'(s):  the first derivative w.r.t. s
 *   - rho.z = rho''(s): the second derivative w.r.t. s
 */
class LossFunctionBatch {
 public:
  /**
   * @brief Evaluates the loss function for a batch of squared residuals on the GPU.
   *
   * @param s          Device pointer to an array of squared residuals (input/output).
   * @param out        Device pointer to an array of float3 output triplets
   *                   (rho, rho', rho'') for each residual.
   * @param num_losses Number of residuals to evaluate.
   * @param stream     CUDA stream for asynchronous execution.
   * @return True on success, false on failure.
   */
  virtual bool Evaluate(float* s, float3* out, int num_losses,
                        cudaStream_t stream) const = 0;

  /** @brief Virtual destructor. */
  virtual ~LossFunctionBatch() = default;
};

}  // namespace cunls
