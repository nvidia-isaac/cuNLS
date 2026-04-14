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

#include <sstream>
#include <type_traits>
#include <utility>

#include "cunls/common/log.h"
#include "cunls/robustifier/loss_function_batch.h"

namespace cunls {

/**
 * @brief Scales the output of a batched loss evaluation in-place on the GPU.
 *
 * Multiplies every element of the float3 output array by a scalar:
 * out[i] = {a * out[i].x, a * out[i].y, a * out[i].z}.
 *
 * @param a          Positive scale factor.
 * @param out        Device pointer to float3 triplets to scale in-place.
 * @param num_losses Number of triplets.
 * @param stream     CUDA stream for asynchronous execution.
 */
void ApplyScaling(float a, float3 *out, int num_losses, cudaStream_t stream);

/**
 * @brief Wrapper loss function that scales the output of another loss function.
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
 * @tparam T The wrapped loss function type, must derive from LossFunctionBatch.
 *
 * @note The inner loss function is owned by value and constructed from
 *       forwarded arguments, following the same pattern as
 *       InformationFactorBatch.
 */
template <class T, typename std::enable_if_t<
                       std::is_base_of_v<LossFunctionBatch, T>, int> = 0>
class ScaledLossFunctionBatch : public LossFunctionBatch {
public:
  /**
   * @brief Constructs a ScaledLossFunctionBatch wrapper.
   *
   * @param a Positive scale factor applied to all loss outputs. Must be > 0.
   * @param loss_args Arguments forwarded to the wrapped loss function
   *                  constructor (e.g. pass ``delta`` when ``T`` is
   *                  ``HuberLossFunctionBatch``).
   * @throws std::invalid_argument if a <= 0.
   */
  template <class... Args>
  ScaledLossFunctionBatch(float a, Args &&...loss_args)
      : a_(a), loss_function_(std::forward<Args>(loss_args)...) {
    if (a_ <= 0.0f) {
      std::stringstream ss;
      ss << "ScaledLossFunctionBatch: scale factor a (" << a_
         << ") must be positive";
      LogError(ss.str());
      throw std::invalid_argument(ss.str());
    }
  }

  /**
   * @brief Evaluates the scaled loss for a batch of squared residuals.
   *
   * First evaluates the wrapped loss function, then scales all output
   * components (rho, rho', rho'') by the scale factor a.
   *
   * @copydetails LossFunctionBatch::Evaluate
   */
  bool Evaluate(float *s, float3 *out, int num_losses,
                cudaStream_t stream) const override {
    if (num_losses <= 0) {
      return true;
    }
    loss_function_.Evaluate(s, out, num_losses, stream);
    ApplyScaling(a_, out, num_losses, stream);
    return true;
  }

private:
  T loss_function_; ///< Wrapped loss function, owned by value.
  float a_;         ///< Positive scale factor.
};

} // namespace cunls
