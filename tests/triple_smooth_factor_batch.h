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

#include <cstddef>

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {
namespace test_utils {

/**
 * @brief Test copy of cuSFM's triple smooth factor (cuSFM
 * `src/core/bundle_adjustment/cunls_triple_smooth_factor.cu`), so that the
 * synthetic cuSFM-like bundle adjustment tests build the same factor graph as
 * the cuSFM Mapper.
 *
 * The middle one of three timestamp-adjacent world_from_rig poses has to stay
 * on the SE(3) interpolation of the outer two. With the interpolation ratio
 * r = (t_1 - t_0) / (t_2 - t_0):
 *
 *     xi = Log(T_0^-1 T_2),   E = T_1^-1 T_0 Exp(r xi),
 *     residual = [Log_SO3(rotation(E)); translation(E)].
 *
 * The Jacobians with respect to [T_0, T_1, T_2] (right perturbations, tangent
 * [rotation; translation]) are central differences of the residual in double
 * precision, as in cuSFM. Wrap it in InformationFactorBatch for a 6x6
 * square-root information matrix.
 */
class TripleSmoothFactorBatch : public SizedFactorBatch<6, 6, 6, 6> {
 public:
  /**
   * @brief Constructs the batch.
   *
   * @param ratios Device pointer to the per-factor interpolation ratios, in
   *        (0, 1); at least `capacity` floats, not owned.
   * @param capacity Number of factors the ratio buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors().
   */
  TripleSmoothFactorBatch(const float *ratios, size_t capacity);

  /** @brief Residuals (6 per item) and optional 6x18 Jacobians; see FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *ratios_;  ///< Device per-factor interpolation ratios (not owned).
};

}  // namespace test_utils
}  // namespace cunls
