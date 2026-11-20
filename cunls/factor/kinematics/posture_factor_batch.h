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

#include "cunls/factor/kinematics/articulation_factor_batch.h"

namespace cunls {

/**
 * @brief The articulation's coordinates should stay near a nominal
 * configuration q*: a regularizer, a null-space posture for redundant robots,
 * or "stay close to the requested state" when projecting a state onto
 * constraints.
 *
 * @verbatim
 *   r = W (q ⊟ q*)                                  DofCount() rows
 * @endverbatim
 *
 * Per joint, in DOF order: scalar DOFs q - q*; BALL Log(R(quat) R(quat*)ᵀ);
 * FREE (p - p*, Log(R R*ᵀ)); FIXED none. Rotation residuals are world-frame
 * rotation vectors in the joint's parent-anchor frame; their Jacobian block
 * is J_l⁻¹(r) (the state's left perturbation), and the others are identity.
 *
 * Data per factor (user-owned device arrays):
 * - targets: float, CoordCount() per factor: q* in joint_q layout;
 * - weights: float, DofCount() per factor, or nullptr for 1 (0 frees a DOF).
 */
class PostureFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @throws std::invalid_argument if targets is null.
   */
  PostureFactorBatch(const ArticulationTopology &topology, const float *targets,
                     const float *weights, size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *targets_, *weights_;
};

}  // namespace cunls
