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

/**
 * @brief Two SE(3) poses keep one local axis aligned: the rows of a hinge
 * (revolute) joint that a position equality does not cover.
 *
 * @verbatim
 *   r = R_1 a - R_0 a                                3 rows (world frame)
 * @endverbatim
 *
 * with X_0 = (R_0, p_0), X_1 = (R_1, p_1) the two poses (SE3StateBatch, or
 * frame views of LinkFrameFactorBatch) and a the unit axis in each pose's
 * frame. r = 0 iff the axes point the same way (unlike projections onto
 * vectors perpendicular to a, which also vanish for opposite axes). The rows
 * have rank 2 (both terms are unit vectors). Usually an equality constraint:
 * wrap it (or the LinkFrameFactorBatch around it) in
 * ConstraintFactorBatch(..., ConstraintKind::kEquality). A revolute loop
 * closure is this plus a position equality of the two frames' origins.
 *
 * Jacobian (tangent [ω, v] per pose, right perturbation X Exp(δ)):
 * ∂r/∂ω_0 = R_0 [a]×, ∂r/∂ω_1 = -R_1 [a]×, zero for v.
 *
 * Data per factor (user-owned device array): axes, 3 floats per factor (the
 * axis a; normalized when read).
 */
class SE3AxisAlignmentFactorBatch : public SizedFactorBatch<3, 6, 6> {
 public:
  /**
   * @param axes Device array of 3 floats per factor: the axis a in the poses' frames.
   * @param capacity Factors the buffer holds (active count starts at 0).
   * @throws std::invalid_argument if axes is null.
   */
  SE3AxisAlignmentFactorBatch(const float *axes, size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *axes_;
};

/**
 * @brief The origin of SE(3) pose X_1 stays on the axis through pose X_0: the
 * rows of a slider (prismatic) joint that an orientation equality does not
 * cover.
 *
 * @verbatim
 *   r_k = (R_0 b_k) · (p_1 - p_0),   k = 1, 2        2 rows
 * @endverbatim
 *
 * with a the unit axis in X_0's frame and b_1, b_2 two unit vectors
 * perpendicular to it (b_1 = normalize(a × e), e the x axis, or the y axis
 * when |a_x| >= 0.9; b_2 = a × b_1). Usually an equality constraint (see
 * SE3AxisAlignmentFactorBatch); a prismatic loop closure is this plus an
 * orientation equality of the two frames.
 *
 * Jacobian (tangent [ω, v] per pose, right perturbation): with Δ = p_1 - p_0,
 * w_k = R_0 b_k: ∂r_k/∂ω_0 = -Δᵀ R_0 [b_k]×, ∂r_k/∂v_0 = -b_kᵀ,
 * ∂r_k/∂ω_1 = 0, ∂r_k/∂v_1 = w_kᵀ R_1.
 *
 * Data per factor (user-owned device array): axes, 3 floats per factor (the
 * axis a in X_0's frame; normalized when read).
 */
class SE3AxisOffsetFactorBatch : public SizedFactorBatch<2, 6, 6> {
 public:
  /**
   * @param axes Device array of 3 floats per factor: the axis a in X_0's frame.
   * @param capacity Factors the buffer holds (active count starts at 0).
   * @throws std::invalid_argument if axes is null.
   */
  SE3AxisOffsetFactorBatch(const float *axes, size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *axes_;
};

}  // namespace cunls
