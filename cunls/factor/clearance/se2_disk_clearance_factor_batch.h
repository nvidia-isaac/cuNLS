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

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Clearance between the origin of an SE(2) pose and a disk
 * obstacle, as a constraint function:
 *
 * @verbatim
 *   c = (radius + margin) - ‖p - center‖        (feasible: c <= 0)
 * @endverbatim
 *
 * with p the pose translation. Wrap it as
 * ConstraintFactorBatch(&factor, ConstraintKind::kInequality); the robot's
 * own radius goes into `margin` (or the obstacle's radius). State: the pose
 * (SE2StateBatch). Jacobian: -(p - center)ᵀ R / ‖p - center‖ on the translation part
 * of the tangent, 0 on the rotation part (0 when p is at the center).
 */
class SE2DiskClearanceFactorBatch : public SizedFactorBatch<1, 3> {
 public:
  /**
   * @param obstacles Device array of 3 floats per factor: the center
   *        (x, y) and the radius [m].
   * @param margin Extra clearance added to every radius [m].
   * @param capacity Number of factors the buffer holds. The active count
   *        starts at 0: call SetNumActiveFactors(n) before solving.
   * @throws std::invalid_argument if obstacles is null.
   */
  SE2DiskClearanceFactorBatch(const float *obstacles, float margin, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *obstacles_;
  float margin_;
};

}  // namespace cunls
