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
 * @brief A mimic (coupled) joint: one scalar DOF follows another,
 *
 * @verbatim
 *   c = q_m - (a + b q_ref)                         1 row
 * @endverbatim
 *
 * as in Newton's joint_mimic_joint / joint_mimic_coeffs and URDF `<mimic>`
 * (offset a, multiplier b). Usually a hard constraint: wrap it as
 * ConstraintFactorBatch(&factor, ConstraintKind::kEquality). Jacobian: 1 at
 * the mimic DOF, -b at the reference DOF.
 *
 * Data per factor (user-owned device arrays):
 * - mimic_dofs:     int,   1 per factor: the DOF m (a PRISMATIC, REVOLUTE or D6 DOF);
 * - reference_dofs: int,   1 per factor: the DOF ref (scalar as well);
 * - coefficients:   float, 2 per factor: (a, b).
 * A DOF of a BALL or FREE joint gives a NaN residual.
 */
class MimicJointFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @throws std::invalid_argument if any array is null.
   */
  MimicJointFactorBatch(const ArticulationTopology &topology, const int *mimic_dofs,
                        const int *reference_dofs, const float *coefficients, size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const int *mimic_dofs_, *reference_dofs_;
  const float *coefficients_;
};

}  // namespace cunls
