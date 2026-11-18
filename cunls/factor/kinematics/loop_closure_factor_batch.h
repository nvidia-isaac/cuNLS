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
 * @brief A loop-closing joint between two links of one articulation instance
 * (or a link and the world): the constraint rows that keep its two joint
 * frames together in every direction the joint does not allow. Wrap it as
 * ConstraintFactorBatch(&factor, ConstraintKind::kEquality).
 *
 * With F_P = X_P(q) X_p (X_P: the parent link's world pose, the identity for
 * parent -1) and F_C = X_C(q) X_c, Δp = p(F_C) - p(F_P), and the joint axis
 * a with two unit vectors b₁, b₂ ⊥ a, all in the joint frame:
 *
 * @verbatim
 *   BALL       3 rows: Δp
 *   REVOLUTE   5 rows: Δp;  (R_FP b₁)·(R_FC a),  (R_FP b₂)·(R_FC a)        (axes stay aligned)
 *   PRISMATIC  5 rows: Log(R_FC R_FPᵀ);  (R_FP b₁)·Δp,  (R_FP b₂)·Δp      (no motion off the axis)
 *   FIXED      6 rows: Δp;  Log(R_FC R_FPᵀ)
 * @endverbatim
 *
 * Positions in meters, rotations in radians. The Jacobian takes the motion
 * columns of the child branch with + and of the parent branch with -; DOFs
 * both branches share cancel. Rows may be redundant (a planar four-bar closed
 * by a REVOLUTE joint has 5 rows of rank 2); the augmented Lagrangian
 * tolerates that.
 *
 * Data per factor (user-owned device arrays):
 * - parent_links: int,   1 per factor: link P, or -1 for the world;
 * - child_links:  int,   1 per factor: link C;
 * - frames:       float, 14 per factor: X_p (position, quaternion x, y, z, w) then X_c;
 * - axes:         float, 3 per factor: the unit joint axis a (REVOLUTE, PRISMATIC; may be
 *                 nullptr for BALL and FIXED).
 */
class LoopClosureFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @param joint_type BALL, REVOLUTE, PRISMATIC or FIXED: every factor of the batch.
   * @throws std::invalid_argument for another joint type, or a null array
   *         (axes may be null only for BALL and FIXED).
   */
  LoopClosureFactorBatch(const ArticulationTopology &topology, ArticulationJointType joint_type,
                         const int *parent_links, const int *child_links, const float *frames,
                         const float *axes, size_t capacity);

  /** @brief The loop-closing joint type of every factor. */
  ArticulationJointType JointType() const { return joint_type_; }

  /** @brief Rows of one factor for `joint_type` (3, 5, 5, 6); throws for other types. */
  static size_t RowsOf(ArticulationJointType joint_type);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  ArticulationJointType joint_type_;
  const int *parent_links_, *child_links_;
  const float *frames_, *axes_;
};

}  // namespace cunls
