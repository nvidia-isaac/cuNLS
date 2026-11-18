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
 * @brief World position of a point on a link should equal a target:
 *
 * @verbatim
 *   r = W (X_L(q) c - p*)                          3 rows
 * @endverbatim
 *
 * X_L(q): world pose of link L of the articulation instance; c: the point in
 * the link frame; p*: the world target; W = diag(weights). Jacobian column of
 * DOF j: W (ω_j × x + v_j) for the DOFs that move L (x = X_L c), 0 otherwise.
 * Newton: IKObjectivePosition.
 *
 * Data per factor (user-owned device arrays, read at every evaluation, not
 * copied; factor f at the offsets shown):
 * - links:   int,   1 per factor: the link L (0 .. LinkCount()-1);
 * - points:  float, 3 per factor: c in the link frame;
 * - targets: float, 3 per factor: p* in the world frame;
 * - weights: float, 3 per factor, or nullptr for 1.
 */
class LinkPositionFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @throws std::invalid_argument if links, points or targets is null.
   */
  LinkPositionFactorBatch(const ArticulationTopology &topology, const int *links,
                          const float *points, const float *targets, const float *weights,
                          size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const int *links_;
  const float *points_, *targets_, *weights_;
};

/**
 * @brief World orientation of a frame on a link should equal a target:
 *
 * @verbatim
 *   r = W Log(R_L(q) R_off R*ᵀ)                    3 rows (rotation vector, world frame)
 * @endverbatim
 *
 * R_L(q): world rotation of link L; R_off: the frame's rotation in the link
 * frame; R*: the world target. Jacobian column of DOF j: W J_l⁻¹(r) ω_j for the
 * DOFs that move L. Newton: IKObjectiveRotation.
 *
 * Data per factor (user-owned device arrays):
 * - links:     int,   1 per factor;
 * - offsets:   float, 4 per factor: R_off as a quaternion (x, y, z, w), or nullptr for identity;
 * - targets:   float, 4 per factor: R* as a quaternion (x, y, z, w);
 * - weights:   float, 3 per factor, or nullptr for 1.
 */
class LinkOrientationFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @throws std::invalid_argument if links or targets is null.
   */
  LinkOrientationFactorBatch(const ArticulationTopology &topology, const int *links,
                             const float *offsets, const float *targets, const float *weights,
                             size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const int *links_;
  const float *offsets_, *targets_, *weights_;
};

/**
 * @brief World pose of a frame on a link should equal a target: the rows of
 * LinkPositionFactorBatch (the frame's origin) followed by those of
 * LinkOrientationFactorBatch, in one pass over the link's cache:
 *
 * @verbatim
 *   r = W [ X_L(q) p_off - p* ;  Log(R_L(q) R_off R*ᵀ) ]      6 rows
 * @endverbatim
 *
 * Data per factor (user-owned device arrays):
 * - links:   int,   1 per factor;
 * - offsets: float, 7 per factor: the frame in the link frame, (p_off, quaternion x, y, z, w),
 *            or nullptr for the link frame itself;
 * - targets: float, 7 per factor: (p*, quaternion R*) in the world frame;
 * - weights: float, 6 per factor, or nullptr for 1.
 */
class LinkPoseFactorBatch : public ArticulationFactorBatch {
 public:
  /**
   * @throws std::invalid_argument if links or targets is null.
   */
  LinkPoseFactorBatch(const ArticulationTopology &topology, const int *links, const float *offsets,
                      const float *targets, const float *weights, size_t capacity);

  /** @brief See FactorBatch::Evaluate. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const int *links_;
  const float *offsets_, *targets_, *weights_;
};

}  // namespace cunls
