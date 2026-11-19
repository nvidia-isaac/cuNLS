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
#include <vector>

#include "cunls/common/types.h"
#include "cunls/factor/kinematics/articulation_factor_batch.h"

namespace cunls {

/**
 * @brief Puts the SE(3) pose states of any factor on a robot: chosen pose
 * slots of a wrapped factor receive the world pose of a frame fixed on a link
 * of an articulation instance, and the wrapped factor's Jacobian is carried to
 * the instance's joint DOFs. A custom kinematic objective is then an ordinary
 * factor over SE(3) poses that knows nothing about robots.
 *
 * <b>Terms</b>
 *
 * - <b>Inner factor</b>: the wrapped factor (a built-in SE(3) factor such as
 *   SE3PriorFactorBatch, SE3BetweenFactorBatch, SE3SphereClearanceFactorBatch,
 *   or a custom one: C++, CustomFactorBatch, pycunls.warp.WarpFactorBatch).
 * - <b>State slot</b> k of the inner factor: its k-th state, in the order of
 *   its StateSizes() (the order of the pointers it is connected with).
 * - <b>Pose slots</b> (`pose_slots`): the inner state slots that do not
 *   connect to a state of the problem but receive a link frame pose instead.
 *   Each must be an SE(3) slot (tangent 6) in SE3StateBatch's convention: a
 *   row-major 4x4 world_from_body matrix, tangent [ω, v] in the body frame,
 *   update X' = X Exp(δ). Write the inner factor exactly as for SE3 states.
 * - <b>Topology link</b>: a link index of the articulation topology,
 *   0 .. LinkCount()-1 (pycunls: the position in `link_names`).
 * - <b>Frame offset</b>: where the frame sits on its link, in the link frame
 *   (position, quaternion x, y, z, w; e.g. a camera mount or a tool tip).
 *
 * <b>Example</b> (nP = pose_slots.size() = 1): a look-at factor over one
 * SE(3) pose, put on a wrist camera, one factor per articulation instance:
 *
 * @verbatim
 *   inner factor state slots:   [0: camera pose (SE3, 6)]
 *   pose_slots      = {0}       slot 0 receives a link frame pose
 *   topology_links  = [wrist, wrist, ...]           1 int per factor (nP per factor)
 *   frame_offsets   = [camera mount, ...]           7 floats per factor (7 nP per factor)
 *   this batch's state slots:   [0: articulation instance (DofCount())]
 * @endverbatim
 *
 * With an object pose as a second inner slot (pose_slots = {0}, inner slots
 * [0: link frame, 1: object (SE3, 6)]), this batch's slots are
 * [0: articulation instance, 1: object] and the object stays a state of the
 * problem, solved together with the joints.
 *
 * <b>Per factor f and pose slot s</b> (s = position in pose_slots):
 *
 * @verbatim
 *   L   = topology_links[f nP + s]
 *   F_s = X_L(q) X_off            X_off = frame_offsets[(f nP + s) 7 ..] (identity when null)
 * @endverbatim
 *
 * with X_L(q) the world pose of link L of the instance in slot 0, read from
 * its forward-kinematics cache. Several pose slots of one factor are frames
 * on the same instance (e.g. the relative pose of two links).
 *
 * <b>States of this batch</b>: slot 0 is the articulation instance
 * (TangentSize() = the topology's DOF count), followed by the inner factor's
 * slots that are not pose slots, in their order.
 *
 * <b>Jacobian</b>: for DOF j with world motion column (ω_j, v_j) (see
 * ArticulationTopology), the body twist of F_s = (R, p) per unit of q_j is
 * b_s,j = (Rᵀω_j, Rᵀ(ω_j × p + v_j)), so
 *
 * @verbatim
 *   ∂r/∂q_j = Σ_s [DOF j moves link L_s] (∂r/∂δ_s) b_s,j
 * @endverbatim
 *
 * and the columns of the other slots are those of the inner factor.
 *
 * <b>Evaluation</b>: three steps on the stream, per item: write each F_s
 * into a scratch pose and build the inner factor's state pointers, evaluate
 * the inner factor (its residuals are this batch's), then carry its Jacobian
 * to the DOFs. The scratch is sized for the largest item count seen
 * (allocated on the first evaluation and when that count grows). A topology
 * link outside 0 .. LinkCount()-1 gives a NaN pose (so, usually, a NaN
 * residual) and zero DOF columns.
 *
 * <b>Sizes</b>: ResidualsSize(), the active count and the capacity are the
 * inner factor's (SetNumActiveFactors() sets the inner one). `topology_links`
 * and `frame_offsets` are read at every evaluation and may be rewritten in
 * place between solves, like any factor data. For constraints, wrap this
 * batch in ConstraintFactorBatch (e.g. clearance as an inequality), never the
 * inner factor.
 */
class LinkFrameFactorBatch : public ArticulationFactorBatch {
 public:
  /** @brief Most state slots of the inner factor (the problem's slot limit). */
  static constexpr int kMaxInnerSlots = 8;

  /**
   * @param topology Kinematic tree of the articulation state (must outlive the batch).
   * @param inner Factor over SE(3) poses (not owned; must outlive the batch).
   * @param pose_slots State slots of `inner` that receive a link frame pose:
   *        distinct, each of tangent size 6 (an SE(3) pose); nP =
   *        pose_slots.size() >= 1.
   * @param topology_links Device array of nP ints per factor: the topology
   *        link carrying the frame of each pose slot (factor f, pose slot s at
   *        f nP + s).
   * @param frame_offsets Device array of 7 nP floats per factor: the frame of
   *        each pose slot in its link frame (position, quaternion x, y, z, w),
   *        or nullptr for the link frames themselves.
   * @throws std::invalid_argument if inner or topology_links is null, inner is
   *         a constraint batch (wrap this batch in ConstraintFactorBatch
   *         instead), `pose_slots` is empty, repeats a slot, names a slot
   *         `inner` does not have or one whose tangent size is not 6, or
   *         `inner` has more than kMaxInnerSlots slots.
   */
  LinkFrameFactorBatch(const ArticulationTopology &topology, FactorBatch *inner,
                       const std::vector<int> &pose_slots, const int *topology_links,
                       const float *frame_offsets = nullptr);

  /** @brief The wrapped (inner) factor. */
  FactorBatch *Inner() const { return inner_; }

  /** @brief State slots of the inner factor that receive a link frame pose. */
  const std::vector<int> &PoseSlots() const { return pose_slots_; }

  /** @brief The articulation instance, then the inner factor's slots that are not pose slots. */
  std::vector<size_t> StateSizes() const override;

  /** @brief The inner factor's active count. */
  size_t NumActiveFactors() const override { return inner_->NumActiveFactors(); }

  /** @brief The inner factor's capacity. */
  size_t Capacity() const override { return inner_->Capacity(); }

  /** @brief Sets the inner factor's active count. */
  void SetNumActiveFactors(size_t num_active_factors) override {
    inner_->SetNumActiveFactors(num_active_factors);
  }

  /** @brief See FactorBatch::Evaluate and the class comment. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  FactorBatch *inner_;
  std::vector<int> pose_slots_;
  const int *topology_links_;
  const float *frame_offsets_;
  mutable dvector<float> poses_;             ///< 16 floats per item and pose slot: F_s.
  mutable dvector<float *> inner_pointers_;  ///< Inner state pointers, item-major.
  mutable dvector<float> inner_jacobians_;   ///< The inner factor's Jacobians.
};

}  // namespace cunls
