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
 * @brief What a frame slot of LinkFrameFactorBatch hands its inner factor: the
 * pose of a frame on a link, its origin or its orientation, each in the layout
 * and tangent convention of an existing state type, so that the factors of that
 * state type work on robot links unchanged.
 */
enum class FrameView : int {
  /**
   * The frame's pose as an SE3StateBatch state: row-major 4x4 world_from_frame,
   * tangent [ω, v] in the frame (body), update X' = X Exp(δ). Slot tangent 6.
   */
  kPose = 0,
  /**
   * The frame's origin p in the world, as a VectorStateBatch<3> state (3 floats,
   * tangent: an additive world-frame step). Slot tangent 3.
   */
  kPosition = 1,
  /**
   * The frame's orientation R as an SO3StateBatch state: row-major 3x3
   * world_from_frame, tangent ω in the frame (body), update R' = R Exp(ω).
   * Slot tangent 3.
   */
  kOrientation = 2,
};

/** @brief One frame slot of LinkFrameFactorBatch: which inner state slot, fed with which view. */
struct LinkFrameSlot {
  int inner_slot = 0;                 ///< Position in the inner factor's StateSizes().
  FrameView view = FrameView::kPose;  ///< What the slot receives.
};

/**
 * @brief Puts the pose, position or orientation states of any factor on a
 * robot: chosen state slots of a wrapped factor (the frame slots) receive a
 * view of a frame fixed on a link of an articulation instance, and the wrapped
 * factor's Jacobian is carried to the instance's joint DOFs. A custom kinematic
 * objective is then an ordinary factor over SE(3) poses, 3D points or SO(3)
 * rotations that knows nothing about robots.
 *
 * <b>Terms</b>
 *
 * - <b>Inner factor</b>: the wrapped factor, built-in (SE3PriorFactorBatch,
 *   SE3BetweenFactorBatch, SE3SphereClearanceFactorBatch,
 *   PriorVectorFactorBatch<3>, HalfspaceFactorBatch<3>,
 *   VectorBetweenFactorBatch<3>, SO3PriorFactorBatch, SO3BetweenFactorBatch,
 *   ...) or custom (C++, CustomFactorBatch, pycunls.warp.WarpFactorBatch).
 * - <b>State slot</b> k of the inner factor: its k-th state, in the order of
 *   its StateSizes() (the order of the pointers it is connected with).
 * - <b>Frame slots</b> (`frame_slots`): the inner state slots that do not
 *   connect to a state of the problem but receive a view of a link frame, each
 *   with its FrameView (kPose: tangent 6; kPosition, kOrientation: tangent 3).
 *   The view is explicit because the tangent size alone is ambiguous (6: SE(3)
 *   or a 6-vector; 3: SO(3) or a 3-vector).
 * - <b>Topology link</b>: a link index of the articulation topology,
 *   0 .. LinkCount()-1 (pycunls: the position in `link_names`), or -1 for the
 *   world (identity pose, moved by no DOF: the frame is the frame offset itself,
 *   e.g. the fixed side of a loop closed to the world).
 * - <b>Frame offset</b>: where the frame sits on its link, in the link frame
 *   (position, quaternion x, y, z, w; e.g. a camera mount or a tool tip).
 *
 * <b>Example</b> (nF = frame_slots.size() = 1): a look-at factor over one
 * SE(3) pose, put on a wrist camera, one factor per articulation instance:
 *
 * @verbatim
 *   inner factor state slots:   [0: camera pose (SE3, 6)]
 *   frame_slots     = {{0, kPose}}                  slot 0 receives a link frame pose
 *   topology_links  = [wrist, wrist, ...]           1 int per factor (nF per factor)
 *   frame_offsets   = [camera mount, ...]           7 floats per factor (7 nF per factor)
 *   this batch's state slots:   [0: articulation instance (DofCount())]
 * @endverbatim
 *
 * "The tool tip stays above the table" is HalfspaceFactorBatch<3> with
 * frame_slots = {{0, kPosition}} and the tip as the frame offset, wrapped as
 * an inequality. With an object pose as a second inner slot (frame_slots =
 * {{0, kPose}}, inner slots [0: link frame, 1: object (SE3, 6)]), this
 * batch's slots are [0: articulation instance, 1: object] and the object stays
 * a state of the problem, solved together with the joints.
 *
 * <b>Per factor f and frame slot s</b> (s = position in frame_slots):
 *
 * @verbatim
 *   L   = topology_links[f nF + s]
 *   F_s = X_L(q) X_off = (R, p)    X_off = frame_offsets[(f nF + s) 7 ..] (identity when null)
 *   kPose: F_s (4x4)    kPosition: p    kOrientation: R (3x3)
 * @endverbatim
 *
 * with X_L(q) the world pose of link L of the instance in slot 0, read from
 * its forward-kinematics cache. Several frame slots of one factor are frames
 * on the same instance (e.g. the relative pose of two links).
 *
 * <b>States of this batch</b>: slot 0 is the articulation instance
 * (TangentSize() = the topology's DOF count), followed by the inner factor's
 * slots that are not frame slots, in their order.
 *
 * <b>Jacobian</b>: for DOF j with world motion column (ω_j, v_j) (see
 * ArticulationTopology), the frame moves per unit of q_j by
 *
 * @verbatim
 *   kPose:        b = (Rᵀω_j, Rᵀ(ω_j × p + v_j))     (body twist)
 *   kPosition:    b = ω_j × p + v_j                  (world velocity of p)
 *   kOrientation: b = Rᵀω_j                          (body angular velocity)
 *   ∂r/∂q_j = Σ_s [DOF j moves link L_s] (∂r/∂δ_s) b_s,j
 * @endverbatim
 *
 * and the columns of the other slots are those of the inner factor.
 *
 * <b>Evaluation</b>: three steps on the stream, per item: write each view
 * into scratch and build the inner factor's state pointers, evaluate the inner
 * factor (its residuals are this batch's), then carry its Jacobian to the
 * DOFs. The scratch is sized for the largest item count seen (allocated on the
 * first evaluation and when that count grows). A topology link outside
 * -1 .. LinkCount()-1 gives a NaN view (so, usually, a NaN residual) and zero
 * DOF columns.
 *
 * <b>Sizes</b>: ResidualsSize(), the active count and the capacity are the
 * inner factor's (SetNumActiveFactors() sets the inner one). `topology_links`
 * and `frame_offsets` are read at every evaluation and may be rewritten in
 * place between solves, like any factor data. For constraints, wrap this
 * batch in ConstraintFactorBatch (e.g. a halfspace as an inequality), never
 * the inner factor.
 */
class LinkFrameFactorBatch : public ArticulationFactorBatch {
 public:
  /** @brief Most state slots of the inner factor (the problem's slot limit). */
  static constexpr int kMaxInnerSlots = 8;

  /**
   * @param topology Kinematic tree of the articulation state (must outlive the batch).
   * @param inner Factor over the views (not owned; must outlive the batch).
   * @param frame_slots Inner state slots that receive a view of a link frame:
   *        distinct, each with the tangent size of its view (kPose 6,
   *        kPosition 3, kOrientation 3); nF = frame_slots.size() >= 1.
   * @param topology_links Device array of nF ints per factor: the topology
   *        link carrying the frame of each frame slot (factor f, frame slot s
   *        at f nF + s), or -1 for the world.
   * @param frame_offsets Device array of 7 nF floats per factor: the frame of
   *        each frame slot in its link frame (position, quaternion x, y, z, w),
   *        or nullptr for the link frames themselves.
   * @throws std::invalid_argument if inner or topology_links is null, inner is
   *         a constraint batch (wrap this batch in ConstraintFactorBatch
   *         instead), `frame_slots` is empty, repeats a slot, names a slot
   *         `inner` does not have or one whose tangent size does not match its
   *         view, or `inner` has more than kMaxInnerSlots slots.
   */
  LinkFrameFactorBatch(const ArticulationTopology &topology, FactorBatch *inner,
                       const std::vector<LinkFrameSlot> &frame_slots, const int *topology_links,
                       const float *frame_offsets = nullptr);

  /** @brief The wrapped (inner) factor. */
  FactorBatch *Inner() const { return inner_; }

  /** @brief Inner state slots that receive a view of a link frame, with their views. */
  const std::vector<LinkFrameSlot> &FrameSlots() const { return frame_slots_; }

  /** @brief The articulation instance, then the inner factor's slots that are not frame slots. */
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
  /** @brief Whether inner state slot `k` is a frame slot. */
  bool IsFrameSlot(int k) const;

  FactorBatch *inner_;
  std::vector<LinkFrameSlot> frame_slots_;
  const int *topology_links_;
  const float *frame_offsets_;
  /** 16 floats per item and frame slot: the view for the inner factor, in its state layout. */
  mutable dvector<float> frames_;
  mutable dvector<float *> inner_pointers_;  ///< Inner state pointers, item-major.
  mutable dvector<float> inner_jacobians_;   ///< The inner factor's Jacobians.
};

}  // namespace cunls
