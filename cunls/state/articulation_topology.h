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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Joint types of an articulation; `ArticulationDescription::joint_type`
 * holds these integer values.
 */
enum class ArticulationJointType : int {
  kPrismatic = 0,  ///< 1 coordinate: translation along the axis.
  kRevolute = 1,   ///< 1 coordinate: rotation about the axis.
  kBall = 2,       ///< 4 coordinates: unit quaternion (x, y, z, w); 3 DOFs.
  kFixed = 3,      ///< 0 coordinates.
  kFree = 4,       ///< 7 coordinates: position (3) then unit quaternion (x, y, z, w); 6 DOFs.
  kD6 = 6,         ///< up to 3 linear then up to 3 angular scalar coordinates (joint_dof_dim).
};

/**
 * @brief Host description of one kinematic tree as flat joint arrays (the
 * cuNLS articulation convention; docs/sphinx/articulations.rst).
 *
 * J joints and J links: link i is the child of exactly one tree joint. Joints
 * may be listed in any order; the topology sorts them parents first. A
 * loop-closing joint is not part of the tree: model it as equality constraints
 * on LinkFrameFactorBatch views of its two joint frames
 * (pycunls.articulation.loop_closure; docs/sphinx/articulations.rst).
 *
 * Forward kinematics, per joint, with A the parent-anchor frame:
 *
 * @verbatim
 *   A          = X_world(parent) · joint_X_p      (root_pose · joint_X_p for parent -1)
 *   X_world(c) = A · X_joint(q) · joint_X_c⁻¹
 * @endverbatim
 *
 * X_joint(q): PRISMATIC translation axis·q; REVOLUTE rotation about axis by q;
 * BALL rotation quat; FREE (p, quat); FIXED identity; D6 translation
 * Σ axis_k q_k along the linear axes, then the angular axes composed as
 * intrinsic rotations, R = R(a_n, q_n) ··· R(a_0, q_0) with a_k the k-th axis
 * rotated by the rotations before it.
 */
struct ArticulationDescription {
  /** @brief Joint type per joint (ArticulationJointType values), J entries. */
  std::vector<int> joint_type;
  /** @brief Parent link per joint, or -1 for root_pose (the world), J entries. */
  std::vector<int> joint_parent;
  /** @brief Child link per joint, J entries: a permutation of 0 .. J-1. */
  std::vector<int> joint_child;
  /** @brief Joint frame in the parent link frame, 7 floats per joint: position, quaternion xyzw. */
  std::vector<float> joint_X_p;
  /** @brief Joint frame in the child link frame, 7 floats per joint: position, quaternion xyzw. */
  std::vector<float> joint_X_c;
  /** @brief First coordinate of each joint plus the total coordinate count, J + 1 entries. */
  std::vector<int> joint_q_start;
  /** @brief First DOF of each joint plus the total DOF count, J + 1 entries. */
  std::vector<int> joint_qd_start;
  /**
   * @brief (linear, angular) axis counts per joint, 2 entries per joint. Read
   * for D6 joints only (may be empty when there is none).
   */
  std::vector<int> joint_dof_dim;
  /** @brief Axis of each DOF in the joint frame, 3 floats per DOF (read for scalar DOFs). */
  std::vector<float> joint_axis;
  /**
   * @brief Lower and upper limit per DOF. ±inf, or |value| >= 1e5, means
   * unbounded. Empty: no limits. BALL and FREE DOFs
   * must be unbounded.
   */
  std::vector<float> limit_lower, limit_upper;
  /** @brief Pose of the frame root joints attach to (parent -1): position, quaternion xyzw. */
  std::array<float, 7> root_pose{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f};
};

/**
 * @brief An immutable kinematic tree on the device, shared by every
 * ArticulationStateBatch instance and kinematic factor built on it.
 *
 * Built once on the host from an ArticulationDescription (validated, sorted
 * parents first) and uploaded. It holds no state values. Sizes:
 * CoordCount() floats of joint_q per instance,
 * DofCount() tangent dimensions, LinkCount() links (== joints).
 *
 * The state ambient layout per instance (ArticulationStateBatch) is
 *
 * @verbatim
 *   [0, C)                  joint_q
 *   [C + 12 l, + 12)        world pose of link l: row-major 3x4 [R | p]
 *   [C + 12 L + 6 j, + 6)   world motion column of DOF j: (ω, v), the world
 *                           twist at the world origin per unit of δ_j
 * @endverbatim
 *
 * with C = CoordCount(), L = LinkCount().
 */
class ArticulationTopology {
 public:
  /**
   * @brief Plain pointers and sizes for kernels (no ownership); see the
   * member comments for the array layouts. All arrays are on the device.
   */
  struct DeviceView {
    int num_joints = 0;              ///< J (== number of links).
    int num_coords = 0;              ///< C.
    int num_dofs = 0;                ///< T (tangent size).
    int ambient = 0;                 ///< floats per instance: C + 12 J + 6 T.
    int ancestor_words = 0;          ///< W = ceil(T / 32): words of one link's ancestor-DOF bitset.
    const int *order = nullptr;      ///< J joint indices, parents first.
    const int *type = nullptr;       ///< J joint types.
    const int *parent = nullptr;     ///< J parent links (-1: root frame).
    const int *child = nullptr;      ///< J child links.
    const int *q_start = nullptr;    ///< J + 1 coordinate offsets.
    const int *qd_start = nullptr;   ///< J + 1 DOF offsets.
    const int *dof_dim = nullptr;    ///< 2 J (linear, angular) axis counts.
    const float *X_p = nullptr;      ///< 12 J: joint_X_p as row-major 3x4.
    const float *X_c_inv = nullptr;  ///< 12 J: joint_X_c⁻¹ as row-major 3x4.
    const float *axis = nullptr;     ///< 3 T: unit axis of each DOF (0 for BALL/FREE DOFs).
    const float *root = nullptr;     ///< 12: root_pose as row-major 3x4.
    /**
     * @brief T entries, per DOF: the coordinate holding it for a scalar DOF
     * (PRISMATIC, REVOLUTE, D6), -1 for BALL and FREE DOFs. The box bound of a
     * DOF applies to that coordinate.
     */
    const int *scalar_coord = nullptr;
    const int *dof_joint = nullptr;  ///< T: the joint that owns each DOF.
    const unsigned *ancestors =
        nullptr;  ///< J W words: bit j of link l is set iff DOF j moves link l.
  };

  /**
   * @brief Validates `description`, sorts the joints parents first and uploads
   * the topology to the current CUDA device.
   * @throws std::invalid_argument with "ArticulationTopology: <what> (got X,
   *         expected Y). Fix: <action>." for any inconsistency.
   */
  explicit ArticulationTopology(const ArticulationDescription &description);

  /** @brief Number of joints J, which is also the number of links. */
  int JointCount() const { return view_.num_joints; }
  /** @brief Number of links (== JointCount()). */
  int LinkCount() const { return view_.num_joints; }
  /** @brief Coordinates per instance C (joint_q floats). */
  int CoordCount() const { return view_.num_coords; }
  /** @brief Degrees of freedom per instance T: the tangent size of an articulation state. */
  int DofCount() const { return view_.num_dofs; }
  /** @brief Floats stored per instance: C + 12 J + 6 T (see the class comment). */
  int AmbientSize() const { return view_.ambient; }
  /** @brief Offset of link `link`'s world pose (12 floats) in an instance's storage. */
  int LinkPoseOffset(int link) const { return view_.num_coords + 12 * link; }
  /** @brief Offset of DOF `dof`'s world motion column (6 floats) in an instance's storage. */
  int DofColumnOffset(int dof) const { return view_.num_coords + 12 * view_.num_joints + 6 * dof; }

  /** @brief Whether any DOF has a finite limit. */
  bool HasLimits() const { return has_limits_; }
  /** @brief Lower limit per DOF (T entries, host; -inf when unbounded). */
  const std::vector<float> &LowerLimits() const { return lower_; }
  /** @brief Upper limit per DOF (T entries, host; +inf when unbounded). */
  const std::vector<float> &UpperLimits() const { return upper_; }
  /** @brief Coordinate of each scalar DOF, -1 for BALL and FREE DOFs (host copy, T entries). */
  const std::vector<int> &ScalarCoord() const { return scalar_coord_; }

  /** @brief Device arrays for kernels (valid for the lifetime of this object). */
  const DeviceView &View() const { return view_; }

 private:
  DeviceView view_;
  bool has_limits_ = false;
  std::vector<float> lower_, upper_;
  std::vector<int> scalar_coord_;
  dvector<int> d_ints_;      ///< All integer arrays of view_, back to back.
  dvector<float> d_floats_;  ///< All float arrays of view_, back to back.
  dvector<unsigned> d_ancestors_;
};

}  // namespace cunls
