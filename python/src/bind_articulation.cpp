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

// Bindings for articulations: ArticulationJointType, ArticulationTopology
// (built from host arrays), ArticulationStateBatch and the kinematic factor
// batches (link targets, posture, loop closure, mimic joint).

#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/log.h"
#include "cunls/factor/kinematics/link_target_factor_batch.h"
#include "cunls/factor/kinematics/loop_closure_factor_batch.h"
#include "cunls/factor/kinematics/mimic_joint_factor_batch.h"
#include "cunls/factor/kinematics/posture_factor_batch.h"
#include "cunls/state/articulation_state_batch.h"
#include "cunls/state/articulation_topology.h"

namespace {

// Host values of a sequence, numpy array or GPU array as a std::vector of T (via numpy, so lists,
// tuples and arrays of any numeric dtype work). None gives an empty vector.
template <typename T>
std::vector<T> HostValues(nb::handle obj, const char *name) {
  if (obj.is_none()) return {};
  nb::module_ np = nb::module_::import_("numpy");
  nb::object flat;
  nb::object source = nb::borrow(obj);
  if (nb::hasattr(obj, "__cuda_array_interface__")) {  // GPU array: pycunls's host copy
    source = nb::module_::import_("pycunls._arrays")
                 .attr("to_numpy")(obj, std::is_same_v<T, int> ? "int32" : "float32", name,
                                   "ArticulationTopology");
  }
  try {
    flat = np.attr("ascontiguousarray")(
                 source, nb::arg("dtype") = std::is_same_v<T, int> ? "int32" : "float32")
               .attr("reshape")(-1);
  } catch (const nb::python_error &) {
    throw nb::value_error((std::string("ArticulationTopology: ") + name +
                           " is not a numeric array (expected a list, numpy array or GPU array). "
                           "Fix: pass numeric values.")
                              .c_str());
  }
  auto array = nb::cast<nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>>(flat);
  return std::vector<T>(array.data(), array.data() + array.shape(0));
}

// Optional device float array (None: nullptr) of count x per_item values.
const float *OptionalFloats(nb::handle obj, const char *name, size_t count, size_t per_item,
                            const char *item) {
  return obj.is_none() ? nullptr : device_floats(obj, name, count, per_item, item);
}

// Device float32 array holding `num_states` instances of `per_instance` floats, instance i
// starting at i * stride (stride 0: per_instance). Returns the base pointer.
float *StridedFloats(nb::handle obj, const char *name, size_t num_states, size_t per_instance,
                     size_t stride) {
  const size_t pitch = stride == 0 ? per_instance : stride;
  const size_t needed = num_states == 0 ? 0 : (num_states - 1) * pitch + per_instance;
  const std::string what = std::to_string(num_states) + " instances of " +
                           std::to_string(per_instance) + " floats, " + std::to_string(pitch) +
                           " apart";
  return reinterpret_cast<float *>(extract_device_ptr(obj, "float32", name, needed, what.c_str()));
}

}  // namespace

using cunls::LogError;  // used by THROW_ON_CUDA_ERROR

void bind_articulation(nb::module_ &m) {
  nb::enum_<cunls::ArticulationJointType>(
      m, "ArticulationJointType",
      "Joint types of an articulation; the values are Newton's JointType values.")
      .value("Prismatic", cunls::ArticulationJointType::kPrismatic,
             "1 coordinate: translation along the axis.")
      .value("Revolute", cunls::ArticulationJointType::kRevolute,
             "1 coordinate: rotation about the axis.")
      .value("Ball", cunls::ArticulationJointType::kBall,
             "4 coordinates (quaternion x, y, z, w), 3 DOFs.")
      .value("Fixed", cunls::ArticulationJointType::kFixed, "0 coordinates.")
      .value("Free", cunls::ArticulationJointType::kFree,
             "7 coordinates (position, quaternion x, y, z, w), 6 DOFs.")
      .value("D6", cunls::ArticulationJointType::kD6,
             "Up to 3 linear then up to 3 angular scalar coordinates (joint_dof_dim).");

  nb::class_<cunls::ArticulationTopology>(
      m, "ArticulationTopology",
      "An immutable kinematic tree (Newton's array layout), shared by every "
      "ArticulationStateBatch instance and kinematic factor built on it. Built once from "
      "arrays (lists, numpy, or GPU arrays such as wp.array, copied to the host): joint_type, "
      "joint_parent (parent link, -1: root frame), "
      "joint_child (child link; link i is the child of exactly one joint), joint_X_p / joint_X_c "
      "(7 floats per joint: position, quaternion x, y, z, w), joint_q_start / joint_qd_start "
      "(J + 1 offsets), joint_axis (3 floats per DOF), joint_dof_dim ((linear, angular) per joint, "
      "for D6), limit_lower / limit_upper (per DOF; |value| >= 1e5 or inf: unbounded), root_pose "
      "(the frame root joints attach to). Loop-closing joints are not part of the tree: add them "
      "as LoopClosureFactorBatch. Forward kinematics: X_world(child) = X_world(parent) joint_X_p "
      "X_joint(q) joint_X_c^-1 (docs/sphinx/articulations.rst).")
      .def(
          "__init__",
          [](cunls::ArticulationTopology *self, nb::handle joint_type, nb::handle joint_parent,
             nb::handle joint_child, nb::handle joint_X_p, nb::handle joint_X_c,
             nb::handle joint_q_start, nb::handle joint_qd_start, nb::handle joint_axis,
             nb::handle joint_dof_dim, nb::handle limit_lower, nb::handle limit_upper,
             nb::handle root_pose) {
            cunls::ArticulationDescription d;
            d.joint_type = HostValues<int>(joint_type, "joint_type");
            d.joint_parent = HostValues<int>(joint_parent, "joint_parent");
            d.joint_child = HostValues<int>(joint_child, "joint_child");
            d.joint_X_p = HostValues<float>(joint_X_p, "joint_X_p");
            d.joint_X_c = HostValues<float>(joint_X_c, "joint_X_c");
            d.joint_q_start = HostValues<int>(joint_q_start, "joint_q_start");
            d.joint_qd_start = HostValues<int>(joint_qd_start, "joint_qd_start");
            d.joint_axis = HostValues<float>(joint_axis, "joint_axis");
            d.joint_dof_dim = HostValues<int>(joint_dof_dim, "joint_dof_dim");
            d.limit_lower = HostValues<float>(limit_lower, "limit_lower");
            d.limit_upper = HostValues<float>(limit_upper, "limit_upper");
            const auto root = HostValues<float>(root_pose, "root_pose");
            if (!root.empty()) {
              if (root.size() != 7) {
                throw nb::value_error(
                    ("ArticulationTopology: root_pose has " + std::to_string(root.size()) +
                     " values (expected 7: position, quaternion x, y, z, w). Fix: pass a pose.")
                        .c_str());
              }
              for (int i = 0; i < 7; ++i) d.root_pose[i] = root[i];
            }
            new (self) cunls::ArticulationTopology(d);
          },
          nb::arg("joint_type"), nb::arg("joint_parent"), nb::arg("joint_child"),
          nb::arg("joint_X_p"), nb::arg("joint_X_c"), nb::arg("joint_q_start"),
          nb::arg("joint_qd_start"), nb::arg("joint_axis"),
          nb::arg("joint_dof_dim").none() = nb::none(), nb::arg("limit_lower").none() = nb::none(),
          nb::arg("limit_upper").none() = nb::none(), nb::arg("root_pose").none() = nb::none())
      .def_prop_ro("joint_count", &cunls::ArticulationTopology::JointCount,
                   "Number of joints J (== number of links).")
      .def_prop_ro("link_count", &cunls::ArticulationTopology::LinkCount,
                   "Number of links (== joint_count).")
      .def_prop_ro("coord_count", &cunls::ArticulationTopology::CoordCount,
                   "Coordinates per instance (joint_q floats, Newton's layout).")
      .def_prop_ro("dof_count", &cunls::ArticulationTopology::DofCount,
                   "Degrees of freedom per instance: the tangent size of an articulation state.")
      .def_prop_ro("ambient_size", &cunls::ArticulationTopology::AmbientSize,
                   "float32 values stored per instance: joint_q, 12 per link pose, 6 per DOF "
                   "motion column.")
      .def_prop_ro("has_limits", &cunls::ArticulationTopology::HasLimits,
                   "Whether any DOF has a finite limit.")
      .def_prop_ro("lower_limits", &cunls::ArticulationTopology::LowerLimits,
                   "Lower limit per DOF (-inf: unbounded).")
      .def_prop_ro("upper_limits", &cunls::ArticulationTopology::UpperLimits,
                   "Upper limit per DOF (+inf: unbounded).")
      .def("link_pose_offset", &cunls::ArticulationTopology::LinkPoseOffset, nb::arg("link"),
           "Offset of a link's world pose (row-major 3x4 [R | p], 12 floats) in an instance's "
           "storage.")
      .def("dof_column_offset", &cunls::ArticulationTopology::DofColumnOffset, nb::arg("dof"),
           "Offset of a DOF's world motion column (omega, v: 6 floats) in an instance's storage.");

  nb::class_<cunls::ArticulationStateBatch, cunls::StateBatch>(
      m, "ArticulationStateBatch",
      "Instances of one articulation (topology): each instance's joint_q plus a cache of link "
      "poses and motion columns that the kinematic factors read. The storage is owned (capacity "
      "x topology.ambient_size); write coordinates with set_joint_q and read them with "
      "get_joint_q. Tangent per joint, in joint_qd order: one scalar per PRISMATIC/REVOLUTE/D6 "
      "axis; BALL a rotation vector in the parent-anchor frame (quat <- Exp(d) * quat); FREE "
      "(dp, dtheta) in the parent-anchor frame. With apply_joint_limits the topology's limits are "
      "box bounds (solve with AugmentedLagrangianMinimizer). const_state_ids (int32 device array) "
      "with const_capacity entries holds instances constant. Starts with 0 active states: call "
      "set_num_active_states.")
      .def(
          "__init__",
          [](cunls::ArticulationStateBatch *self, const cunls::ArticulationTopology &topology,
             size_t capacity, bool apply_joint_limits) {
            new (self) cunls::ArticulationStateBatch(topology, capacity, apply_joint_limits);
          },
          nb::arg("topology"), nb::arg("capacity"), nb::arg("apply_joint_limits") = true,
          nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](cunls::ArticulationStateBatch *self, const cunls::ArticulationTopology &topology,
             size_t capacity, nb::handle const_ids, size_t const_capacity,
             bool apply_joint_limits) {
            auto cids =
                device_ints(const_ids, "const_state_ids", const_capacity, "const_capacity ids");
            new (self) cunls::ArticulationStateBatch(topology, capacity, cids, const_capacity,
                                                     apply_joint_limits);
          },
          nb::arg("topology"), nb::arg("capacity"), nb::arg("const_state_ids"),
          nb::arg("const_capacity"), nb::arg("apply_joint_limits") = true, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 4>())
      .def_prop_ro("topology", &cunls::ArticulationStateBatch::Topology,
                   nb::rv_policy::reference_internal, "The kinematic tree of every instance.")
      .def(
          "set_joint_q",
          [](cunls::ArticulationStateBatch &self, nb::handle joint_q, size_t num_states,
             cunls::CudaStream &stream, size_t stride) {
            const size_t C = self.Topology().CoordCount();
            self.SetJointQ(StridedFloats(joint_q, "joint_q", num_states, C, stride), num_states,
                           stream.GetStream(), stride);
          },
          nb::arg("joint_q"), nb::arg("num_states"), nb::arg("stream"), nb::kw_only(),
          nb::arg("stride") = 0,
          "Copies the coordinates of the first num_states instances from joint_q (device float32; "
          "instance i's coord_count values start at i * stride, stride 0: packed) and recomputes "
          "their cache (forward kinematics), asynchronous on stream. A stride reads instances "
          "embedded in a larger array in place, e.g. one articulation per replicated world of a "
          "scene-wide joint_q: set_joint_q(scene_q[first:], worlds, stream, stride=per_world).")
      .def(
          "get_joint_q",
          [](cunls::ArticulationStateBatch &self, nb::handle joint_q, size_t num_states,
             cunls::CudaStream &stream, size_t stride) {
            const size_t C = self.Topology().CoordCount();
            self.GetJointQ(StridedFloats(joint_q, "joint_q", num_states, C, stride), num_states,
                           stream.GetStream(), stride);
          },
          nb::arg("joint_q"), nb::arg("num_states"), nb::arg("stream"), nb::kw_only(),
          nb::arg("stride") = 0,
          "Copies the coordinates of the first num_states instances into joint_q (device "
          "float32; instance i at i * stride, stride 0: packed; the values between instances "
          "are left untouched), asynchronous on stream. A stride writes a solve back into a "
          "larger (scene-wide) array in place.")
      .def(
          "get_link_transforms",
          [](cunls::ArticulationStateBatch &self, nb::handle transforms, size_t num_states,
             cunls::CudaStream &stream, size_t stride) {
            const size_t per = 7 * static_cast<size_t>(self.Topology().LinkCount());
            self.GetLinkTransforms(StridedFloats(transforms, "transforms", num_states, per, stride),
                                   num_states, stream.GetStream(), stride);
          },
          nb::arg("transforms"), nb::arg("num_states"), nb::arg("stream"), nb::kw_only(),
          nb::arg("stride") = 0,
          "Writes the world transform of every link of the first num_states instances (forward "
          "kinematics of their current joint_q): 7 float32 per link, the pose convention of the "
          "factor targets (Newton's body_q layout), "
          "position then quaternion (x, y, z, w), links in topology order; instance i starts at "
          "i * stride (stride 0: packed, 7 * link_count). Asynchronous on stream. A "
          "wp.array(dtype=wp.transform) of shape (num_states * link_count,) is accepted.")
      .def(
          "set_bounds",
          [](cunls::ArticulationStateBatch &self, nb::handle lower, nb::handle upper) {
            if (lower.is_none() && upper.is_none()) {
              self.SetBounds(nullptr, nullptr);
              return;
            }
            if (lower.is_none() || upper.is_none()) {
              throw std::invalid_argument(
                  "ArticulationStateBatch.set_bounds: got one bound and None (expected both "
                  "arrays, or None, None). Fix: pass -inf / +inf for an unbounded side.");
            }
            self.SetBounds(
                device_floats(lower, "lower", self.Capacity(), self.TangentSize(), "state"),
                device_floats(upper, "upper", self.Capacity(), self.TangentSize(), "state"));
          },
          nb::arg("lower").none(), nb::arg("upper").none(), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(),
          "Box bounds per instance and DOF (device float32 arrays of capacity x dof_count; "
          "±inf: unbounded; BALL/FREE entries ignored), replacing the joint limits. None, None "
          "removes them; use_joint_limits() restores the topology's limits.")
      .def("use_joint_limits", &cunls::ArticulationStateBatch::UseJointLimits,
           "Restores the topology's joint limits as the bounds (none if it has no finite limit).")
      .def_prop_ro(
          "has_bounds",
          [](const cunls::ArticulationStateBatch &self) { return cunls::HasBoxBounds(&self); },
          "Whether the batch has box bounds (solve with AugmentedLagrangianMinimizer).")
      .def(
          "refresh",
          [](cunls::ArticulationStateBatch &self, cunls::CudaStream &stream) {
            self.RefreshDerivedData(stream.GetStream());
          },
          nb::arg("stream"),
          "Recomputes the cache of the active instances from their joint_q (after writing "
          "state_device_ptr storage directly; the minimizers refresh at every solve).");

  nb::class_<cunls::LinkPositionFactorBatch, cunls::FactorBatch>(
      m, "LinkPositionFactorBatch",
      "r = W (X_L(q) c - p*): the world position of point c on link L should equal p* (3 rows). "
      "Device arrays per factor: links (int32, 1), points (3 floats, link frame), targets "
      "(3 floats, world), weights (3 floats, or None for 1). One articulation state per factor.")
      .def(
          "__init__",
          [](cunls::LinkPositionFactorBatch *self, const cunls::ArticulationTopology &topology,
             nb::handle links, nb::handle points, nb::handle targets, nb::handle weights,
             size_t capacity) {
            new (self) cunls::LinkPositionFactorBatch(
                topology, device_ints(links, "links", capacity, "capacity links"),
                device_floats(points, "points", capacity, 3, "factor"),
                device_floats(targets, "targets", capacity, 3, "factor"),
                OptionalFloats(weights, "weights", capacity, 3, "factor"), capacity);
          },
          nb::arg("topology"), nb::arg("links"), nb::arg("points"), nb::arg("targets"),
          nb::arg("weights").none(), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          nb::keep_alive<1, 6>());

  nb::class_<cunls::LinkOrientationFactorBatch, cunls::FactorBatch>(
      m, "LinkOrientationFactorBatch",
      "r = W Log(R_L(q) R_off R*^T): the world orientation of a frame on link L should equal R* "
      "(3 rows, world rotation vector). Device arrays per factor: links (int32, 1), offsets "
      "(R_off as quaternion x, y, z, w: 4 floats, or None for identity), targets (R* as "
      "quaternion: 4 floats), weights (3 floats, or None).")
      .def(
          "__init__",
          [](cunls::LinkOrientationFactorBatch *self, const cunls::ArticulationTopology &topology,
             nb::handle links, nb::handle offsets, nb::handle targets, nb::handle weights,
             size_t capacity) {
            new (self) cunls::LinkOrientationFactorBatch(
                topology, device_ints(links, "links", capacity, "capacity links"),
                OptionalFloats(offsets, "offsets", capacity, 4, "factor"),
                device_floats(targets, "targets", capacity, 4, "factor"),
                OptionalFloats(weights, "weights", capacity, 3, "factor"), capacity);
          },
          nb::arg("topology"), nb::arg("links"), nb::arg("offsets").none(), nb::arg("targets"),
          nb::arg("weights").none(), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          nb::keep_alive<1, 6>());

  nb::class_<cunls::LinkPoseFactorBatch, cunls::FactorBatch>(
      m, "LinkPoseFactorBatch",
      "r = W [X_L(q) p_off - p*; Log(R_L(q) R_off R*^T)]: the world pose of a frame on link L "
      "should equal a target (6 rows). Device arrays per factor: links (int32, 1), offsets (the "
      "frame in the link frame: position, quaternion x, y, z, w = 7 floats, or None), targets "
      "(7 floats: position, quaternion), weights (6 floats, or None).")
      .def(
          "__init__",
          [](cunls::LinkPoseFactorBatch *self, const cunls::ArticulationTopology &topology,
             nb::handle links, nb::handle offsets, nb::handle targets, nb::handle weights,
             size_t capacity) {
            new (self) cunls::LinkPoseFactorBatch(
                topology, device_ints(links, "links", capacity, "capacity links"),
                OptionalFloats(offsets, "offsets", capacity, 7, "factor"),
                device_floats(targets, "targets", capacity, 7, "factor"),
                OptionalFloats(weights, "weights", capacity, 6, "factor"), capacity);
          },
          nb::arg("topology"), nb::arg("links"), nb::arg("offsets").none(), nb::arg("targets"),
          nb::arg("weights").none(), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          nb::keep_alive<1, 6>());

  nb::class_<cunls::PostureFactorBatch, cunls::FactorBatch>(
      m, "PostureFactorBatch",
      "r = W (q (-) q*): the coordinates should stay near a nominal configuration (dof_count "
      "rows; scalar DOFs q - q*, BALL/FREE rotations Log(R R*^T)). A regularizer, null-space "
      "posture, or 'stay close to the requested state'. Device arrays per factor: targets "
      "(coord_count floats, joint_q layout), weights (dof_count floats, or None for 1).")
      .def(
          "__init__",
          [](cunls::PostureFactorBatch *self, const cunls::ArticulationTopology &topology,
             nb::handle targets, nb::handle weights, size_t capacity) {
            new (self) cunls::PostureFactorBatch(
                topology,
                device_floats(targets, "targets", capacity, topology.CoordCount(), "factor"),
                OptionalFloats(weights, "weights", capacity, topology.DofCount(), "factor"),
                capacity);
          },
          nb::arg("topology"), nb::arg("targets"), nb::arg("weights").none(), nb::arg("capacity"),
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>());

  nb::class_<cunls::LoopClosureFactorBatch, cunls::FactorBatch>(
      m, "LoopClosureFactorBatch",
      "A loop-closing joint between two links of an articulation instance (parent -1: the "
      "world): the constraint rows keeping its two joint frames together (BALL 3, REVOLUTE 5, "
      "PRISMATIC 5, FIXED 6). Wrap in ConstraintFactorBatch(..., ConstraintKind.Equality). "
      "joint_type: the ArticulationJointType of every factor. Device arrays per factor: "
      "parent_links, child_links (int32, 1 each), frames (14 floats: joint_X_p then joint_X_c, "
      "each position + quaternion x, y, z, w), axes (3 floats; None allowed for BALL/FIXED).")
      .def(
          "__init__",
          [](cunls::LoopClosureFactorBatch *self, const cunls::ArticulationTopology &topology,
             cunls::ArticulationJointType joint_type, nb::handle parent_links,
             nb::handle child_links, nb::handle frames, nb::handle axes, size_t capacity) {
            new (self) cunls::LoopClosureFactorBatch(
                topology, joint_type,
                device_ints(parent_links, "parent_links", capacity, "capacity links"),
                device_ints(child_links, "child_links", capacity, "capacity links"),
                device_floats(frames, "frames", capacity, 14, "factor"),
                OptionalFloats(axes, "axes", capacity, 3, "factor"), capacity);
          },
          nb::arg("topology"), nb::arg("joint_type"), nb::arg("parent_links"),
          nb::arg("child_links"), nb::arg("frames"), nb::arg("axes").none(), nb::arg("capacity"),
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          nb::keep_alive<1, 6>(), nb::keep_alive<1, 7>());

  nb::class_<cunls::MimicJointFactorBatch, cunls::FactorBatch>(
      m, "MimicJointFactorBatch",
      "c = q_m - (a + b q_ref): a mimic (coupled) scalar DOF, as Newton's joint_mimic_* and URDF "
      "<mimic>. Wrap in ConstraintFactorBatch(..., ConstraintKind.Equality). Device arrays per "
      "factor: mimic_dofs, reference_dofs (int32 DOF indices, 1 each), coefficients (a, b: 2 "
      "floats).")
      .def(
          "__init__",
          [](cunls::MimicJointFactorBatch *self, const cunls::ArticulationTopology &topology,
             nb::handle mimic_dofs, nb::handle reference_dofs, nb::handle coefficients,
             size_t capacity) {
            new (self) cunls::MimicJointFactorBatch(
                topology, device_ints(mimic_dofs, "mimic_dofs", capacity, "capacity DOFs"),
                device_ints(reference_dofs, "reference_dofs", capacity, "capacity DOFs"),
                device_floats(coefficients, "coefficients", capacity, 2, "factor"), capacity);
          },
          nb::arg("topology"), nb::arg("mimic_dofs"), nb::arg("reference_dofs"),
          nb::arg("coefficients"), nb::arg("capacity"), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>());
}
