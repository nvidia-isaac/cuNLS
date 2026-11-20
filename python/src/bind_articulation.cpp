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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "bindings.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/log.h"
#include "cunls/factor/kinematics/link_frame_factor_batch.h"
#include "cunls/factor/kinematics/link_target_factor_batch.h"
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
      "Joint types of an articulation (the integer values of ArticulationTopology's joint_type).")
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
      "An immutable kinematic tree (flat joint arrays), shared by every "
      "ArticulationStateBatch instance and kinematic factor built on it. Built once from "
      "arrays (lists, numpy, or GPU arrays such as wp.array, copied to the host): joint_type, "
      "joint_parent (parent link, -1: root frame), "
      "joint_child (child link; link i is the child of exactly one joint), joint_X_p / joint_X_c "
      "(7 floats per joint: position, quaternion x, y, z, w), joint_q_start / joint_qd_start "
      "(J + 1 offsets), joint_axis (3 floats per DOF), joint_dof_dim ((linear, angular) per joint, "
      "for D6), limit_lower / limit_upper (per DOF; |value| >= 1e5 or inf: unbounded), root_pose "
      "(the frame root joints attach to). Loop-closing joints are not part of the tree: add them "
      "as constraints with pycunls.articulation.loop_closure. Forward kinematics: X_world(child) = "
      "X_world(parent) joint_X_p "
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
                   "Coordinates per instance (joint_q floats).")
      .def_prop_ro("dof_count", &cunls::ArticulationTopology::DofCount,
                   "Degrees of freedom per instance: the tangent size of an articulation state.")
      .def_prop_ro("ambient_size", &cunls::ArticulationTopology::AmbientSize,
                   "float32 values stored per instance: joint_q, 12 per link pose, 6 per DOF "
                   "motion column.")
      .def_prop_ro("has_limits", &cunls::ArticulationTopology::HasLimits,
                   "Whether any DOF has a finite limit.")
      .def_prop_ro("scalar_coord", &cunls::ArticulationTopology::ScalarCoord,
                   "Per DOF, the joint_q coordinate holding it for a scalar DOF (PRISMATIC, "
                   "REVOLUTE, D6), -1 for BALL and FREE DOFs. In a state, DOF j of a scalar "
                   "joint is the float at scalar_coord[j] of its instance (the joint_q part "
                   "comes first), which is what joint-space custom factors read.")
      .def_prop_ro("lower_limits", &cunls::ArticulationTopology::LowerLimits,
                   "Lower limit per DOF (-inf: unbounded).")
      .def_prop_ro("upper_limits", &cunls::ArticulationTopology::UpperLimits,
                   "Upper limit per DOF (+inf: unbounded).")
      .def("link_pose_offset", &cunls::ArticulationTopology::LinkPoseOffset, nb::arg("link"),
           "Offset of a link's world pose (row-major 3x4 [R | p], 12 floats) in an instance's "
           "storage.")
      .def("dof_column_offset", &cunls::ArticulationTopology::DofColumnOffset, nb::arg("dof"),
           "Offset of a DOF's world motion column (omega, v: 6 floats) in an instance's storage.");

  auto articulation_states =
      nb::class_<cunls::ArticulationStateBatch, cunls::StateBatch>(
          m, "ArticulationStateBatch", nb::dynamic_attr(),
          "Instances of one articulation (topology): each instance's joint_q plus a cache of link "
          "poses and motion columns that the kinematic factors read. The storage is owned "
          "(capacity "
          "x topology.ambient_size); write coordinates with set_joint_q and read them with "
          "get_joint_q. Tangent per joint, in joint_qd order: one scalar per PRISMATIC/REVOLUTE/D6 "
          "axis; BALL a rotation vector in the parent-anchor frame (quat <- Exp(d) * quat); FREE "
          "(dp, dtheta) in the parent-anchor frame. With apply_joint_limits the topology's limits "
          "are "
          "box bounds (solve with AugmentedLagrangianMinimizer). const_state_ids (int32 device "
          "array) "
          "with const_capacity entries holds instances constant. Starts with 0 active states: call "
          "set_num_active_states.")
          .def(
              "__init__",
              [](cunls::ArticulationStateBatch *self, const cunls::ArticulationTopology &topology,
                 size_t capacity, nb::handle const_ids, size_t const_capacity,
                 bool apply_joint_limits) {
                new (self) cunls::ArticulationStateBatch(topology, capacity,
                                                         const_state_ids(const_ids, const_capacity),
                                                         const_capacity, apply_joint_limits);
              },
              nb::arg("topology"), nb::arg("capacity"),
              nb::arg("const_state_ids").none() = nb::none(), nb::arg("const_capacity") = 0,
              nb::arg("apply_joint_limits") = true, nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>())
          .def_prop_ro("topology", &cunls::ArticulationStateBatch::Topology,
                       nb::rv_policy::reference_internal, "The kinematic tree of every instance.")
          .def(
              "set_joint_q",
              [](cunls::ArticulationStateBatch &self, nb::handle joint_q, size_t num_states,
                 cunls::CudaStream &stream, size_t stride) {
                const size_t C = self.Topology().CoordCount();
                DeviceScope scope(device_of(self.StateDevicePtr(0)));
                self.SetJointQ(StridedFloats(joint_q, "joint_q", num_states, C, stride), num_states,
                               stream.GetStream(), stride);
              },
              nb::arg("joint_q"), nb::arg("num_states"), nb::arg("stream"), nb::kw_only(),
              nb::arg("stride") = 0,
              "Copies the coordinates of the first num_states instances from joint_q (device "
              "float32; "
              "instance i's coord_count values start at i * stride, stride 0: packed) and "
              "recomputes "
              "their cache (forward kinematics), asynchronous on stream. A stride reads instances "
              "embedded in a larger array in place, e.g. one articulation per replicated world of "
              "a "
              "scene-wide joint_q: set_joint_q(scene_q[first:], worlds, stream, stride=per_world).")
          .def(
              "get_joint_q",
              [](cunls::ArticulationStateBatch &self, nb::handle joint_q, size_t num_states,
                 cunls::CudaStream &stream, size_t stride) {
                const size_t C = self.Topology().CoordCount();
                DeviceScope scope(device_of(self.StateDevicePtr(0)));
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
                DeviceScope scope(device_of(self.StateDevicePtr(0)));
                self.GetLinkTransforms(
                    StridedFloats(transforms, "transforms", num_states, per, stride), num_states,
                    stream.GetStream(), stride);
              },
              nb::arg("transforms"), nb::arg("num_states"), nb::arg("stream"), nb::kw_only(),
              nb::arg("stride") = 0,
              "Writes the world transform of every link of the first num_states instances, read "
              "from "
              "the cached poses (no forward kinematics runs: set_joint_q keeps the cache current; "
              "after writing state_device_ptr storage directly, call refresh(stream) first): 7 "
              "float32 per link, the pose convention of the "
              "factor targets, "
              "position then quaternion (x, y, z, w), links in topology order; instance i starts "
              "at "
              "i * stride (stride 0: packed, 7 * link_count). Asynchronous on stream. A "
              "wp.array(dtype=wp.transform) of shape (num_states * link_count,) is accepted.")
          .def(
              "use_joint_limits",
              [](cunls::ArticulationStateBatch &self) {
                self.UseJointLimits();
                hold_bounds(self, nb::none(), nb::none());
              },
              "Restores the topology's joint limits as the bounds (none if it has no finite limit) "
              "and releases the arrays of set_bounds.")
          .def(
              "refresh",
              [](cunls::ArticulationStateBatch &self, cunls::CudaStream &stream) {
                DeviceScope scope(device_of(self.StateDevicePtr(0)));
                self.RefreshDerivedData(stream.GetStream());
              },
              nb::arg("stream"),
              "Recomputes the cache of the active instances from their joint_q (after writing "
              "state_device_ptr storage directly; the minimizers refresh at every solve).");
  bind_box_bounds(
      articulation_states, "ArticulationStateBatch",
      "Box bounds per instance and DOF, replacing the joint limits: either two device "
      "float32 arrays of capacity x dof_count (an unbounded side is a ±inf-filled array; "
      "BALL/FREE entries ignored), or None, None to remove them (one None alone is "
      "rejected); use_joint_limits() restores the topology's limits. Keeps only the current "
      "arrays alive (replaced ones are released).");

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

  nb::enum_<cunls::FrameView>(
      m, "FrameView",
      "What a frame slot of LinkFrameFactorBatch hands its inner factor, in the layout and "
      "tangent convention of an existing state type, so that factors of that type work on robot "
      "links unchanged.")
      .value("Pose", cunls::FrameView::kPose,
             "The frame's pose as an SE3StateBatch state: row-major 4x4 world_from_frame, tangent "
             "[omega, v] in the frame, X' = X Exp(delta). Slot tangent 6.")
      .value("Position", cunls::FrameView::kPosition,
             "The frame's origin in the world as a VectorStateBatch3 state (3 floats, additive "
             "world-frame step). Slot tangent 3.")
      .value("Orientation", cunls::FrameView::kOrientation,
             "The frame's rotation as an SO3StateBatch state: row-major 3x3 world_from_frame, "
             "tangent omega in the frame, R' = R Exp(omega). Slot tangent 3.");

  nb::class_<cunls::LinkFrameFactorBatch, cunls::FactorBatch>(
      m, "LinkFrameFactorBatch",
      "Puts the pose, position or orientation states of any factor on a robot, for custom "
      "kinematic objectives. The wrapped inner_factor is an ordinary factor over SE(3) poses, 3D "
      "points or SO(3) rotations (built-in: SE3PriorFactorBatch, SE3BetweenFactorBatch, "
      "SE3SphereClearanceFactorBatch, PriorVectorFactorBatch3, HalfspaceFactorBatch3, "
      "VectorBetweenFactorBatch3, SO3PriorFactorBatch, SO3BetweenFactorBatch, ...; or a "
      "pycunls.warp.WarpFactorBatch) that knows nothing about robots. Each frame slot (an inner "
      "state slot with a FrameView) does not connect to a state of the problem: it receives a view "
      "of a frame fixed on a link of the articulation instance, and the inner Jacobian is carried "
      "to the joint DOFs (chain rule through the forward-kinematics cache).\n\n"
      "For factor f and frame slot s (the position in frame_slots, nF = len(frame_slots)) the "
      "frame is X_L(q) X_off with L = topology_links[f * nF + s] and X_off = "
      "frame_offsets[f * nF + s]; FrameView.Pose hands it as SE3StateBatch does, "
      "FrameView.Position its origin (VectorStateBatch3), FrameView.Orientation its rotation "
      "(SO3StateBatch). Write inner_factor exactly as for those states.\n\n"
      "States of this batch: [articulation instance] followed by inner_factor's slots that are "
      "not frame slots, in order (e.g. an object pose solved with the joints). Size and active "
      "count are inner_factor's. A topology link out of range gives a NaN view. For a "
      "constraint (e.g. a halfspace as an inequality), wrap this batch in ConstraintFactorBatch, "
      "never inner_factor. The fused LinkPose / LinkPosition / LinkOrientation factors are "
      "faster for what they cover. Guide: docs/sphinx/articulations.rst, 'Custom kinematic "
      "objectives'.\n\n"
      "Parameters\n"
      "----------\n"
      "topology : ArticulationTopology\n"
      "    Kinematic tree of the articulation state.\n"
      "inner_factor : FactorBatch\n"
      "    The factor over the views (kept alive by this batch).\n"
      "frame_slots : list[tuple[int, FrameView]]\n"
      "    (inner state slot, view) pairs: the slots of inner_factor (positions in its\n"
      "    state_sizes) that receive a view of a link frame; distinct, each of its view's tangent\n"
      "    size (Pose 6, Position 3, Orientation 3). Example: [(0, FrameView.Position)].\n"
      "topology_links : int32 GPU array\n"
      "    capacity x nF link indices of the topology (positions in UrdfArticulation.link_names):\n"
      "    the link carrying the frame of each frame slot, factor-major.\n"
      "frame_offsets : float32 GPU array or None\n"
      "    capacity x nF x 7: each frame in its link frame (position, quaternion x, y, z, w;\n"
      "    e.g. a camera mount or a tool tip), or None for the link frames themselves.")
      .def(
          "__init__",
          [](cunls::LinkFrameFactorBatch *self, const cunls::ArticulationTopology &topology,
             cunls::FactorBatch *inner,
             const std::vector<std::pair<int, cunls::FrameView>> &frame_slots,
             nb::handle topology_links, nb::handle frame_offsets) {
            std::vector<cunls::LinkFrameSlot> slots;
            for (const auto &[inner_slot, view] : frame_slots) slots.push_back({inner_slot, view});
            const size_t capacity = inner->Capacity(), nF = slots.size();
            new (self) cunls::LinkFrameFactorBatch(
                topology, inner, slots,
                device_ints(topology_links, "topology_links", capacity * nF,
                            "capacity x len(frame_slots) links"),
                OptionalFloats(frame_offsets, "frame_offsets", capacity, 7 * nF, "factor"));
          },
          nb::arg("topology"), nb::arg("inner_factor"), nb::arg("frame_slots"),
          nb::arg("topology_links"), nb::arg("frame_offsets").none() = nb::none(),
          nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 5>(),
          nb::keep_alive<1, 6>())
      .def_prop_ro("inner_factor", &cunls::LinkFrameFactorBatch::Inner, nb::rv_policy::reference,
                   "The wrapped factor over the views.")
      .def_prop_ro(
          "frame_slots",
          [](const cunls::LinkFrameFactorBatch &self) {
            std::vector<std::pair<int, cunls::FrameView>> out;
            for (const auto &slot : self.FrameSlots()) out.emplace_back(slot.inner_slot, slot.view);
            return out;
          },
          "(inner state slot, FrameView) pairs: the inner slots that receive a view of a link "
          "frame.");

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

  nb::class_<cunls::MimicJointFactorBatch, cunls::FactorBatch>(
      m, "MimicJointFactorBatch",
      "c = q_m - (a + b q_ref): a mimic (coupled) scalar DOF, as URDF "
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
