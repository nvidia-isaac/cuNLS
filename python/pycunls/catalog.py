# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Catalog of the built-in state and factor batches: what each one connects and expects.

Use it to pick a class and to get its data layout right without reading the C++ headers::

    print(pycunls.describe("SE3BetweenFactorBatch"))   # states, residual, measurement layout
    pycunls.catalog(kind="factor", query="SE3")         # list of dicts, filterable

Conventions shared by every class:

* Arrays are GPU arrays (e.g. ``wp.array``), C-contiguous, one item after another: float32 for
  values, int32 for indices (links, DOFs, state ids); each argument's stated dtype rules.
* Poses are ``world_from_body`` matrices stored row-major (SE3: 4x4 = 16 floats per pose,
  SE2: 3x3, SO3: 3x3, SO2: 2x2). The update is right-multiplicative, ``T' = T Exp(delta)``.
* Tangent order: se(3) = ``[omega (3), v (3)]`` (rotation first), se(2) = ``[v_x, v_y, theta]``
  (translation first) -- see each state's ``tangent_order``.
* Between factors measure ``Delta = T_right^-1 T_left`` (residual ``Log(Delta T_left^-1
  T_right)``), i.e. the *inverse* of the usual relative pose ``T_left^-1 T_right``.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Tuple, Union


@dataclasses.dataclass(frozen=True)
class StateInfo:
    """A built-in state batch.

    Attributes:
        name: Class name in :mod:`pycunls`.
        ambient: float32 values stored per state (0: set by the constructor's topology).
        tangent: Degrees of freedom (size of one update; 0: set by the topology).
        layout: How one state is stored.
        tangent_order: Meaning of the tangent coordinates.
        update: The update x' = x (+) delta, when it is not the Lie-group one.
        constructor: The constructor's arguments, when it does not take a data array.
    """
    name: str
    ambient: int
    tangent: int
    layout: str
    tangent_order: str
    kind: str = "state"
    update: str = "x' = x (+) delta (right-multiplicative for Lie groups)"
    constructor: str = ""


@dataclasses.dataclass(frozen=True)
class FactorInfo:
    """A built-in factor batch.

    Attributes:
        name: Class name in :mod:`pycunls`.
        states: State batch class of each connected state, in connection order.
        residual_size: Residual dimension of one factor.
        residual: The residual formula.
        arrays: Constructor array arguments and the layout of one item of each.
        relative: Every connection constrains states only relative to each other (between-type
            factors, motion models): a group of states touched only by such factors has a
            gauge freedom unless one state is constant or has a prior.
        use: When to use it.
        notes: Conventions and traps.
    """
    name: str
    states: List[str]
    residual_size: int
    residual: str
    arrays: Dict[str, str]
    relative: bool
    use: str
    notes: str = ""
    kind: str = "factor"


_POSE = {
    "SE3": ("SE3StateBatch", 16, 6, "4x4 matrix [R t; 0 1], row-major (16 floats)",
            "[omega_x, omega_y, omega_z, v_x, v_y, v_z]"),
    "SE2": ("SE2StateBatch", 9, 3, "3x3 matrix [R t; 0 1], row-major (9 floats)",
            "[v_x, v_y, theta]"),
    "SO3": ("SO3StateBatch", 9, 3, "3x3 rotation matrix, row-major", "[omega_x, omega_y, omega_z]"),
    "SO2": ("SO2StateBatch", 4, 1, "2x2 rotation matrix, row-major", "[omega]"),
    "Similarity2": ("Similarity2StateBatch", 9, 4, "3x3 matrix [sR t; 0 1], row-major",
                    "[u_x, u_y, theta, lambda (log scale)]"),
    "Similarity3": ("Similarity3StateBatch", 16, 7, "4x4 matrix [sR t; 0 1], row-major",
                    "[omega (3), u (3), lambda (log scale)]"),
    "SL4": ("SL4StateBatch", 16, 15, "4x4 matrix with det 1, row-major", "15 sl(4) coordinates"),
}
# The numbered names of the vector classes (VectorStateBatch3, BoundFactorBatch12, ...): the
# runtime-sized class with dim fixed. The catalog lists the runtime-sized classes; a lookup by a
# numbered name gets the entry with the size filled in.
VECTOR_DIMS = (1, 2, 3, 4, 6, 12)  # sizes with a numbered name
_NUMBERED = re.compile(r"^(VectorStateBatch|PriorVectorFactorBatch|VectorBetweenFactorBatch|"
                       r"BoundFactorBatch|HalfspaceFactorBatch)(\d+)$")


def _vector_entries(dim: Optional[int]) -> List[Union[StateInfo, FactorInfo]]:
    """The vector state and factor entries: runtime-sized (dim None, sizes 0) or numbered."""
    sfx, d, size = ("", "dim", 0) if dim is None else (str(dim), str(dim), dim)
    state = f"VectorStateBatch{sfx}"
    sized = "" if dim else (" Size: the dim keyword, else the trailing extent of the first "
                            "array ((M, dim) or wp.vecN); the numbered names (e.g. "
                            "PriorVectorFactorBatch3) fix it.")
    return [
        StateInfo(state, size, size, f"{d} floats", f"the {d} components (Plus is addition)",
                  constructor="" if dim else (
                      "data (float32 GPU array of capacity x dim), capacity, optionally "
                      "const_state_ids (int32 GPU array of the states held fixed) with "
                      "const_capacity, and the keyword dim (read from data shaped (N, dim) or "
                      "wp.vecN when omitted; VectorStateBatch3 and the other numbered names "
                      "fix it)")),
        FactorInfo(f"PriorVectorFactorBatch{sfx}", [state], size, "x - observation",
                   {"observations": f"{d} floats per factor"}, False,
                   "Absolute measurement of a vector state; also a weak prior/regularizer (wrap "
                   "in WeightedFactorBatch).", sized.strip()),
        FactorInfo(f"VectorBetweenFactorBatch{sfx}", [state] * 2, size,
                   "x_left - x_right - delta", {"deltas": f"{d} floats per factor"}, True,
                   "Relative measurement between two vector states.",
                   "Sign: residual is left - right - delta, so delta = x_left - x_right." + sized),
        FactorInfo(f"BoundFactorBatch{sfx}", [state], 2 * size,
                   "box constraint lower <= x <= upper",
                   {"lower": f"{d} floats per factor", "upper": f"{d} floats per factor"}, False,
                   "Box bounds on a vector state; needs AugmentedLagrangianMinimizer.",
                   "For plain parameter bounds prefer VectorStateBatch.set_bounds (projected)." +
                   sized),
        FactorInfo(f"HalfspaceFactorBatch{sfx}", [state], 1, "a^T x - b",
                   {"normals": f"{d} floats per factor (a)", "offsets": "1 float per factor (b)"},
                   False, "Linear inequality a^T x <= b: wrap in ConstraintFactorBatch(..., "
                   "ConstraintKind.Inequality) and solve with AugmentedLagrangianMinimizer.",
                   sized.strip()),
    ]


def _build() -> Dict[str, Union[StateInfo, FactorInfo]]:
    entries: List[Union[StateInfo, FactorInfo]] = []
    for group, (cls, amb, tan, layout, order) in _POSE.items():
        entries.append(StateInfo(cls, amb, tan, layout, order))

    lie_label = {"SE3": "SE(3)", "SE2": "SE(2)", "SO3": "SO(3)", "SO2": "SO(2)",
                 "Similarity2": "Sim(2)", "Similarity3": "Sim(3)", "SL4": "SL(4)"}
    for group, (cls, amb, tan, layout, _order) in _POSE.items():
        entries.append(FactorInfo(
            f"{group}BetweenFactorBatch", [cls, cls], tan,
            "Log(Delta * T_left^-1 * T_right)",
            {"deltas": f"Delta per factor: {layout}"}, True,
            f"Relative {lie_label[group]} measurement between two states (odometry, loop "
            "closures, pose graphs).",
            "Delta = T_right^-1 * T_left (the inverse of the usual T_left^-1 * T_right). "
            "Connect [left, right] per factor. Fix one state (constant id) or add a prior."))
        entries.append(FactorInfo(
            f"{group}PriorFactorBatch", [cls], tan, "Log(T_target^-1 * T)",
            {"observations": f"T_target per factor: {layout}"}, False,
            f"Absolute {lie_label[group]} measurement or anchor of one state."))
    entries += _vector_entries(None)
    entries += [
        FactorInfo("ReprojectionFactorBatch", ["SE3StateBatch", "VectorStateBatch3"], 2,
                   "P_c = T^-1 P_w; r = P_c.xy / P_c.z - observation",
                   {"observations": "2 floats per factor: normalized image coords (K^-1 applied)"},
                   True, "Bundle adjustment: camera pose (world_from_camera) and 3D point.",
                   "Observations are normalized (divide pixel coords by the intrinsics first). "
                   "Points behind the camera (z < z_threshold) give a zero residual. Has a 7-DOF "
                   "gauge (6 + scale): fix a pose and the scale (a second pose or a prior)."),
        FactorInfo("PnPFactorBatch", ["SE3StateBatch"], 2,
                   "P_c = E T^-1 P_w; r = P_c.xy / P_c.z - observation",
                   {"observations": "2 floats per factor: normalized image coords",
                    "points_world": "3 floats per factor: the known 3D point",
                    "poses_camera_from_rig": "optional E: 16 floats (row-major 4x4) per camera"},
                   False, "Camera pose from known 3D points (PnP); with outliers use the RANSAC "
                   "minimizers.", "Points are constants, not states. z_threshold: minimum "
                   "depth; points closer or behind the camera give a zero residual."),
        FactorInfo("PointToPointFactorBatch", ["SE3StateBatch"], 3, "p - T q",
                   {"p_observations": "3 floats per factor (target point)",
                    "q_observations": "3 floats per factor (source point)"}, False,
                   "Point-set registration (ICP) with known correspondences."),
        FactorInfo("PointToPlaneFactorBatch", ["SE3StateBatch"], 1, "n_q^T (p - T q)",
                   {"p_observations": "3 floats per factor",
                    "q_observations": "3 floats per factor",
                    "nq_observations": "3 floats per factor: plane normal in the target frame"},
                   False, "Point-to-plane ICP."),
        FactorInfo("SymmetricPointToPlaneFactorBatch", ["SE3StateBatch"], 1,
                   "symmetric point-to-plane distance",
                   {"p_observations": "3 floats", "q_observations": "3 floats",
                    "np_observations": "3 floats (normal at p)",
                    "nq_observations": "3 floats per factor (normal at q)"}, False,
                   "Symmetric point-to-plane ICP."),
        FactorInfo("ImuFactorBatch", ["SE3StateBatch", "VectorStateBatch3", "VectorStateBatch6",
                                      "SE3StateBatch", "VectorStateBatch3", "VectorStateBatch6"],
                   15, "preintegrated IMU error between two keyframes (pose, velocity, biases)",
                   {"imu_samples": "raw samples, see docs/sphinx/imu.rst",
                    "sample_offsets": "int32 per factor"}, True,
                   "Visual-inertial estimation; guide docs/sphinx/imu.rst.",
                   "Gravity makes roll and pitch observable; yaw and position still need an "
                   "anchor."),
    ]
    for g, (cls, _a, tan, _l, _o) in (("SE3", _POSE["SE3"]), ("SO3", _POSE["SO3"]),
                                      ("SE2", _POSE["SE2"]), ("SO2", _POSE["SO2"])):
        for model, derivs in (("Velocity", 1), ("Acceleration", 2)):
            # pose_k, pose_{k+1}, then each derivative at k and k+1.
            states = [cls, cls] + [f"VectorStateBatch{tan}"] * (2 * derivs)
            for info in ("", "Information"):
                entries.append(FactorInfo(
                    f"Constant{model}{info}{g}FactorBatch", states,
                    tan * (derivs + 1),
                    f"constant-{model.lower()} motion prior between consecutive {g} states"
                    + (" weighted by Q(dt)^-1" if info else ""),
                    {"time_steps": "1 float per factor (dt)"}
                    | ({"qc_diag": f"{tan} floats (power spectral density)"} if info else {}),
                    True, "Smoothness prior on trajectories (with motion derivatives as states).",
                    "Information variants take a stream first and need "
                    "update(stream, num) after time_steps change." if info else ""))
    for name, states, use in (
            ("SE2KinematicsFactorBatch", ["SE2StateBatch", "VectorStateBatch3", "SE2StateBatch"],
             "X_{k+1} = X_k Exp(dt xi_k) with the body twist as control (MPC, trajectories)."),
            ("SO3KinematicsFactorBatch", ["SO3StateBatch", "VectorStateBatch3", "SO3StateBatch"],
             "R_{k+1} = R_k Exp(dt omega_k) with the body rate as control."),
            ("SE3KinematicsFactorBatch", ["SE3StateBatch", "VectorStateBatch6", "SE3StateBatch"],
             "T_{k+1} = T_k Exp(dt xi_k) with the body twist [omega, v] as control.")):
        entries.append(FactorInfo(name, states, int(states[1][-1]), "Log(X_{k+1}^-1 X_k Exp(dt "
                                  "u_k))-type dynamics residual", {"time_steps": "1 float per "
                                  "factor"}, True, use, "See docs/sphinx/dynamics.rst."))
    for name, use in (
            ("SE2DifferentialDriveFactorBatch", "Differential-drive robot dynamics (MPC)."),
            ("SE2KinematicBicycleFactorBatch", "Car-like (Ackermann) dynamics (MPC)."),
            ("SE3DifferentialDriveFactorBatch", "Differential drive on non-planar terrain."),
            ("SE3KinematicBicycleFactorBatch", "Car-like dynamics on non-planar roads."),
            ("QuadrotorFactorBatch", "Quadrotor rigid-body dynamics with rotor thrusts."),
            ("QuadrupedFactorBatch", "Quadruped base driven by foot forces.")):
        entries.append(FactorInfo(name, [], 0, "dynamics residual (see docs/sphinx/dynamics.rst)",
                                  {"time_steps": "1 float per factor"}, True, use,
                                  "Prefer pycunls.mpc, which wires these for you."))
    entries += [
        FactorInfo("SE2DiskClearanceFactorBatch", ["SE2StateBatch"], 1,
                   "(radius + margin) - |p - center|",
                   {"obstacles": "3 floats per factor: center x, y, radius"},
                   False, "Obstacle avoidance constraint (wrap in ConstraintFactorBatch)."),
        FactorInfo("SE3SphereClearanceFactorBatch", ["SE3StateBatch"], 1,
                   "(radius + margin) - |p - center|",
                   {"obstacles": "4 floats per factor: center x, y, z, radius"},
                   False, "Obstacle avoidance constraint (wrap in ConstraintFactorBatch)."),
        FactorInfo("SE3AxisAlignmentFactorBatch", ["SE3StateBatch", "SE3StateBatch"], 3,
                   "R_1 a - R_0 a", {"axes": "3 floats per factor: the axis a in each pose's frame"},
                   True, "Hinge rows: two poses keep a local axis aligned (with a position "
                   "equality: a revolute joint); usually an equality constraint.",
                   "On robot links through LinkFrameFactorBatch (FrameView.Pose); "
                   "pycunls.articulation.loop_closure builds revolute loops from it."),
        FactorInfo("SE3AxisOffsetFactorBatch", ["SE3StateBatch", "SE3StateBatch"], 2,
                   "(R_0 b_k) . (p_1 - p_0), b_1, b_2 perpendicular to a",
                   {"axes": "3 floats per factor: the axis a in pose 0's frame"},
                   True, "Slider rows: pose 1's origin stays on the axis through pose 0 (with an "
                   "orientation equality: a prismatic joint); usually an equality constraint.",
                   "On robot links through LinkFrameFactorBatch (FrameView.Pose); "
                   "pycunls.articulation.loop_closure builds prismatic loops from it."),
        FactorInfo("InformationFactorBatch", [], 0, "L_f * r_inner",
                   {"sqrt_information_matrices": "residual_size^2 floats per factor (row-major)"},
                   False, "Weight any factor by a per-factor square-root information matrix "
                   "(measurement covariance^-1/2).", "Inherits the wrapped factor's states."),
        FactorInfo("WeightedFactorBatch", [], 0, "w * r_inner",
                   {"weights": "optional 1 float per factor"}, False,
                   "Scale any factor by a weight (1 / sigma).",
                   "Inherits the wrapped factor's states."),
        FactorInfo("ConstraintFactorBatch", [], 0, "scale * r_inner = 0 or <= 0", {}, False,
                   "Turn any factor into hard equality / inequality constraints "
                   "(AugmentedLagrangianMinimizer)."),
    ]
    # Articulations: sizes depend on the ArticulationTopology (0 = per topology).
    entries.append(StateInfo(
        "ArticulationStateBatch", 0, 0,
        "owned: joint_q (topology.coord_count, quaternions x, y, z, w), then a cache "
        "of link poses (12 floats each) and DOF motion columns (6 each)",
        "topology.dof_count values in joint_qd order: a scalar per PRISMATIC/REVOLUTE/D6 axis, "
        "BALL a rotation vector and FREE (dp, dtheta) in the parent-anchor frame",
        update="per joint: scalar coordinates add delta (clamped into the bounds), BALL "
        "quat <- Exp(delta) * quat, FREE position += dp and quat <- Exp(dtheta) * quat; then "
        "forward kinematics refreshes the cache",
        constructor="topology (ArticulationTopology), capacity, and optionally const_state_ids "
        "(int32 GPU array of the instances held fixed) with const_capacity, and "
        "apply_joint_limits (default True); storage is owned: write joint_q with set_joint_q"))
    art = ["ArticulationStateBatch"]
    articulation_note = ("Build ArticulationTopology from the robot's joint arrays or a URDF, "
                         "then ArticulationStateBatch(topology, n); write joint_q with set_joint_q. "
                         "Joint limits are hard bounds: solve with AugmentedLagrangianMinimizer.")
    entries += [
        FactorInfo("LinkPositionFactorBatch", art, 3, "W (X_L(q) c - p*)",
                   {"topology": "ArticulationTopology", "links": "int32 link per factor",
                    "points": "3 floats per factor (link frame)",
                    "targets": "3 floats per factor (world)",
                    "weights": "3 floats per factor or None"}, False,
                   "Inverse kinematics: a point on a link should reach a world position.",
                   articulation_note),
        FactorInfo("LinkOrientationFactorBatch", art, 3, "W Log(R_L(q) R_off R*^T)",
                   {"topology": "ArticulationTopology", "links": "int32 link per factor",
                    "offsets": "quaternion x, y, z, w per factor or None",
                    "targets": "quaternion x, y, z, w per factor",
                    "weights": "3 floats per factor or None"}, False,
                   "Inverse kinematics: a link frame should reach a world orientation.",
                   articulation_note),
        FactorInfo("LinkPoseFactorBatch", art, 6, "W [X_L p_off - p*; Log(R_L R_off R*^T)]",
                   {"topology": "ArticulationTopology", "links": "int32 link per factor",
                    "offsets": "7 floats (position, quaternion) per factor or None",
                    "targets": "7 floats (position, quaternion) per factor",
                    "weights": "6 floats per factor or None"}, False,
                   "Inverse kinematics: a link frame should reach a world pose.",
                   articulation_note),
        FactorInfo("LinkFrameFactorBatch", art, 0,
                   "the inner factor's residual, its frame slots fed a view of X_L(q) X_off",
                   {"topology": "ArticulationTopology",
                    "inner_factor": "factor over SE(3) poses, 3D points or SO(3) rotations "
                                    "(built-in or WarpFactorBatch)",
                    "frame_slots": "list of (inner state slot, FrameView): Pose (SE3, 6), "
                                   "Position (3-vector, 3) or Orientation (SO3, 3)",
                    "topology_links": "int32, len(frame_slots) per factor: topology link index "
                                      "(position in link_names) carrying each frame",
                    "frame_offsets": "capacity x len(frame_slots) x 7 floats (position, "
                                     "quaternion per factor and frame slot, not broadcast): the "
                                     "frame in its link, or None"}, False,
                   "Custom kinematic objectives: existing factors on frames of robot links (SE3 "
                   "prior/between/sphere clearance on a Pose, vector prior/halfspace/between on "
                   "a Position, SO3 prior/between on an Orientation, a Warp factor); the inner "
                   "factor's other slots (an object pose, a camera) stay states.",
                   "States: [articulation instance] + inner_factor's slots that are not frame "
                   "slots. Views use the convention of their state type (Pose: row-major 4x4, "
                   "body tangent [omega, v], X' = X Exp(delta); Position: world 3-vector; "
                   "Orientation: row-major 3x3, R' = R Exp(omega)). For constraints wrap this "
                   "batch in ConstraintFactorBatch, never the inner one. LinkPose/Position/"
                   "OrientationFactorBatch are faster for what they cover. " + articulation_note),
        FactorInfo("PostureFactorBatch", art, 0, "W (q (-) q*)  (topology.dof_count rows)",
                   {"topology": "ArticulationTopology",
                    "targets": "topology.coord_count floats per factor (joint_q layout)",
                    "weights": "topology.dof_count floats per factor or None"}, False,
                   "Regularizer / null-space posture / stay near a requested configuration.",
                   articulation_note),
        FactorInfo("MimicJointFactorBatch", art, 1, "q_m - (a + b q_ref)",
                   {"topology": "ArticulationTopology", "mimic_dofs": "int32 DOF per factor",
                    "reference_dofs": "int32 DOF per factor",
                    "coefficients": "2 floats per factor (a, b)"}, False,
                   "Coupled joints (URDF <mimic>); wrap in "
                   "ConstraintFactorBatch(..., ConstraintKind.Equality).", articulation_note),
    ]
    return {e.name: e for e in entries}


_ENTRIES = _build()


def _entry(name: str) -> Optional[Union[StateInfo, FactorInfo]]:
    entry = _ENTRIES.get(name)
    numbered = _NUMBERED.match(name) if entry is None else None
    if numbered and int(numbered[2]) in VECTOR_DIMS:
        entry = next(e for e in _vector_entries(int(numbered[2])) if e.name == name)
    return entry


def state_info(name: str) -> Optional[StateInfo]:
    """The :class:`StateInfo` of a built-in state batch class name, or None."""
    entry = _entry(name)
    return entry if isinstance(entry, StateInfo) else None


def factor_info(name: str) -> Optional[FactorInfo]:
    """The :class:`FactorInfo` of a built-in factor batch class name, or None."""
    entry = _entry(name)
    return entry if isinstance(entry, FactorInfo) else None


def effective_factor(batch) -> Tuple[object, Optional[FactorInfo], str]:
    """What a factor batch in a problem does, seen through its wrappers.

    Follows ``inner_factor`` of ``WeightedFactorBatch``, ``InformationFactorBatch`` and
    ``ConstraintFactorBatch`` (nested too) to the wrapped batch; stops at a
    ``LinkFrameFactorBatch`` (its inner factor reads link frames, not the problem's states).

    Returns:
        ``(inner batch, its FactorInfo or None, role)`` with role ``"relative"`` (constrains its
        states only relative to each other: between, reprojection, motion factors),
        ``"absolute"`` (a built-in prior or measurement that fixes its states), ``"inactive"``
        (an inequality constraint, which may hold without pulling on its states, so it neither
        connects nor fixes them) or ``"unknown"`` (a custom factor; diagnostics assume it fixes
        its states).
    """
    inner, inequality = batch, False
    while True:
        kind = getattr(inner, "kind", None)
        inequality |= getattr(kind, "name", None) == "Inequality"
        if type(inner).__name__ == "LinkFrameFactorBatch":
            break  # its inner factor reads link frames, not the problem's states
        nested = getattr(inner, "inner_factor", None)
        if nested is None:
            break
        inner = nested
    info = factor_info(type(inner).__name__)
    if inequality:
        role = "inactive"
    elif info is None:
        role = "unknown"
    else:
        role = "relative" if info.relative else "absolute"
    return inner, info, role


def catalog(kind: Optional[str] = None, query: Optional[str] = None) -> List[Dict]:
    """The built-in state and factor batches as a list of dicts.

    Args:
        kind: ``"state"``, ``"factor"`` or None for both.
        query: Case-insensitive text that must appear in the name, formula or use.

    Example::

        [e["name"] for e in pycunls.catalog("factor", "between")]
    """
    out = []
    for entry in _ENTRIES.values():
        if kind is not None and entry.kind != kind:
            continue
        d = dataclasses.asdict(entry)
        if query is not None:
            text = " ".join(str(v) for v in d.values()).lower()
            if query.lower() not in text:
                continue
        out.append(d)
    return out


def describe(name) -> str:
    """A readable description of a built-in state or factor batch (a name or the class).

    Example::

        print(pycunls.describe(pycunls.SE3BetweenFactorBatch))
    """
    if not isinstance(name, str):
        name = name.__name__
    entry = _entry(name)
    if entry is None:
        close = [n for n in _ENTRIES if name.lower() in n.lower()]
        hint = f" Did you mean: {', '.join(close[:8])}?" if close else ""
        raise KeyError(f"{name} is not a built-in state or factor batch.{hint}")
    if isinstance(entry, StateInfo):
        if entry.name == "VectorStateBatch":
            sizes, tangent = "dim floats per state", "dim"
        elif entry.ambient == 0:  # sized by the topology
            sizes = ("topology.ambient_size floats per state (e.g. 19 for one revolute joint: "
                     "1 coordinate, 12 pose floats, 6 motion-column floats)")
            tangent = "topology.dof_count (e.g. 1 for one revolute joint)"
        else:
            sizes, tangent = f"{entry.ambient} floats per state", str(entry.tangent)
        constructor = entry.constructor or (
            f"data (float32 GPU array of capacity x {entry.ambient}), capacity, and optionally "
            "const_state_ids (int32 GPU array of the states held fixed) with const_capacity "
            "(entries it holds)")
        return (f"{entry.name} (state)\n  stored: {entry.layout}, {sizes}\n  tangent ({tangent}): "
                f"{entry.tangent_order}\n  update: {entry.update}\n  constructor: {constructor}")
    lines = [f"{entry.name} (factor)"]
    if entry.states:
        lines.append(f"  states: [{', '.join(entry.states)}]")
    if entry.residual_size:
        lines.append(f"  residual ({entry.residual_size}): {entry.residual}")
    else:
        lines.append(f"  residual: {entry.residual}")
    for arg, layout in entry.arrays.items():
        lines.append(f"  {arg}: {layout}")
    lines.append(f"  relative: {'yes (needs an anchor: a constant state or a prior)' if entry.relative else 'no'}")
    lines.append(f"  use: {entry.use}")
    if entry.notes:
        lines.append(f"  notes: {entry.notes}")
    return "\n".join(lines)


_CAPACITY_NOTE = ("capacity: how many factors (or states) the arrays hold. Batches start with 0 "
                  "active items: call set_num_active_factors(n) / set_num_active_states(n) "
                  "before solving. Arrays are GPU arrays (e.g. wp.array), C-contiguous: "
                  "float32 values, int32 indices, as each argument states.")


def annotate_classes(module) -> None:
    """Appends each built-in class's catalog description to its docstring (``help(cls)``)."""
    for name in _ENTRIES:
        cls = getattr(module, name, None)
        if cls is None or getattr(cls, "_catalog_annotated", False):
            continue
        base = (cls.__doc__ or "").rstrip()
        cls.__doc__ = f"{base}\n\n{describe(name)}\n  {_CAPACITY_NOTE}"
        cls._catalog_annotated = True
