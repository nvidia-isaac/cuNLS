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

* All arrays are float32 GPU arrays (e.g. ``wp.array``), C-contiguous, one item after another.
* Poses are ``world_from_body`` matrices stored row-major (SE3: 4x4 = 16 floats per pose,
  SE2: 3x3, SO3: 3x3, SO2: 2x2). The update is right-multiplicative, ``T' = T Exp(delta)``.
* Tangent order: se(3) = ``[omega (3), v (3)]`` (rotation first), se(2) = ``[v_x, v_y, theta]``
  (translation first) -- see each state's ``tangent_order``.
* Between factors measure ``Delta = T_right^-1 T_left`` (residual ``Log(Delta T_left^-1
  T_right)``), i.e. the *inverse* of the usual relative pose ``T_left^-1 T_right``.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Union


@dataclasses.dataclass(frozen=True)
class StateInfo:
    """A built-in state batch.

    Attributes:
        name: Class name in :mod:`pycunls`.
        ambient: float32 values stored per state.
        tangent: Degrees of freedom (size of one update).
        layout: How one state is stored.
        tangent_order: Meaning of the tangent coordinates.
    """
    name: str
    ambient: int
    tangent: int
    layout: str
    tangent_order: str
    kind: str = "state"


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
_VECTOR_DIMS = (1, 2, 3, 4, 6, 12)


def _build() -> Dict[str, Union[StateInfo, FactorInfo]]:
    entries: List[Union[StateInfo, FactorInfo]] = []
    for group, (cls, amb, tan, layout, order) in _POSE.items():
        entries.append(StateInfo(cls, amb, tan, layout, order))
    for d in _VECTOR_DIMS:
        entries.append(StateInfo(f"VectorStateBatch{d}", d, d, f"{d} floats",
                                 f"the {d} components (Plus is addition)"))

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
    for d in _VECTOR_DIMS:
        entries.append(FactorInfo(
            f"PriorVectorFactorBatch{d}", [f"VectorStateBatch{d}"], d, "x - observation",
            {"observations": f"{d} floats per factor"}, False,
            "Absolute measurement of a vector state; also a weak prior/regularizer (wrap in "
            "WeightedFactorBatch)."))
        entries.append(FactorInfo(
            f"VectorBetweenFactorBatch{d}", [f"VectorStateBatch{d}"] * 2, d,
            "x_left - x_right - delta", {"deltas": f"{d} floats per factor"}, True,
            "Relative measurement between two vector states.",
            "Sign: residual is left - right - delta, so delta = x_left - x_right."))
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
    for d in _VECTOR_DIMS:
        entries.append(FactorInfo(
            f"BoundFactorBatch{d}", [f"VectorStateBatch{d}"], 2 * d,
            "box constraint lower <= x <= upper",
            {"lower": f"{d} floats per factor", "upper": f"{d} floats per factor"}, False,
            "Box bounds on a vector state; needs AugmentedLagrangianMinimizer.",
            "For plain parameter bounds prefer VectorStateBatch.set_bounds (projected)."))
    for d in _VECTOR_DIMS:
        entries.append(FactorInfo(
            f"HalfspaceFactorBatch{d}", [f"VectorStateBatch{d}"], 1, "a^T x - b",
            {"normals": f"{d} floats per factor (a)", "offsets": "1 float per factor (b)"},
            False, "Linear inequality a^T x <= b: wrap in ConstraintFactorBatch(..., "
            "ConstraintKind.Inequality) and solve with AugmentedLagrangianMinimizer."))
    entries += [
        FactorInfo("SE2DiskClearanceFactorBatch", ["SE2StateBatch"], 1,
                   "(radius + margin) - |p - center|",
                   {"obstacles": "3 floats per factor: center x, y, radius"},
                   False, "Obstacle avoidance constraint (wrap in ConstraintFactorBatch)."),
        FactorInfo("SE3SphereClearanceFactorBatch", ["SE3StateBatch"], 1,
                   "(radius + margin) - |p - center|",
                   {"obstacles": "4 floats per factor: center x, y, z, radius"},
                   False, "Obstacle avoidance constraint (wrap in ConstraintFactorBatch)."),
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
    return {e.name: e for e in entries}


_ENTRIES = _build()


def state_info(name: str) -> Optional[StateInfo]:
    """The :class:`StateInfo` of a built-in state batch class name, or None."""
    entry = _ENTRIES.get(name)
    return entry if isinstance(entry, StateInfo) else None


def factor_info(name: str) -> Optional[FactorInfo]:
    """The :class:`FactorInfo` of a built-in factor batch class name, or None."""
    entry = _ENTRIES.get(name)
    return entry if isinstance(entry, FactorInfo) else None


def effective_factor(batch) -> Tuple[object, Optional[FactorInfo], str]:
    """What a factor batch in a problem does, seen through its wrappers.

    Follows ``inner_factor`` of ``WeightedFactorBatch``, ``InformationFactorBatch`` and
    ``ConstraintFactorBatch`` (nested too) to the wrapped batch.

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
    entry = _ENTRIES.get(name)
    if entry is None:
        close = [n for n in _ENTRIES if name.lower() in n.lower()]
        hint = f" Did you mean: {', '.join(close[:8])}?" if close else ""
        raise KeyError(f"{name} is not a built-in state or factor batch.{hint}")
    if isinstance(entry, StateInfo):
        return (f"{entry.name} (state)\n  stored: {entry.layout}, {entry.ambient} floats per state"
                f"\n  tangent ({entry.tangent}): {entry.tangent_order}\n  update: x' = x (+) "
                "delta (right-multiplicative for Lie groups)\n  constructor: data (float32 GPU "
                f"array of capacity x {entry.ambient}), capacity, and optionally "
                "const_state_ids (int32 GPU array of the states held fixed) with const_capacity "
                "(entries it holds)")
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
                  "before solving. Arrays are float32 GPU arrays (e.g. wp.array), C-contiguous.")


def annotate_classes(module) -> None:
    """Appends each built-in class's catalog description to its docstring (``help(cls)``)."""
    for name in _ENTRIES:
        cls = getattr(module, name, None)
        if cls is None or getattr(cls, "_catalog_annotated", False):
            continue
        base = (cls.__doc__ or "").rstrip()
        cls.__doc__ = f"{base}\n\n{describe(name)}\n  {_CAPACITY_NOTE}"
        cls._catalog_annotated = True
