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

"""Articulations from URDF files.

:func:`topology_from_urdf` reads the kinematic tree of a URDF file (joints, origins, axes,
limits, mimic joints; no meshes, inertias or transmissions) into an
:class:`pycunls.ArticulationTopology`, so cuNLS can be used for inverse kinematics without a
simulator. Robots already loaded elsewhere (a simulator, a robot description library) pass their joint arrays to
``ArticulationTopology`` directly.

Example::

    import pycunls
    from pycunls.articulation import topology_from_urdf

    robot = topology_from_urdf("fr3.urdf")
    states = pycunls.ArticulationStateBatch(robot.topology, 1024)   # limits are hard bounds
    hand = robot.link_names.index("fr3_hand")
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ElementTree
from typing import Dict, List, Tuple

import numpy as np

from pycunls._pycunls_core import ArticulationJointType, ArticulationTopology

__all__ = ["UrdfArticulation", "MimicJoint", "topology_from_urdf"]


@dataclasses.dataclass(frozen=True)
class MimicJoint:
    """A URDF ``<mimic>`` coupling, ready for :class:`pycunls.MimicJointFactorBatch`.

    Attributes:
        mimic_dof: DOF index of the mimic joint.
        reference_dof: DOF index of the joint it follows.
        offset: ``a`` in ``q_m = a + b q_ref``.
        multiplier: ``b``.
    """
    mimic_dof: int
    reference_dof: int
    offset: float
    multiplier: float


@dataclasses.dataclass
class UrdfArticulation:
    """A URDF robot as an articulation.

    Attributes:
        topology: The :class:`pycunls.ArticulationTopology`.
        link_names: Name of each link (index = link index of the factors).
        joint_names: Name of each topology joint; the first is the added root joint
            (``"<root>"``), FIXED or FREE.
        dof_names: Joint name of each DOF (FREE root DOFs: ``"<root>:x"`` ...).
        coord_names: Joint name of each joint_q coordinate.
        mimic_joints: The ``<mimic>`` couplings (enforce them with MimicJointFactorBatch).
    """
    topology: ArticulationTopology
    link_names: List[str]
    joint_names: List[str]
    dof_names: List[str]
    coord_names: List[str]
    mimic_joints: List[MimicJoint]


def _floats(text: str, count: int, default: Tuple[float, ...]) -> List[float]:
    if text is None:
        return list(default)
    values = [float(v) for v in text.split()]
    if len(values) != count:
        raise ValueError(f"topology_from_urdf: expected {count} numbers (got '{text}'). "
                         "Fix: correct the URDF attribute.")
    return values


def _rpy_quaternion(r: float, p: float, y: float) -> List[float]:
    """Quaternion (x, y, z, w) of URDF roll-pitch-yaw (R = Rz(y) Ry(p) Rx(r))."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def topology_from_urdf(path: str, floating_base: bool = False) -> UrdfArticulation:
    """Reads the kinematic tree of a URDF file.

    The root link (the one that is no joint's child) is attached to the world by an added root
    joint: FIXED, or FREE with ``floating_base``. Joint types: revolute, continuous (revolute
    without limits), prismatic, fixed, floating (FREE). ``<limit lower upper>`` become joint
    limits (hard bounds of :class:`pycunls.ArticulationStateBatch`); an omitted ``lower`` or
    ``upper`` is 0, as the URDF specification defines. Continuous joints and revolute joints
    without ``<limit>`` are unbounded, as are limits of magnitude 1e5 or more (the topology's
    "unbounded" convention, which URDFs also use for "no limit"; ``topology.lower_limits`` /
    ``upper_limits`` report them as infinite).

    Args:
        path: The URDF file.
        floating_base: Attach the root link with a FREE joint (6 DOFs) instead of FIXED.

    Returns:
        :class:`UrdfArticulation`.

    Raises:
        ValueError: An unsupported joint type (planar), several roots, a duplicate link, a joint
            naming an undeclared link, a mimic joint that names no revolute or prismatic joint,
            or a malformed number.
    """
    root = ElementTree.parse(path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    duplicates = sorted({name for name in links if links.count(name) > 1})
    if duplicates:
        raise ValueError(f"topology_from_urdf: links declared more than once (got {duplicates}, "
                         "expected unique names). Fix: rename or remove the duplicates.")
    joints = root.findall("joint")
    for j in joints:
        for end in ("parent", "child"):
            name = j.find(end).get("link")
            if name not in links:
                raise ValueError(f"topology_from_urdf: joint '{j.get('name')}' names {end} link "
                                 f"'{name}', which no <link> declares. Fix: declare the link or "
                                 "correct the joint.")
    children = {j.find("child").get("link") for j in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"topology_from_urdf: the URDF has {len(roots)} root links (got "
                         f"{roots}, expected exactly 1). Fix: connect the links into one tree.")
    link_index: Dict[str, int] = {}
    order = [roots[0]] + [j.find("child").get("link") for j in joints]
    for name in order:
        link_index[name] = len(link_index)

    joint_type, parent, child, X_p, X_c, axes, lower, upper, dof_dim = ([] for _ in range(9))
    q_start, qd_start = [0], [0]
    joint_names, dof_names, coord_names = [], [], []
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    inf = float("inf")

    def add(name, jtype, p, c, origin, axis, lo, hi):
        joint_type.append(jtype.value)
        parent.append(p)
        child.append(c)
        X_p.extend(origin)
        X_c.extend(identity)
        dof_dim.extend([0, 0])
        joint_names.append(name)
        coords, dofs = {ArticulationJointType.Revolute: (1, 1),
                        ArticulationJointType.Prismatic: (1, 1),
                        ArticulationJointType.Fixed: (0, 0),
                        ArticulationJointType.Free: (7, 6)}[jtype]
        q_start.append(q_start[-1] + coords)
        qd_start.append(qd_start[-1] + dofs)
        for k in range(dofs):
            axes.extend(axis if dofs == 1 else [0.0, 0.0, 0.0])
            lower.append(lo if dofs == 1 else -inf)
            upper.append(hi if dofs == 1 else inf)
            dof_names.append(name if dofs == 1 else f"{name}:{'xyzabc'[k]}")
        coord_names.extend([name] * coords)

    add("<root>", ArticulationJointType.Free if floating_base else ArticulationJointType.Fixed,
        -1, 0, identity, None, -inf, inf)
    dof_of_joint: Dict[str, int] = {}
    mimics = []
    for j in joints:
        name, kind = j.get("name"), j.get("type")
        origin = j.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0))
        rpy = _floats(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0))
        axis_el = j.find("axis")
        axis = _floats(axis_el.get("xyz") if axis_el is not None else None, 3, (1.0, 0.0, 0.0))
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis = [a / norm for a in axis]
        limit = j.find("limit")
        lo, hi = -inf, inf
        if kind in ("revolute", "prismatic") and limit is not None:
            lo = float(limit.get("lower", "0"))
            hi = float(limit.get("upper", "0"))
        jtype = {"revolute": ArticulationJointType.Revolute,
                 "continuous": ArticulationJointType.Revolute,
                 "prismatic": ArticulationJointType.Prismatic,
                 "fixed": ArticulationJointType.Fixed,
                 "floating": ArticulationJointType.Free}.get(kind)
        if jtype is None:
            raise ValueError(f"topology_from_urdf: joint '{name}' has unsupported type '{kind}' "
                             "(expected revolute, continuous, prismatic, fixed or floating). "
                             "Fix: model it with a supported joint.")
        if jtype in (ArticulationJointType.Revolute, ArticulationJointType.Prismatic):
            dof_of_joint[name] = qd_start[-1]
        add(name, jtype, link_index[j.find("parent").get("link")],
            link_index[j.find("child").get("link")], xyz + _rpy_quaternion(*rpy), axis, lo, hi)
        mimic = j.find("mimic")
        if mimic is not None:
            mimics.append((name, mimic.get("joint"), float(mimic.get("offset", "0")),
                           float(mimic.get("multiplier", "1"))))

    topology = ArticulationTopology(
        joint_type=joint_type, joint_parent=parent, joint_child=child, joint_X_p=X_p,
        joint_X_c=X_c, joint_q_start=q_start, joint_qd_start=qd_start,
        joint_axis=np.asarray(axes, dtype=np.float32), joint_dof_dim=dof_dim,
        limit_lower=lower, limit_upper=upper)
    for m, r, _, _ in mimics:
        if m not in dof_of_joint or r not in dof_of_joint:
            raise ValueError(f"topology_from_urdf: joint '{m}' mimics '{r}', but both must be "
                             "revolute, continuous or prismatic joints of the file (one DOF "
                             "each). Fix: correct the <mimic joint> name or remove the coupling.")
    mimic_joints = [MimicJoint(dof_of_joint[m], dof_of_joint[r], off, mul)
                    for m, r, off, mul in mimics]
    return UrdfArticulation(topology, [n for n in order], joint_names, dof_names, coord_names,
                            mimic_joints)
