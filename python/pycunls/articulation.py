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

"""Articulations from URDF files or from a scene's joint arrays, and loop-closing joints as
constraints.

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from pycunls import _pycunls_core as _core
from pycunls._arrays import to_numpy
from pycunls._pycunls_core import ArticulationJointType, ArticulationTopology

__all__ = ["UrdfArticulation", "MimicJoint", "topology_from_urdf", "SceneArticulation",
           "topology_from_joint_arrays", "LoopClosure", "loop_closure"]


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
            if math.isnan(lo) or math.isnan(hi):
                raise ValueError(f"topology_from_urdf: joint '{name}' has a NaN limit (got lower "
                                 f"{lo}, upper {hi}, expected numbers). Fix: correct the "
                                 "<limit> or remove it for an unbounded joint.")
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


@dataclasses.dataclass
class LoopClosure:
    """A loop-closing joint as equality constraints (see :func:`loop_closure`).

    Attributes:
        constraints: The ``ConstraintFactorBatch`` objects to add to the problem, each with
            ``problem.add_factor_batch(c, [states], instance)``.
        factors: The ``LinkFrameFactorBatch`` rows inside them, unconstrained (for
            ``pycunls.check_jacobians`` or ``pycunls.evaluate``).
    """
    constraints: List[object]
    factors: List[object]
    _keep: List[object] = dataclasses.field(default_factory=list, repr=False)

    def set_num_active_factors(self, n: int) -> None:
        """Sets the active loop count of every row batch (batches start with 0 active)."""
        for factor in self.factors:
            factor.set_num_active_factors(n)


def loop_closure(topology: ArticulationTopology, joint_type: ArticulationJointType, links,
                 frames, axes=None, capacity: int = 0) -> LoopClosure:
    """A loop-closing joint of every articulation instance, as equality constraints.

    A loop-closing joint keeps two joint frames together in the directions its type does not allow:
    the parent frame ``X_P X_p`` (``X_P`` the parent link's world pose, the identity for the world)
    and the child frame ``X_C X_c``. It is built from existing factors on
    :class:`pycunls.LinkFrameFactorBatch` views of the two frames, each wrapped as an equality:

    * Ball (3 rows): the origins coincide (``VectorBetweenFactorBatch`` on two Position views).
    * Revolute (3 + 3): the origins coincide, and ``SE3AxisAlignmentFactorBatch`` on two Pose
      views keeps the axes aligned.
    * Prismatic (3 + 2): the rotations are equal (``SO3BetweenFactorBatch`` on two Orientation
      views), and ``SE3AxisOffsetFactorBatch`` on two Pose views keeps the child on the axis.
    * Fixed (6): the poses are equal (``SE3BetweenFactorBatch`` on two Pose views).

    Rows may be redundant (a planar four-bar's revolute loop has 6 rows of rank 2; the augmented
    Lagrangian tolerates that). The arrays are used in place, read at every evaluation.

    Args:
        topology: The articulation topology.
        joint_type: ``ArticulationJointType.Ball``, ``Revolute``, ``Prismatic`` or ``Fixed``.
        links: int32 GPU array, 2 per loop: the parent link (-1: the world) and the child link.
        frames: float32 GPU array, 14 per loop: ``joint_X_p`` then ``joint_X_c`` (position,
            quaternion x, y, z, w), the joint frame in the parent and in the child link.
        axes: float32 GPU array, 3 per loop: the joint axis in the joint frame (Revolute,
            Prismatic; may be None for Ball and Fixed).
        capacity: Loops the arrays hold (0: the number of entries of ``links`` / 2, for a flat or
            an (n, 2) array).

    Returns:
        A :class:`LoopClosure`; call ``set_num_active_factors(n)`` and add its ``constraints``.

    Raises:
        ValueError: For another joint type, or a Revolute / Prismatic loop without axes.

    Example::

        loop = loop_closure(topo, pycunls.ArticulationJointType.Revolute, links, frames, axes)
        loop.set_num_active_factors(n)
        for c in loop.constraints:
            problem.add_factor_batch(c, [states], instance)   # one loop per instance
    """
    JT = ArticulationJointType
    if joint_type not in (JT.Ball, JT.Revolute, JT.Prismatic, JT.Fixed):
        raise ValueError(f"loop_closure: unsupported loop-closing joint type (got {joint_type}, "
                         "expected Ball, Revolute, Prismatic or Fixed). Fix: model the joint with "
                         "one of these types.")
    if joint_type in (JT.Revolute, JT.Prismatic) and axes is None:
        raise ValueError(f"loop_closure: axes is None for a {joint_type.name} loop (expected 3 "
                         "floats per loop: the joint axis). Fix: pass the axes array.")
    size = getattr(links, "size", None)
    n = int(capacity) if capacity else int(size if size is not None else len(links)) // 2
    # The Warp device of `links` (a zero-copy view also resolves CuPy / PyTorch arrays).
    device = links.device if isinstance(links, wp.array) else None
    if device is None:
        try:
            device = wp.array(links, dtype=wp.int32, copy=False).device
        except Exception:
            device = None
        if device is None or not device.is_cuda:
            device = wp.get_cuda_device()
    keep: List[object] = []

    def identity(rows: int) -> wp.array:
        values = np.tile(np.eye(rows, dtype=np.float32).reshape(-1), n)
        array = wp.array(values, dtype=wp.float32, device=device)
        keep.append(array)
        return array

    constraints, factors = [], []

    def add(inner, view) -> None:
        frame = _core.LinkFrameFactorBatch(topology, inner, frame_slots=[(0, view), (1, view)],
                                           topology_links=links, frame_offsets=frames)
        factors.append(frame)
        constraints.append(_core.ConstraintFactorBatch(frame, _core.ConstraintKind.Equality))
        keep.append(inner)

    if joint_type in (JT.Ball, JT.Revolute):
        zeros = wp.zeros(3 * n, dtype=wp.float32, device=device)
        keep.append(zeros)
        add(_core.VectorBetweenFactorBatch(zeros, n, dim=3), _core.FrameView.Position)
    if joint_type == JT.Revolute:
        add(_core.SE3AxisAlignmentFactorBatch(axes, n), _core.FrameView.Pose)
    if joint_type == JT.Prismatic:
        add(_core.SO3BetweenFactorBatch(identity(3), n), _core.FrameView.Orientation)
        add(_core.SE3AxisOffsetFactorBatch(axes, n), _core.FrameView.Pose)
    if joint_type == JT.Fixed:
        add(_core.SE3BetweenFactorBatch(identity(4), n), _core.FrameView.Pose)
    wp.synchronize_device(device)  # the uploads above run on Warp's stream
    return LoopClosure(constraints, factors, keep)


@dataclasses.dataclass(frozen=True)
class SceneArticulation:
    """One articulation selected from a scene's joint arrays (see
    :func:`topology_from_joint_arrays`), with the maps between scene and topology indices.

    Attributes:
        topology: The :class:`pycunls.ArticulationTopology` (local joints, links, coordinates
            and DOFs).
        joints: int32, per local joint (= per local link): its scene joint index.
        bodies: int32, per local link: its scene body index (use it to read targets for link
            factors from scene bodies, or to scatter ``get_link_transforms`` into a scene pose
            array).
        link_of_body: int32, per scene body (up to the largest index the arrays name): its local
            link, or -1 for a body outside this articulation (use it to turn scene body indices
            into the ``links`` / ``topology_links`` of factors).
        coord_index: int32, per local coordinate: its index in the scene's ``joint_q``.
        dof_index: int32, per local DOF: its index in the scene's ``joint_qd`` (and per-DOF
            arrays such as axes and limits).
        coord_start: The scene ``joint_q`` index of local coordinate 0 when ``coord_index`` is a
            contiguous range (then ``set_joint_q(scene_q[coord_start:], ...)`` reads the
            articulation in place, with a ``stride`` for replicated worlds), else None.
        loop_joints: int32, scene joints of the selection left out of the tree because their child
            body is already the child of an earlier selected joint: loop-closing joints, to add as
            constraints (:func:`loop_closure`, with parent and child mapped by ``link_of_body``).
    """
    topology: ArticulationTopology
    joints: np.ndarray
    bodies: np.ndarray
    link_of_body: np.ndarray
    coord_index: np.ndarray
    dof_index: np.ndarray
    coord_start: Optional[int]
    loop_joints: np.ndarray


def topology_from_joint_arrays(joint_type, joint_parent, joint_child, joint_X_p, joint_X_c,
                               joint_q_start, joint_qd_start, joint_axis, joint_dof_dim=None,
                               limit_lower=None, limit_upper=None, joints=None,
                               root_pose=None) -> SceneArticulation:
    """The topology of one articulation of a scene, from the scene's joint arrays.

    A scene (a simulator model with several robots or replicated worlds) stores every joint of
    every articulation in flat arrays with **scene** indices: parents and children are scene
    bodies, ``joint_q_start`` / ``joint_qd_start`` offsets into the scene's ``joint_q`` /
    ``joint_qd``, and per-DOF arrays (axes, limits) are indexed by scene DOF.
    :class:`pycunls.ArticulationTopology` wants one articulation with **local** indices (link
    *i* is the child of joint *i*, offsets from 0). This function selects the joints of one
    articulation and does that remapping:

    * ``joints`` selects scene joints (default: all); their order is the local joint order.
    * The first selected joint with a given child body is a tree joint; a later one with the same
      child is a loop-closing joint, returned in ``loop_joints`` and left out of the tree.
    * Local link *i* is the child body of local joint *i*; a parent body that is -1 stays -1 (the
      root frame, ``root_pose``).
    * Coordinates, DOFs, axes and limits are sliced and rebased per joint.

    Arrays may be NumPy, lists, or GPU arrays (copied once to the host). Shapes follow the
    :class:`pycunls.ArticulationTopology` arguments with scene sizes: ``joint_X_p`` /
    ``joint_X_c`` 7 floats per joint (position, quaternion x, y, z, w; e.g. ``wp.transform``),
    ``joint_q_start`` / ``joint_qd_start`` one offset per joint plus the total, ``joint_axis``
    3 floats per DOF, ``joint_dof_dim`` 2 ints per joint (needed with D6 joints), limits one float
    per DOF.

    Args:
        joints: Scene joint indices of the articulation (an int array, a list or a ``range``,
            e.g. ``range(articulation_start[a], articulation_start[a + 1])``), or None for all.
        root_pose: Pose of the frame joints with parent -1 attach to (7 floats), default identity.

    Returns:
        A :class:`SceneArticulation`: the topology and the scene/local index maps.

    Raises:
        ValueError: A selected joint whose parent body is neither -1 nor the child of a
            selected tree joint (the articulation is cut off from its root), or a joint index out
            of range; topology errors (cycles, bad offsets) come from ``ArticulationTopology``.

    Example::

        art = topology_from_joint_arrays(model.joint_type, model.joint_parent, model.joint_child,
                                         model.joint_X_p, model.joint_X_c, model.joint_q_start,
                                         model.joint_qd_start, model.joint_axis,
                                         model.joint_dof_dim, model.joint_limit_lower,
                                         model.joint_limit_upper, joints=range(0, 9))
        hand = art.link_of_body[hand_body]                       # a link index for the factors
        states.set_joint_q(scene_q[art.coord_start:], worlds, stream, stride=coords_per_world)
    """
    owner = "topology_from_joint_arrays"
    jt = to_numpy(joint_type, np.int32, "joint_type", owner).reshape(-1)
    parent = to_numpy(joint_parent, np.int32, "joint_parent", owner).reshape(-1)
    child = to_numpy(joint_child, np.int32, "joint_child", owner).reshape(-1)
    xp = to_numpy(joint_X_p, np.float32, "joint_X_p", owner).reshape(-1, 7)
    xc = to_numpy(joint_X_c, np.float32, "joint_X_c", owner).reshape(-1, 7)
    q_start = to_numpy(joint_q_start, np.int32, "joint_q_start", owner).reshape(-1)
    qd_start = to_numpy(joint_qd_start, np.int32, "joint_qd_start", owner).reshape(-1)
    axis = to_numpy(joint_axis, np.float32, "joint_axis", owner).reshape(-1, 3)
    num_joints = jt.size
    for name, offsets, total in (("joint_q_start", q_start, None),
                                 ("joint_qd_start", qd_start, axis.shape[0])):
        if (offsets.size != num_joints + 1 or offsets[0] < 0 or np.any(np.diff(offsets) < 0)
                or (total is not None and offsets[-1] > total)):
            raise ValueError(f"{owner}: {name} is not a valid offset array (got {offsets.size} "
                             f"entries from {int(offsets[0]) if offsets.size else 'none'}, expected "
                             f"{num_joints + 1} nonnegative, nondecreasing offsets, one per joint "
                             "plus the total). Fix: pass the scene's offset arrays unchanged.")
    selected = (np.arange(num_joints, dtype=np.int32) if joints is None
                else np.asarray(list(joints) if isinstance(joints, range) else
                                to_numpy(joints, np.int32, "joints", owner)).reshape(-1)
                .astype(np.int32))
    bad = selected[(selected < 0) | (selected >= num_joints)]
    if bad.size:
        raise ValueError(f"{owner}: joint {int(bad[0])} is out of range (got {int(bad[0])}, "
                         f"expected 0 .. {num_joints - 1}). Fix: select scene joint indices.")

    tree, loops, claimed = [], [], set()
    for j in selected:
        (loops if int(child[j]) in claimed else tree).append(int(j))
        claimed.add(int(child[j]))
    tree = np.asarray(tree, np.int32)
    bodies = child[tree].astype(np.int32)
    size = int(max(parent.max(initial=-1), child.max(initial=-1))) + 1
    link_of_body = np.full(size, -1, np.int32)
    link_of_body[bodies] = np.arange(tree.size, dtype=np.int32)
    local_parent = np.empty(tree.size, np.int32)
    for i, j in enumerate(tree):
        p = int(parent[j])
        local_parent[i] = -1 if p < 0 else link_of_body[p]
        if p >= 0 and local_parent[i] < 0:
            raise ValueError(f"{owner}: joint {int(j)}'s parent body {p} is not a link of the "
                             f"selection (got parent {p}, expected -1 or the child body of a "
                             "selected tree joint). Fix: select the joints from the root, or "
                             "give the joint parent -1 and place the root with root_pose.")

    coords = [np.arange(q_start[j], q_start[j + 1], dtype=np.int32) for j in tree]
    dofs = [np.arange(qd_start[j], qd_start[j + 1], dtype=np.int32) for j in tree]
    coord_index = np.concatenate(coords) if coords else np.zeros(0, np.int32)
    dof_index = np.concatenate(dofs) if dofs else np.zeros(0, np.int32)
    local_q_start = np.concatenate([[0], np.cumsum([c.size for c in coords])]).astype(np.int32)
    local_qd_start = np.concatenate([[0], np.cumsum([d.size for d in dofs])]).astype(np.int32)
    contiguous = coord_index.size == 0 or np.array_equal(
        coord_index, np.arange(coord_index[0], coord_index[0] + coord_index.size))
    kwargs = {}
    if joint_dof_dim is not None:
        kwargs["joint_dof_dim"] = to_numpy(joint_dof_dim, np.int32, "joint_dof_dim",
                                           owner).reshape(-1, 2)[tree].reshape(-1)
    for name, values in (("limit_lower", limit_lower), ("limit_upper", limit_upper)):
        if values is not None:
            kwargs[name] = to_numpy(values, np.float32, name, owner).reshape(-1)[dof_index]
    if root_pose is not None:
        kwargs["root_pose"] = to_numpy(root_pose, np.float32, "root_pose", owner).reshape(-1)
    topology = ArticulationTopology(
        joint_type=jt[tree], joint_parent=local_parent,
        joint_child=np.arange(tree.size, dtype=np.int32), joint_X_p=xp[tree].reshape(-1),
        joint_X_c=xc[tree].reshape(-1), joint_q_start=local_q_start,
        joint_qd_start=local_qd_start, joint_axis=axis[dof_index].reshape(-1), **kwargs)
    return SceneArticulation(topology, tree, bodies, link_of_body, coord_index, dof_index,
                             int(coord_index[0]) if contiguous and coord_index.size else None,
                             np.asarray(loops, np.int32))
