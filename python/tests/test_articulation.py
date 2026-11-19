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

"""Articulations from Python: topology from arrays and URDF, bounded IK, a closed loop, Jacobian
checks of every kinematic factor."""

import gc
import math
import weakref

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.articulation import topology_from_urdf

JT = pycunls.ArticulationJointType

URDF = """<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="tool"/>
  <link name="finger_a"/><link name="finger_b"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/><limit lower="-2.5" upper="2.5"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.6" upper="1.6"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/><limit lower="-2.0" upper="2.0"/></joint>
  <joint name="j4" type="continuous"><parent link="l3"/><child link="tool"/>
    <origin xyz="0 0 0.2" rpy="0 0.3 0"/><axis xyz="0 0 1"/></joint>
  <joint name="fa" type="prismatic"><parent link="tool"/><child link="finger_a"/>
    <origin xyz="0 0.02 0.1"/><axis xyz="0 1 0"/><limit lower="0" upper="0.04"/></joint>
  <joint name="fb" type="prismatic"><parent link="tool"/><child link="finger_b"/>
    <origin xyz="0 -0.02 0.1"/><axis xyz="0 -1 0"/><limit lower="0" upper="0.04"/>
    <mimic joint="fa" multiplier="1" offset="0"/></joint>
</robot>
"""


@pytest.fixture()
def arm(tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text(URDF)
    return topology_from_urdf(str(path))


def _one_factor_per_instance(problem, states, factors, n):
    """Factor i of every batch acts on instance i; each instance is its own subproblem."""
    instance = wp.array(np.arange(n, dtype=np.int32))
    for fb in factors:
        problem.add_factor_batch(fb, [states], instance)
    problem.set_problem_partition(n, [instance])
    return instance


def test_urdf_topology(arm):
    topo = arm.topology
    assert topo.joint_count == 7 and topo.dof_count == 6 and topo.coord_count == 6
    assert arm.link_names[0] == "base" and arm.joint_names[0] == "<root>"
    assert arm.dof_names == ["j1", "j2", "j3", "j4", "fa", "fb"]
    assert math.isinf(topo.lower_limits[3])  # continuous
    assert arm.mimic_joints[0].mimic_dof == 5 and arm.mimic_joints[0].reference_dof == 4


def _urdf(body):
    return f'<robot name="r"><link name="base"/>{body}</robot>'


def test_urdf_limits_follow_the_specification(tmp_path):
    """An omitted lower/upper is 0; |limit| >= 1e5 is the unbounded convention."""
    path = tmp_path / "r.urdf"
    path.write_text(_urdf(
        '<link name="a"/><link name="b"/>'
        '<joint name="ja" type="prismatic"><parent link="base"/><child link="a"/>'
        '<limit upper="0.2" effort="1" velocity="1"/></joint>'
        '<joint name="jb" type="revolute"><parent link="a"/><child link="b"/>'
        '<limit lower="-1e16" upper="1e16" effort="1" velocity="1"/></joint>'))
    topo = topology_from_urdf(str(path)).topology
    assert list(topo.lower_limits[:1]) == [0.0] and topo.upper_limits[0] == pytest.approx(0.2)
    assert math.isinf(topo.lower_limits[1]) and math.isinf(topo.upper_limits[1])


@pytest.mark.parametrize("body, match", [
    ('<link name="a"/><link name="a"/>', "more than once"),
    ('<joint name="j" type="fixed"><parent link="base"/><child link="ghost"/></joint>',
     "no <link> declares"),
    ('<link name="a"/><joint name="j" type="revolute"><parent link="base"/><child link="a"/>'
     '<mimic joint="missing"/></joint>', "mimics 'missing'"),
    ('<link name="a"/><joint name="j" type="revolute"><parent link="base"/><child link="a"/>'
     '<limit lower="nan" upper="1"/></joint>', "NaN"),
])
def test_urdf_rejects_inconsistent_files(tmp_path, body, match):
    path = tmp_path / "r.urdf"
    path.write_text(_urdf(body))
    with pytest.raises(ValueError, match=match):
        topology_from_urdf(str(path))


def test_set_bounds_keeps_only_the_current_arrays(arm, stream):
    """Replaced bounds arrays are released, not accumulated for the batch's lifetime."""
    topo, n = arm.topology, 2
    states = pycunls.ArticulationStateBatch(topo, n)
    shape = n * topo.dof_count

    def install(lo, hi):  # the test keeps only weak references
        lower, upper = wp.full(shape, lo, dtype=float), wp.full(shape, hi, dtype=float)
        states.set_bounds(lower, upper)
        return weakref.ref(lower), weakref.ref(upper)

    first = install(-1.0, 1.0)
    gc.collect()
    assert all(r() is not None for r in first)          # held while they are the bounds
    second = install(-0.5, 0.5)
    gc.collect()
    assert all(r() is None for r in first)              # released once replaced
    assert all(r() is not None for r in second)
    states.use_joint_limits()
    gc.collect()
    assert all(r() is None for r in second) and states.has_bounds


def test_topology_errors_say_what_and_how_to_fix():
    with pytest.raises(ValueError, match="Fix:"):
        pycunls.ArticulationTopology(
            joint_type=[JT.Revolute.value], joint_parent=[-1], joint_child=[0],
            joint_X_p=[0, 0, 0, 0, 0, 0, 1], joint_X_c=[0, 0, 0, 0, 0, 0, 1],
            joint_q_start=[0, 1], joint_qd_start=[0, 1], joint_axis=[0, 0, 2])


def _link_transforms(states, n, stream):
    """World pose (position, quaternion x, y, z, w) of every link: (n, link_count, 7)."""
    out = wp.zeros((n, states.topology.link_count), dtype=wp.transform)
    states.get_link_transforms(out, n, stream)
    wp.synchronize()
    return out.numpy()


def _fk_tool(states, arm, n, stream):
    """World position of the tool link of the first n instances."""
    return _link_transforms(states, n, stream)[:, arm.link_names.index("tool"), :3]


def test_bounded_ik_with_mimic(arm, stream):
    topo, n = arm.topology, 128
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(topo.lower_limits), topo.lower_limits, -1.0)
    hi = np.where(np.isfinite(topo.upper_limits), topo.upper_limits, 1.0)
    q_true = (lo + (hi - lo) * rng.uniform(0.1, 0.9, (n, topo.coord_count))).astype(np.float32)
    q_true[:, 5] = q_true[:, 4]  # mimic

    # Targets: FK of q_true, computed by a constant-free solve-free path (the state cache).
    probe = pycunls.ArticulationStateBatch(topo, n, apply_joint_limits=False)
    probe.set_num_active_states(n)
    probe.set_joint_q(wp.array(q_true), n, stream)
    wp.synchronize()
    targets = wp.array(_fk_tool(probe, arm, n, stream).astype(np.float32))

    states = pycunls.ArticulationStateBatch(topo, n)
    states.set_num_active_states(n)
    q0 = np.clip(q_true + rng.uniform(-0.3, 0.3, q_true.shape), lo, hi).astype(np.float32)
    states.set_joint_q(wp.array(q0), n, stream)
    links = wp.array(np.full(n, arm.link_names.index("tool"), np.int32))
    position = pycunls.LinkPositionFactorBatch(topo, links, wp.zeros(3 * n, dtype=float), targets,
                                               None, n)
    position.set_num_active_factors(n)
    nominal = wp.array(np.repeat(q0, 1, axis=0).astype(np.float32))
    posture = pycunls.PostureFactorBatch(topo, nominal, wp.full(n * topo.dof_count, 1e-3,
                                                                dtype=float), n)
    posture.set_num_active_factors(n)
    mimic = arm.mimic_joints[0]
    mim = pycunls.MimicJointFactorBatch(topo, wp.full(n, mimic.mimic_dof, dtype=wp.int32),
                                        wp.full(n, mimic.reference_dof, dtype=wp.int32),
                                        wp.array(np.tile([mimic.offset, mimic.multiplier],
                                                         n).astype(np.float32)), n)
    mim.set_num_active_factors(n)
    coupled = pycunls.ConstraintFactorBatch(mim, pycunls.ConstraintKind.Equality)

    problem = pycunls.Problem()
    problem.add_state_batch(states)
    _one_factor_per_instance(problem, states, (position, posture, coupled), n)
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    base = lm.base_options
    base.cost_tolerance = 1e-12  # absolute: residuals are millimeters
    base.state_tolerance = 1e-12
    base.max_num_iterations = 100
    lm.base_options = base
    solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(lm))
    wp.synchronize()
    summary = solver.minimize(stream, problem)
    assert summary.max_violation < 1e-4, summary.message

    q = wp.zeros(n * topo.coord_count, dtype=float)
    states.get_joint_q(q, n, stream)
    wp.synchronize()
    q = q.numpy().reshape(n, topo.coord_count)
    finite = np.isfinite(topo.lower_limits)
    assert np.all(q[:, finite] >= np.asarray(topo.lower_limits)[finite] - 1e-6)
    assert np.all(q[:, finite] <= np.asarray(topo.upper_limits)[finite] + 1e-6)
    np.testing.assert_allclose(q[:, 5], q[:, 4], atol=1e-4)
    err = np.linalg.norm(_fk_tool(states, arm, n, stream) - targets.numpy(), axis=1)
    assert np.mean(err < 1e-3) > 0.95, np.sort(err)[-10:]


def test_every_kinematic_factor_passes_check_jacobians(arm, stream):
    topo, n = arm.topology, 16
    rng = np.random.default_rng(1)
    states = pycunls.ArticulationStateBatch(topo, n, apply_joint_limits=False)
    states.set_num_active_states(n)
    states.set_joint_q(wp.array(rng.uniform(-0.8, 0.8, (n, topo.coord_count)).astype(np.float32)),
                       n, stream)
    links = wp.array((np.arange(n) % topo.link_count).astype(np.int32))

    def quats(k):
        v = np.concatenate([rng.uniform(-0.3, 0.3, (k, 3)), np.ones((k, 1))], axis=1)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    poses = np.concatenate([rng.uniform(-1, 1, (n, 3)), quats(n)], axis=1).astype(np.float32)
    frames = np.concatenate([poses, poses], axis=1).astype(np.float32)
    factors = [
        pycunls.LinkPositionFactorBatch(topo, links, wp.array(rng.uniform(-.1, .1, 3 * n)
                                                              .astype(np.float32)),
                                        wp.array(rng.uniform(-1, 1, 3 * n).astype(np.float32)),
                                        None, n),
        pycunls.LinkOrientationFactorBatch(topo, links, None, wp.array(quats(n)), None, n),
        pycunls.LinkPoseFactorBatch(topo, links, None, wp.array(poses), None, n),
        pycunls.PostureFactorBatch(topo, wp.zeros(n * topo.coord_count, dtype=float), None, n),
        pycunls.LoopClosureFactorBatch(topo, JT.Revolute, wp.array(np.full(n, 2, np.int32)),
                                       wp.array(np.full(n, 5, np.int32)), wp.array(frames),
                                       wp.array(np.tile([0, 0, 1], n).astype(np.float32)), n),
        pycunls.MimicJointFactorBatch(topo, wp.full(n, 5, dtype=wp.int32),
                                      wp.full(n, 4, dtype=wp.int32),
                                      wp.array(np.tile([0.1, 2.0], n).astype(np.float32)), n),
    ]
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    ptrs = [states.state_device_ptr(i) for i in range(n)]
    for fb in factors:
        fb.set_num_active_factors(n)
        problem.add_factor_batch(fb, ptrs)
    wp.synchronize()
    for check in pycunls.check_jacobians(problem, stream):
        assert check.ok, check


def test_check_jacobians_at_a_joint_limit(arm, stream):
    """Perturbations ignore the bounds: a central difference at a limit is not one-sided."""
    topo, n = arm.topology, 4
    states = pycunls.ArticulationStateBatch(topo, n)              # joint limits as bounds
    states.set_num_active_states(n)
    q = np.zeros((n, topo.coord_count), np.float32)
    q[:, 4] = topo.upper_limits[4]                                # finger "fa" at its upper limit
    states.set_joint_q(wp.array(q), n, stream)
    target = wp.array(np.full(n * topo.coord_count, 0.01, np.float32))
    posture = pycunls.PostureFactorBatch(topo, target, None, n)
    posture.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(posture, [states.state_device_ptr(i) for i in range(n)])
    for check in pycunls.check_jacobians(problem, stream):
        assert check.ok, check


def test_four_bar_closes(arm, stream):
    """Ground, crank, coupler, rocker; the coupler-rocker hinge closes the loop."""
    ident = [0, 0, 0, 0, 0, 0, 1]
    topo = pycunls.ArticulationTopology(
        joint_type=[JT.Fixed.value, JT.Revolute.value, JT.Revolute.value, JT.Revolute.value],
        joint_parent=[-1, 0, 1, 0], joint_child=[0, 1, 2, 3],
        joint_X_p=ident + ident + [0.4, 0, 0, 0, 0, 0, 1] + [1.0, 0, 0, 0, 0, 0, 1],
        joint_X_c=ident * 4, joint_q_start=[0, 0, 1, 2, 3], joint_qd_start=[0, 0, 1, 2, 3],
        joint_axis=[0, 0, 1] * 3)
    n = 16
    states = pycunls.ArticulationStateBatch(topo, n)
    states.set_num_active_states(n)
    q0 = np.stack([np.linspace(0.3, 1.5, n), np.zeros(n), np.full(n, 1.2)], 1).astype(np.float32)
    states.set_joint_q(wp.array(q0), n, stream)
    posture = pycunls.PostureFactorBatch(topo, wp.array(q0), wp.array(
        np.tile([100.0, 0.01, 0.01], n).astype(np.float32)), n)
    frames = np.tile([1.0, 0, 0, 0, 0, 0, 1, 0.8, 0, 0, 0, 0, 0, 1], n).astype(np.float32)
    loop = pycunls.LoopClosureFactorBatch(topo, JT.Revolute, wp.full(n, 2, dtype=wp.int32),
                                          wp.full(n, 3, dtype=wp.int32), wp.array(frames),
                                          wp.array(np.tile([0, 0, 1], n).astype(np.float32)), n)
    for fb in (posture, loop):
        fb.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    _one_factor_per_instance(
        problem, states,
        (posture, pycunls.ConstraintFactorBatch(loop, pycunls.ConstraintKind.Equality)), n)
    options = pycunls.AugmentedLagrangianMinimizerOptions()
    options.constraint_tolerance = 1e-5
    solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(), options)
    wp.synchronize()
    summary = solver.minimize(stream, problem)
    assert summary.converged(), summary.message
    assert summary.max_violation < 1e-5


def test_strided_joint_q_and_link_transforms(arm, stream):
    """Instances inside a scene-wide array: read and written in place, gaps untouched."""
    topo, n, pad = arm.topology, 8, 3
    stride = topo.coord_count + pad
    rng = np.random.default_rng(2)
    scene = np.full((n, stride), -7.0, np.float32)
    scene[:, :topo.coord_count] = rng.uniform(-0.5, 0.5, (n, topo.coord_count))
    states = pycunls.ArticulationStateBatch(topo, n, apply_joint_limits=False)
    states.set_num_active_states(n)
    states.set_joint_q(wp.array(scene.reshape(-1)), n, stream, stride=stride)
    back = wp.full(n * stride, -7.0, dtype=float)
    states.get_joint_q(back, n, stream, stride=stride)
    packed = _link_transforms(states, n, stream)
    poses = wp.full(n * (7 * topo.link_count + 2), -7.0, dtype=float)
    states.get_link_transforms(poses, n, stream, stride=7 * topo.link_count + 2)
    wp.synchronize()
    np.testing.assert_array_equal(back.numpy().reshape(n, stride), scene)
    strided = poses.numpy().reshape(n, -1)
    np.testing.assert_array_equal(strided[:, :7 * topo.link_count], packed.reshape(n, -1))
    assert np.all(strided[:, 7 * topo.link_count:] == -7.0)
    np.testing.assert_allclose(np.linalg.norm(packed[:, :, 3:], axis=2), 1.0, atol=1e-5)
    # Joint j1 rotates link l1 about z at height 0.1: its pose is (0, 0, 0.1), Rz(q).
    l1, q = arm.link_names.index("l1"), scene[:, 0]
    np.testing.assert_allclose(packed[:, l1, :3], np.tile([0, 0, 0.1], (n, 1)), atol=1e-6)
    np.testing.assert_allclose(np.abs(packed[:, l1, 5]), np.abs(np.sin(q / 2)), atol=1e-6)
    with pytest.raises(ValueError, match="Fix:"):
        states.set_joint_q(wp.zeros(n * topo.coord_count, dtype=float), n, stream, stride=stride)


def test_topology_accepts_gpu_arrays():
    ident = [0, 0, 0, 0, 0, 0, 1]
    host = dict(joint_type=[JT.Fixed.value, JT.Revolute.value], joint_parent=[-1, 0],
                joint_child=[0, 1], joint_X_p=ident + [0, 0, 0.5, 0, 0, 0, 1],
                joint_X_c=ident * 2, joint_q_start=[0, 0, 1], joint_qd_start=[0, 0, 1],
                joint_axis=[0, 0, 1], limit_lower=[-1.0], limit_upper=[1.0])
    ints = ("joint_type", "joint_parent", "joint_child", "joint_q_start", "joint_qd_start")
    gpu = {k: wp.array(np.asarray(v, np.int32 if k in ints else np.float32))
           for k, v in host.items()}
    gpu["joint_X_p"] = wp.array(np.asarray(host["joint_X_p"], np.float32).reshape(2, 7),
                                dtype=wp.transform)                  # a scene's transform array
    a, b = pycunls.ArticulationTopology(**host), pycunls.ArticulationTopology(**gpu)
    assert (a.joint_count, a.coord_count, a.dof_count) == (b.joint_count, b.coord_count,
                                                           b.dof_count)
    assert list(a.lower_limits) == list(b.lower_limits) == [-1.0]


def test_inverse_kinematics_recipe(arm, stream):
    """IK as an application builds it (docs/sphinx/articulations.rst, "Building an
    inverse-kinematics solver"): problems x seeds instances, one subproblem each, targets
    rewritten in place for a second solve with reuse_structure, the best seed per problem picked
    from the link transforms."""
    topo, problems, seeds = arm.topology, 32, 4
    n = problems * seeds
    rng = np.random.default_rng(3)
    tool = arm.link_names.index("tool")
    lo = np.where(np.isfinite(topo.lower_limits), topo.lower_limits, -np.pi)
    hi = np.where(np.isfinite(topo.upper_limits), topo.upper_limits, np.pi)

    def reachable_targets():
        q = (lo + (hi - lo) * rng.uniform(0.1, 0.9, (problems, topo.coord_count)))
        probe = pycunls.ArticulationStateBatch(topo, problems, apply_joint_limits=False)
        probe.set_num_active_states(problems)
        probe.set_joint_q(wp.array(q.astype(np.float32)), problems, stream)
        return np.repeat(_link_transforms(probe, problems, stream)[:, tool], seeds, axis=0)

    targets = wp.array(reachable_targets())                         # (n, 7), seeds adjacent
    states = pycunls.ArticulationStateBatch(topo, n)
    states.set_num_active_states(n)
    pose = pycunls.LinkPoseFactorBatch(topo, wp.full(n, tool, dtype=wp.int32), None, targets,
                                       wp.array(np.tile([1, 1, 1, 0.1, 0.1, 0.1], n)
                                                .astype(np.float32)), n)
    pose.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    _one_factor_per_instance(problem, states, (pose,), n)
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    base = lm.base_options
    base.cost_tolerance, base.state_tolerance, base.max_num_iterations = 1e-12, 1e-12, 60
    base.reuse_structure = True
    lm.base_options = base
    al = pycunls.AugmentedLagrangianMinimizerOptions()
    al.reuse_structure = True                                        # the wrapper's own flag
    solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(lm), al)

    for _ in range(2):                                               # second round: new targets
        seeds_q = lo + (hi - lo) * rng.uniform(0.0, 1.0, (n, topo.coord_count))
        states.set_joint_q(wp.array(seeds_q.astype(np.float32)), n, stream)
        wp.synchronize()
        solver.minimize(stream, problem)
        links = _link_transforms(states, n, stream)[:, tool]
        err = np.linalg.norm(links[:, :3] - targets.numpy()[:, :3], axis=1)
        best = err.reshape(problems, seeds).min(axis=1)
        assert np.mean(best < 1e-3) > 0.95, np.sort(best)[-5:]
        # The orientation of the picked seed follows the new targets too (q and -q: one rotation).
        pick = err.reshape(problems, seeds).argmin(axis=1) + seeds * np.arange(problems)
        dot = np.abs(np.sum(links[pick, 3:] * targets.numpy()[pick, 3:], axis=1))
        angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
        assert np.mean(angle < 1e-2) > 0.95, np.sort(angle)[-5:]
        wp.copy(targets, wp.array(reachable_targets()))             # same array, new values
