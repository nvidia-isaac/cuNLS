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

import math

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


def _partition(problem, states, n):
    problem.set_problem_partition(n, [wp.array(np.arange(n, dtype=np.int32))])


def test_urdf_topology(arm):
    topo = arm.topology
    assert topo.joint_count == 7 and topo.dof_count == 6 and topo.coord_count == 6
    assert arm.link_names[0] == "base" and arm.joint_names[0] == "<root>"
    assert arm.dof_names == ["j1", "j2", "j3", "j4", "fa", "fb"]
    assert math.isinf(topo.lower_limits[3])  # continuous
    assert arm.mimic_joints[0].mimic_dof == 5 and arm.mimic_joints[0].reference_dof == 4


def test_topology_errors_say_what_and_how_to_fix():
    with pytest.raises(ValueError, match="Fix:"):
        pycunls.ArticulationTopology(
            joint_type=[JT.Revolute.value], joint_parent=[-1], joint_child=[0],
            joint_X_p=[0, 0, 0, 0, 0, 0, 1], joint_X_c=[0, 0, 0, 0, 0, 0, 1],
            joint_q_start=[0, 1], joint_qd_start=[0, 1], joint_axis=[0, 0, 2])


def _fk_tool(states, arm, n):
    """World position of the tool link of the first n instances (from the cache)."""
    topo = arm.topology
    tool = arm.link_names.index("tool")
    view = wp.array(ptr=states.state_device_ptr(0), dtype=float, shape=(n, topo.ambient_size),
                    device=wp.get_cuda_device(), copy=False).numpy()
    pose = view[:, topo.link_pose_offset(tool):topo.link_pose_offset(tool) + 12]
    return pose[:, [3, 7, 11]]


def test_bounded_ik_with_mimic(arm):
    topo, n = arm.topology, 128
    stream = pycunls.CudaStream()
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
    targets = wp.array(_fk_tool(probe, arm, n).astype(np.float32))

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
    ptrs = [states.state_device_ptr(i) for i in range(n)]
    for fb in (position, posture, coupled):
        problem.add_factor_batch(fb, ptrs)
    _partition(problem, states, n)
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
    err = np.linalg.norm(_fk_tool(states, arm, n) - targets.numpy(), axis=1)
    assert np.mean(err < 1e-3) > 0.95, np.sort(err)[-10:]


def test_every_kinematic_factor_passes_check_jacobians(arm):
    topo, n = arm.topology, 16
    stream = pycunls.CudaStream()
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


def test_four_bar_closes(arm):
    """Ground, crank, coupler, rocker; the coupler-rocker hinge closes the loop."""
    ident = [0, 0, 0, 0, 0, 0, 1]
    topo = pycunls.ArticulationTopology(
        joint_type=[JT.Fixed.value, JT.Revolute.value, JT.Revolute.value, JT.Revolute.value],
        joint_parent=[-1, 0, 1, 0], joint_child=[0, 1, 2, 3],
        joint_X_p=ident + ident + [0.4, 0, 0, 0, 0, 0, 1] + [1.0, 0, 0, 0, 0, 0, 1],
        joint_X_c=ident * 4, joint_q_start=[0, 0, 1, 2, 3], joint_qd_start=[0, 0, 1, 2, 3],
        joint_axis=[0, 0, 1] * 3)
    n = 16
    stream = pycunls.CudaStream()
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
    ptrs = [states.state_device_ptr(i) for i in range(n)]
    problem.add_factor_batch(posture, ptrs)
    problem.add_factor_batch(pycunls.ConstraintFactorBatch(loop, pycunls.ConstraintKind.Equality),
                             ptrs)
    _partition(problem, states, n)
    options = pycunls.AugmentedLagrangianMinimizerOptions()
    options.constraint_tolerance = 1e-5
    solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(), options)
    wp.synchronize()
    summary = solver.minimize(stream, problem)
    assert summary.converged(), summary.message
    assert summary.max_violation < 1e-5
