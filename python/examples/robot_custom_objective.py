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

"""Custom kinematic objectives on a robot arm: a Warp factor and built-in SE(3) and vector factors
on link frames (``LinkFrameFactorBatch``).

A 7-DOF arm carries a camera on its wrist. For each of 512 inspection points, find joint angles
such that

* the camera looks at the point from 0.35 m: a **custom Warp factor** ``LookAtFactor`` over one
  SE(3) pose (the camera), with r = (u - e_z, |d| - 0.35), d = R^T (target - p), u = d / |d|;
* the elbow keeps 0.15 m from a spherical obstacle: the **built-in**
  ``SE3SphereClearanceFactorBatch`` on the elbow's pose, an inequality constraint;
* the camera stays above a table plane: the **built-in** ``HalfspaceFactorBatch3`` on the
  camera's *position*, an inequality constraint;
* the joints stay within their limits (hard bounds of the state);
* the arm stays near a nominal posture (a weak ``PostureFactorBatch``, for redundancy).

None of these factors knows about robots. ``LinkFrameFactorBatch`` feeds each one a view of a
frame on a link (X_link(q) X_offset, from the state's forward-kinematics cache): its pose
(``FrameView.Pose``, as an ``SE3StateBatch`` state), its position (``FrameView.Position``, as a
``VectorStateBatch3`` state) or its rotation (``FrameView.Orientation``, as an ``SO3StateBatch``
state), and carries the factor's Jacobian to the joint angles through the chain rule. So a custom
objective is an ordinary factor written for one of those state types.
"""

import os
import tempfile

import numpy as np
import warp as wp

import pycunls
from pycunls.articulation import topology_from_urdf
from pycunls.warp import WarpFactorBatch, jacobian_index, pose_at
from example_utils import report

wp.init()

NUM_PROBLEMS = 512
STANDOFF = 0.35                                   # camera-to-point distance [m]
OBSTACLE = np.array([0.35, 0.0, 0.55, 0.08], np.float32)  # center (x, y, z), radius [m]
ELBOW_MARGIN = 0.07                               # elbow radius: clearance = 0.15 m
TABLE_HEIGHT = 0.42                               # the camera stays above z = 0.42 m
CAMERA_ON_WRIST = np.array([0.0, 0.05, 0.08, 0.0, 0.0, 0.0, 1.0], np.float32)  # p, quat xyzw

# A 7-DOF arm: revolute joints alternating about z and y, then a flange.
ARM_URDF = """<robot name="arm7">
  <link name="base"/> <link name="l1"/> <link name="l2"/> <link name="l3"/> <link name="l4"/>
  <link name="l5"/> <link name="l6"/> <link name="l7"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-1.7" upper="1.7"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-2.9" upper="0.1"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 1 0"/><limit lower="-0.1" upper="3.7"/></joint>
  <joint name="j7" type="revolute"><parent link="l6"/><child link="l7"/>
    <origin xyz="0 0 0.08"/><axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8"/></joint>
</robot>
"""


# ── The custom factor: an ordinary SE(3) factor, written in Warp ─────────────────────────────

@wp.kernel
def look_at_kernel(targets: wp.array(dtype=wp.vec3), standoff: float,
                   ids: wp.array(dtype=wp.int32),
                   state_pointers: wp.array(dtype=wp.uint64),
                   residuals: wp.array(dtype=float), jacobians: wp.array(dtype=float),
                   write_jacobians: int):
    t = wp.tid()
    X = pose_at(state_pointers, t)                 # camera pose: item t's only state (K = 1)
    R = wp.quat_to_matrix(wp.transform_get_rotation(X))
    p = wp.transform_get_translation(X)
    d = wp.transpose(R) * (targets[ids[t]] - p)                    # target in the camera frame
    dist = wp.length(d)
    u = d / dist
    residuals[4 * t + 0] = u[0]                                    # u - e_z: look along +z
    residuals[4 * t + 1] = u[1]
    residuals[4 * t + 2] = u[2] - 1.0
    residuals[4 * t + 3] = dist - standoff
    if write_jacobians != 0:
        # Right perturbation X Exp([w, v]): d' = d + d x w - v, so dd/dw = [d]x, dd/dv = -I.
        # du/dd = (I - u u^T) / |d|, d|d|/dd = u^T (and u^T [d]x = 0).
        P = (wp.identity(n=3, dtype=float) - wp.outer(u, u)) / dist
        Jw = P * wp.skew(d)
        for r in range(3):
            for k in range(3):
                jacobians[jacobian_index(t, r, k, 4, 6)] = Jw[r, k]          # columns 0-2: w
                jacobians[jacobian_index(t, r, 3 + k, 4, 6)] = -P[r, k]      # columns 3-5: v
        for k in range(3):
            jacobians[jacobian_index(t, 3, k, 4, 6)] = 0.0
            jacobians[jacobian_index(t, 3, 3 + k, 4, 6)] = -u[k]


class LookAtFactor(WarpFactorBatch):
    """4 rows on one SE(3) pose: look at targets[f] along +z from `standoff` meters."""

    def __init__(self, targets, standoff, capacity):
        super().__init__(residual_size=4, state_sizes=[6], capacity=capacity)
        self.targets, self.standoff = targets, standoff

    def evaluate(self, *args):
        """Item t looks up its factor's data by ids[t] and reads its state through the
        pointer table (see WarpFactorBatch.items)."""
        return self.launch(look_at_kernel, self.items(*args), [self.targets, self.standoff])


# ── Checks in NumPy ──────────────────────────────────────────────────────────────────────────

def rotation(q):
    """(n, 3, 3) rotation matrices of (n, 4) quaternions x, y, z, w."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    axis=1).reshape(-1, 3, 3)


def main():
    # 1. The robot, and inspection points in a band in front of it.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "arm7.urdf")
        with open(path, "w") as f:
            f.write(ARM_URDF)
        robot = topology_from_urdf(path)
    topo, n = robot.topology, NUM_PROBLEMS
    wrist, elbow = robot.link_names.index("l7"), robot.link_names.index("l4")
    rng = np.random.default_rng(0)
    angle = rng.uniform(-1.2, 1.2, n)
    radius = rng.uniform(0.45, 0.7, n)
    points = np.stack([radius * np.cos(angle), radius * np.sin(angle),
                       rng.uniform(0.1, 0.5, n)], axis=1).astype(np.float32)

    # 2. States: one arm per problem, from a bent nominal posture; joint limits are hard bounds.
    nominal = np.array([0.0, 0.6, 0.0, -1.6, 0.0, 1.6, 0.0], np.float32)
    stream = wp.get_stream()   # solve on Warp's stream: uploads, solve and reads stay in order
    states = pycunls.ArticulationStateBatch(topo, n)
    states.set_num_active_states(n)
    states.set_joint_q(wp.array(np.tile(nominal, n)), n, stream)

    # 3. Factors. Each one is an ordinary factor over one state (slot 0). LinkFrameFactorBatch
    #    feeds that slot a view of a frame on a robot link instead of a state: frame_slots names
    #    the slot and the view, topology_links the link per factor (an index into
    #    robot.link_names), frame_offsets where the frame sits on the link.
    look_at = LookAtFactor(wp.array(points, dtype=wp.vec3), STANDOFF, n)
    wrist_links = wp.full(n, wrist, dtype=wp.int32)                # the camera is on the wrist
    camera_mounts = wp.array(np.tile(CAMERA_ON_WRIST, n))          # ... at this mount
    camera = pycunls.LinkFrameFactorBatch(
        topo, look_at, frame_slots=[(0, pycunls.FrameView.Pose)],   # an SE(3) factor: the pose
        topology_links=wrist_links, frame_offsets=camera_mounts)
    camera.set_num_active_factors(n)                               # sets look_at's count

    obstacles = wp.array(np.tile(OBSTACLE, n))
    sphere = pycunls.SE3SphereClearanceFactorBatch(obstacles, ELBOW_MARGIN, n)
    elbow_frame = pycunls.LinkFrameFactorBatch(
        topo, sphere, frame_slots=[(0, pycunls.FrameView.Pose)],
        topology_links=wp.full(n, elbow, dtype=wp.int32))          # the elbow link's own frame
    elbow_frame.set_num_active_factors(n)

    # a^T p - b <= 0 with a = (0, 0, -1), b = -TABLE_HEIGHT: the camera's z stays above the table.
    plane = pycunls.HalfspaceFactorBatch3(
        wp.array(np.tile([0.0, 0.0, -1.0], n).astype(np.float32)),
        wp.full(n, -TABLE_HEIGHT, dtype=float), n)
    camera_height = pycunls.LinkFrameFactorBatch(
        topo, plane, frame_slots=[(0, pycunls.FrameView.Position)],  # a vector factor: the origin
        topology_links=wrist_links, frame_offsets=camera_mounts)
    camera_height.set_num_active_factors(n)
    above_table = pycunls.ConstraintFactorBatch(camera_height, pycunls.ConstraintKind.Inequality)
    clearance = pycunls.ConstraintFactorBatch(elbow_frame, pycunls.ConstraintKind.Inequality)

    posture = pycunls.PostureFactorBatch(topo, wp.array(np.tile(nominal, n)),
                                         wp.full(n * topo.dof_count, 0.003, dtype=float), n)
    posture.set_num_active_factors(n)

    # 4. Problem: factor i acts on arm i, and each arm is its own subproblem.
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    instance = wp.array(np.arange(n, dtype=np.int32))
    for factor in (camera, clearance, above_table, posture):
        problem.add_factor_batch(factor, [states], instance)
    problem.set_problem_partition(n, [instance])
    for issue in pycunls.diagnose(problem, stream):
        print(issue)

    # 5. Solve: augmented Lagrangian (bounds + clearance) around Levenberg-Marquardt.
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    lm.base_options.max_num_iterations = 100
    solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(lm))
    summary = solver.minimize(stream, problem)

    # 6. Check every solution in NumPy from the link transforms.
    poses = wp.zeros((n, topo.link_count), dtype=wp.transform)
    q = wp.zeros(n * topo.coord_count, dtype=float)
    states.get_link_transforms(poses, n, stream)
    states.get_joint_q(q, n, stream)
    poses, q = poses.numpy(), q.numpy().reshape(n, -1)
    R_wrist = rotation(poses[:, wrist, 3:])
    cam_p = poses[:, wrist, :3] + R_wrist @ CAMERA_ON_WRIST[:3]
    cam_z = (R_wrist @ rotation(CAMERA_ON_WRIST[None, 3:])[0])[:, :, 2]
    to_point = points - cam_p
    dist = np.linalg.norm(to_point, axis=1)
    view_error = np.degrees(np.arccos(np.clip(np.sum(cam_z * to_point, 1) / dist, -1, 1)))
    elbow_gap = (np.linalg.norm(poses[:, elbow, :3] - OBSTACLE[:3], axis=1)
                 - OBSTACLE[3] - ELBOW_MARGIN)
    lower, upper = np.asarray(topo.lower_limits), np.asarray(topo.upper_limits)
    in_limits = np.all((q >= lower - 1e-5) & (q <= upper + 1e-5), axis=1)
    above = cam_p[:, 2] - TABLE_HEIGHT
    solved = ((view_error < 0.5) & (np.abs(dist - STANDOFF) < 2e-3) & (elbow_gap > -2e-3)
              & (above > -2e-3))

    print(f"Wrist camera aims at {n} points (custom Warp factor, SE(3) clearance and a table "
          "halfspace on links)")
    print(f"  Outcome      : {summary.status.name}, {summary.num_converged} / {n} subproblems "
          f"converged, {summary.outer_iterations} outer / {summary.inner_iterations} inner "
          "iterations")
    print(f"  Solved       : {int(solved.sum())} / {n} (view < 0.5 deg, standoff +-2 mm, "
          "clear elbow, above the table)")
    print(f"  View error   : median {np.median(view_error):.3f} deg, max {view_error.max():.3f} deg")
    print(f"  Standoff     : max |dist - {STANDOFF}| = {np.abs(dist - STANDOFF).max() * 1e3:.2f} mm")
    print(f"  Elbow gap    : min {elbow_gap.min() * 1e3:.2f} mm (>= 0 is clear)")
    print(f"  Table gap    : min {above.min() * 1e3:.2f} mm, {int((above < 1e-3).sum())} cameras "
          "on the plane")
    print(f"  In limits    : {int(in_limits.sum())} / {n}")
    report.check(bool(np.all(in_limits)), "a solution left the joint limits")
    report.check(solved.mean() >= 0.9, f"only {int(solved.sum())} of {n} problems solved")


if __name__ == "__main__":
    main()
