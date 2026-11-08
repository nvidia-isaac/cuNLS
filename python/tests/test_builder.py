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

"""ProblemBuilder and pycunls.solve: problems from arrays, checked before solving."""

import numpy as np
import pytest

import pycunls


def _chain(n=20, seed=0):
    """Ground-truth SE3 chain, exact between measurements (cuNLS convention) and a bad start."""
    rng = np.random.default_rng(seed)
    truth = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    truth[:, 0, 3] = np.arange(n)
    truth[:, 1, 3] = np.sin(np.arange(n) * 0.3)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
    deltas = np.stack([np.linalg.inv(truth[j]) @ truth[i] for i, j in edges]).astype(np.float32)
    initial = truth.copy()
    initial[1:, :3, 3] += rng.normal(0, 0.2, (n - 1, 3))
    return truth, initial, edges, deltas


def test_pose_graph_with_the_builder():
    truth, initial, edges, deltas = _chain()
    pb = pycunls.ProblemBuilder()
    poses = pb.add_states(pycunls.SE3StateBatch, initial, constant=[0])
    pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges))
    # Exact measurements: the final cost is ~0, below the default absolute cost_tolerance.
    result = pb.solve(cost_tolerance=1e-12)
    assert result.converged, result.message
    np.testing.assert_allclose(result[poses], truth, atol=1e-3)
    assert result.evaluate().total_cost < 1e-6


def test_one_call_solve_infers_the_state_type():
    truth, initial, edges, deltas = _chain()
    result = pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=initial, connect=edges,
                           constant=[0], max_num_iterations=30, cost_tolerance=1e-12)
    assert result.converged
    np.testing.assert_allclose(result.states, truth, atol=1e-3)


def test_gauge_freedom_warns_before_solving():
    _, initial, edges, deltas = _chain()
    with pytest.warns(UserWarning, match="Gauge freedom"):
        pycunls.solve("SE3BetweenFactorBatch", deltas, initial=initial, connect=edges)


def test_two_state_types_and_per_slot_connectivity():
    # Points observed by fixed cameras: ReprojectionFactorBatch [pose, point].
    rng = np.random.default_rng(1)
    cams = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    cams[1, 0, 3] = 1.0
    points = rng.uniform([-1, -1, 4], [1, 1, 6], (30, 3)).astype(np.float32)
    cam_ids = np.repeat([0, 1], 30)
    pt_ids = np.tile(np.arange(30), 2)
    p_cam = points[pt_ids] - cams[cam_ids, :3, 3]
    obs = (p_cam[:, :2] / p_cam[:, 2:]).astype(np.float32)
    pb = pycunls.ProblemBuilder()
    c = pb.add_states("SE3StateBatch", cams, constant=[0, 1])
    p = pb.add_states("VectorStateBatch3", points + rng.normal(0, 0.1, points.shape))
    pb.add_factors(pycunls.ReprojectionFactorBatch, obs, connect=[(c, cam_ids), (p, pt_ids)])
    result = pb.solve(method="lm", max_num_iterations=50)
    assert result.converged
    np.testing.assert_allclose(result[p], points, atol=1e-2)


def test_bounds_use_the_augmented_lagrangian():
    pb = pycunls.ProblemBuilder()
    x = pb.add_states(pycunls.VectorStateBatch2, np.zeros((3, 2)), bounds=(-1.0, [1.0, 0.5]))
    target = np.array([[3, 3], [0.2, 0.2], [-5, 0]], np.float32)
    pb.add_factors(pycunls.PriorVectorFactorBatch2, target, connect=(x, np.arange(3)))
    result = pb.solve()
    assert result.converged
    np.testing.assert_allclose(result[x], [[1, 0.5], [0.2, 0.2], [-1, 0]], atol=1e-3)


def test_mistakes_are_explained():
    _, initial, edges, deltas = _chain()
    pb = pycunls.ProblemBuilder()
    with pytest.raises(ValueError, match="stores 16 floats per state"):
        pb.add_states(pycunls.SE3StateBatch, initial[:, :3, :])
    poses = pb.add_states(pycunls.SE3StateBatch, initial, constant=[0])
    with pytest.raises(ValueError, match=r"indices must be in \[0, 20\)"):
        pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges + 1))
    with pytest.raises(ValueError, match="tangent size 3.*SE3StateBatch has 6"):
        pb.add_factors(pycunls.SE2BetweenFactorBatch, deltas[:, :3, :3], connect=(poses, edges))
    pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges))
    with pytest.raises(ValueError, match="unknown option.*max_iters"):
        pb.solve(max_iters=5)


def test_nan_initial_state_is_caught_before_solving():
    _, initial, edges, deltas = _chain()
    initial[4, 0, 3] = np.nan
    with pytest.raises(ValueError, match="first state 4"):
        pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=initial, connect=edges,
                      constant=[0])


def test_resolve_with_new_measurements():
    truth, initial, edges, deltas = _chain()
    pb = pycunls.ProblemBuilder()
    poses = pb.add_states(pycunls.SE3StateBatch, initial, constant=[0])
    between = pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas, connect=(poses, edges))
    pb.solve(cost_tolerance=1e-12)
    shifted = truth.copy()
    shifted[1:, 2, 3] += 0.5  # every pose after the first moves up
    new = np.stack([np.linalg.inv(shifted[j]) @ shifted[i] for i, j in edges]).astype(np.float32)
    between.set_measurements(new)
    poses.set(initial)
    result = pb.solve(reuse_structure=True, cost_tolerance=1e-12)
    np.testing.assert_allclose(result[poses], shifted, atol=1e-3)
