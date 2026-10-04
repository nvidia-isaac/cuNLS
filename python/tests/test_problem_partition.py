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

"""Problem.set_problem_partition and the SE2 / Sim2 / Sim3 prior factor batches."""

import cupy as cp
import numpy as np
import pytest

import pycunls


def se2(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float32)


def solve_se2_priors(stream, targets, init, partition):
    """One SE2 state per subproblem, pulled to its target by an SE2 prior."""
    b = len(targets)
    poses = cp.asarray(np.stack(init))
    obs = cp.asarray(np.stack(targets))
    states = pycunls.SE2StateBatch(poses, b)
    states.set_num_active_states(b)
    prior = pycunls.SE2PriorFactorBatch(obs, b)
    prior.set_num_active_factors(b)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(prior, [states], cp.arange(b, dtype=cp.int32))
    ids = cp.arange(b, dtype=cp.int32)
    if partition:
        problem.set_problem_partition(b, [ids])
        assert problem.num_problems == b
    options = pycunls.MinimizerOptions()
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    summary = pycunls.GaussNewtonMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())
    return cp.asnumpy(poses), summary


@pytest.mark.parametrize("partition", [False, True])
def test_partitioned_se2_priors_reach_their_targets(stream, partition):
    rng = np.random.default_rng(0)
    targets = [se2(*rng.normal(size=2), rng.uniform(-2.5, 2.5)) for _ in range(16)]
    init = [se2(*rng.normal(size=2), rng.uniform(-2.5, 2.5)) for _ in range(16)]
    poses, summary = solve_se2_priors(stream, targets, init, partition)
    np.testing.assert_allclose(poses, np.stack(targets), atol=1e-4)
    assert summary.final_cost < 1e-6


def test_partition_rejects_factor_across_subproblems(stream):
    poses = cp.asarray(np.stack([se2(0, 0, 0), se2(1, 0, 0)]))
    states = pycunls.SE2StateBatch(poses, 2)
    states.set_num_active_states(2)
    deltas = cp.asarray(se2(-1, 0, 0)[None])
    between = pycunls.SE2BetweenFactorBatch(deltas, 1)
    between.set_num_active_factors(1)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(between, [states, states], cp.asarray([0, 1], dtype=cp.int32))
    problem.set_problem_partition(2, [cp.arange(2, dtype=cp.int32)])
    with pytest.raises(ValueError, match="different subproblems"):
        pycunls.GaussNewtonMinimizer().minimize(stream, problem)
    with pytest.raises(ValueError, match="one id array per state batch"):
        problem.set_problem_partition(2, [])
    problem.set_problem_partition(0, [])
    assert problem.num_problems == 1


@pytest.mark.parametrize("state_cls, prior_cls, dim", [
    (pycunls.Similarity2StateBatch, pycunls.Similarity2PriorFactorBatch, 3),
    (pycunls.Similarity3StateBatch, pycunls.Similarity3PriorFactorBatch, 4),
])
def test_similarity_priors_reach_their_targets(stream, state_cls, prior_cls, dim):
    # Targets: rotation about z, a translation, scale s stored as 1/s.
    targets = []
    for k in range(4):
        m = np.eye(dim, dtype=np.float32)
        c, s = np.cos(0.3 * k), np.sin(0.3 * k)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        m[: dim - 1, dim - 1] = 0.5 * (k + 1)
        m[dim - 1, dim - 1] = 1.0 / (1.0 + 0.2 * k)
        targets.append(m)
    poses = cp.asarray(np.stack([np.eye(dim, dtype=np.float32)] * 4))
    obs = cp.asarray(np.stack(targets))
    states = state_cls(poses, 4)
    states.set_num_active_states(4)
    prior = prior_cls(obs, 4)
    prior.set_num_active_factors(4)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(prior, [states], cp.arange(4, dtype=cp.int32))
    problem.set_problem_partition(4, [cp.arange(4, dtype=cp.int32)])
    options = pycunls.MinimizerOptions()
    options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    pycunls.GaussNewtonMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())
    np.testing.assert_allclose(cp.asnumpy(poses), np.stack(targets), atol=1e-4)
