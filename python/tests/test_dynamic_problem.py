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

"""Reusable buffers: batches and connectivity bound once at capacity, then
rewritten in place between solves (the Python mirror of
tests/dynamic_problem_test.cpp)."""

import cupy as cp
import numpy as np
import pytest

import pycunls

CAP_STATES, CAP_EDGES = 300, 900


def _make_frame(num_states, extra_edges, seed):
    """A linear 3D chain-with-loops problem whose solution is ``gt``.

    Returns ``(gt, initial, deltas, edges)``; edge ``e`` reads states
    ``edges[2e]`` and ``edges[2e + 1]``.
    """
    rng = np.random.default_rng(seed)
    gt = rng.uniform(-10, 10, (num_states, 3)).astype(np.float32)
    initial = gt + rng.uniform(-0.5, 0.5, gt.shape).astype(np.float32)
    initial[0] = gt[0]
    order = np.concatenate([[0], 1 + rng.permutation(num_states - 1)])
    pairs = list(zip(order[:-1], order[1:]))  # a random spanning chain
    for _ in range(extra_edges):
        i, j = rng.integers(0, num_states, 2)
        if i != j:
            pairs.append((i, j))
    edges = np.array(pairs, dtype=np.int32).reshape(-1)
    # VectorBetweenFactorBatch: r = x_left - x_right - delta.
    deltas = (gt[edges[0::2]] - gt[edges[1::2]]).astype(np.float32)
    return gt, initial, deltas, edges


def _options(jacobian_mode=pycunls.JacobianMode.analytic):
    o = pycunls.LevenbergMarquardtMinimizerOptions()
    o.base_options.max_num_iterations = 20
    o.base_options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.DenseLDLT
    o.base_options.jacobian_mode = jacobian_mode
    return o


def _solve_fresh(gt, initial, deltas, edges, stream):
    """Solves the frame with a problem built from scratch at exactly its size."""
    n, m = len(gt), len(deltas)
    states = cp.asarray(initial.reshape(-1))
    deltas_gpu = cp.asarray(deltas.reshape(-1))
    anchor = cp.asarray(gt[0])
    state_batch = pycunls.VectorStateBatch3(states, n)
    between = pycunls.VectorBetweenFactorBatch3(deltas_gpu, m)
    prior = pycunls.PriorVectorFactorBatch3(anchor, 1)
    state_batch.set_num_state_blocks(n)
    between.set_num_factors(m)
    prior.set_num_factors(1)
    problem = pycunls.Problem()
    problem.add_state_batch(state_batch)
    problem.add_factor_batch(between, [state_batch.state_block_device_ptr(int(i)) for i in edges])
    problem.add_factor_batch(prior, [state_batch.state_block_device_ptr(0)])
    pycunls.LevenbergMarquardtMinimizer(_options()).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())
    return cp.asnumpy(states).reshape(-1, 3)


class BoundProblem:
    """Buffers bound once at capacity; each frame rewrites contents and sizes."""

    def __init__(self, form, options=None):
        self.form = form
        self.states = cp.zeros(CAP_STATES * 3, dtype=cp.float32)
        self.deltas = cp.zeros(CAP_EDGES * 3, dtype=cp.float32)
        self.anchor = cp.zeros(3, dtype=cp.float32)
        self.indices = cp.zeros(2 * CAP_EDGES, dtype=cp.int32)
        self.pointers = cp.zeros(2 * CAP_EDGES, dtype=cp.uint64)
        self.anchor_index = cp.zeros(1, dtype=cp.int32)
        self.state_batch = pycunls.VectorStateBatch3(self.states, CAP_STATES)
        self.between = pycunls.VectorBetweenFactorBatch3(self.deltas, CAP_EDGES)
        self.prior = pycunls.PriorVectorFactorBatch3(self.anchor, 1)
        self.prior.set_num_factors(1)  # the anchor; states and edges are sized per frame
        self.problem = pycunls.Problem()
        self.problem.add_state_batch(self.state_batch)
        if form == "host_list":
            first = self.state_batch.state_block_device_ptr(0)
            self.problem.add_factor_batch(self.between, [first] * (2 * CAP_EDGES))
        elif form == "device_pointers":
            self.problem.add_factor_batch(self.between, state_pointer_table=self.pointers)
        else:
            self.problem.add_factor_batch(self.between, [self.state_batch, self.state_batch],
                                          self.indices)
        self.problem.add_factor_batch(self.prior, [self.state_batch], self.anchor_index)
        self.minimizer = pycunls.LevenbergMarquardtMinimizer(options or _options())
        self.stream = pycunls.CudaStream()

    def solve(self, gt, initial, deltas, edges):
        n, m = len(gt), len(deltas)
        # 1. Rewrite contents in place.
        self.states[:n * 3] = cp.asarray(initial.reshape(-1))
        self.deltas[:m * 3] = cp.asarray(deltas.reshape(-1))
        self.anchor[:] = cp.asarray(gt[0])
        ptrs = [self.state_batch.state_block_device_ptr(int(i)) for i in edges]
        # 2. Sizes.
        self.state_batch.set_num_state_blocks(n)
        self.between.set_num_factors(m)
        # 3. Connectivity.
        if self.form == "host_list":
            self.problem.set_state_pointers(0, ptrs)
        elif self.form == "device_pointers":
            self.pointers[:len(ptrs)] = cp.asarray(np.array(ptrs, dtype=np.uint64))
        else:
            self.indices[:len(edges)] = cp.asarray(edges)
        assert self.problem.validate(self.stream)
        self.minimizer.minimize(self.stream, self.problem)
        cp.cuda.runtime.streamSynchronize(self.stream.get_stream())
        return cp.asnumpy(self.states[:n * 3]).reshape(-1, 3)


@pytest.mark.parametrize("form", ["host_list", "device_pointers", "device_indices"])
def test_every_frame_matches_a_fresh_problem(form, stream):
    bound = BoundProblem(form)
    # Sizes grow, shrink and grow again; connectivity is new every frame.
    for seed, (num_states, extra) in enumerate([(120, 200), (300, 600), (40, 10), (250, 400),
                                                (300, 0)]):
        frame = _make_frame(num_states, extra, seed + 11)
        solved = bound.solve(*frame)
        fresh = _solve_fresh(*frame, stream)
        gt = frame[0]
        assert np.abs(solved - gt).max() < 5e-2, f"frame with {num_states} states"
        assert np.abs(solved - fresh).max() < 1e-4, f"frame with {num_states} states"


def test_batches_start_empty():
    bound = BoundProblem("device_indices")
    assert bound.state_batch.capacity == CAP_STATES
    assert bound.state_batch.num_state_blocks == 0
    assert bound.between.capacity == CAP_EDGES
    assert bound.between.num_factors == 0
    with pytest.raises(ValueError):
        bound.between.set_num_factors(CAP_EDGES + 1)
    with pytest.raises(ValueError):
        bound.state_batch.set_num_state_blocks(CAP_STATES + 1)


def test_validation_catches_bad_connectivity():
    bound = BoundProblem("device_indices")
    gt, initial, deltas, edges = _make_frame(50, 20, 3)
    bound.solve(gt, initial, deltas, edges)
    assert bound.problem.validate(bound.stream)
    assert bound.problem.check_consistency()

    # An index beyond the active states.
    bound.indices[0] = 50
    assert not bound.problem.validate(bound.stream)

    # States no factor reads: keep only the first between factor.
    bound.indices[:len(edges)] = cp.asarray(edges)
    assert bound.problem.validate(bound.stream)
    bound.between.set_num_factors(1)
    assert not bound.problem.validate(bound.stream)


def test_numeric_jacobians_follow_connectivity_changes():
    bound = BoundProblem("device_indices", _options(pycunls.JacobianMode.numeric))
    for seed in (21, 22, 23):
        frame = _make_frame(80, 40, seed)
        solved = bound.solve(*frame)
        assert np.abs(solved - frame[0]).max() < 1e-2, f"seed {seed}"
