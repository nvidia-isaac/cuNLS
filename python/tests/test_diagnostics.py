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

"""pycunls.diagnose / evaluate / check_jacobians find what is wrong before (and after) a solve."""

import json

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls.catalog import effective_factor


def _pose_graph(n=5, constant_first=False, nan_pose=None):
    """A chain of SE3 poses with exact between measurements but a perturbed start."""
    T = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        T[i, 0, 3] = i + 0.1 * i * i  # off the measured unit steps
    if nan_pose is not None:
        T[nan_pose, 1, 3] = np.nan
    D = np.tile(np.eye(4, dtype=np.float32), (n - 1, 1, 1))
    D[:, 0, 3] = -1.0  # Delta = T_{i+1}^-1 T_i for unit steps along x
    poses, deltas = wp.array(T), wp.array(D)
    if constant_first:
        ids = wp.zeros(1, wp.int32)
        sb = pycunls.SE3StateBatch(poses, n, ids, 1)
        sb.set_num_active_states(n, 1)
    else:
        ids = None
        sb = pycunls.SE3StateBatch(poses, n)
        sb.set_num_active_states(n)
    fb = pycunls.SE3BetweenFactorBatch(deltas, n - 1)
    fb.set_num_active_factors(n - 1)
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    pointers = []
    for i in range(n - 1):
        pointers += [sb.state_device_ptr(i), sb.state_device_ptr(i + 1)]
    problem.add_factor_batch(fb, pointers)
    return problem, (poses, deltas, ids, sb, fb)


@wp.kernel
def _wrong_jacobian_prior(state_ptrs: wp.array(dtype=wp.uint64), obs: wp.array2d(dtype=float),
                          scale: float, res: wp.array2d(dtype=float), jac_ptr: wp.uint64):
    t = wp.tid()
    x = wp.array(ptr=state_ptrs[t], shape=(2,), dtype=float)
    res[t, 0] = x[0] - obs[t, 0]
    res[t, 1] = x[1] - obs[t, 1]
    if jac_ptr != wp.uint64(0):
        jac = wp.array(ptr=jac_ptr, shape=(4 * t + 4,), dtype=float)  # row-major 2x2 per item
        jac[4 * t + 0] = scale
        jac[4 * t + 1] = 0.0
        jac[4 * t + 2] = 0.0
        jac[4 * t + 3] = scale


class _WrongJacobianPrior(pycunls.CustomFactorBatch):
    """residual = x - obs on 2-vectors, with a deliberately wrong Jacobian (2 I, or I)."""

    def __init__(self, obs, capacity, scale):
        super().__init__(2, [2], capacity)
        self.obs, self.scale = obs, scale

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream, ids_ptr, num_ids):
        n = num_ids if num_ids else self.num_active_factors
        device = wp.get_cuda_device()
        state_ptrs = wp.array(ptr=sp_ptr, shape=(n,), dtype=wp.uint64, device=device)
        res = wp.array(ptr=res_ptr, shape=(n, 2), dtype=float, device=device)
        wp.launch(_wrong_jacobian_prior, dim=n,
                  inputs=[state_ptrs, self.obs, float(self.scale), res, wp.uint64(jac_ptr)],
                  stream=wp.Stream(device, cuda_stream=stream))
        return True


def _custom_problem(scale):
    states = wp.array(np.array([[3, 4], [5, -1]], np.float32))
    obs = wp.array(np.array([[1, 2], [0, 0]], np.float32))
    sb = pycunls.VectorStateBatch2(states, 2)
    sb.set_num_active_states(2)
    fb = _WrongJacobianPrior(obs, 2, scale)
    fb.set_num_active_factors(2)
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_device_ptr(0), sb.state_device_ptr(1)])
    return problem, (states, obs, sb, fb)


def test_diagnose_finds_a_gauge_freedom_and_its_fix():
    problem, _keep = _pose_graph()
    issues = pycunls.diagnose(problem)
    assert len(issues) == 1 and issues[0].severity == "warning"
    assert "Gauge freedom" in issues[0].message and "constant" in issues[0].message


def test_diagnose_is_quiet_on_an_anchored_graph():
    problem, _keep = _pose_graph(constant_first=True)
    assert pycunls.diagnose(problem) == []


def test_diagnose_reports_nan_states_with_their_index():
    problem, _keep = _pose_graph(constant_first=True, nan_pose=3)
    issues = pycunls.diagnose(problem)
    assert issues[0].severity == "error"
    assert "NaN or Inf" in issues[0].message and "first state 3" in issues[0].message


def test_diagnose_reports_missing_active_counts():
    problem, keep = _pose_graph(constant_first=True)
    keep[4].set_num_active_factors(0)
    issues = pycunls.diagnose(problem)
    assert issues[0].severity == "error"
    assert "set_num_active_factors" in issues[0].message


def test_evaluate_reports_cost_and_the_worst_factor():
    problem, _keep = _pose_graph(constant_first=True)
    ev = pycunls.evaluate(problem, keep_residuals=True)
    (b,) = ev.batches
    assert b.num_factors == 4 and b.residual_size == 6 and b.num_non_finite == 0
    assert b.worst_factor == 3  # the steps grow along the chain
    norms = b.factor_norms()
    assert isinstance(norms, np.ndarray) and isinstance(b.residuals, np.ndarray)
    assert b.residuals.shape == (4, 6)
    assert np.argmax(norms) == 3
    assert ev.total_cost == pytest.approx(0.5 * float(np.sum(norms ** 2)), rel=1e-4)
    json.dumps(ev.to_dict())
    assert "SE3BetweenFactorBatch" in str(ev)
    # After the solve the residuals are gone.
    pycunls.LevenbergMarquardtMinimizer().minimize(pycunls.CudaStream(), problem)
    assert pycunls.evaluate(problem).total_cost < 1e-8


def test_evaluate_counts_non_finite_residuals():
    problem, _keep = _pose_graph(constant_first=True, nan_pose=2)
    (b,) = pycunls.evaluate(problem).batches
    assert b.num_non_finite == 2 and b.first_non_finite == 1  # factors 1 and 2 read pose 2
    assert not pycunls.evaluate(problem).all_finite


def test_check_jacobians_accepts_built_in_factors():
    problem, _keep = _pose_graph(constant_first=True)
    (check,) = pycunls.check_jacobians(problem)
    assert check.ok, str(check)


def test_check_jacobians_catches_a_wrong_custom_jacobian():
    good, _k1 = _custom_problem(scale=1.0)
    bad, _k2 = _custom_problem(scale=2.0)
    assert pycunls.check_jacobians(good)[0].ok
    (check,) = pycunls.check_jacobians(bad)
    assert not check.ok and check.relative_error > 0.5
    assert "MISMATCH" in str(check)


def test_states_are_unchanged_by_the_checks():
    problem, keep = _pose_graph(constant_first=True)
    before = keep[0].numpy().copy()
    pycunls.diagnose(problem)
    pycunls.evaluate(problem)
    pycunls.check_jacobians(problem)
    assert np.array_equal(before, keep[0].numpy())


def _wrapped_chain(wrap):
    """The 5-pose chain with its between factors wrapped by `wrap(fb)` (no anchor)."""
    problem, (poses, deltas, ids, sb, fb) = _pose_graph()
    wrapped = pycunls.Problem()
    wrapped.add_state_batch(sb)
    outer, keep = wrap(fb)
    wrapped.add_factor_batch(outer, [sb.state_device_ptr(i + k) for i in range(4) for k in (0, 1)])
    return wrapped, (problem, poses, deltas, sb, fb, outer, keep)


def test_diagnose_sees_through_wrapped_factors():
    # A weighted between factor is still relative: the gauge warning must stay.
    wrapped, keep = _wrapped_chain(lambda fb: (pycunls.WeightedFactorBatch(fb, 2.0), None))
    sb, fb, weighted = keep[3:6]
    assert weighted.inner_factor is fb
    issues = pycunls.diagnose(wrapped)
    assert len(issues) == 1 and "Gauge freedom" in issues[0].message

    # A weighted prior still anchors.
    target = wp.array(np.eye(4, dtype=np.float32)[None])
    prior = pycunls.SE3PriorFactorBatch(target, 1)
    prior.set_num_active_factors(1)
    wrapped.add_factor_batch(pycunls.WeightedFactorBatch(prior, 1.0), [sb.state_device_ptr(0)])
    assert pycunls.diagnose(wrapped) == []


def test_nested_wrappers_keep_the_gauge_warning():
    def wrap(fb):
        sqrt_info = wp.array(np.tile(np.eye(6, dtype=np.float32), (4, 1, 1)))
        info = pycunls.InformationFactorBatch(fb, sqrt_info)
        return pycunls.WeightedFactorBatch(info, 3.0), (info, sqrt_info)

    wrapped, keep = _wrapped_chain(wrap)
    inner, info, role = effective_factor(keep[5])
    assert inner is keep[4] and info.name == "SE3BetweenFactorBatch" and role == "relative"
    issues = pycunls.diagnose(wrapped)
    assert len(issues) == 1 and "Gauge freedom" in issues[0].message


def _anchor_with(problem, sb, kind):
    target = wp.array(np.eye(4, dtype=np.float32)[None])
    prior = pycunls.SE3PriorFactorBatch(target, 1)
    prior.set_num_active_factors(1)
    constraint = pycunls.ConstraintFactorBatch(prior, kind)
    problem.add_factor_batch(constraint, [sb.state_device_ptr(0)])
    return target, prior, constraint


def test_an_equality_constraint_anchors_but_an_inequality_does_not():
    wrapped, keep = _wrapped_chain(lambda fb: (pycunls.WeightedFactorBatch(fb, 1.0), None))
    sb = keep[3]
    held = _anchor_with(wrapped, sb, pycunls.ConstraintKind.Inequality)
    assert effective_factor(held[2])[2] == "inactive"
    assert any("Gauge freedom" in i.message for i in pycunls.diagnose(wrapped))
    held2 = _anchor_with(wrapped, sb, pycunls.ConstraintKind.Equality)
    assert effective_factor(held2[2])[2] == "absolute"
    assert pycunls.diagnose(wrapped) == []


def test_custom_factors_are_assumed_to_anchor():
    problem, keep = _custom_problem(1.0)
    assert effective_factor(keep[3])[2] == "unknown"
    assert pycunls.diagnose(problem) == []
