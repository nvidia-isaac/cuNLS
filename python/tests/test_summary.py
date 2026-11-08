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

"""Summaries say why a solve stopped: termination_reason, converged(), message,
per-subproblem reasons, to_dict() and report()."""

import json

import cupy as cp
import numpy as np

import pycunls
from pycunls import TerminationReason


def _priors(initial, partition=False):
    """n 2-vectors pulled to (1, 2), optionally one subproblem per vector."""
    initial = np.asarray(initial, np.float32).reshape(-1)
    n = initial.size // 2
    states = cp.asarray(initial)
    obs = cp.asarray(np.tile(np.array([1, 2], np.float32), n))
    sb = pycunls.VectorStateBatch2(states, n)
    sb.set_num_active_states(n)
    fb = pycunls.PriorVectorFactorBatch2(obs, n)
    fb.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_device_ptr(i) for i in range(n)])
    ids = cp.arange(n, dtype=cp.int32)
    if partition:
        problem.set_problem_partition(n, [ids])
    return problem, (states, obs, sb, fb, ids)


def test_converged_solve_reports_a_convergence_reason(stream):
    problem, _keep = _priors([5, -3])
    summary = pycunls.GaussNewtonMinimizer().minimize(stream, problem)
    assert summary.converged()
    assert summary.termination_reason in (TerminationReason.CostTolerance,
                                          TerminationReason.StateTolerance)
    assert summary.message.startswith("Converged")
    assert summary.num_free_parameters == 2
    assert summary.solve_time_ms > 0
    assert "converged=True" in repr(summary)


def test_iteration_cap_is_reported_with_a_fix(stream):
    problem, _keep = _priors([50, -30])
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 1
    options.initial_lambda = 10.0
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    assert not summary.converged()
    assert summary.termination_reason == TerminationReason.MaxIterations
    assert "max_num_iterations" in summary.message


def test_nan_initial_state_is_not_reported_as_converged(stream):
    problem, _keep = _priors([np.nan, 0])
    summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
    assert not summary.converged()
    assert summary.termination_reason == TerminationReason.NonFiniteCost
    assert "not finite" in summary.message


def test_partition_reasons_per_subproblem(stream):
    problem, _keep = _priors([5, -3, np.nan, 0, 7, 7, 1, 2], partition=True)
    minimizer = pycunls.LevenbergMarquardtMinimizer()
    summary = minimizer.minimize(stream, problem)
    assert summary.termination_reason == TerminationReason.NonFiniteCost
    counts = summary.to_dict()["num_problems_by_reason"]
    assert counts["NonFiniteCost"] == 1 and sum(counts.values()) == 4
    reasons = minimizer.subproblem_termination_reasons()
    assert reasons.shape == (4,)
    assert reasons[1] == int(TerminationReason.NonFiniteCost)
    bad = np.flatnonzero(reasons >= int(TerminationReason.MaxIterations))
    assert bad.tolist() == [1]


def test_to_dict_is_json_ready_and_report_leads_with_the_outcome(stream):
    problem, _keep = _priors([np.nan, 0])
    summary = pycunls.GaussNewtonMinimizer().minimize(stream, problem)
    d = summary.to_dict()
    json.dumps(d)  # NaN costs become None, enums names
    assert d["converged"] is False
    assert d["termination_reason"] == "NonFiniteCost"
    assert d["initial_cost"] is None
    text = summary.report()
    assert text.splitlines()[0] == "MinimizerSummary: NOT CONVERGED"
    assert "not finite" in text.splitlines()[1]


def test_augmented_lagrangian_summary_has_message_and_converged(stream):
    problem, _keep = _priors([5, -3])
    inner = pycunls.LevenbergMarquardtMinimizer()
    summary = pycunls.AugmentedLagrangianMinimizer(inner).minimize(stream, problem)
    assert summary.converged()
    assert summary.message  # the wrapped minimizer's message without constraints
    assert summary.to_dict()["status"] == "Converged"
    assert summary.report().startswith("AugmentedLagrangianMinimizerSummary: converged")
