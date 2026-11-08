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
    json.dumps(d, allow_nan=False)  # NaN costs become None, enums names
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


def test_real_time_augmented_lagrangian_without_constraints_is_not_determined(stream):
    problem, (states, *_rest) = _priors([50, -30])
    lm = pycunls.LevenbergMarquardtMinimizerOptions()
    lm.initial_lambda = 1e4  # one heavily damped step: far from the minimum
    options = pycunls.AugmentedLagrangianMinimizerOptions()
    options.real_time = True
    options.inner_iterations = 1
    inner = pycunls.LevenbergMarquardtMinimizer(lm)
    summary = pycunls.AugmentedLagrangianMinimizer(inner, options).minimize(stream, problem)
    assert summary.status == pycunls.AugmentedLagrangianMinimizerStatus.NotDetermined
    assert not summary.converged() and summary.num_converged == 0
    d = summary.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["status"] == "NotDetermined" and d["converged"] is False
    assert "NOT CONVERGED" in summary.report().splitlines()[0]


def test_json_helper_is_strict_for_nested_values():
    from pycunls._summary import _plain
    value = {"a": np.array([[1.0, np.nan], [np.inf, 2.0]]), "b": (np.float32(np.nan), 3),
             "c": [np.int64(4), {"d": -np.inf}], "e": TerminationReason.MaxIterations}
    out = _plain(value)
    json.dumps(out, allow_nan=False)
    assert out == {"a": [[1.0, None], [None, 2.0]], "b": [None, 3], "c": [4, {"d": None}],
                   "e": "MaxIterations"}


def test_evaluation_to_dict_is_strict_json(stream):
    problem, _keep = _priors([np.nan, 0])
    d = pycunls.evaluate(problem).to_dict()
    json.dumps(d, allow_nan=False)
    # The cost covers the finite residuals only; the NaN factor is counted instead.
    assert d["total_cost"] == 0.0 and d["all_finite"] is False
    assert d["batches"][0]["num_non_finite"] == 1
