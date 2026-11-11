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

"""The native log reaches Python's logging (log_to_python) and CUNLS_LOG_LEVEL sets the level."""

import logging
import os
import subprocess
import sys

import numpy as np
import warp as wp

import pycunls


def _solve(stream):
    states = wp.array(np.array([5, -3], np.float32))
    obs = wp.array(np.array([1, 2], np.float32))
    sb = pycunls.VectorStateBatch2(states, 1)
    sb.set_num_active_states(1)
    fb = pycunls.PriorVectorFactorBatch2(obs, 1)
    fb.set_num_active_factors(1)
    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_device_ptr(0)])
    return pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)


def test_log_to_python_forwards_iterations_and_the_outcome(stream, caplog):
    try:
        pycunls.log_to_python(verbosity=pycunls.Verbosity.Message)
        with caplog.at_level(logging.INFO, logger="pycunls"):
            summary = _solve(stream)
        messages = [r.getMessage() for r in caplog.records if r.name == "pycunls"]
        assert any(m.startswith("Iteration 1:") for m in messages), messages
        assert messages[-1] == summary.message
    finally:
        pycunls.log_to_python(False)
        pycunls.set_log_verbosity(pycunls.Verbosity.Silent)


def test_silent_by_default_forwards_nothing(stream, caplog):
    try:
        pycunls.set_log_verbosity(pycunls.Verbosity.Silent)
        pycunls.log_to_python()
        with caplog.at_level(logging.DEBUG, logger="pycunls"):
            _solve(stream)
        assert not [r for r in caplog.records if r.name == "pycunls"]
    finally:
        pycunls.log_to_python(False)


def test_env_var_sets_the_initial_level():
    code = ("import numpy as np, warp as wp, pycunls\n"
            "s = wp.array(np.array([5, -3], np.float32)); o = wp.array(np.array([1, 2], "
            "np.float32))\n"
            "sb = pycunls.VectorStateBatch2(s, 1); sb.set_num_active_states(1)\n"
            "fb = pycunls.PriorVectorFactorBatch2(o, 1); fb.set_num_active_factors(1)\n"
            "p = pycunls.Problem(); p.add_state_batch(sb)\n"
            "p.add_factor_batch(fb, [sb.state_device_ptr(0)])\n"
            "pycunls.GaussNewtonMinimizer().minimize(pycunls.CudaStream(), p)\n")
    env = dict(os.environ, CUNLS_LOG_LEVEL="message")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert "Converged" in out.stdout + out.stderr
