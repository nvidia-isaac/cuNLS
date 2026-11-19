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

"""Streams of other libraries (wp.Stream, __cuda_stream__ objects) wherever pycunls takes one."""

import numpy as np
import pytest
import warp as wp

import pycunls


def _prior_problem(n):
    """n 3-vectors pulled to targets by a prior: one LM step solves it."""
    x = wp.zeros(3 * n, dtype=float)
    targets = wp.array(np.arange(3 * n, dtype=np.float32))
    states = pycunls.VectorStateBatch3(x, n)
    states.set_num_active_states(n)
    prior = pycunls.PriorVectorFactorBatch3(targets, n)
    prior.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(n)])
    return problem, x, targets


def test_a_warp_stream_is_accepted_as_the_solver_stream():
    """Uploads, the solve and the read-back on Warp's own stream: no extra synchronization."""
    n = 64
    stream = wp.get_stream()
    problem, x, targets = _prior_problem(n)
    x.fill_(5.0)                                  # on Warp's stream, before the solve
    assert not pycunls.diagnose(problem, stream)
    summary = pycunls.LevenbergMarquardtMinimizer().minimize(stream, problem)
    assert summary.converged(), summary.message
    np.testing.assert_allclose(x.numpy(), targets.numpy(), atol=1e-4)


def test_cuda_stream_wraps_other_streams_without_owning_them():
    s = wp.Stream(wp.get_cuda_device())
    assert pycunls.CudaStream(s).get_stream() == s.cuda_stream            # .cuda_stream
    assert pycunls.CudaStream(s.cuda_stream).get_stream() == s.cuda_stream  # raw handle

    class Protocol:                               # e.g. a PyTorch / CuPy stream
        def __cuda_stream__(self):
            return (0, s.cuda_stream)

    wrapped = pycunls.CudaStream(Protocol())
    assert wrapped.get_stream() == s.cuda_stream
    del wrapped                                   # must not destroy s
    problem, x, targets = _prior_problem(8)
    pycunls.LevenbergMarquardtMinimizer().minimize(Protocol(), problem)   # implicit conversion
    np.testing.assert_allclose(x.numpy(), targets.numpy(), atol=1e-4)
    with pytest.raises(TypeError, match="Fix:"):
        pycunls.CudaStream("not a stream")
