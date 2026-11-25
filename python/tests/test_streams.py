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

import gc

import numpy as np
import pytest
import warp as wp

import pycunls
from pycunls._arrays import borrow_stream
from pycunls.warp import WarpFactorBatch


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



@pytest.mark.parametrize("augmented_lagrangian", [False, True])
def test_a_captured_call_that_would_synchronize_raises_with_the_fix(augmented_lagrangian):
    """The structure setup synchronizes: a captured first call (and a captured initialize())
    raises before enqueuing anything (the capture ends normally). After initialize() outside the
    capture, which leaves the states unchanged, the first call is captured and replays the
    solve."""
    problem, x, targets = _prior_problem(4)
    if augmented_lagrangian:
        options = pycunls.AugmentedLagrangianMinimizerOptions()
        options.max_outer_iterations = 1
    else:
        options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.real_time = True
    options.base_options.reuse_structure = True
    options.base_options.max_num_iterations = 3
    minimizer = (pycunls.AugmentedLagrangianMinimizer(options) if augmented_lagrangian
                 else pycunls.LevenbergMarquardtMinimizer(options))
    stream = wp.get_stream()
    with pytest.raises(ValueError, match=r"call Initialize\(stream, problem\) before the capture"):
        with wp.ScopedCapture():
            minimizer.minimize(stream, problem)
    with pytest.raises(ValueError, match=r"call Initialize\(\) before the capture"):
        with wp.ScopedCapture():
            minimizer.initialize(stream, problem)
    start = x.numpy().copy()
    minimizer.initialize(stream, problem)
    np.testing.assert_array_equal(x.numpy(), start)
    with wp.ScopedCapture() as capture:
        minimizer.minimize(stream, problem)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(x.numpy(), targets.numpy(), atol=1e-3)


def test_cuda_stream_keeps_a_temporary_warp_stream_alive():
    """CudaStream(wp.Stream(...)): the temporary Warp stream must outlive the wrapper's use."""
    wrapped = pycunls.CudaStream(wp.Stream(wp.get_cuda_device()))
    gc.collect()
    with wp.ScopedStream(borrow_stream(wrapped.get_stream())):
        problem, x, targets = _prior_problem(8)
    pycunls.LevenbergMarquardtMinimizer().minimize(wrapped, problem)
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
    with wp.ScopedStream(s):                      # uploads on s, ordered before the solve
        problem, x, targets = _prior_problem(8)
    pycunls.LevenbergMarquardtMinimizer().minimize(Protocol(), problem)   # implicit conversion
    np.testing.assert_allclose(x.numpy(), targets.numpy(), atol=1e-4)
    with pytest.raises(TypeError, match="Fix:"):
        pycunls.CudaStream("not a stream")


# ---------------------------------------------------------------------------
# Callbacks borrow the solver stream: Warp keeps knowing the stream's owner
# ---------------------------------------------------------------------------


@wp.kernel
def _prior3_kernel(observations: wp.array(dtype=wp.vec3), factor_ids: wp.array(dtype=wp.int32),
                   states: wp.array2d(dtype=wp.uint64), residuals: wp.array2d(dtype=float),
                   jacobians: wp.array3d(dtype=float)):
    t = wp.tid()
    x = wp.array(ptr=states[t, 0], shape=(3,), dtype=float)
    obs = observations[factor_ids[t]]
    for d in range(3):
        residuals[t, d] = x[d] - obs[d]
    if jacobians.shape[0] > 0:
        for d in range(3):
            for c in range(3):
                jacobians[t, d, c] = wp.where(d == c, 1.0, 0.0)


def _assert_capturable(stream):
    """A graph captured on ``stream`` replays: Warp still knows the stream."""
    a = wp.zeros(4, dtype=float)
    with wp.ScopedCapture(stream=stream) as capture:
        a.fill_(1.0)
    wp.capture_launch(capture.graph, stream=stream)
    np.testing.assert_array_equal(a.numpy(), 1.0)


def _solve_with_kernel_factor(stream):
    """Solves an 8-state prior problem with a kernel WarpFactorBatch; everything is collected."""
    n = 8
    x = wp.zeros(3 * n, dtype=float)
    states = pycunls.VectorStateBatch3(x, n)
    states.set_num_active_states(n)
    targets = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    prior = WarpFactorBatch(_prior3_kernel, inputs=[targets], residual_size=3, state_sizes=[3],
                            capacity=n)
    prior.set_num_active_factors(n)
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(n)])
    assert pycunls.GaussNewtonMinimizer().minimize(stream, problem).converged()
    np.testing.assert_allclose(x.numpy(), 1.0, atol=1e-5)


@pytest.mark.parametrize("which", ["warp", "user"])
def test_callbacks_leave_the_solver_stream_registered_with_warp(which):
    """Callbacks launch on a borrowed view of the solver stream; collecting it must not make
    Warp forget the stream's owner (captures on it failed with "unknown stream", and Warp's
    current stream became NULL)."""
    device = wp.get_cuda_device()
    stream = wp.get_stream(device) if which == "warp" else wp.Stream(device)
    _solve_with_kernel_factor(stream)
    gc.collect()
    assert wp.get_stream(device).cuda_stream == device.runtime.core.wp_cuda_context_get_stream(
        device.context)
    _assert_capturable(stream)


def test_fit_and_diagnose_leave_warp_streams_registered():
    """fit() on Warp's stream and diagnose() on a CudaStream wrapping a user wp.Stream."""

    @wp.func
    def line(x: float, p: wp.vec2) -> float:
        return p[0] * x + p[1]

    xs = np.linspace(0.0, 1.0, 32).astype(np.float32)
    result = pycunls.fit(line, xs, 2.0 * xs + 1.0, [0.0, 0.0], stream=wp.get_stream())
    np.testing.assert_allclose(np.asarray(result.params), [2.0, 1.0], atol=1e-4)
    user = wp.Stream(wp.get_cuda_device())
    problem, _, _ = _prior_problem(4)
    pycunls.diagnose(problem, pycunls.CudaStream(user))
    gc.collect()
    _assert_capturable(wp.get_stream())
    _assert_capturable(user)


def test_a_borrowed_stream_has_the_attributes_of_a_warp_stream():
    """borrow_stream sets wp.Stream's attributes by hand (running wp.Stream.__init__ would
    register the handle): an attribute a Warp release adds must be added there too."""
    device = wp.get_cuda_device()
    owner = wp.Stream(device)
    assert set(vars(borrow_stream(owner.cuda_stream, device))) == set(vars(owner))


# ---------------------------------------------------------------------------
# Devices: a problem runs on the device of its states
# ---------------------------------------------------------------------------


def _prior_problem_on(device):
    with wp.ScopedDevice(device):
        x = wp.zeros(6, dtype=wp.float32)
        states = pycunls.VectorStateBatch3(x, 2)
        states.set_num_active_states(2)
        prior = pycunls.PriorVectorFactorBatch3(wp.array(np.full(6, 2.0, dtype=np.float32)), 2)
        prior.set_num_active_factors(2)
        problem = pycunls.Problem()
        problem.add_state_batch(states)
        problem.add_factor_batch(prior, [states.state_device_ptr(i) for i in range(2)])
    return x, problem, (states, prior)


def test_problem_device_is_the_device_of_its_states():
    assert pycunls.Problem().device == -1
    x, problem, _ = _prior_problem_on(wp.get_cuda_device())
    assert problem.device == x.device.ordinal


@pytest.mark.skipif(wp.get_cuda_device_count() < 2, reason="needs two GPUs")
def test_minimize_runs_on_the_problems_device_whatever_is_current():
    x, problem, _ = _prior_problem_on("cuda:1")
    minimizer = pycunls.LevenbergMarquardtMinimizer()
    with wp.ScopedDevice("cuda:0"):  # another device is current: minimize switches itself
        summary = minimizer.minimize(wp.get_stream("cuda:1"), problem)
        assert summary.converged(), summary.message
        with pytest.raises(ValueError, match="stream"):
            minimizer.minimize(wp.get_stream("cuda:0"), problem)
    np.testing.assert_allclose(x.numpy(), 2.0, atol=1e-4)
