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

"""Tests for CuPy array interoperability.

pycunls constructors accept ``cupy.ndarray`` objects as well as raw integer device pointers
(both resolve to the same GPU address), check a CuPy array's dtype, and a full round trip works:
CuPy → pycunls optimise → CuPy read-back. CuPy is optional (the ``test`` extra): skipped without
it. Every other argument check goes through the same ``__cuda_array_interface__`` path as Warp
arrays and is tested in ``test_argument_errors.py``.
"""

import numpy as np
import pytest

import pycunls

cp = pytest.importorskip("cupy")


def test_cupy_array_and_its_int_pointer_give_the_same_state_pointer():
    data = cp.zeros(30, dtype=cp.float32)
    sb_cp = pycunls.VectorStateBatch3(data, 10)
    sb_cp.set_num_active_states(sb_cp.capacity, sb_cp.const_capacity)
    sb_int = pycunls.VectorStateBatch3(int(data.data.ptr), 10)
    sb_int.set_num_active_states(sb_int.capacity, sb_int.const_capacity)
    assert sb_cp.num_active_states == sb_int.num_active_states == 10
    assert sb_cp.state_device_ptr(0) == sb_int.state_device_ptr(0) == data.data.ptr


def test_cupy_dtype_is_checked():
    with pytest.raises(TypeError, match="data must have dtype float32, got float64"):
        pycunls.VectorStateBatch3(cp.zeros(9, dtype=cp.float64), 3)
    with pytest.raises(TypeError, match="const_ids must have dtype int32, got int64"):
        pycunls.SE3StateBatch(cp.zeros(16 * 2, dtype=cp.float32), 2, cp.arange(1), 1)


def test_optimize_and_read_back(stream):
    """Write data via CuPy, optimize, read back updated data via CuPy."""
    target = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    states_gpu = cp.zeros(3, dtype=cp.float32)
    obs_gpu = cp.asarray(target)

    sb = pycunls.VectorStateBatch3(states_gpu, 1)
    sb.set_num_active_states(sb.capacity, sb.const_capacity)
    fb = pycunls.PriorVectorFactorBatch3(obs_gpu, 1)
    fb.set_num_active_factors(fb.capacity)

    problem = pycunls.Problem()
    problem.add_state_batch(sb)
    problem.add_factor_batch(fb, [sb.state_device_ptr(0)])
    assert problem.check_consistency()

    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 10
    opts.disable_safety_checks = False
    pycunls.GaussNewtonMinimizer(opts).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    np.testing.assert_allclose(cp.asnumpy(states_gpu), target, atol=1e-3)


def test_host_copy_waits_for_the_producer_stream():
    """A CuPy array still being written on a non-blocking stream is copied after the write."""
    from pycunls._arrays import to_numpy
    late_fill = cp.RawKernel(r"""
    extern "C" __global__ void late_fill(float *a, int n) {
      const long long start = clock64();
      while (clock64() - start < 200000000) {}  // ~0.1 s: the copy must wait for it
      for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] = 1.0f;
    }""", "late_fill")
    with cp.cuda.Stream(non_blocking=True):
        a = cp.zeros(1024, dtype=cp.float32)
        late_fill((1,), (256,), (a, np.int32(a.size)))
        host = to_numpy(a, np.float32, "a", "test")
    assert (host == 1.0).all()
