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

"""Custom factor defined with NVIDIA Warp, solved via pycunls.

This is a Python port of ``examples/custom_factor/main.cu``.  It builds a
chain of scalar states connected by "difference" constraints:

    residual_i = (x_{i+1} - x_i) - measurement_i

The Warp kernel computes residuals and Jacobians, and pycunls runs
Levenberg-Marquardt to recover the ground truth.

Pointer-gathering strategy
--------------------------
Warp kernels operate on contiguous ``wp.array`` objects and cannot perform
the double-pointer indirection that raw CUDA kernels do (reading a
``float*`` from a ``float const* const*`` and then dereferencing it).  The
``evaluate()`` method therefore uses a CuPy ``RawKernel`` to gather the
scattered scalar state values into a contiguous ``cp.ndarray`` before
wrapping it as a ``wp.array`` and launching the Warp kernel.

Problem structure
-----------------
* **States**: N scalar values (VectorStateBatch1).
* **Factors**: (N-1) ScalarDiffFactor + 1 PriorVectorFactorBatch1 anchor.
"""

import numpy as np
import cupy as cp
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch

wp.init()


# ── Warp kernel ─────────────────────────────────────────────────────────────

@wp.kernel
def scalar_diff_kernel(
    measurements: wp.array(dtype=wp.float32),
    left_vals: wp.array(dtype=wp.float32),
    right_vals: wp.array(dtype=wp.float32),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    num_factors: int,
    write_jacobians: int,
):
    i = wp.tid()
    if i >= num_factors:
        return

    residuals[i] = (right_vals[i] - left_vals[i]) - measurements[i]

    if write_jacobians != 0:
        jacobians[i * 2] = -1.0
        jacobians[i * 2 + 1] = 1.0


# ── CuPy gather kernel ─────────────────────────────────────────────────────
# Reads float values from scattered device pointers into a contiguous buffer.

_gather_kernel = cp.RawKernel(r"""
extern "C" __global__
void gather_floats(const unsigned long long* ptrs,
                   float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* p = reinterpret_cast<const float*>(ptrs[i]);
        out[i] = p[0];
    }
}
""", "gather_floats")


def _gather_state_values(state_ptrs_ptr: int, count: int) -> cp.ndarray:
    """Read *count* float values from scattered pointers in device memory."""
    ptrs = cp.ndarray(
        shape=(count,), dtype=cp.uint64,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(state_ptrs_ptr, count * 8, None), 0))
    out = cp.empty(count, dtype=cp.float32)
    threads = 256
    blocks = (count + threads - 1) // threads
    _gather_kernel((blocks,), (threads,), (ptrs, out, np.int32(count)))
    return out


# ── WarpFactorBatch subclass ───────────────────────────────────────────────

class ScalarDiffFactor(WarpFactorBatch):
    """Between factor on scalar states: residual_i = (x_{right} - x_{left}) - m_i.

    Each factor connects two scalar state blocks (block sizes = [1, 1]).
    The Jacobian is constant: [-1, +1] per row (one row per factor).
    """

    def __init__(self, measurements_wp: wp.array, num_factors: int):
        super().__init__(
            residual_size=1,
            state_block_sizes=[1, 1],
            num_factors=num_factors,
        )
        self.measurements = measurements_wp
        self._num_factors = num_factors

    def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle):
        n = self._num_factors

        all_vals = _gather_state_values(state_pointers_ptr, n * 2)
        left_vals = all_vals[0::2].copy()
        right_vals = all_vals[1::2].copy()

        left_wp = wp.array(ptr=int(left_vals.data.ptr), dtype=wp.float32,
                           shape=(n,), device=self._device, copy=False)
        right_wp = wp.array(ptr=int(right_vals.data.ptr), dtype=wp.float32,
                            shape=(n,), device=self._device, copy=False)

        res = self.wrap_array(residuals_ptr, wp.float32, n)
        write_jac = 1 if jacobians_ptr != 0 else 0
        jac = (self.wrap_array(jacobians_ptr, wp.float32, n * 2)
               if jacobians_ptr != 0
               else wp.zeros(1, dtype=wp.float32, device=self._device))

        stream = self.make_warp_stream(stream_handle)
        wp.launch(
            scalar_diff_kernel,
            dim=n,
            inputs=[self.measurements, left_wp, right_wp, res, jac,
                    n, write_jac],
            stream=stream,
        )
        return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    num_states = 256
    num_diff = num_states - 1

    rng = np.random.default_rng(121314)

    # Monotonic ground truth.
    gt = np.zeros(num_states, dtype=np.float32)
    gt[0] = 0.5
    for i in range(1, num_states):
        gt[i] = gt[i - 1] + rng.uniform(0.2, 0.6)

    measurements_np = np.diff(gt).astype(np.float32)

    # Noisy initial guess.
    initial = gt + rng.uniform(-0.35, 0.35, num_states).astype(np.float32)

    # Prior on first state to anchor the chain.
    prior_obs_np = gt[:1].copy()

    # ── Upload to GPU ───────────────────────────────────────────────────────
    states_gpu = cp.asarray(initial)
    measurements_wp = wp.array(measurements_np, dtype=wp.float32, device="cuda:0")
    prior_obs_gpu = cp.asarray(prior_obs_np)

    # ── Build cuNLS problem ─────────────────────────────────────────────────
    stream = pycunls.CudaStream()

    state_batch = pycunls.VectorStateBatch1(states_gpu, num_states)
    diff_factor = ScalarDiffFactor(measurements_wp, num_diff)
    prior_factor = pycunls.PriorVectorFactorBatch1(prior_obs_gpu, 1)

    diff_ptrs = []
    for i in range(num_diff):
        diff_ptrs.append(state_batch.state_block_device_ptr(i))
        diff_ptrs.append(state_batch.state_block_device_ptr(i + 1))

    prior_ptrs = [state_batch.state_block_device_ptr(0)]

    problem = pycunls.Problem()
    problem.add_state_batch(state_batch)
    problem.add_factor_batch(diff_factor, diff_ptrs)
    problem.add_factor_batch(prior_factor, prior_ptrs)
    assert problem.check_consistency(), "Problem consistency check failed"

    # ── Solve ───────────────────────────────────────────────────────────────
    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 50
    opts.state_tolerance = 1e-8
    opts.cost_tolerance = 1e-8

    lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
    lm_opts.base_options = opts
    lm_opts.initial_lambda = 1e-3

    minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
    summary = minimizer.minimize(stream, problem)

    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # ── Report ──────────────────────────────────────────────────────────────
    optimized = cp.asnumpy(states_gpu)
    mse_before = float(np.mean((initial - gt) ** 2))
    mse_after = float(np.mean((optimized - gt) ** 2))

    print("Custom Warp Factor Example (pycunls)")
    print(f"  Initial cost : {summary.initial_cost:.6f}")
    print(f"  Final cost   : {summary.final_cost:.6f}")
    print(f"  Iterations   : {summary.num_iterations}")
    print(f"  State MSE    : {mse_before:.6f} -> {mse_after:.6f}")


if __name__ == "__main__":
    main()
