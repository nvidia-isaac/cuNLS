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

"""Custom Warp-based state batch: positive-scalar manifold.

Demonstrates ``WarpStateBatch`` by defining a **positive-scalar** state
where the Plus (retraction) operation is multiplicative:

    x (+) delta = x * exp(delta)

This makes the tangent space the reals (delta ∈ R), while states stay
strictly positive.  It is the natural parameterisation for quantities like
scales, variances, or rates.

The example builds a chain of positive scalars connected by log-ratio
between-factors, with a prior on the first element to fix the gauge:

    prior_factor:   residual = log(x_0) - log(target_0),  J = [1]
    between_factor: residual = log(x_{i+1}/x_i) - log(m_i),  J = [-1, +1]

All Jacobians equal ±1 because the problem is linear in the tangent
(log) space, so Gauss-Newton converges in a single iteration.
"""

import numpy as np
import cupy as cp
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, WarpStateBatch

wp.init()


# ── Warp kernel: multiplicative Plus ────────────────────────────────────────

@wp.kernel
def positive_plus_kernel(
    x: wp.array(dtype=wp.float32),
    delta: wp.array(dtype=wp.float32),
    x_plus_delta: wp.array(dtype=wp.float32),
    n: int,
):
    """x_plus_delta[i] = x[i] * exp(delta[i])."""
    i = wp.tid()
    if i >= n:
        return
    x_plus_delta[i] = x[i] * wp.exp(delta[i])


# ── Warp kernels: factors ───────────────────────────────────────────────────

@wp.kernel
def log_prior_kernel(
    observations: wp.array(dtype=wp.float32),
    states: wp.array(dtype=wp.float32),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    n: int,
    write_jac: int,
):
    """residual = log(x) - log(obs), Jacobian = 1."""
    i = wp.tid()
    if i >= n:
        return
    residuals[i] = wp.log(states[i]) - wp.log(observations[i])
    if write_jac != 0:
        jacobians[i] = 1.0


@wp.kernel
def log_ratio_kernel(
    measurements: wp.array(dtype=wp.float32),
    left_vals: wp.array(dtype=wp.float32),
    right_vals: wp.array(dtype=wp.float32),
    residuals: wp.array(dtype=wp.float32),
    jacobians: wp.array(dtype=wp.float32),
    n: int,
    write_jac: int,
):
    """residual = log(right/left) - measurement, Jacobian = [-1, +1]."""
    i = wp.tid()
    if i >= n:
        return
    residuals[i] = wp.log(right_vals[i]) - wp.log(left_vals[i]) - measurements[i]
    if write_jac != 0:
        jacobians[i * 2] = -1.0
        jacobians[i * 2 + 1] = 1.0


# ── CuPy gather kernel ─────────────────────────────────────────────────────
# Reads scalar float values from an array of scattered device pointers.

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
    """Read *count* scalar float values from scattered device pointers."""
    ptrs = cp.ndarray(
        shape=(count,), dtype=cp.uint64,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(state_ptrs_ptr, count * 8, None), 0))
    out = cp.empty(count, dtype=cp.float32)
    threads = 256
    blocks = (count + threads - 1) // threads
    _gather_kernel((blocks,), (threads,), (ptrs, out, np.int32(count)))
    return out


# ── Custom state batch ──────────────────────────────────────────────────────

class PositiveScalarStateBatch(WarpStateBatch):
    """Positive-scalar manifold: Plus(x, delta) = x * exp(delta).

    Ambient dimension = 1 (the positive scalar itself).
    Tangent dimension  = 1 (delta ∈ R).
    """

    def __init__(self, data, num_blocks, **kwargs):
        super().__init__(data, ambient_size=1, tangent_size=1,
                         num_blocks=num_blocks, **kwargs)
        self._num = num_blocks

    def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle):
        n = self._num
        x = self.wrap_array(x_ptr, wp.float32, n)
        delta = self.wrap_array(delta_ptr, wp.float32, n)
        x_out = self.wrap_array(x_plus_delta_ptr, wp.float32, n)
        stream = self.make_warp_stream(stream_handle)
        wp.launch(positive_plus_kernel, dim=n,
                  inputs=[x, delta, x_out, n], stream=stream)


# ── Custom factors ──────────────────────────────────────────────────────────

class LogPriorFactor(WarpFactorBatch):
    """Prior in log-space: residual = log(x) - log(target), Jacobian = 1."""

    def __init__(self, observations_wp: wp.array, num_factors: int):
        super().__init__(residual_size=1, state_block_sizes=[1],
                         num_factors=num_factors)
        self.observations = observations_wp
        self._num = num_factors

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle):
        n = self._num
        vals = _gather_state_values(sp_ptr, n)
        states_wp = wp.array(ptr=int(vals.data.ptr), dtype=wp.float32,
                             shape=(n,), device=self._device, copy=False)

        res = self.wrap_array(res_ptr, wp.float32, n)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n)
               if jac_ptr != 0
               else wp.zeros(1, dtype=wp.float32, device=self._device))

        stream = self.make_warp_stream(stream_handle)
        wp.launch(log_prior_kernel, dim=n,
                  inputs=[self.observations, states_wp, res, jac,
                          n, write_jac],
                  stream=stream)
        return True


class LogRatioBetweenFactor(WarpFactorBatch):
    """Between factor in log-space: residual = log(x_right/x_left) - m.

    Jacobian = [-1, +1] (constant because the problem is linear in log-space).
    """

    def __init__(self, measurements_wp: wp.array, num_factors: int):
        super().__init__(residual_size=1, state_block_sizes=[1, 1],
                         num_factors=num_factors)
        self.measurements = measurements_wp
        self._num = num_factors

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle):
        n = self._num

        all_vals = _gather_state_values(sp_ptr, n * 2)
        left_vals = all_vals[0::2].copy()
        right_vals = all_vals[1::2].copy()

        left_wp = wp.array(ptr=int(left_vals.data.ptr), dtype=wp.float32,
                           shape=(n,), device=self._device, copy=False)
        right_wp = wp.array(ptr=int(right_vals.data.ptr), dtype=wp.float32,
                            shape=(n,), device=self._device, copy=False)

        res = self.wrap_array(res_ptr, wp.float32, n)
        write_jac = 1 if jac_ptr != 0 else 0
        jac = (self.wrap_array(jac_ptr, wp.float32, n * 2)
               if jac_ptr != 0
               else wp.zeros(1, dtype=wp.float32, device=self._device))

        stream = self.make_warp_stream(stream_handle)
        wp.launch(log_ratio_kernel, dim=n,
                  inputs=[self.measurements, left_wp, right_wp, res, jac,
                          n, write_jac],
                  stream=stream)
        return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    num_states = 128
    num_between = num_states - 1

    rng = np.random.default_rng(314159)

    # Ground-truth: a monotonically growing positive chain.
    gt = np.ones(num_states, dtype=np.float32)
    gt[0] = 2.0
    for i in range(1, num_states):
        gt[i] = gt[i - 1] * rng.uniform(1.05, 1.25)

    # Log-ratio measurements between consecutive elements.
    log_ratios = np.log(gt[1:] / gt[:-1]).astype(np.float32)

    # Noisy initial guess (still positive, but perturbed).
    initial = gt * rng.uniform(0.6, 1.6, num_states).astype(np.float32)

    # ── Upload to GPU ───────────────────────────────────────────────────────
    states_gpu = cp.asarray(initial)
    log_ratios_wp = wp.array(log_ratios, dtype=wp.float32, device="cuda:0")
    prior_obs_wp = wp.array(gt[:1], dtype=wp.float32, device="cuda:0")

    # ── Build cuNLS problem ─────────────────────────────────────────────────
    stream = pycunls.CudaStream()

    state_batch = PositiveScalarStateBatch(states_gpu, num_states)
    between_factor = LogRatioBetweenFactor(log_ratios_wp, num_between)
    prior_factor = LogPriorFactor(prior_obs_wp, 1)

    between_ptrs = []
    for i in range(num_between):
        between_ptrs.append(state_batch.state_block_device_ptr(i))
        between_ptrs.append(state_batch.state_block_device_ptr(i + 1))

    prior_ptrs = [state_batch.state_block_device_ptr(0)]

    problem = pycunls.Problem()
    problem.add_state_batch(state_batch)
    problem.add_factor_batch(between_factor, between_ptrs)
    problem.add_factor_batch(prior_factor, prior_ptrs)
    assert problem.check_consistency(), "Problem consistency check failed"

    # ── Solve ───────────────────────────────────────────────────────────────
    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 30
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
    mse_before = float(np.mean((np.log(initial) - np.log(gt)) ** 2))
    mse_after = float(np.mean((np.log(optimized) - np.log(gt)) ** 2))

    print("Custom Warp State Batch Example (positive-scalar manifold)")
    print(f"  Num states   : {num_states}")
    print(f"  Initial cost : {summary.initial_cost:.6f}")
    print(f"  Final cost   : {summary.final_cost:.6f}")
    print(f"  Iterations   : {summary.num_iterations}")
    print(f"  Log MSE      : {mse_before:.6f} -> {mse_after:.6f}")
    print(f"  Range [gt]   : [{gt.min():.2f}, {gt.max():.2f}]")
    print(f"  Range [opt]  : [{optimized.min():.2f}, {optimized.max():.2f}]")


if __name__ == "__main__":
    main()
