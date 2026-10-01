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

"""Custom state batch written with NVIDIA Warp: a positive-scalar manifold.

The Plus (retraction) is multiplicative, ``x (+) delta = x * exp(delta)``, so
states stay strictly positive while the tangent space is all of R: the natural
parametrization for scales, variances or rates.

A chain of positive scalars is solved from log-ratio measurements plus a
prior on the first element (both custom Warp factors):

    prior:   residual = log(x_0) - log(target_0),         J = [1]
    between: residual = log(x_{i+1} / x_i) - log(m_i),    J = [-1, +1]

The Jacobians are constant because the problem is linear in log space.
"""

import cupy as cp
import numpy as np
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, WarpStateBatch
from example_utils import datasets, metrics, report
from example_utils.gpu import gather_state_pairs, gather_state_values

wp.init()


# ── Custom state: x (+) delta = x * exp(delta) ──────────────────────────────

@wp.kernel
def positive_plus_kernel(x: wp.array(dtype=wp.float32), delta: wp.array(dtype=wp.float32),
                         x_plus_delta: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x_plus_delta[i] = x[i] * wp.exp(delta[i])


class PositiveScalarStateBatch(WarpStateBatch):
    """Ambient size 1 (the positive value), tangent size 1 (delta in R)."""

    def __init__(self, data, capacity, **kwargs):
        super().__init__(data, ambient_size=1, tangent_size=1, capacity=capacity, **kwargs)

    def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas):
        # The arrays hold num_replicas contiguous copies of the active blocks
        # (num_state_blocks, at most the capacity); every block is independent,
        # so all copies are one flat launch.
        n = self.num_state_blocks * num_replicas
        wp.launch(positive_plus_kernel, dim=n,
                  inputs=[self.wrap_array(x_ptr, wp.float32, n),
                          self.wrap_array(delta_ptr, wp.float32, n),
                          self.wrap_array(x_plus_delta_ptr, wp.float32, n)],
                  stream=self.make_warp_stream(stream_handle))


# ── Custom factors ──────────────────────────────────────────────────────────

@wp.kernel
def log_prior_kernel(observations: wp.array(dtype=wp.float32), ids: wp.array(dtype=wp.int32),
                     states: wp.array(dtype=wp.float32), residuals: wp.array(dtype=wp.float32),
                     jacobians: wp.array(dtype=wp.float32), write_jacobians: int):
    t = wp.tid()                                                     # item
    residuals[t] = wp.log(states[t]) - wp.log(observations[ids[t]])  # observation of its factor
    if write_jacobians != 0:
        jacobians[t] = 1.0


@wp.kernel
def log_ratio_kernel(measurements: wp.array(dtype=wp.float32), ids: wp.array(dtype=wp.int32),
                     left: wp.array(dtype=wp.float32), right: wp.array(dtype=wp.float32),
                     residuals: wp.array(dtype=wp.float32), jacobians: wp.array(dtype=wp.float32),
                     write_jacobians: int):
    t = wp.tid()
    residuals[t] = wp.log(right[t]) - wp.log(left[t]) - measurements[ids[t]]
    if write_jacobians != 0:
        jacobians[2 * t] = -1.0
        jacobians[2 * t + 1] = 1.0


class LogPriorFactor(WarpFactorBatch):
    """residual = log(x) - log(target); one state block."""

    def __init__(self, observations, capacity):
        super().__init__(residual_size=1, state_block_sizes=[1], capacity=capacity)
        self.observations = observations

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids                                   # number of items
        ids = self.factor_ids(factor_ids_ptr, n)             # factor of each item
        states = gather_state_values(sp_ptr, n, stream_handle)  # item t's state
        write_jacobians = 1 if jac_ptr != 0 else 0           # 0 = residuals only
        jacobians = (self.wrap_array(jac_ptr, wp.float32, n) if write_jacobians
                     else wp.zeros(1, dtype=wp.float32, device=self._device))
        wp.launch(log_prior_kernel, dim=n,
                  inputs=[self.observations, ids, wp.from_dlpack(states),
                          self.wrap_array(res_ptr, wp.float32, n), jacobians, write_jacobians],
                  stream=self.make_warp_stream(stream_handle))
        return True


class LogRatioBetweenFactor(WarpFactorBatch):
    """residual = log(x_right / x_left) - m; two state blocks."""

    def __init__(self, measurements, capacity):
        super().__init__(residual_size=1, state_block_sizes=[1, 1], capacity=capacity)
        self.measurements = measurements

    def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        ids = self.factor_ids(factor_ids_ptr, n)
        left, right = gather_state_pairs(sp_ptr, n, stream_handle)
        write_jacobians = 1 if jac_ptr != 0 else 0
        jacobians = (self.wrap_array(jac_ptr, wp.float32, 2 * n) if write_jacobians
                     else wp.zeros(1, dtype=wp.float32, device=self._device))
        wp.launch(log_ratio_kernel, dim=n,
                  inputs=[self.measurements, ids, wp.from_dlpack(left), wp.from_dlpack(right),
                          self.wrap_array(res_ptr, wp.float32, n), jacobians, write_jacobians],
                  stream=self.make_warp_stream(stream_handle))
        return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # 1. Synthetic data: growing positive chain, log-ratio measurements, noisy guess.
    chain = datasets.positive_chain(num_states=128)
    num_states = len(chain.gt)

    # 2. Upload: states with CuPy, factor data with Warp.
    states_gpu = cp.asarray(chain.initial)
    log_ratios_wp = wp.array(chain.measurements, dtype=wp.float32, device="cuda:0")
    prior_wp = wp.array(chain.gt[:1], dtype=wp.float32, device="cuda:0")

    # 3. The custom state batch and the two custom factor batches.
    #    Capacity vs. active count. A batch is constructed with its capacity: how many state blocks
    #    (or factors) its bound device buffers hold. The capacity is fixed for the batch's lifetime;
    #    size it once for the largest problem you expect. Right after construction nothing is
    #    active: set_num_state_blocks / set_num_factors set the active count, how many of the first
    #    slots the next solve uses (a solve without it throws). The setter is host-only (no
    #    allocation, no device work) and may change the count between solves up to the capacity,
    #    which is what lets a real-time application allocate once and reuse the same buffers every
    #    frame while the problem size changes. This example solves every slot once, so each active
    #    count equals its capacity.
    states_capacity = num_states  # every slot solved: active = capacity
    between_capacity = num_states - 1
    prior_capacity = 1
    states = PositiveScalarStateBatch(states_gpu, states_capacity)
    between = LogRatioBetweenFactor(log_ratios_wp, between_capacity)
    prior = LogPriorFactor(prior_wp, prior_capacity)
    num_between_factors = between_capacity
    num_prior_factors = 1
    states.set_num_state_blocks(num_states)  # active counts
    between.set_num_factors(num_between_factors)
    prior.set_num_factors(num_prior_factors)
    between_pointers = []
    for i in range(num_states - 1):
        between_pointers.append(states.state_block_device_ptr(i))
        between_pointers.append(states.state_block_device_ptr(i + 1))

    # 4. Problem.
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(between, between_pointers)
    problem.add_factor_batch(prior, [states.state_block_device_ptr(0)])
    assert problem.check_consistency(), "Problem consistency check failed"

    # 5. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 30
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # 6. Report and check (errors measured in log space).
    optimized = cp.asnumpy(states_gpu)
    mse_before = metrics.mse(np.log(chain.initial), np.log(chain.gt))
    mse_after = metrics.mse(np.log(optimized), np.log(chain.gt))
    report.print_summary(
        "Custom Warp State Batch Example (positive-scalar manifold)", summary,
        Num_states=num_states, Log_MSE=f"{mse_before:.6f} -> {mse_after:.6f}",
        Range_gt=f"[{chain.gt.min():.2f}, {chain.gt.max():.2f}]",
        Range_opt=f"[{optimized.min():.2f}, {optimized.max():.2f}]")
    report.check(mse_after < 0.01 * mse_before and optimized.min() > 0,
                 "log error did not decrease or a state left the manifold")


if __name__ == "__main__":
    main()
