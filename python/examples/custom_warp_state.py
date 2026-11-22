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

import numpy as np
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, WarpStateBatch, scalar_at
from example_utils import datasets, metrics, report

wp.init()


# ── Custom state: x (+) delta = x * exp(delta) ──────────────────────────────
# A WarpStateBatch is its Plus kernel plus sizes. pycunls fills the kernel's parameters with
# x (n, A), delta (n, T) and x_plus_delta (n, A): A = ambient_size floats stored per state,
# T = tangent_size floats per step, n = every active state (of every RANSAC replica).

@wp.kernel
def positive_plus_kernel(x: wp.array2d(dtype=wp.float32),               # (n, 1)
                         delta: wp.array2d(dtype=wp.float32),           # (n, 1)
                         x_plus_delta: wp.array2d(dtype=wp.float32)):   # (n, 1)
    i = wp.tid()                                                        # one thread per state
    x_plus_delta[i, 0] = x[i, 0] * wp.exp(delta[i, 0])


# ── Custom factors ──────────────────────────────────────────────────────────
# Each is a WarpFactorBatch: its kernel plus sizes. pycunls fills the last four kernel
# parameters: factor_ids (n,), states (n, K), residuals (n, M), jacobians (n, M, J).

@wp.kernel
def log_prior_kernel(observations: wp.array(dtype=wp.float32),
                     factor_ids: wp.array(dtype=wp.int32), states: wp.array2d(dtype=wp.uint64),
                     residuals: wp.array2d(dtype=wp.float32),
                     jacobians: wp.array3d(dtype=wp.float32)):
    """residual = log(x) - log(target); one state."""
    t = wp.tid()
    x = scalar_at(states[t, 0])
    residuals[t, 0] = wp.log(x) - wp.log(observations[factor_ids[t]])
    if jacobians.shape[0] > 0:
        jacobians[t, 0, 0] = 1.0           # d r / d delta (the tangent of x * exp(delta))


@wp.kernel
def log_ratio_kernel(measurements: wp.array(dtype=wp.float32),
                     factor_ids: wp.array(dtype=wp.int32), states: wp.array2d(dtype=wp.uint64),
                     residuals: wp.array2d(dtype=wp.float32),
                     jacobians: wp.array3d(dtype=wp.float32)):
    """residual = log(x_right / x_left) - m; two states."""
    t = wp.tid()
    left = scalar_at(states[t, 0])
    right = scalar_at(states[t, 1])
    residuals[t, 0] = wp.log(right) - wp.log(left) - measurements[factor_ids[t]]
    if jacobians.shape[0] > 0:
        jacobians[t, 0, 0] = -1.0
        jacobians[t, 0, 1] = 1.0


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # 1. Synthetic data: growing positive chain, log-ratio measurements, noisy guess.
    chain = datasets.positive_chain(num_states=128)
    num_states = len(chain.gt)

    # 2. Upload states and factor data as Warp arrays.
    states_gpu = wp.array(chain.initial, dtype=wp.float32)
    log_ratios_wp = wp.array(chain.measurements, dtype=wp.float32)
    prior_wp = wp.array(chain.gt[:1], dtype=wp.float32)

    # 3. The custom state batch and the two custom factor batches.
    #    Capacity vs. active count. A batch is constructed with its capacity: how many states (or
    #    factors) its bound device buffers hold. The capacity is fixed for the batch's lifetime;
    #    size it once for the largest problem you expect. Right after construction nothing is
    #    active: set_num_active_states / set_num_active_factors set the active count, how many of
    #    the first slots the next solve uses (a solve without it throws). The setter is host-only
    #    (no allocation, no device work) and may change the count between solves up to the capacity,
    #    which is what lets a real-time application allocate once and reuse the same buffers every
    #    frame while the problem size changes. This example solves every slot once, so each active
    #    count equals its capacity.
    states_capacity = num_states  # every slot solved: active = capacity
    between_capacity = num_states - 1
    prior_capacity = 1
    states = WarpStateBatch(positive_plus_kernel, data=states_gpu, ambient_size=1,
                            tangent_size=1, capacity=states_capacity)
    between = WarpFactorBatch(log_ratio_kernel, inputs=[log_ratios_wp], residual_size=1,
                              state_sizes=[1, 1], capacity=between_capacity)
    prior = WarpFactorBatch(log_prior_kernel, inputs=[prior_wp], residual_size=1,
                            state_sizes=[1], capacity=prior_capacity)
    num_between_factors = between_capacity
    num_prior_factors = 1
    states.set_num_active_states(num_states)  # active counts
    between.set_num_active_factors(num_between_factors)
    prior.set_num_active_factors(num_prior_factors)
    between_pointers = []
    for i in range(num_states - 1):
        between_pointers.append(states.state_device_ptr(i))
        between_pointers.append(states.state_device_ptr(i + 1))

    # 4. Problem.
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    problem.add_factor_batch(between, between_pointers)
    problem.add_factor_batch(prior, [states.state_device_ptr(0)])
    assert problem.check_consistency(), "Problem consistency check failed"

    # 5. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 30
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    wp.synchronize_stream(wp.get_stream())  # uploads ran on Warp's stream: finish them first
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)  # syncs

    # 6. Report and check (errors measured in log space).
    optimized = states_gpu.numpy()
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
