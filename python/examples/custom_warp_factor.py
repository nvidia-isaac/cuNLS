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

"""Custom factor written with NVIDIA Warp (Python port of
``examples/custom_factor/main.cu``).

Solves a chain of scalars x_0..x_{N-1} from difference measurements

    residual_i = (x_{i+1} - x_i) - measurement_i

plus a built-in prior on x_0, twice:

* Part 1 (``scalar_diff_kernel``): the Warp kernel computes the residual and
  its (constant) analytic Jacobian [-1, +1].
* Part 2 (``scalar_diff_residual_only_kernel``): the kernel computes only the
  residual; the factor is registered with
  ``jacobian_mode_override=pycunls.JacobianMode.numeric`` and pycunls
  differentiates it by finite differences.

A ``WarpFactorBatch`` is its kernel plus sizes: ``residual_size=1`` (one
residual per factor) and ``state_sizes=[1, 1]`` (two scalar states). The
kernel's last four parameters are filled by pycunls with arrays of exactly those
shapes: ``factor_ids`` (n), ``states`` (n, 2), ``residuals`` (n, 1) and
``jacobians`` (n, 1, 2); thread t evaluates factor ``factor_ids[t]``.
"""

import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch, scalar_at
from example_utils import datasets, metrics, report

wp.init()


# ── Part 1: residual and analytic Jacobian ──────────────────────────────────

@wp.kernel
def scalar_diff_kernel(measurements: wp.array(dtype=wp.float32),    # the factor's own input
                       factor_ids: wp.array(dtype=wp.int32),          # (n,)       filled by
                       states: wp.array2d(dtype=wp.uint64),           # (n, 2)     pycunls
                       residuals: wp.array2d(dtype=wp.float32),       # (n, 1)
                       jacobians: wp.array3d(dtype=wp.float32)):      # (n, 1, 2)
    t = wp.tid()                                                # one thread per factor
    left = scalar_at(states[t, 0])                              # x_i: state 0
    right = scalar_at(states[t, 1])                             # x_{i+1}: state 1
    residuals[t, 0] = (right - left) - measurements[factor_ids[t]]
    if jacobians.shape[0] > 0:                                  # no rows: residuals only
        jacobians[t, 0, 0] = -1.0                               # d r / d x_i
        jacobians[t, 0, 1] = 1.0                                # d r / d x_{i+1}


# ── Part 2: residual only, numeric Jacobian ─────────────────────────────────

@wp.kernel
def scalar_diff_residual_only_kernel(measurements: wp.array(dtype=wp.float32),
                                     factor_ids: wp.array(dtype=wp.int32),
                                     states: wp.array2d(dtype=wp.uint64),
                                     residuals: wp.array2d(dtype=wp.float32),
                                     jacobians: wp.array3d(dtype=wp.float32)):
    t = wp.tid()
    left = scalar_at(states[t, 0])
    right = scalar_at(states[t, 1])
    residuals[t, 0] = (right - left) - measurements[factor_ids[t]]
    # No Jacobian: with JacobianMode.numeric, jacobians never has rows.


# ── Main ────────────────────────────────────────────────────────────────────

def run_chain_example(title, use_numeric_jacobian):
    # 1. Synthetic data: monotonic chain, exact differences, noisy initial guess.
    chain = datasets.scalar_chain(num_states=256)
    num_states = len(chain.gt)

    # 2. Upload states, the anchor value and the measurements as Warp arrays.
    states_gpu = wp.array(chain.initial, dtype=wp.float32)
    prior_gpu = wp.array(chain.gt[:1], dtype=wp.float32)
    measurements_wp = wp.array(chain.measurements, dtype=wp.float32)

    # 3. Scalar states; one difference factor per pair (x_i, x_{i+1}); a
    #    built-in prior anchoring x_0.
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
    states = pycunls.VectorStateBatch1(states_gpu, states_capacity)
    states.set_num_active_states(num_states)  # active count
    diff_pointers = []
    for i in range(num_states - 1):
        diff_pointers.append(states.state_device_ptr(i))
        diff_pointers.append(states.state_device_ptr(i + 1))
    num_diff_factors = num_states - 1
    diff_capacity = num_diff_factors  # used by the difference factor in step 4
    prior_capacity = 1
    prior = pycunls.PriorVectorFactorBatch1(prior_gpu, prior_capacity)
    num_prior_factors = 1
    prior.set_num_active_factors(num_prior_factors)

    # 4. Problem. Part 2 overrides the Jacobian mode of the difference factors
    #    only; the prior keeps its analytic Jacobian.
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    kernel = scalar_diff_residual_only_kernel if use_numeric_jacobian else scalar_diff_kernel
    diff = WarpFactorBatch(kernel, inputs=[measurements_wp],
                           residual_size=1, state_sizes=[1, 1], capacity=diff_capacity)
    diff.set_num_active_factors(num_diff_factors)  # active count
    if use_numeric_jacobian:
        problem.add_factor_batch(diff, diff_pointers,
                                 jacobian_mode_override=pycunls.JacobianMode.numeric)
    else:
        problem.add_factor_batch(diff, diff_pointers)
    problem.add_factor_batch(prior, [states.state_device_ptr(0)])
    assert problem.check_consistency(), "Problem consistency check failed"

    # 5. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 50
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    wp.synchronize_stream(wp.get_stream())  # uploads ran on Warp's stream: finish them first
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)  # syncs

    # 6. Report and check.
    mse_before = metrics.mse(chain.initial, chain.gt)
    mse_after = metrics.mse(states_gpu.numpy(), chain.gt)
    report.print_summary(title, summary, State_MSE=f"{mse_before:.6f} -> {mse_after:.6f}")
    report.check(mse_after < 0.01 * mse_before, "state error did not decrease")


def main():
    run_chain_example("Part 1: analytic Jacobian (Warp)", use_numeric_jacobian=False)
    print()
    run_chain_example("Part 2: residual-only, numeric Jacobian (Warp)", use_numeric_jacobian=True)


if __name__ == "__main__":
    main()
