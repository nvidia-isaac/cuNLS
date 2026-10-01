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

* Part 1 (``ScalarDiffFactor``): the Warp kernel computes the residual and
  its (constant) analytic Jacobian [-1, +1].
* Part 2 (``ScalarDiffResidualOnlyFactor``): the kernel computes only the
  residual; the factor is registered with
  ``jacobian_mode_override=pycunls.JacobianMode.numeric`` and pycunls
  differentiates it by finite differences.

Both factors follow the item contract of ``WarpFactorBatch.evaluate``: item t
reads the measurement of its factor ``ids[t]`` and its own two states.
"""

import cupy as cp
import warp as wp

import pycunls
from pycunls.warp import WarpFactorBatch
from example_utils import datasets, metrics, report
from example_utils.gpu import gather_state_pairs

wp.init()


# ── Part 1: residual and analytic Jacobian ──────────────────────────────────

@wp.kernel
def scalar_diff_kernel(measurements: wp.array(dtype=wp.float32),
                       ids: wp.array(dtype=wp.int32),
                       left: wp.array(dtype=wp.float32),
                       right: wp.array(dtype=wp.float32),
                       residuals: wp.array(dtype=wp.float32),
                       jacobians: wp.array(dtype=wp.float32),
                       write_jacobians: int):
    t = wp.tid()                                                # item
    residuals[t] = (right[t] - left[t]) - measurements[ids[t]]  # measurement of its factor
    if write_jacobians != 0:
        jacobians[2 * t] = -1.0                                 # d r / d x_i
        jacobians[2 * t + 1] = 1.0                              # d r / d x_{i+1}


class ScalarDiffFactor(WarpFactorBatch):
    """residual = (x_right - x_left) - m; one residual, two scalar state blocks."""

    def __init__(self, measurements, num_factors):
        super().__init__(residual_size=1, state_block_sizes=[1, 1], num_factors=num_factors)
        self.measurements = measurements

    def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
                 factor_ids_ptr, num_factor_ids):
        n = num_factor_ids                                  # number of items
        ids = self.factor_ids(factor_ids_ptr, n)            # factor of each item
        # Item t's states are state_pointers[2t] (x_i) and [2t + 1] (x_{i+1}).
        left, right = gather_state_pairs(state_pointers_ptr, n, stream_handle)

        write_jacobians = 1 if jacobians_ptr != 0 else 0    # 0 = residuals only
        jacobians = (self.wrap_array(jacobians_ptr, wp.float32, 2 * n) if write_jacobians
                     else wp.zeros(1, dtype=wp.float32, device=self._device))
        wp.launch(scalar_diff_kernel, dim=n,
                  inputs=[self.measurements, ids, wp.from_dlpack(left), wp.from_dlpack(right),
                          self.wrap_array(residuals_ptr, wp.float32, n), jacobians,
                          write_jacobians],
                  stream=self.make_warp_stream(stream_handle))
        return True


# ── Part 2: residual only, numeric Jacobian ─────────────────────────────────

@wp.kernel
def scalar_diff_residual_only_kernel(measurements: wp.array(dtype=wp.float32),
                                     ids: wp.array(dtype=wp.int32),
                                     left: wp.array(dtype=wp.float32),
                                     right: wp.array(dtype=wp.float32),
                                     residuals: wp.array(dtype=wp.float32)):
    t = wp.tid()
    residuals[t] = (right[t] - left[t]) - measurements[ids[t]]


class ScalarDiffResidualOnlyFactor(WarpFactorBatch):
    """Same residual, no Jacobian code: register it with JacobianMode.numeric."""

    def __init__(self, measurements, num_factors):
        super().__init__(residual_size=1, state_block_sizes=[1, 1], num_factors=num_factors)
        self.measurements = measurements

    def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr, stream_handle,
                 factor_ids_ptr, num_factor_ids):
        n = num_factor_ids
        ids = self.factor_ids(factor_ids_ptr, n)
        left, right = gather_state_pairs(state_pointers_ptr, n, stream_handle)
        wp.launch(scalar_diff_residual_only_kernel, dim=n,
                  inputs=[self.measurements, ids, wp.from_dlpack(left), wp.from_dlpack(right),
                          self.wrap_array(residuals_ptr, wp.float32, n)],
                  stream=self.make_warp_stream(stream_handle))
        return True


# ── Main ────────────────────────────────────────────────────────────────────

def run_chain_example(title, use_numeric_jacobian):
    # 1. Synthetic data: monotonic chain, exact differences, noisy initial guess.
    chain = datasets.scalar_chain(num_states=256)
    num_states = len(chain.gt)

    # 2. Upload: states and the anchor value with CuPy, measurements with Warp.
    states_gpu = cp.asarray(chain.initial)
    prior_gpu = cp.asarray(chain.gt[:1])
    measurements_wp = wp.array(chain.measurements, dtype=wp.float32, device="cuda:0")

    # 3. Scalar states; one difference factor per pair (x_i, x_{i+1}); a
    #    built-in prior anchoring x_0.
    states = pycunls.VectorStateBatch1(states_gpu, num_states)
    diff_pointers = []
    for i in range(num_states - 1):
        diff_pointers.append(states.state_block_device_ptr(i))
        diff_pointers.append(states.state_block_device_ptr(i + 1))
    prior = pycunls.PriorVectorFactorBatch1(prior_gpu, 1)

    # 4. Problem. Part 2 overrides the Jacobian mode of the difference factors
    #    only; the prior keeps its analytic Jacobian.
    problem = pycunls.Problem()
    problem.add_state_batch(states)
    if use_numeric_jacobian:
        diff = ScalarDiffResidualOnlyFactor(measurements_wp, num_states - 1)
        problem.add_factor_batch(diff, diff_pointers,
                                 jacobian_mode_override=pycunls.JacobianMode.numeric)
    else:
        diff = ScalarDiffFactor(measurements_wp, num_states - 1)
        problem.add_factor_batch(diff, diff_pointers)
    problem.add_factor_batch(prior, [states.state_block_device_ptr(0)])
    assert problem.check_consistency(), "Problem consistency check failed"

    # 5. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 50
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # 6. Report and check.
    mse_before = metrics.mse(chain.initial, chain.gt)
    mse_after = metrics.mse(cp.asnumpy(states_gpu), chain.gt)
    report.print_summary(title, summary, State_MSE=f"{mse_before:.6f} -> {mse_after:.6f}")
    report.check(mse_after < 0.01 * mse_before, "state error did not decrease")


def main():
    run_chain_example("Part 1: analytic Jacobian (Warp)", use_numeric_jacobian=False)
    print()
    run_chain_example("Part 2: residual-only, numeric Jacobian (Warp)", use_numeric_jacobian=True)


if __name__ == "__main__":
    main()
