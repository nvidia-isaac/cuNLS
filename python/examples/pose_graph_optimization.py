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

"""Pose Graph Optimization with pycunls (Python port of
``examples/pose_graph_optimization/main.cpp``).

A chain of SE(3) poses connected by relative-transform ("between") factors.
Pose 0 is held constant to anchor the chain.
"""

import cupy as cp

import pycunls
from example_utils import datasets, report


def main():
    # 1. Synthetic data: ground-truth chain, its relative measurements, a
    #    perturbed initial guess.
    chain = datasets.pose_chain(num_poses=201)
    num_poses = len(chain.gt_poses)
    num_constraints = num_poses - 1

    # 2. Upload to the GPU (row-major 4x4 matrices).
    poses_gpu = cp.asarray(chain.initial_poses.reshape(-1))
    deltas_gpu = cp.asarray(chain.deltas.reshape(-1))
    const_ids_gpu = cp.array([0], dtype=cp.int32)  # pose 0 is constant

    # 3. One SE(3) state batch; one between factor per consecutive pair.
    cublas = pycunls.CublasHandle()
    pose_states = pycunls.SE3StateBatch(cublas, poses_gpu, num_poses, const_ids_gpu, 1)
    between = pycunls.SE3BetweenFactorBatch(deltas_gpu, num_constraints)
    state_pointers = []
    for i in range(num_constraints):
        state_pointers.append(pose_states.state_block_device_ptr(i))
        state_pointers.append(pose_states.state_block_device_ptr(i + 1))

    # 4. Problem.
    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_factor_batch(between, state_pointers)
    assert problem.check_consistency(), "Problem consistency check failed"

    # 5. Solve with Levenberg-Marquardt.
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 60
    options.base_options.state_tolerance = 1e-8
    options.base_options.cost_tolerance = 1e-8
    options.initial_lambda = 1e-3
    stream = pycunls.CudaStream()
    summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)
    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # 6. Report and check.
    report.print_summary("Pose Graph Optimization (pycunls)", summary,
                         Num_poses=num_poses, Num_factors=num_constraints)
    report.check(summary.final_cost < 1e-3 * summary.initial_cost, "cost did not decrease")


if __name__ == "__main__":
    main()
