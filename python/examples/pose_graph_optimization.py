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

"""Pose Graph Optimization with pycunls.ProblemBuilder (the Python version of
``examples/pose_graph_optimization/main.cpp``).

A chain of SE(3) poses connected by relative-transform ("between") factors.
Pose 0 is held constant to anchor the chain. The builder takes NumPy arrays and
an edge list; it sets up the state and factor batches, the capacities, active
counts and state pointers, and checks the problem before solving.
"""

import numpy as np

import pycunls
from example_utils import datasets, report


def main():
    # 1. Synthetic data: ground-truth chain, its relative measurements, a
    #    perturbed initial guess.
    chain = datasets.pose_chain(num_poses=201)
    num_poses = len(chain.gt_poses)

    # 2. The graph: states from their initial values, factors from their
    #    measurements and the states each one connects. Between factor i reads
    #    [T_i, T_{i+1}] and measures Delta_i = T_{i+1}^-1 T_i.
    pb = pycunls.ProblemBuilder()
    poses = pb.add_states(pycunls.SE3StateBatch, chain.initial_poses,  # (N, 4, 4)
                          constant=[0])  # the gauge anchor
    edges = np.stack([np.arange(num_poses - 1), np.arange(1, num_poses)], axis=1)  # (N-1, 2)
    pb.add_factors(pycunls.SE3BetweenFactorBatch, chain.deltas, connect=(poses, edges))

    # 3. Solve with Levenberg-Marquardt (the default method). diagnose() runs
    #    first: a missing anchor, NaN data or a wrong state type raise or warn
    #    here. Each step factors the normal equations with BlockSparseCholesky
    #    (the default; exact steps).
    result = pb.solve(max_num_iterations=60, state_tolerance=1e-8, cost_tolerance=1e-8,
                      initial_lambda=1e-3)

    # 4. Results: the solved poses as NumPy, the outcome, residuals per batch.
    optimized = result[poses]  # (N, 4, 4)
    summary = result.summary
    report.print_summary("Pose Graph Optimization (pycunls)", summary,
                         Num_poses=num_poses, Num_factors=len(edges))
    print(result.message)
    print(result.evaluate())
    report.check(result.converged, result.message)
    report.check(summary.final_cost < 1e-3 * summary.initial_cost, "cost did not decrease")
    # Position error against the ground truth; along an open chain it grows
    # slowly with the distance from the anchor (float32 composition).
    before = np.abs(chain.initial_poses[:, :3, 3] - chain.gt_poses[:, :3, 3]).max()
    after = np.abs(optimized[:, :3, 3] - chain.gt_poses[:, :3, 3]).max()
    print(f"  Max position error: {before:.3f} -> {after:.4f}")
    report.check(after < 0.1 * before, "position error did not decrease")

    # 5. Underneath is an ordinary pycunls.Problem: pb.problem, poses.batch (the
    #    SE3StateBatch) and poses.data (the Warp array the solve updates) are the
    #    full API, e.g. for real-time re-solves with fixed buffers (see
    #    tartan_vio.py and the quick start).


if __name__ == "__main__":
    main()
