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

"""Pose Graph Optimization using pycunls with CuPy arrays.

This is a Python port of ``examples/pose_graph_optimization/main.cpp``.
It builds a chain of SE(3) poses connected by between constraints, adds
noise, and recovers the ground-truth chain using Levenberg-Marquardt.

Problem structure
-----------------
* **States**: A single SE3StateBatch with N poses (ambient=16, tangent=6).
* **Factors**: (N-1) SE3BetweenFactorBatch entries.  Each factor connects
  two consecutive poses and measures the relative transform between them.
* **Gauge fix**: Pose 0 is held constant via ``const_ids`` to anchor the
  chain.

The ground-truth chain is computed as T_{i+1} = T_i * inv(delta_i), where
delta_i is a random relative transform.  The initial guess is obtained by
perturbing the ground-truth with small random SE(3) elements.
"""

import numpy as np
import cupy as cp

import pycunls
from se3_utils import twist_to_se3, se3_inverse, compose_se3


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    num_poses = 201
    num_constraints = num_poses - 1
    rng = np.random.default_rng(9012)

    # Random anchor and relative transforms.
    def random_se3(rot_scale=0.3, trans_scale=1.0):
        tw = np.zeros(6, dtype=np.float32)
        tw[:3] = rng.uniform(-rot_scale, rot_scale, 3).astype(np.float32)
        tw[3:] = rng.uniform(-trans_scale, trans_scale, 3).astype(np.float32)
        return twist_to_se3(tw)

    anchor = random_se3()
    deltas = [random_se3() for _ in range(num_constraints)]

    # Ground-truth chain:  T_{i+1} = T_i * inv(delta_i)
    gt_poses = [anchor]
    for i in range(num_constraints):
        gt_poses.append(compose_se3(gt_poses[-1], se3_inverse(deltas[i])))

    # Perturbed initial poses (anchor is fixed).
    initial_poses = [gt_poses[0].copy()]
    for i in range(1, num_poses):
        perturbation = random_se3(rot_scale=0.05, trans_scale=0.3)
        initial_poses.append(compose_se3(perturbation, gt_poses[i]))

    # ── Upload to GPU ───────────────────────────────────────────────────────
    poses_flat = np.stack(initial_poses).reshape(-1).astype(np.float32)
    deltas_flat = np.stack(deltas).reshape(-1).astype(np.float32)

    poses_gpu = cp.asarray(poses_flat)
    deltas_gpu = cp.asarray(deltas_flat)
    const_ids_gpu = cp.array([0], dtype=cp.int32)

    # ── Build cuNLS problem ─────────────────────────────────────────────────
    cublas = pycunls.CublasHandle()
    stream = pycunls.CudaStream()

    pose_states = pycunls.SE3StateBatch(
        cublas, poses_gpu, num_poses, const_ids_gpu, 1)

    between_factor = pycunls.SE3BetweenFactorBatch(
        deltas_gpu, num_constraints)

    state_pointers = []
    for i in range(num_constraints):
        state_pointers.append(pose_states.state_block_device_ptr(i))
        state_pointers.append(pose_states.state_block_device_ptr(i + 1))

    problem = pycunls.Problem()
    problem.add_state_batch(pose_states)
    problem.add_factor_batch(between_factor, state_pointers)
    assert problem.check_consistency(), "Problem consistency check failed"

    # ── Solve ───────────────────────────────────────────────────────────────
    opts = pycunls.MinimizerOptions()
    opts.max_num_iterations = 60
    opts.state_tolerance = 1e-8
    opts.cost_tolerance = 1e-8

    lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
    lm_opts.base_options = opts
    lm_opts.initial_lambda = 1e-3

    minimizer = pycunls.LevenbergMarquardtMinimizer(lm_opts)
    summary = minimizer.minimize(stream, problem)

    cp.cuda.runtime.streamSynchronize(stream.get_stream())

    # ── Report ──────────────────────────────────────────────────────────────
    print("Pose Graph Optimization (pycunls)")
    print(f"  Num poses      : {num_poses}")
    print(f"  Num constraints: {num_constraints}")
    print(f"  Initial cost   : {summary.initial_cost:.6f}")
    print(f"  Final cost     : {summary.final_cost:.6f}")
    print(f"  Iterations     : {summary.num_iterations}")


if __name__ == "__main__":
    main()
