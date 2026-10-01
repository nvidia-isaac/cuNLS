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

"""Robust PnP with RansacLevenbergMarquardtMinimizer.

Python port of ``examples/ransac_pnp/main.cpp``. Recovers a camera pose from
3D-2D correspondences of which a large fraction are gross outliers (wrong
matches). The problem is built exactly as for the regular minimizers; only the
minimizer and its options differ. A plain LevenbergMarquardtMinimizer runs on
the same data for comparison: the outliers pull it away, RANSAC is not.

Usage::

    python ransac_pnp.py [--num-points N] [--outlier-ratio R]
"""

import argparse

import cupy as cp
import numpy as np

import pycunls
from se3_utils import compose_se3, compute_depth, project_normalized, twist_to_se3

PIXEL_NOISE = 3e-3        # inlier noise per axis, normalized image units
INLIER_THRESHOLD = 0.01   # ~3.3 sigma on the 2D residual norm
Z_THRESHOLD = 1e-3        # PnPFactorBatch guard against z ~ 0


def generate_dataset(num_points, outlier_ratio, rng):
    """Noisy projections of random points; outliers are random image points."""
    twist = np.concatenate([rng.uniform(-0.3, 0.3, 3),
                            rng.uniform(-1.0, 1.0, 2),
                            [8.0 + rng.uniform(-1.0, 1.0)]]).astype(np.float32)
    gt_pose = twist_to_se3(twist)

    points = np.empty((num_points, 3), dtype=np.float32)
    observations = np.empty((num_points, 2), dtype=np.float32)
    is_outlier = rng.uniform(size=num_points) < outlier_ratio
    for i in range(num_points):
        p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        while compute_depth(gt_pose, p) < 1.0:
            p = rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        proj = project_normalized(gt_pose, p)
        if is_outlier[i]:
            obs = rng.uniform(-0.5, 0.5, 2)
            while np.linalg.norm(obs - proj) < 4 * INLIER_THRESHOLD:
                obs = rng.uniform(-0.5, 0.5, 2)
        else:
            obs = proj + rng.normal(0.0, PIXEL_NOISE, 2)
        points[i] = p
        observations[i] = obs

    delta = np.concatenate([rng.uniform(-0.1, 0.1, 3),
                            rng.uniform(-0.3, 0.3, 3)]).astype(np.float32)
    initial_pose = compose_se3(twist_to_se3(delta), gt_pose)
    return gt_pose, initial_pose, points, observations, is_outlier


class PnPProblem:
    """One SE(3) pose state and one PnP factor per correspondence.

    Built exactly as for any minimizer. The GPU arrays are kept as attributes
    so they outlive the problem.
    """

    def __init__(self, initial_pose, points, observations):
        self.cublas = pycunls.CublasHandle()
        self.pose_gpu = cp.asarray(initial_pose.reshape(-1).astype(np.float32))
        self.points_gpu = cp.asarray(points.reshape(-1))
        self.observations_gpu = cp.asarray(observations.reshape(-1))
        n = len(points)

        self.pose_state = pycunls.SE3StateBatch(self.cublas, self.pose_gpu, 1)
        self.pnp = pycunls.PnPFactorBatch(self.observations_gpu, self.points_gpu, n,
                                          Z_THRESHOLD)
        self.problem = pycunls.Problem()
        self.problem.add_state_batch(self.pose_state)
        # Every factor reads the same (single) pose.
        self.problem.add_factor_batch(self.pnp, [self.pose_state.state_block_device_ptr(0)] * n)
        assert self.problem.check_consistency()

    def pose(self):
        return cp.asnumpy(self.pose_gpu).reshape(4, 4)


def solve_with_lm(data):
    """Plain Levenberg-Marquardt: every correspondence is treated as an inlier."""
    _, initial_pose, points, observations, _ = data
    p = PnPProblem(initial_pose, points, observations)
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 60
    stream = pycunls.CudaStream()
    pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, p.problem)
    return p.pose()


def solve_with_ransac_lm(data):
    """RANSAC + Levenberg-Marquardt. Returns (pose, inlier mask, summary)."""
    _, initial_pose, points, observations, _ = data
    p = PnPProblem(initial_pose, points, observations)

    options = pycunls.RansacLevenbergMarquardtMinimizerOptions()
    ransac = options.base_options  # a reference: edits below change `options`
    # One entry per residual batch, in the order they were added. The PnP batch
    # is sampled: its factors may be outliers and are classified by
    # |r| <= inlier_threshold. Assign the whole list (not .append()).
    ransac.factor_batches = [
        pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, INLIER_THRESHOLD)
    ]
    ransac.hypotheses_per_round = 256  # hypotheses generated in parallel per round
    ransac.max_rounds = 8              # adaptive stopping usually needs fewer
    ransac.confidence = 0.999          # P(at least one all-inlier sample)
    ransac.seed = 1                    # same seed -> bitwise identical result

    minimizer = pycunls.RansacLevenbergMarquardtMinimizer(options)
    stream = pycunls.CudaStream()
    summary = minimizer.minimize(stream, p.problem)  # writes the pose back
    mask = minimizer.inlier_mask(0)                  # numpy uint8, 1 = inlier
    return p.pose(), mask, summary


def rotation_error_deg(a, b):
    c = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def translation_error(a, b):
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--outlier-ratio", type=float, default=0.5)
    args = parser.parse_args()

    rng = np.random.default_rng(2024)
    data = generate_dataset(args.num_points, args.outlier_ratio, rng)
    gt_pose, initial_pose, _, _, is_outlier = data

    def report(label, pose):
        print(f"  {label:<20}: rotation error {rotation_error_deg(pose, gt_pose):.4f} deg, "
              f"translation error {translation_error(pose, gt_pose):.4f}")

    print(f"RANSAC PnP example (pycunls): {args.num_points} correspondences, "
          f"{args.outlier_ratio * 100:.0f}% outliers")
    report("Initial guess", initial_pose)
    report("LevenbergMarquardt", solve_with_lm(data))
    pose, mask, summary = solve_with_ransac_lm(data)
    report("RansacLM", pose)

    kept = int(np.sum((mask == 1) & ~is_outlier))
    accepted_outliers = int(np.sum((mask == 1) & is_outlier))
    print(f"  RANSAC: {summary.num_rounds} round(s), {summary.num_hypotheses} hypotheses, "
          f"{summary.num_inliers} inliers ({summary.inlier_ratio * 100:.1f}%)")
    print(f"  Inlier mask: {kept} of {int(np.sum(~is_outlier))} true inliers kept, "
          f"{accepted_outliers} outliers accepted")

    ok = (rotation_error_deg(pose, gt_pose) < 0.5 and translation_error(pose, gt_pose) < 0.05
          and accepted_outliers == 0)
    if not ok:
        raise SystemExit("RANSAC quality check failed")


if __name__ == "__main__":
    main()
