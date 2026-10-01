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

"""Robust PnP with RansacLevenbergMarquardtMinimizer (Python port of
``examples/ransac_pnp/main.cpp``).

Recovers a camera pose from 3D-2D correspondences of which a large fraction
are gross outliers (wrong matches). The problem is built exactly as for the
regular minimizers; only the minimizer and its options differ. A plain
LevenbergMarquardtMinimizer runs on the same data for comparison.

Usage::

    python ransac_pnp.py [--num-points N] [--outlier-ratio R]
"""

import argparse

import cupy as cp

import pycunls
from example_utils import datasets, metrics, report

INLIER_THRESHOLD = 0.01  # |r| in normalized image units; ~3.3 sigma of the 3e-3 noise


class PnPProblem:
    """One SE(3) pose state and one PnP factor per correspondence, all reading
    that pose. The GPU arrays are attributes so they outlive the problem."""

    def __init__(self, scene):
        n = len(scene.points)
        self.cublas = pycunls.CublasHandle()
        self.pose_gpu = cp.asarray(scene.initial_pose.reshape(-1))
        self.points_gpu = cp.asarray(scene.points.reshape(-1))
        self.observations_gpu = cp.asarray(scene.observations.reshape(-1))

        self.pose_state = pycunls.SE3StateBatch(self.cublas, self.pose_gpu, 1)
        self.pnp = pycunls.PnPFactorBatch(self.observations_gpu, self.points_gpu, n, 1e-3)
        self.problem = pycunls.Problem()
        self.problem.add_state_batch(self.pose_state)
        self.problem.add_factor_batch(self.pnp, [self.pose_state.state_block_device_ptr(0)] * n)
        assert self.problem.check_consistency()

    def pose(self):
        return cp.asnumpy(self.pose_gpu).reshape(4, 4)


def solve_with_lm(scene):
    """Plain Levenberg-Marquardt: every correspondence is treated as an inlier."""
    p = PnPProblem(scene)
    options = pycunls.LevenbergMarquardtMinimizerOptions()
    options.base_options.max_num_iterations = 60
    pycunls.LevenbergMarquardtMinimizer(options).minimize(pycunls.CudaStream(), p.problem)
    return p.pose()


def solve_with_ransac_lm(scene):
    """RANSAC + Levenberg-Marquardt. Returns (pose, inlier mask, summary)."""
    p = PnPProblem(scene)

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
    summary = minimizer.minimize(pycunls.CudaStream(), p.problem)  # writes the pose back
    mask = minimizer.inlier_mask(0)                                # numpy uint8, 1 = inlier
    return p.pose(), mask, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-points", type=int, default=2000)
    parser.add_argument("--outlier-ratio", type=float, default=0.5)
    args = parser.parse_args()

    # 1. Synthetic data with gross outliers.
    scene = datasets.pnp_scene(args.num_points, args.outlier_ratio, INLIER_THRESHOLD)
    print(f"RANSAC PnP example (pycunls): {args.num_points} correspondences, "
          f"{args.outlier_ratio * 100:.0f}% outliers")

    # 2. Solve with plain LM and with RANSAC-LM.
    lm_pose = solve_with_lm(scene)
    ransac_pose, mask, summary = solve_with_ransac_lm(scene)

    # 3. Report and check.
    for label, pose in [("Initial guess", scene.initial_pose), ("LevenbergMarquardt", lm_pose),
                        ("RansacLM", ransac_pose)]:
        report.print_pose_error(label, metrics.rotation_error_deg(pose, scene.gt_pose),
                                metrics.translation_error(pose, scene.gt_pose))
    kept, true_inliers, accepted = metrics.inlier_mask_stats(mask, scene.is_outlier)
    print(f"  RANSAC: {summary.num_rounds} round(s), {summary.num_hypotheses} hypotheses, "
          f"{summary.num_inliers} inliers ({summary.inlier_ratio * 100:.1f}%)")
    print(f"  Inlier mask: {kept} of {true_inliers} true inliers kept, "
          f"{accepted} outliers accepted")
    report.check(metrics.rotation_error_deg(ransac_pose, scene.gt_pose) < 0.5 and
                 metrics.translation_error(ransac_pose, scene.gt_pose) < 0.05 and accepted == 0,
                 "RANSAC pose or inlier mask")


if __name__ == "__main__":
    main()
