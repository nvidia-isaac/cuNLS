/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Robust PnP with RansacLevenbergMarquardtMinimizer: recover a camera pose
// from 3D-2D correspondences of which a large fraction are gross outliers
// (wrong matches). The problem is built exactly as for the regular
// minimizers; only the minimizer and its options differ. A plain
// LevenbergMarquardtMinimizer runs on the same data for comparison.

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/cli.h"
#include "utils/datasets.h"
#include "utils/report.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

namespace {

constexpr float kInlierThreshold = 0.01f;  // ~3.3 sigma of the 2D residual norm

// The PnP problem on the GPU: one SE(3) pose state and one PnP factor per
// correspondence, all reading that pose. Built exactly as for any minimizer.
struct PnPProblem {
  // Capacity vs. active count. A batch is constructed with its capacity: how many states (or
  // factors) its bound device buffers hold. The capacity is fixed for the batch's lifetime; size it
  // once for the largest problem you expect. Right after construction nothing is active:
  // SetNumActiveStates / SetNumActiveFactors set the active count, how many of the first slots the
  // next solve uses (a solve without it throws). The setter is host-only (no allocation, no device
  // work) and may change the count between solves up to the capacity, which is what lets a
  // real-time application allocate once and reuse the same buffers every frame while the problem
  // size changes. This example solves every slot once, so each active count equals its capacity.
  static constexpr size_t kPoseCapacity = 1;
  size_t points_capacity;  // correspondences the observation / point buffers hold
  dvector<Vector<3>> points;
  dvector<Vector<2>> observations;
  dvector<SE3Transform> pose;
  cunls::SE3StateBatch pose_state;
  cunls::PnPFactorBatch pnp;
  cunls::Problem problem;

  explicit PnPProblem(const examples::PnPScene &scene)
      : points_capacity(scene.points_world.size()),
        points(scene.points_world),
        observations(scene.observations),
        pose(std::vector<SE3Transform>{scene.initial_pose}),
        pose_state(reinterpret_cast<const float *>(pose.data()), kPoseCapacity),
        pnp(observations.data(), points.data(), points_capacity, /*z_threshold=*/1e-3f) {
    const size_t num_poses = 1;
    const size_t num_points = points_capacity;  // every slot solved: active = capacity
    pose_state.SetNumActiveStates(num_poses);
    pnp.SetNumActiveFactors(num_points);
    problem.AddStateBatch(&pose_state);
    problem.AddFactorBatch(&pnp, std::vector<float *>(num_points, pose_state.StateDevicePtr(0)));
  }

  SE3Transform Pose() const {
    SE3Transform out;
    pose.CopyToHost(&out, 1);
    return out;
  }
};

}  // namespace

int main(int argc, char **argv) {
  try {
    const examples::CommandLine cli(argc, argv, {"--num-points", "--outlier-ratio"});
    examples::PnPSceneOptions data;
    data.num_points = cli.GetSize("--num-points", 2000);
    data.outlier_ratio = cli.GetFloat("--outlier-ratio", 0.5f);
    if (data.num_points < 10 || !(data.outlier_ratio >= 0.f && data.outlier_ratio < 1.f)) {
      cli.Fail("Need --num-points >= 10 and --outlier-ratio in [0, 1)");
    }
    data.min_outlier_distance = 4 * kInlierThreshold;
    data.init_rotation = 0.1f;
    data.init_translation = 0.3f;
    data.seed = 2024;
    const examples::PnPScene scene = examples::MakePnPScene(data);

    examples::PrintTitle(examples::Str("RANSAC PnP example: ", data.num_points,
                                       " correspondences, ", data.outlier_ratio * 100,
                                       "% outliers"));
    examples::PrintPoseError("Initial guess", scene.initial_pose, scene.gt_pose);
    cunls::CudaStream stream;

    // Plain Levenberg-Marquardt treats every correspondence as an inlier.
    PnPProblem lm_problem(scene);
    cunls::LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options.max_num_iterations = 60;
    cunls::LevenbergMarquardtMinimizer(lm_options).Minimize(stream.GetStream(), lm_problem.problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    examples::PrintPoseError("LevenbergMarquardt", lm_problem.Pose(), scene.gt_pose);

    // RANSAC + Levenberg-Marquardt on a fresh copy of the same problem.
    PnPProblem ransac_problem(scene);
    cunls::RansacLevenbergMarquardtMinimizerOptions options;
    cunls::RansacMinimizerOptions &ransac = options.base_options;
    // One entry per residual batch, in the order they were added to the problem.
    // The PnP batch is sampled: its factors may be outliers and are classified
    // with |r| <= inlier_threshold.
    ransac.factor_batches = {{cunls::RansacRole::kSampled, kInlierThreshold}};
    ransac.hypotheses_per_round = 256;  // hypotheses generated in parallel per round
    ransac.max_rounds = 8;              // adaptive stopping usually needs fewer
    ransac.confidence = 0.999f;         // P(at least one all-inlier sample)
    ransac.seed = 1;                    // same seed -> bitwise identical result
    cunls::RansacLevenbergMarquardtMinimizer minimizer(options);
    const cunls::RansacSummary summary =
        minimizer.Minimize(stream.GetStream(), ransac_problem.problem);  // writes the pose back

    // Inlier mask of residual batch 0: one byte per factor (device), 1 = inlier.
    std::vector<uint8_t> mask(minimizer.InlierMaskSize(0));
    THROW_ON_CUDA_ERROR(
        cudaMemcpy(mask.data(), minimizer.InlierMask(0), mask.size(), cudaMemcpyDeviceToHost));

    const SE3Transform pose = ransac_problem.Pose();
    const examples::InlierMaskStats stats = examples::CompareInlierMask(mask, scene.is_outlier);
    examples::PrintPoseError("RansacLM", pose, scene.gt_pose);
    examples::PrintRansacResult(summary, stats);
    return examples::QualityExitCode(examples::RotationErrorDeg(pose, scene.gt_pose) < 0.5f &&
                                     examples::TranslationError(pose, scene.gt_pose) < 0.05f &&
                                     stats.accepted_outliers == 0);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
