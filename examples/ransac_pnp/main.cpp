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

// Robust PnP with RansacLevenbergMarquardtMinimizer.
//
// Recovers a camera pose from 3D-2D correspondences of which a large fraction
// are gross outliers (wrong matches). The problem is built exactly as for the
// regular minimizers (see examples/pnp); only the minimizer and its options
// differ. A plain LevenbergMarquardtMinimizer is run on the same problem for
// comparison: it is pulled away by the outliers, RANSAC is not.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/camera_utils.h"
#include "utils/se3_utils.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

namespace {

constexpr float kMinDepth = 1.0f;          // generated points are at least this far in front
constexpr float kZThreshold = 1e-3f;       // PnPFactorBatch guard against z ~ 0
constexpr float kPixelNoise = 3e-3f;       // inlier noise (normalized image units, per axis)
constexpr float kInlierThreshold = 0.01f;  // ~3.3 sigma on the 2D residual norm

struct Dataset {
  SE3Transform gt_pose;
  SE3Transform initial_pose;
  std::vector<Vector<3>> points_world;
  std::vector<Vector<2>> observations;
  std::vector<uint8_t> is_outlier;
};

// Synthetic PnP data: inliers are noisy projections; outliers are random
// image points at least 4 thresholds away from the true projection.
Dataset GenerateDataset(size_t num_points, float outlier_ratio, std::mt19937 &rng) {
  Dataset data;
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f), trans(-1.0f, 1.0f);
  Vector<6> twist;
  twist[0] = rot(rng);
  twist[1] = rot(rng);
  twist[2] = rot(rng);
  twist[3] = trans(rng);
  twist[4] = trans(rng);
  twist[5] = 8.0f + trans(rng);
  std::vector<SE3Transform> gt;
  examples::TwistsToSE3({twist}, gt);
  data.gt_pose = gt[0];

  std::uniform_real_distribution<float> coord(-3.0f, 3.0f), image(-0.5f, 0.5f), unit(0.f, 1.f);
  std::normal_distribution<float> noise(0.0f, kPixelNoise);
  for (size_t i = 0; i < num_points; ++i) {
    Vector<3> p;
    do {
      p[0] = coord(rng);
      p[1] = coord(rng);
      p[2] = coord(rng);
    } while (examples::ComputeDepth(data.gt_pose, p) < kMinDepth);
    const Vector<2> proj = examples::ProjectNormalized(data.gt_pose, p);
    Vector<2> obs = proj;
    const bool outlier = unit(rng) < outlier_ratio;
    if (outlier) {
      do {
        obs[0] = image(rng);
        obs[1] = image(rng);
      } while (std::hypot(obs[0] - proj[0], obs[1] - proj[1]) < 4 * kInlierThreshold);
    } else {
      obs[0] += noise(rng);
      obs[1] += noise(rng);
    }
    data.points_world.push_back(p);
    data.observations.push_back(obs);
    data.is_outlier.push_back(outlier ? 1 : 0);
  }

  std::vector<SE3Transform> perturbation;
  examples::GenerateRandomSE3(1, rng, perturbation, 0.1f, 0.3f);
  data.initial_pose = examples::ComposeSE3(perturbation[0], data.gt_pose);
  return data;
}

// One PnP problem on the GPU: a single SE(3) pose state and one PnP factor per
// correspondence, all pointing to that pose. Built exactly as for any minimizer.
struct PnPProblem {
  dvector<Vector<3>> points;
  dvector<Vector<2>> observations;
  dvector<SE3Transform> pose;
  cunls::cuBLASHandle cublas;
  cunls::SE3StateBatch pose_state;
  cunls::PnPFactorBatch pnp;
  cunls::Problem problem;

  explicit PnPProblem(const Dataset &data)
      : points(data.points_world),
        observations(data.observations),
        pose(std::vector<SE3Transform>{data.initial_pose}),
        pose_state(cublas, reinterpret_cast<const float *>(pose.data()), 1),
        pnp(observations.data(), points.data(), data.points_world.size(), kZThreshold) {
    problem.AddStateBatch(&pose_state);
    problem.AddFactorBatch(
        &pnp, std::vector<float *>(data.points_world.size(), pose_state.StateBlockDevicePtr(0)));
    if (!problem.CheckConsistency()) {
      throw std::runtime_error("PnP problem consistency check failed");
    }
  }

  SE3Transform Pose() const {
    SE3Transform out;
    pose.CopyToHost(&out, 1);
    return out;
  }
};

// Plain Levenberg-Marquardt: treats every correspondence as an inlier.
SE3Transform SolveWithLM(const Dataset &data) {
  PnPProblem p(data);
  cunls::LevenbergMarquardtMinimizerOptions options;
  options.base_options.max_num_iterations = 60;
  cunls::LevenbergMarquardtMinimizer lm(options);
  cunls::CudaStream stream;
  lm.Minimize(stream.GetStream(), p.problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return p.Pose();
}

// RANSAC + Levenberg-Marquardt. Returns the pose and fills the inlier mask.
SE3Transform SolveWithRansacLM(const Dataset &data, std::vector<uint8_t> &inlier_mask,
                               cunls::RansacSummary &summary) {
  PnPProblem p(data);

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
  cunls::CudaStream stream;
  summary = minimizer.Minimize(stream.GetStream(), p.problem);  // writes the pose back

  // Inlier mask of residual batch 0: one byte per factor, 1 = inlier.
  inlier_mask.resize(data.points_world.size());
  THROW_ON_CUDA_ERROR(cudaMemcpy(inlier_mask.data(), minimizer.InlierMask(0), inlier_mask.size(),
                                 cudaMemcpyDeviceToHost));
  return p.Pose();
}

float RotationErrorDeg(const SE3Transform &a, const SE3Transform &b) {
  double trace = 0;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) trace += static_cast<double>(a[i * 4 + k]) * b[i * 4 + k];
  }
  const double c = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
  return static_cast<float>(std::acos(c) * 180.0 / M_PI);
}

float TranslationError(const SE3Transform &a, const SE3Transform &b) {
  return std::hypot(std::hypot(a[3] - b[3], a[7] - b[7]), a[11] - b[11]);
}

void PrintPose(const char *label, const SE3Transform &pose, const SE3Transform &gt) {
  std::cout << "  " << label << ": rotation error " << RotationErrorDeg(pose, gt)
            << " deg, translation error " << TranslationError(pose, gt) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    size_t num_points = 2000;
    float outlier_ratio = 0.5f;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--num-points") == 0 && i + 1 < argc) {
        num_points = std::strtoul(argv[++i], nullptr, 10);
      } else if (std::strcmp(argv[i], "--outlier-ratio") == 0 && i + 1 < argc) {
        outlier_ratio = std::strtof(argv[++i], nullptr);
      } else {
        std::cout << "Usage: " << argv[0] << " [--num-points N] [--outlier-ratio R]\n";
        return std::strcmp(argv[i], "--help") == 0 ? 0 : 1;
      }
    }
    if (num_points < 10 || !(outlier_ratio >= 0.f && outlier_ratio < 1.f)) {
      std::cerr << "Need --num-points >= 10 and --outlier-ratio in [0, 1)\n";
      return 1;
    }

    std::mt19937 rng(2024);
    const Dataset data = GenerateDataset(num_points, outlier_ratio, rng);
    std::cout << "RANSAC PnP example: " << num_points << " correspondences, " << outlier_ratio * 100
              << "% outliers\n";
    PrintPose("Initial guess      ", data.initial_pose, data.gt_pose);

    const SE3Transform lm_pose = SolveWithLM(data);
    PrintPose("LevenbergMarquardt ", lm_pose, data.gt_pose);

    std::vector<uint8_t> mask;
    cunls::RansacSummary summary;
    const SE3Transform ransac_pose = SolveWithRansacLM(data, mask, summary);
    PrintPose("RansacLM           ", ransac_pose, data.gt_pose);

    // Classification quality against the generated ground truth.
    size_t true_pos = 0, false_pos = 0, true_inliers = 0;
    for (size_t i = 0; i < mask.size(); ++i) {
      true_inliers += !data.is_outlier[i];
      true_pos += mask[i] && !data.is_outlier[i];
      false_pos += mask[i] && data.is_outlier[i];
    }
    std::cout << "  RANSAC: " << summary.num_rounds << " round(s), " << summary.num_hypotheses
              << " hypotheses, " << summary.num_inliers << " inliers ("
              << summary.inlier_ratio * 100 << "%)\n"
              << "  Inlier mask: " << true_pos << " of " << true_inliers << " true inliers kept, "
              << false_pos << " outliers accepted\n";

    const bool ok = RotationErrorDeg(ransac_pose, data.gt_pose) < 0.5f &&
                    TranslationError(ransac_pose, data.gt_pose) < 0.05f && false_pos == 0;
    if (!ok) {
      std::cerr << "RANSAC quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
