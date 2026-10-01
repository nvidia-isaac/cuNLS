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

#pragma once

// Synthetic datasets for the examples. Every generator is deterministic for a
// given seed and returns plain host data: ground truth, an initial guess, and
// the measurements the factors consume.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/so_se_lie_math.h"
#include "utils/camera_utils.h"
#include "utils/se3_utils.h"

namespace examples {

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

// ---------------------------------------------------------------------------
// PnP: one camera, known 3D points, 2D observations (optionally with outliers)
// ---------------------------------------------------------------------------

struct PnPScene {
  SE3Transform gt_pose;       // world -> camera
  SE3Transform initial_pose;  // perturbed gt_pose
  std::vector<Vector<3>> points_world;
  std::vector<Vector<2>> observations;  // normalized image coordinates
  std::vector<uint8_t> is_outlier;      // 1 = gross outlier (wrong match)
};

struct PnPSceneOptions {
  size_t num_points = 2000;
  float outlier_ratio = 0.0f;          // fraction of observations replaced by outliers
  float pixel_noise = 3e-3f;           // inlier noise sigma per axis (normalized units)
  float min_outlier_distance = 0.04f;  // outliers lie at least this far from the truth
  float init_rotation = 0.05f;         // initial-guess perturbation ranges
  float init_translation = 0.2f;
  uint32_t seed = 13579;
};

// Points are drawn in front of the camera (depth >= 1). Inliers are noisy
// projections; outliers are random image points.
inline PnPScene MakePnPScene(const PnPSceneOptions &o) {
  PnPScene s;
  std::mt19937 rng(o.seed);
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f), trans(-1.0f, 1.0f);
  Vector<6> twist;
  twist[0] = rot(rng);
  twist[1] = rot(rng);
  twist[2] = rot(rng);
  twist[3] = trans(rng);
  twist[4] = trans(rng);
  twist[5] = 8.0f + trans(rng);
  std::vector<SE3Transform> gt;
  TwistsToSE3({twist}, gt);
  s.gt_pose = gt[0];

  std::uniform_real_distribution<float> coord(-3.0f, 3.0f), image(-0.5f, 0.5f), unit(0.f, 1.f);
  std::normal_distribution<float> noise(0.0f, o.pixel_noise);
  for (size_t i = 0; i < o.num_points; ++i) {
    Vector<3> p;
    do {
      p[0] = coord(rng);
      p[1] = coord(rng);
      p[2] = coord(rng);
    } while (ComputeDepth(s.gt_pose, p) < 1.0f);
    const Vector<2> proj = ProjectNormalized(s.gt_pose, p);
    Vector<2> obs = proj;
    const bool outlier = o.outlier_ratio > 0.f && unit(rng) < o.outlier_ratio;
    if (outlier) {
      do {
        obs[0] = image(rng);
        obs[1] = image(rng);
      } while (std::hypot(obs[0] - proj[0], obs[1] - proj[1]) < o.min_outlier_distance);
    } else {
      obs[0] += noise(rng);
      obs[1] += noise(rng);
    }
    s.points_world.push_back(p);
    s.observations.push_back(obs);
    s.is_outlier.push_back(outlier ? 1 : 0);
  }

  std::vector<SE3Transform> perturbation;
  GenerateRandomSE3(1, rng, perturbation, o.init_rotation, o.init_translation);
  s.initial_pose = ComposeSE3(perturbation[0], s.gt_pose);
  return s;
}

// ---------------------------------------------------------------------------
// Bundle adjustment: several cameras observing every point
// ---------------------------------------------------------------------------

struct BundleAdjustmentScene {
  std::vector<SE3Transform> gt_poses, initial_poses;  // initial_poses[0] == gt_poses[0]
  std::vector<Vector<3>> gt_points, initial_points;
  std::vector<Vector<2>> observations;  // observation of point j by camera i at i * P + j
};

inline BundleAdjustmentScene MakeBundleAdjustmentScene(size_t num_poses, size_t num_points) {
  BundleAdjustmentScene s;
  // Cameras look roughly at the origin (tz ~ 8) so the points stay visible.
  {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> rot(-0.2f, 0.2f), trans(-1.0f, 1.0f);
    std::vector<Vector<6>> twists(num_poses);
    for (auto &t : twists) {
      t[0] = rot(rng);
      t[1] = rot(rng);
      t[2] = rot(rng);
      t[3] = trans(rng);
      t[4] = trans(rng);
      t[5] = 8.0f + trans(rng);
    }
    TwistsToSE3(twists, s.gt_poses);
  }

  std::mt19937 rng(5678);
  std::uniform_real_distribution<float> coord(-3.0f, 3.0f), noise(-0.35f, 0.35f);
  s.gt_points.resize(num_points);
  s.initial_points.resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    bool visible = false;
    while (!visible) {  // visible from every camera
      Vector<3> p;
      p[0] = coord(rng);
      p[1] = coord(rng);
      p[2] = coord(rng);
      visible = true;
      for (const auto &pose : s.gt_poses) {
        if (ComputeDepth(pose, p) < 1.0f) {
          visible = false;
          break;
        }
      }
      if (visible) s.gt_points[i] = p;
    }
    s.initial_points[i] = s.gt_points[i];
    for (int k = 0; k < 3; ++k) s.initial_points[i][k] += noise(rng);
  }

  for (size_t c = 0; c < num_poses; ++c) {
    for (size_t j = 0; j < num_points; ++j) {
      s.observations.push_back(ProjectNormalized(s.gt_poses[c], s.gt_points[j]));
    }
  }

  // Perturb every camera but the first (the gauge anchor).
  std::vector<SE3Transform> perturbations;
  GenerateRandomSE3(num_poses - 1, rng, perturbations, 0.02f, 0.1f);
  s.initial_poses = s.gt_poses;
  for (size_t i = 1; i < num_poses; ++i) {
    s.initial_poses[i] = ComposeSE3(perturbations[i - 1], s.gt_poses[i]);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Pose chain: T_0 -> T_1 -> ... with relative-transform measurements
// ---------------------------------------------------------------------------

struct PoseChainScene {
  std::vector<SE3Transform> gt_poses, initial_poses;  // initial_poses[0] == gt_poses[0]
  std::vector<SE3Transform> deltas;  // delta_i * T_i^{-1} * T_{i+1} = I at the truth
};

inline PoseChainScene MakePoseChainScene(size_t num_poses) {
  PoseChainScene s;
  std::mt19937 rng(9012);
  std::vector<SE3Transform> anchor;
  GenerateRandomSE3(1, rng, anchor);
  GenerateRandomSE3(num_poses - 1, rng, s.deltas);

  s.gt_poses.resize(num_poses);
  s.gt_poses[0] = anchor[0];
  for (size_t i = 0; i + 1 < num_poses; ++i) {
    s.gt_poses[i + 1] = ComposeSE3(s.gt_poses[i], InverseSE3(s.deltas[i]));
  }

  // Disturb every pose but the anchor, within the SE(3) log map's basin.
  std::vector<SE3Transform> disturbance;
  GenerateRandomSE3(num_poses - 1, rng, disturbance, 0.05f, 0.3f);
  s.initial_poses = s.gt_poses;
  for (size_t i = 1; i < num_poses; ++i) {
    s.initial_poses[i] = ComposeSE3(disturbance[i - 1], s.gt_poses[i]);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Constant-velocity trajectory: poses and body velocities
// ---------------------------------------------------------------------------

// J_l(twist) * v for the SE(3) left Jacobian, computed with the GPU math
// library. Under constant body velocity, v_{i+1} = J_l(dt * v_i) * v_i.
inline Vector<6> ApplyLeftJacobianSE3(const Vector<6> &twist, const Vector<6> &v) {
  dvector<Vector<6>> twist_dev({twist});
  dvector<cunls::Matrix<6>> jl_dev(1);
  cunls::CudaStream stream;
  cunls::ComputeJacobianLeftSE3(stream.GetStream(),
                                reinterpret_cast<const float *>(twist_dev.data()), 6, 6, 36, 1,
                                reinterpret_cast<float *>(jl_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  cunls::Matrix<6> jl;
  jl_dev.CopyToHost(&jl, 1);
  Vector<6> out{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) out[i] += jl[i * 6 + j] * v[j];
  }
  return out;
}

struct ConstantVelocityScene {
  std::vector<SE3Transform> gt_poses, initial_poses;  // index 0 is the anchor
  std::vector<Vector<6>> gt_velocities, initial_velocities;
};

// Integrates an exact constant-velocity trajectory, then disturbs every
// pose / velocity except the anchor.
inline ConstantVelocityScene MakeConstantVelocityScene(size_t num_poses, float dt) {
  ConstantVelocityScene s;
  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> rot(-0.15f, 0.15f), trans(-0.5f, 0.5f);
  std::vector<SE3Transform> anchor;
  GenerateRandomSE3(1, rng, anchor);
  Vector<6> anchor_vel{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};

  s.gt_poses.resize(num_poses);
  s.gt_velocities.resize(num_poses);
  s.gt_poses[0] = anchor[0];
  s.gt_velocities[0] = anchor_vel;
  for (size_t i = 0; i + 1 < num_poses; ++i) {
    Vector<6> step;
    for (int d = 0; d < 6; ++d) step[d] = dt * s.gt_velocities[i][d];
    std::vector<SE3Transform> step_pose;
    TwistsToSE3({step}, step_pose);
    s.gt_poses[i + 1] = ComposeSE3(s.gt_poses[i], step_pose[0]);
    s.gt_velocities[i + 1] = ApplyLeftJacobianSE3(step, s.gt_velocities[i]);
  }

  std::vector<SE3Transform> pose_noise;
  GenerateRandomSE3(num_poses - 1, rng, pose_noise, 0.05f, 0.2f);
  std::uniform_real_distribution<float> vel_noise(-0.1f, 0.1f);
  s.initial_poses = s.gt_poses;
  s.initial_velocities = s.gt_velocities;
  for (size_t i = 1; i < num_poses; ++i) {
    s.initial_poses[i] = ComposeSE3(pose_noise[i - 1], s.gt_poses[i]);
    for (int d = 0; d < 6; ++d) s.initial_velocities[i][d] += vel_noise(rng);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Scalar chain: x_0 < x_1 < ... with exact consecutive differences
// ---------------------------------------------------------------------------

struct ScalarChainScene {
  std::vector<Vector<1>> gt_states, initial_states;
  std::vector<float> differences;  // x_{i+1} - x_i at the truth
};

inline ScalarChainScene MakeScalarChainScene(size_t num_states) {
  ScalarChainScene s;
  std::mt19937 rng(121314);
  std::uniform_real_distribution<float> step(0.2f, 0.6f), noise(-0.35f, 0.35f);
  s.gt_states.resize(num_states);
  s.gt_states[0][0] = 0.5f;
  for (size_t i = 1; i < num_states; ++i) {
    s.gt_states[i][0] = s.gt_states[i - 1][0] + step(rng);
  }
  s.initial_states.resize(num_states);
  for (size_t i = 0; i < num_states; ++i) {
    s.initial_states[i][0] = s.gt_states[i][0] + noise(rng);
  }
  for (size_t i = 0; i + 1 < num_states; ++i) {
    s.differences.push_back(s.gt_states[i + 1][0] - s.gt_states[i][0]);
  }
  return s;
}

}  // namespace examples
