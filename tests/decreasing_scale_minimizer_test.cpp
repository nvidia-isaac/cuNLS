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

/**
 * @file decreasing_scale_minimizer_test.cpp
 * @brief Tests that a single LM minimizer instance can sequentially solve
 *        reprojection problems of progressively decreasing scale.
 *
 * A single LevenbergMarquardtMinimizer is created once and reused across
 * 5 optimizations where the number of 3D points decreases from 10000 to 1000
 * in 10 steps. Each optimization builds a fresh Problem with the
 * appropriate subset of points and observations.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {

constexpr float kZThreshold = 1e-3f;
constexpr float kMinPointDepth = 1.0f;

/**
 * @brief Test fixture for sequential minimization with decreasing problem
 * scale.
 *
 * Generates a synthetic bundle-adjustment scenario with enough points for the
 * largest problem (10000) and enough camera poses (10). Smaller problems simply
 * use a prefix of the full point / observation arrays.
 */
class DecreasingScaleMinimizerTest : public ::testing::Test {
 public:
  using Point3D = Vector<3>;
  using Observation2D = Vector<2>;

  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> translation_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> point_dist(-3.0f, 3.0f);

    GenerateRandomPoses(num_poses_, rng, rotation_dist, translation_dist,
                        ground_truth_poses_);
    GenerateRandomPointsVisibleFromAllCameras(max_num_points_, rng, point_dist,
                                              ground_truth_points_);
    ValidateAllPointsVisible(max_num_points_);
    GenerateObservations(max_num_points_);
  }

 protected:
  // ------------------------------------------------------------------
  // Data generation helpers (same logic as ReprojectionFactorBatchTest)
  // ------------------------------------------------------------------

  float ComputePointDepth(const SE3Transform& pose, const Point3D& point) {
    float depth = pose[2 * 4 + 3];
    for (int j = 0; j < 3; j++) {
      depth += pose[2 * 4 + j] * point[j];
    }
    return depth;
  }

  void ValidateAllPointsVisible(size_t num_points) {
    for (size_t pose_idx = 0; pose_idx < ground_truth_poses_.size();
         pose_idx++) {
      for (size_t point_idx = 0; point_idx < num_points; point_idx++) {
        float depth = ComputePointDepth(ground_truth_poses_[pose_idx],
                                        ground_truth_points_[point_idx]);
        if (depth < kMinPointDepth) {
          throw std::runtime_error("Point " + std::to_string(point_idx) +
                                   " has depth " + std::to_string(depth) +
                                   " < " + std::to_string(kMinPointDepth) +
                                   " in camera " + std::to_string(pose_idx));
        }
      }
    }
  }

  void GenerateRandomPoses(
      size_t num_poses, std::mt19937& rng,
      std::uniform_real_distribution<float>& rotation_dist,
      std::uniform_real_distribution<float>& translation_dist,
      std::vector<SE3Transform>& poses) {
    hvector<Vector<6>> twists(num_poses);
    for (size_t i = 0; i < num_poses; i++) {
      Vector<6>& twist = twists[i];
      twist[0] = rotation_dist(rng);
      twist[1] = rotation_dist(rng);
      twist[2] = rotation_dist(rng);
      twist[3] = translation_dist(rng);
      twist[4] = translation_dist(rng);
      twist[5] = 10.0f + translation_dist(rng);
    }

    CudaStream stream;
    constexpr size_t twist_stride = 6;
    constexpr size_t transform_pitch = 4;
    constexpr size_t transform_stride = 16;

    dvector<Vector<6>> twists_device(twists);
    dvector<SE3Transform> poses_device(num_poses);

    auto twists_ptr = reinterpret_cast<const float*>(twists_device.data());
    auto poses_ptr = reinterpret_cast<float*>(poses_device.data());

    ComputeExpSE3(stream.GetStream(), twists_ptr, twist_stride, transform_pitch,
                  transform_stride, num_poses, poses_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    poses.resize(num_poses);
    poses_device.CopyToHost(poses.data(), num_poses);
  }

  void GenerateRandomPointsVisibleFromAllCameras(
      size_t num_points, std::mt19937& rng,
      std::uniform_real_distribution<float>& point_dist,
      std::vector<Point3D>& points) {
    points.resize(num_points);
    for (size_t i = 0; i < num_points; i++) {
      points[i][0] = point_dist(rng);
      points[i][1] = point_dist(rng);
      points[i][2] = point_dist(rng);
    }
  }

  Observation2D ProjectPoint(const SE3Transform& pose, const Point3D& point) {
    float point_cam[3];
    point_cam[0] = pose[0 * 4 + 3];
    point_cam[1] = pose[1 * 4 + 3];
    point_cam[2] = pose[2 * 4 + 3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        point_cam[i] += pose[i * 4 + j] * point[j];
      }
    }
    Observation2D obs;
    float inv_z = 1.0f / point_cam[2];
    obs[0] = point_cam[0] * inv_z;
    obs[1] = point_cam[1] * inv_z;
    return obs;
  }

  void GenerateObservations(size_t num_points) {
    size_t num_observations = num_poses_ * num_points;
    observations_.resize(num_observations);
    for (size_t pose_idx = 0; pose_idx < num_poses_; pose_idx++) {
      for (size_t point_idx = 0; point_idx < num_points; point_idx++) {
        size_t obs_idx = pose_idx * num_points + point_idx;
        observations_[obs_idx] = ProjectPoint(ground_truth_poses_[pose_idx],
                                              ground_truth_points_[point_idx]);
      }
    }
  }

  void DisturbPoints(std::vector<Point3D>& points, float noise_magnitude) {
    std::mt19937 rng(fixed_seed_ + 1);
    std::uniform_real_distribution<float> noise_dist(-noise_magnitude,
                                                     noise_magnitude);
    for (auto& point : points) {
      Point3D disturbed;
      bool valid = false;
      for (int attempt = 0; attempt < 100 && !valid; attempt++) {
        disturbed[0] = point[0] + noise_dist(rng);
        disturbed[1] = point[1] + noise_dist(rng);
        disturbed[2] = point[2] + noise_dist(rng);
        valid = true;
        for (const auto& pose : ground_truth_poses_) {
          if (ComputePointDepth(pose, disturbed) < kMinPointDepth) {
            valid = false;
            break;
          }
        }
      }
      if (valid) {
        point = disturbed;
      }
    }
  }

  /**
   * @brief Extracts the observations for a prefix subset of points.
   *
   * The full observation array is laid out as
   *   observations_[pose_idx * max_num_points_ + point_idx]
   * This helper gathers the first @p num_points columns for every pose into
   * a contiguous array suitable for ReprojectionFactorBatch.
   */
  std::vector<Observation2D> ExtractObservations(size_t num_points) {
    std::vector<Observation2D> subset;
    subset.reserve(num_poses_ * num_points);
    for (size_t pose_idx = 0; pose_idx < num_poses_; pose_idx++) {
      for (size_t point_idx = 0; point_idx < num_points; point_idx++) {
        subset.push_back(observations_[pose_idx * max_num_points_ + point_idx]);
      }
    }
    return subset;
  }

  std::vector<float*> CreateStatePointers(
      SE3StateBatch& state_batch_poses, VectorStateBatch<3>& state_batch_points,
      size_t num_points) {
    std::vector<float*> state_pointers;
    state_pointers.reserve(2 * num_poses_ * num_points);
    for (size_t pose_idx = 0; pose_idx < num_poses_; pose_idx++) {
      for (size_t point_idx = 0; point_idx < num_points; point_idx++) {
        state_pointers.push_back(
            state_batch_poses.StateBlockDevicePtr(pose_idx));
        state_pointers.push_back(
            state_batch_points.StateBlockDevicePtr(point_idx));
      }
    }
    return state_pointers;
  }

  float ComputePointMSE(const std::vector<Point3D>& points_a,
                        const std::vector<Point3D>& points_b, size_t count) {
    float mse = 0.0f;
    for (size_t i = 0; i < count; i++) {
      float dx = points_a[i][0] - points_b[i][0];
      float dy = points_a[i][1] - points_b[i][1];
      float dz = points_a[i][2] - points_b[i][2];
      mse += dx * dx + dy * dy + dz * dz;
    }
    return mse / static_cast<float>(count);
  }

  // Configuration
  const size_t num_poses_ = 10;
  const size_t max_num_points_ = 10000;
  const uint32_t fixed_seed_ = 54321;

  // Ground truth data (generated for max_num_points_)
  std::vector<SE3Transform> ground_truth_poses_;
  std::vector<Point3D> ground_truth_points_;
  std::vector<Observation2D> observations_;

  // cuBLAS handle for factor and state batch constructors
  cuBLASHandle cublas_handle_;
};

/**
 * @brief Reuses one LM minimizer across 10 problems of decreasing point count.
 *
 * Point counts: 10000 -> .. -> 1000.
 * Poses are held constant; only points are optimized.
 * Each problem is constructed from scratch while the minimizer persists.
 */
TEST_F(DecreasingScaleMinimizerTest, SequentialDecreasingScale) {
  // Create the single minimizer instance that is reused for all problems.
  MinimizerOptions options;
  options.max_num_iterations = 100;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-8f;

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  CudaStream stream;

  constexpr size_t kNumOptimizations = 10;
  constexpr size_t kFirstNumPoints = 10000;
  constexpr size_t kLastNumPoints = 1000;
  constexpr size_t kPointStep =
      (kFirstNumPoints - kLastNumPoints) / (kNumOptimizations - 1);

  for (size_t opt_idx = 0; opt_idx < kNumOptimizations; opt_idx++) {
    size_t num_points = kFirstNumPoints - opt_idx * kPointStep;
    size_t num_observations = num_poses_ * num_points;

    SCOPED_TRACE("Optimization " + std::to_string(opt_idx) + " with " +
                 std::to_string(num_points) + " points");

    // Prepare a disturbed copy of the first num_points ground truth points.
    std::vector<Point3D> disturbed_points(
        ground_truth_points_.begin(),
        ground_truth_points_.begin() + num_points);
    DisturbPoints(disturbed_points, 0.5f);

    float initial_mse =
        ComputePointMSE(disturbed_points, ground_truth_points_, num_points);
    ASSERT_GT(initial_mse, 0.01f) << "Points should be significantly disturbed";

    // Extract the observation subset for this point count.
    std::vector<Observation2D> obs_subset = ExtractObservations(num_points);

    // Copy data to device.
    dvector<SE3Transform> poses_device(ground_truth_poses_);
    dvector<Point3D> points_device(disturbed_points);
    dvector<Observation2D> observations_device(obs_subset);

    // Mark all poses as constant.
    std::vector<int> const_pose_ids(num_poses_);
    for (size_t i = 0; i < num_poses_; i++) {
      const_pose_ids[i] = static_cast<int>(i);
    }
    dvector<int> const_pose_ids_device(const_pose_ids);

    const float* poses_ptr =
        reinterpret_cast<const float*>(poses_device.data());
    const float* points_ptr =
        reinterpret_cast<const float*>(points_device.data());

    SE3StateBatch state_batch_poses(cublas_handle_, poses_ptr, num_poses_,
                                    const_pose_ids_device.data(), num_poses_);
    VectorStateBatch<3> state_batch_points(points_ptr, num_points);

    ReprojectionFactorBatch factor_batch(cublas_handle_, observations_device.data(),
                                         num_observations, kZThreshold);

    std::vector<float*> state_pointers =
        CreateStatePointers(state_batch_poses, state_batch_points, num_points);

    Problem problem;
    problem.AddStateBatch(&state_batch_poses);
    problem.AddStateBatch(&state_batch_points);
    problem.AddFactorBatch(&factor_batch, state_pointers);
    ASSERT_TRUE(problem.CheckConsistency());

    // Minimize using the same minimizer instance.
    MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    EXPECT_LT(summary.final_cost, 1e-4f)
        << "Optimization should converge for " << num_points << " points";
    EXPECT_GT(summary.num_iterations, 0)
        << "Should take at least one iteration";

    // Verify recovered points.
    std::vector<Point3D> optimized_points(num_points);
    points_device.CopyToHost(optimized_points.data(), num_points);

    float final_mse =
        ComputePointMSE(optimized_points, ground_truth_points_, num_points);
    EXPECT_LT(final_mse, 1e-4f) << "Points should converge to ground truth for "
                                << num_points << " points";
    EXPECT_LT(final_mse, initial_mse * 0.01f)
        << "Final MSE should be much smaller than initial";
  }
}

}  // namespace cunls
