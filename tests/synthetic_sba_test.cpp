/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * @file synthetic_sba_test.cpp
 * @brief Parameterized synthetic SBA benchmark.
 *
 * Generates a synthetic bundle-adjustment problem (random poses, random
 * landmarks, reprojection observations with Gaussian noise) at a grid of
 * problem sizes (poses x points), runs LM, and lets nsys measure
 * per-stage costs.  Solver / Schur / PCG block size are picked from
 * environment variables — see tests/utils.h.
 *
 * The first pose and the first landmark are held fixed to remove the
 * SBA gauge degeneracy.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

struct SyntheticSbaParams {
  int n_poses;
  int n_points;
  int obs_per_landmark; // visibility per landmark; total obs = n_points * obs_per_landmark.
  const char *label;
};

inline std::ostream &operator<<(std::ostream &os,
                                const SyntheticSbaParams &p) {
  return os << p.label;
}

class SyntheticSbaTest : public ::testing::TestWithParam<SyntheticSbaParams> {
protected:
  /**
   * @brief Generates a random SE3 pose by sampling a small twist and
   *        composing with a forward translation, so all poses look at
   *        roughly the same scene region.
   */
  SE3Transform RandomPose(std::mt19937 &rng) {
    std::uniform_real_distribution<float> r(-0.3f, 0.3f);
    std::uniform_real_distribution<float> t_xy(-2.f, 2.f);
    std::uniform_real_distribution<float> t_z(-1.f, 1.f);
    Vector<6> twist{r(rng), r(rng), r(rng), t_xy(rng), t_xy(rng), t_z(rng)};
    dvector<Vector<6>> d_twist({twist});
    dvector<SE3Transform> d_pose(1);
    CudaStream stream;
    constexpr size_t pitch = 4;
    constexpr size_t stride = 16;
    ComputeExpSE3(stream.GetStream(),
                  reinterpret_cast<const float *>(d_twist.data()), 6, pitch,
                  stride, 1, reinterpret_cast<float *>(d_pose.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    SE3Transform out;
    d_pose.CopyToHost(&out, 1);
    return out;
  }

  /** Projects a world point through a world-from-camera pose to get a
   *  normalized image observation.  Returns false if the point is behind
   *  the camera or too close to it. */
  static bool Project(const SE3Transform &pose_world_from_cam_or_cam_from_world,
                      const Vector<3> &p_world, Vector<2> &out, bool cam_from_world) {
    const SE3Transform &T = pose_world_from_cam_or_cam_from_world;
    Vector<3> p_cam{};
    if (cam_from_world) {
      // T transforms world->cam.
      p_cam[0] = T[0] * p_world[0] + T[1] * p_world[1] + T[2] * p_world[2] + T[3];
      p_cam[1] = T[4] * p_world[0] + T[5] * p_world[1] + T[6] * p_world[2] + T[7];
      p_cam[2] = T[8] * p_world[0] + T[9] * p_world[1] + T[10] * p_world[2] + T[11];
    } else {
      // T transforms cam->world; compute inverse.
      // Inverse of [R t; 0 1] is [R^T -R^T t; 0 1].
      float dx = p_world[0] - T[3];
      float dy = p_world[1] - T[7];
      float dz = p_world[2] - T[11];
      p_cam[0] = T[0] * dx + T[4] * dy + T[8] * dz;
      p_cam[1] = T[1] * dx + T[5] * dy + T[9] * dz;
      p_cam[2] = T[2] * dx + T[6] * dy + T[10] * dz;
    }
    if (!(p_cam[2] > 0.05f)) {
      return false;
    }
    out[0] = p_cam[0] / p_cam[2];
    out[1] = p_cam[1] / p_cam[2];
    return true;
  }

  /** Returns (host) ground-truth poses, perturbed poses, ground-truth
   *  landmarks, perturbed landmarks, observations, and per-obs ids. */
  void GenerateProblem(int n_poses, int n_points, int obs_per_landmark,
                       std::vector<SE3Transform> &gt_poses,
                       std::vector<SE3Transform> &init_poses,
                       std::vector<Vector<3>> &gt_points,
                       std::vector<Vector<3>> &init_points,
                       std::vector<Vector<2>> &observations,
                       std::vector<int> &pose_ids,
                       std::vector<int> &point_ids) {
    std::mt19937 rng(42);

    // Ground-truth poses: random small perturbations.  Treat the pose
    // state as world-from-cam (cuNLS's SE3 state).
    gt_poses.resize(n_poses);
    for (int i = 0; i < n_poses; ++i) {
      gt_poses[i] = RandomPose(rng);
    }

    // Ground-truth landmarks: uniform cloud in front of the cameras.
    gt_points.resize(n_points);
    std::uniform_real_distribution<float> xy_dist(-10.f, 10.f);
    std::uniform_real_distribution<float> z_dist(3.f, 15.f);
    for (int i = 0; i < n_points; ++i) {
      gt_points[i] = {xy_dist(rng), xy_dist(rng), z_dist(rng)};
    }

    // Observations: each landmark gets up to `obs_per_landmark` valid
    // observations from random poses.  Landmarks that fail to obtain at
    // least one valid observation (point behind every randomly sampled
    // camera) are dropped from the problem to keep
    // Problem::CheckConsistency happy — every registered state must be
    // constrained by some factor.
    std::uniform_int_distribution<int> pose_pick(0, n_poses - 1);
    std::normal_distribution<float> noise(0.f, 1e-3f);
    observations.clear();
    pose_ids.clear();
    point_ids.clear();
    observations.reserve(static_cast<size_t>(n_points) *
                         static_cast<size_t>(obs_per_landmark));
    pose_ids.reserve(observations.capacity());
    point_ids.reserve(observations.capacity());
    std::vector<int> keep_landmark;
    keep_landmark.reserve(n_points);
    int next_remap = 0;
    std::vector<int> remap(n_points, -1);
    for (int j = 0; j < n_points; ++j) {
      int got = 0;
      int attempts = 0;
      while (got < obs_per_landmark && attempts < obs_per_landmark * 8) {
        ++attempts;
        int pose_id = pose_pick(rng);
        Vector<2> obs{};
        if (!Project(gt_poses[pose_id], gt_points[j], obs,
                     /*cam_from_world=*/false)) {
          continue;
        }
        if (remap[j] < 0) {
          remap[j] = next_remap++;
          keep_landmark.push_back(j);
        }
        obs[0] += noise(rng);
        obs[1] += noise(rng);
        observations.push_back(obs);
        pose_ids.push_back(pose_id);
        point_ids.push_back(remap[j]); // remap to the kept-landmark index
        ++got;
      }
    }
    // Rebuild gt_points to contain only kept landmarks, in remapped order.
    std::vector<Vector<3>> gt_pts_kept(keep_landmark.size());
    for (size_t i = 0; i < keep_landmark.size(); ++i) {
      gt_pts_kept[i] = gt_points[keep_landmark[i]];
    }
    gt_points.swap(gt_pts_kept);

    // Perturbed initial state: GT + small noise, except first pose / first
    // landmark which stay at GT (and will be held constant to remove gauge).
    // gt_points has been resized to only the kept landmarks; init_points
    // matches.
    std::normal_distribution<float> pose_jitter_twist(0.f, 0.05f);
    std::normal_distribution<float> point_jitter(0.f, 0.05f);
    init_poses = gt_poses;
    init_points = gt_points;
    {
      std::vector<Vector<6>> jitters(n_poses);
      for (int i = 1; i < n_poses; ++i) {
        Vector<6> &j = jitters[i];
        for (int k = 0; k < 6; ++k) {
          j[k] = pose_jitter_twist(rng);
        }
      }
      // Compose init_pose[i] = gt_pose[i] * exp(jitter)  on device.
      dvector<Vector<6>> d_jit(jitters);
      dvector<SE3Transform> d_delta(n_poses);
      CudaStream stream;
      ComputeExpSE3(stream.GetStream(),
                    reinterpret_cast<const float *>(d_jit.data()), 6, 4, 16,
                    n_poses, reinterpret_cast<float *>(d_delta.data()));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
      std::vector<SE3Transform> deltas(n_poses);
      d_delta.CopyToHost(deltas.data(), n_poses);
      // 4x4 matrix multiplication on host.
      auto mul = [](const SE3Transform &a,
                    const SE3Transform &b) -> SE3Transform {
        SE3Transform c{};
        for (int r = 0; r < 4; ++r) {
          for (int cc = 0; cc < 4; ++cc) {
            float s = 0.f;
            for (int k = 0; k < 4; ++k) {
              s += a[r * 4 + k] * b[k * 4 + cc];
            }
            c[r * 4 + cc] = s;
          }
        }
        return c;
      };
      for (int i = 1; i < n_poses; ++i) {
        init_poses[i] = mul(gt_poses[i], deltas[i]);
      }
    }
    for (size_t j = 1; j < init_points.size(); ++j) {
      Vector<3> &p = init_points[j];
      p[0] += point_jitter(rng);
      p[1] += point_jitter(rng);
      p[2] += point_jitter(rng);
    }
  }

  cuBLASHandle cublas_handle_;
  profiler::Domain profiler_domain_ = profiler::Domain("SyntheticSbaTest");
};

/**
 * @brief Builds and runs LM on a synthetic SBA problem of the configured
 *        size.  Holds pose 0 and landmark 0 constant.  Time measurement
 *        is via NVTX ranges installed by the minimizer + the test
 *        fixture.
 */
TEST_P(SyntheticSbaTest, Optimize) {
  auto params = GetParam();
  SCOPED_TRACE(std::string("SyntheticSbaTest: ") + params.label);

  std::vector<SE3Transform> gt_poses, init_poses;
  std::vector<Vector<3>> gt_points, init_points;
  std::vector<Vector<2>> observations;
  std::vector<int> pose_ids, point_ids;
  GenerateProblem(params.n_poses, params.n_points, params.obs_per_landmark,
                  gt_poses, init_poses, gt_points, init_points, observations,
                  pose_ids, point_ids);

  const size_t n_obs = observations.size();
  ASSERT_GT(n_obs, 0u);

  // Identity 2x2 sqrt information per observation.
  std::vector<Matrix<2>> sqrt_info(n_obs);
  for (size_t i = 0; i < n_obs; ++i) {
    sqrt_info[i] = {1.f, 0.f, 0.f, 1.f};
  }

  // Camera-from-rig: identity (single camera, no rig).
  std::vector<SE3Transform> cam_from_rig(n_obs);
  for (size_t i = 0; i < n_obs; ++i) {
    cam_from_rig[i] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  }

  dvector<SE3Transform> poses_d(init_poses);
  dvector<Vector<3>> points_d(init_points);
  dvector<Vector<2>> obs_d(observations);
  dvector<Matrix<2>> info_d(sqrt_info);
  dvector<SE3Transform> cam_from_rig_d(cam_from_rig);

  // Hold pose 0 + landmark 0 constant to fix the gauge.
  std::vector<int> const_pose_ids = {0};
  std::vector<int> const_point_ids = {0};
  dvector<int> const_pose_ids_d(const_pose_ids);
  dvector<int> const_point_ids_d(const_point_ids);

  auto poses_ptr = reinterpret_cast<const float *>(poses_d.data());
  auto points_ptr = reinterpret_cast<const float *>(points_d.data());

  SE3StateBatch pose_batch(cublas_handle_, poses_ptr,
                           static_cast<size_t>(params.n_poses),
                           const_pose_ids_d.data(), const_pose_ids.size());
  VectorStateBatch<3> point_batch(points_ptr, init_points.size(),
                                  const_point_ids_d.data(),
                                  const_point_ids.size());

  InformationFactorBatch<ReprojectionFactorBatch> info_factor(
      cublas_handle_, info_d.data(), n_obs, obs_d.data(),
      cam_from_rig_d.data(), n_obs, 1e-3f);

  std::vector<float *> state_pointers;
  state_pointers.reserve(n_obs * 2);
  for (size_t i = 0; i < n_obs; ++i) {
    state_pointers.push_back(
        pose_batch.StateBlockDevicePtr(static_cast<size_t>(pose_ids[i])));
    state_pointers.push_back(
        point_batch.StateBlockDevicePtr(static_cast<size_t>(point_ids[i])));
  }

  HuberLossFunctionBatch huber(1.0f);
  Problem problem;
  problem.AddStateBatch(&pose_batch);
  problem.AddStateBatch(&point_batch);
  problem.AddFactorBatch(&info_factor, &huber, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 10;
  options.state_tolerance = 1e-10f;
  options.cost_tolerance = 1e-6f;
  options.disable_safety_checks = true;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size = test_utils::PCGBlockSizeFromEnv(6);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations = test_utils::PCGMaxIterFromEnv(200);
  options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance = test_utils::PCGTolFromEnv(1e-3f);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary;
  {
    auto range = profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  EXPECT_TRUE(std::isfinite(summary.initial_cost));
  EXPECT_TRUE(std::isfinite(summary.final_cost));
  EXPECT_LE(summary.final_cost, summary.initial_cost + 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, SyntheticSbaTest,
    ::testing::Values(
        SyntheticSbaParams{10, 1000, 5, "P10_L1k_obs5"},
        SyntheticSbaParams{50, 10000, 5, "P50_L10k_obs5"},
        SyntheticSbaParams{100, 50000, 5, "P100_L50k_obs5"},
        SyntheticSbaParams{250, 200000, 5, "P250_L200k_obs5"},
        // 1M landmarks: obs_per_landmark=2 keeps the Jacobian row count
        // under cuNLS's per-batch grid-dim limit (~4M rows).  Total obs ≈ 2M.
        SyntheticSbaParams{500, 1000000, 2, "P500_L1M_obs2"}));

} // namespace cunls
