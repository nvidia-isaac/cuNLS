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
 * @file numeric_diff_minimizer_test.cpp
 * @brief End-to-end GaussNewtonMinimizer / LevenbergMarquardtMinimizer
 * convergence tests in JacobianMode::kNumeric on a small synthetic SE3 pose
 * graph, plus a mixed-mode (one group numeric, one analytic) test. Adapted
 * from the fixture pattern in synthetic_pgo_test.cpp, but with a much
 * smaller pose count since this exercises correctness, not performance.
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

constexpr size_t kNumPoses = 24;
constexpr uint32_t kFixedSeed = 7;

std::vector<SE3Transform> GenerateRandomPoses(
    size_t num_poses, std::mt19937 &rng, std::uniform_real_distribution<float> &rotation_dist,
    std::uniform_real_distribution<float> &translation_dist) {
  hvector<Vector<6>> twists(num_poses);
  for (size_t i = 0; i < num_poses; i++) {
    Vector<6> &twist = twists[i];
    twist[0] = rotation_dist(rng);
    twist[1] = rotation_dist(rng);
    twist[2] = rotation_dist(rng);
    twist[3] = translation_dist(rng);
    twist[4] = translation_dist(rng);
    twist[5] = translation_dist(rng);
  }

  CudaStream stream;
  dvector<Vector<6>> twists_device(twists);
  dvector<SE3Transform> poses_device(num_poses);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twists_device.data()), 6, 4, 16,
                num_poses, reinterpret_cast<float *>(poses_device.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<SE3Transform> poses(num_poses);
  hvector<SE3Transform> poses_host(num_poses);
  poses_device.CopyToHost(poses_host.data(), num_poses);
  for (size_t i = 0; i < num_poses; i++) poses[i] = poses_host[i];
  return poses;
}

/**
 * @brief Owns a small synthetic PGO problem: two SE3 pose sets connected by
 * one-to-one between constraints (pose_set1[i] <-> pose_set2[i]), matching
 * the pattern in `SyntheticPGOTest.OptimizeConsecutiveBetweenConstraints`.
 */
struct SyntheticPGOProblem {
  explicit SyntheticPGOProblem(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> rotation_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);
    poses_set1 = GenerateRandomPoses(kNumPoses, rng, rotation_dist, translation_dist);
    poses_set2 = GenerateRandomPoses(kNumPoses, rng, rotation_dist, translation_dist);
    pose_deltas = GenerateRandomPoses(kNumPoses, rng, rotation_dist, translation_dist);

    poses_set1_device = dvector<SE3Transform>(poses_set1);
    poses_set2_device = dvector<SE3Transform>(poses_set2);
    pose_deltas_device = dvector<SE3Transform>(pose_deltas);

    state_batch_set1 = std::make_unique<SE3StateBatch>(
        cublas_handle, reinterpret_cast<const float *>(poses_set1_device.data()), kNumPoses);
    state_batch_set2 = std::make_unique<SE3StateBatch>(
        cublas_handle, reinterpret_cast<const float *>(poses_set2_device.data()), kNumPoses);
  }

  /**
   * @brief Registers the full constraint set as a single factor group
   * (optionally with a JacobianMode override) on `problem`.
   */
  void AddSingleGroup(Problem &problem, std::optional<JacobianMode> override_mode) {
    between_factor_batches.push_back(
        std::make_unique<SE3BetweenFactorBatch>(pose_deltas_device.data(), kNumPoses));
    std::vector<float *> state_pointers;
    for (size_t i = 0; i < kNumPoses; i++) {
      state_pointers.push_back(state_batch_set1->StateBlockDevicePtr(i));
      state_pointers.push_back(state_batch_set2->StateBlockDevicePtr(i));
    }
    problem.AddStateBatch(state_batch_set1.get());
    problem.AddStateBatch(state_batch_set2.get());
    problem.AddFactorBatch(between_factor_batches.back().get(), state_pointers, override_mode);
  }

  /**
   * @brief Registers the constraint set split into two disjoint factor
   * groups (first half / second half of the pose indices), each with its
   * own JacobianMode.
   */
  void AddSplitGroups(Problem &problem, JacobianMode first_half_mode,
                      JacobianMode second_half_mode) {
    const size_t half = kNumPoses / 2;

    dvector<SE3Transform> deltas_first(
        std::vector<SE3Transform>(pose_deltas.begin(), pose_deltas.begin() + half));
    dvector<SE3Transform> deltas_second(
        std::vector<SE3Transform>(pose_deltas.begin() + half, pose_deltas.end()));
    // Keep the underlying device storage alive for the lifetime of this
    // object (factor batches only store the raw pointer).
    extra_owned_deltas.push_back(std::move(deltas_first));
    extra_owned_deltas.push_back(std::move(deltas_second));

    between_factor_batches.push_back(
        std::make_unique<SE3BetweenFactorBatch>(extra_owned_deltas[0].data(), half));
    between_factor_batches.push_back(
        std::make_unique<SE3BetweenFactorBatch>(extra_owned_deltas[1].data(), kNumPoses - half));

    std::vector<float *> ptrs_first, ptrs_second;
    for (size_t i = 0; i < half; i++) {
      ptrs_first.push_back(state_batch_set1->StateBlockDevicePtr(i));
      ptrs_first.push_back(state_batch_set2->StateBlockDevicePtr(i));
    }
    for (size_t i = half; i < kNumPoses; i++) {
      ptrs_second.push_back(state_batch_set1->StateBlockDevicePtr(i));
      ptrs_second.push_back(state_batch_set2->StateBlockDevicePtr(i));
    }

    problem.AddStateBatch(state_batch_set1.get());
    problem.AddStateBatch(state_batch_set2.get());
    problem.AddFactorBatch(between_factor_batches[0].get(), ptrs_first, first_half_mode);
    problem.AddFactorBatch(between_factor_batches[1].get(), ptrs_second, second_half_mode);
  }

  cuBLASHandle cublas_handle;
  std::vector<SE3Transform> poses_set1, poses_set2, pose_deltas;
  dvector<SE3Transform> poses_set1_device, poses_set2_device, pose_deltas_device;
  std::vector<dvector<SE3Transform>> extra_owned_deltas;
  std::unique_ptr<SE3StateBatch> state_batch_set1, state_batch_set2;
  std::vector<std::unique_ptr<SE3BetweenFactorBatch>> between_factor_batches;
};

MinimizerOptions MakeBaseOptions() {
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-6f;
  options.cost_tolerance = 1e-6f;
  options.disable_safety_checks = false;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
      test_utils::PCGBlockSizeFromEnv(6);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
      test_utils::PCGMaxIterFromEnv(400);
  options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance =
      test_utils::PCGTolFromEnv(1e-4f);
  return options;
}

constexpr float kConvergedCostThreshold = 5e-2f;

TEST(NumericDiffMinimizerTest, GaussNewtonAnalyticBaseline) {
  SyntheticPGOProblem data(kFixedSeed);
  Problem problem;
  data.AddSingleGroup(problem, std::nullopt);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions options = MakeBaseOptions();
  options.jacobian_mode = JacobianMode::kAnalytic;
  GaussNewtonMinimizer minimizer(options);

  CudaStream stream;
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_LT(summary.final_cost, kConvergedCostThreshold);
}

TEST(NumericDiffMinimizerTest, GaussNewtonNumericMatchesAnalytic) {
  SyntheticPGOProblem data(kFixedSeed);
  Problem problem;
  data.AddSingleGroup(problem, std::nullopt);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions options = MakeBaseOptions();
  options.jacobian_mode = JacobianMode::kNumeric;
  GaussNewtonMinimizer minimizer(options);

  CudaStream stream;
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_GT(summary.num_iterations, 0u);
  ASSERT_LT(summary.final_cost, kConvergedCostThreshold);
}

TEST(NumericDiffMinimizerTest, LevenbergMarquardtNumericMatchesAnalytic) {
  SyntheticPGOProblem data(kFixedSeed);
  Problem problem;
  data.AddSingleGroup(problem, std::nullopt);
  ASSERT_TRUE(problem.CheckConsistency());

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = MakeBaseOptions();
  lm_options.base_options.jacobian_mode = JacobianMode::kNumeric;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  CudaStream stream;
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_GT(summary.num_iterations, 0u);
  ASSERT_LT(summary.final_cost, kConvergedCostThreshold);
}

/**
 * @brief One factor group forced to kNumeric via the per-group override,
 * the other left at the minimizer's global default (kAnalytic), inside the
 * same Problem. Both halves should still converge together.
 */
TEST(NumericDiffMinimizerTest, MixedModeSingleProblem) {
  SyntheticPGOProblem data(kFixedSeed);
  Problem problem;
  data.AddSplitGroups(problem, JacobianMode::kAnalytic, JacobianMode::kNumeric);
  ASSERT_TRUE(problem.CheckConsistency());

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = MakeBaseOptions();
  lm_options.base_options.jacobian_mode = JacobianMode::kAnalytic;  // global default
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  CudaStream stream;
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  ASSERT_GT(summary.num_iterations, 0u);
  ASSERT_LT(summary.final_cost, kConvergedCostThreshold);
}

}  // namespace
}  // namespace cunls
