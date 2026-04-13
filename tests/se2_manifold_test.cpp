/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file se2_manifold_test.cpp
 * @brief Unified SE(2) manifold tests: state batch dimensions, prior and between
 *        factor LM convergence.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/factor/se2_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"

namespace cunls {

namespace {

Matrix<3> MakeSE2(float theta, float tx, float ty) {
  float c = cosf(theta), s = sinf(theta);
  return {c, -s, tx, s, c, ty, 0.0f, 0.0f, 1.0f};
}

Matrix<3> MakeSE2Identity() {
  return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}

}  // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(SE2ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<Matrix<3>> transforms(kN, MakeSE2Identity());
  dvector<Matrix<3>> transforms_dev(transforms);

  cuBLASHandle cublas;
  SE2StateBatch states(cublas,
                       reinterpret_cast<const float*>(transforms_dev.data()),
                       kN);

  EXPECT_EQ(states.TangentSize(), 3u);
  EXPECT_EQ(states.AmbientSize(), 9u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(SE2ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(
      -static_cast<float>(M_PI), static_cast<float>(M_PI));
  std::uniform_real_distribution<float> trans_dist(-2.0f, 2.0f);
  std::uniform_real_distribution<float> pert_angle(-0.3f, 0.3f);
  std::uniform_real_distribution<float> pert_trans(-0.3f, 0.3f);

  hvector<Matrix<3>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    float theta = angle_dist(rng);
    float tx = trans_dist(rng), ty = trans_dist(rng);
    targets[i] = MakeSE2(theta, tx, ty);
    initials[i] = MakeSE2(theta + pert_angle(rng),
                           tx + pert_trans(rng),
                           ty + pert_trans(rng));
  }

  dvector<Matrix<3>> targets_dev(targets), initials_dev(initials);

  cuBLASHandle cublas;
  SE2StateBatch state_batch(
      cublas, reinterpret_cast<const float*>(initials_dev.data()), kN);
  SE2PriorFactorBatch factor_batch(cublas, targets_dev.data(), kN);

  std::vector<float*> ptrs;
  ptrs.reserve(kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs.push_back(state_batch.StateBlockDevicePtr(i));
  }

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, ptrs);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions opts;
  opts.max_num_iterations = 50;
  opts.state_tolerance = 1e-8f;
  opts.cost_tolerance = 1e-8f;
  opts.disable_safety_checks = false;
  LevenbergMarquardtMinimizerOptions lm_opts;
  lm_opts.base_options = opts;
  lm_opts.initial_lambda = 1e-3f;

  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_opts);
  MinimizerSummary summary =
      minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-4f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Matrix<3>> optimized(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(),
                                 state_batch.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<3>),
                                 cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 9; ++j) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], 1e-2f)
          << "transform " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(SE2ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(50), rng2(51);
  std::uniform_real_distribution<float> angle_dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> trans_dist(-1.0f, 1.0f);

  hvector<Matrix<3>> poses_left(kN), poses_right(kN);
  for (size_t i = 0; i < kN; ++i) {
    poses_left[i] =
        MakeSE2(angle_dist(rng1), trans_dist(rng1), trans_dist(rng1));
    poses_right[i] =
        MakeSE2(angle_dist(rng2), trans_dist(rng2), trans_dist(rng2));
  }

  dvector<Matrix<3>> left_dev(poses_left), right_dev(poses_right);
  hvector<Matrix<3>> deltas(kN, MakeSE2Identity());
  dvector<Matrix<3>> deltas_dev(deltas);

  cuBLASHandle cublas;
  SE2StateBatch state_left(
      cublas, reinterpret_cast<const float*>(left_dev.data()), kN);
  SE2StateBatch state_right(
      cublas, reinterpret_cast<const float*>(right_dev.data()), kN);
  SE2BetweenFactorBatch factor_batch(cublas, deltas_dev.data(), kN);

  std::vector<float*> ptrs;
  ptrs.reserve(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs.push_back(state_left.StateBlockDevicePtr(i));
    ptrs.push_back(state_right.StateBlockDevicePtr(i));
  }

  Problem problem;
  problem.AddStateBatch(&state_left);
  problem.AddStateBatch(&state_right);
  problem.AddFactorBatch(&factor_batch, ptrs);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions opts;
  opts.max_num_iterations = 80;
  opts.state_tolerance = 1e-7f;
  opts.cost_tolerance = 1e-7f;
  opts.disable_safety_checks = false;
  LevenbergMarquardtMinimizerOptions lm_opts;
  lm_opts.base_options = opts;
  lm_opts.initial_lambda = 1e-3f;

  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_opts);
  MinimizerSummary summary =
      minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Matrix<3>> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_left.data(),
                                 state_left.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<3>),
                                 cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_right.data(),
                                 state_right.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<3>),
                                 cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 9; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.1f)
          << "transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
