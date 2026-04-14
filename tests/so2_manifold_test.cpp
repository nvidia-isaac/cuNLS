/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file so2_manifold_test.cpp
 * @brief Unified SO(2) manifold tests: state batch dimensions, prior and
 * between factor LM convergence.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/so2_between_factor_batch.h"
#include "cunls/factor/so2_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so2_state_batch.h"

namespace cunls {

namespace {

Matrix<2> MakeSO2(float theta) {
  float c = cosf(theta), s = sinf(theta);
  return {c, -s, s, c};
}

Matrix<2> MakeSO2Identity() { return {1.0f, 0.0f, 0.0f, 1.0f}; }

} // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(SO2ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<Matrix<2>> rots(kN, MakeSO2Identity());
  dvector<Matrix<2>> rots_dev(rots);

  cuBLASHandle cublas;
  SO2StateBatch states(cublas, reinterpret_cast<const float *>(rots_dev.data()),
                       kN);

  EXPECT_EQ(states.TangentSize(), 1u);
  EXPECT_EQ(states.AmbientSize(), 4u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(SO2ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(-static_cast<float>(M_PI),
                                                   static_cast<float>(M_PI));
  std::uniform_real_distribution<float> pert_dist(-0.3f, 0.3f);

  hvector<Matrix<2>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    float a = angle_dist(rng);
    targets[i] = MakeSO2(a);
    initials[i] = MakeSO2(a + pert_dist(rng));
  }

  dvector<Matrix<2>> targets_dev(targets), initials_dev(initials);

  cuBLASHandle cublas;
  SO2StateBatch state_batch(
      cublas, reinterpret_cast<const float *>(initials_dev.data()), kN);
  SO2PriorFactorBatch factor_batch(targets_dev.data(), kN);

  std::vector<float *> ptrs;
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
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-4f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Matrix<2>> optimized(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(optimized.data(), state_batch.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<2>), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], 1e-3f)
          << "rotation " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(SO2ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(60), rng2(61);
  std::uniform_real_distribution<float> angle_dist(-static_cast<float>(M_PI),
                                                   static_cast<float>(M_PI));

  hvector<Matrix<2>> rots_left(kN), rots_right(kN);
  for (size_t i = 0; i < kN; ++i) {
    rots_left[i] = MakeSO2(angle_dist(rng1));
    rots_right[i] = MakeSO2(angle_dist(rng2));
  }

  dvector<Matrix<2>> left_dev(rots_left), right_dev(rots_right);
  hvector<Matrix<2>> deltas(kN, MakeSO2Identity());
  dvector<Matrix<2>> deltas_dev(deltas);

  cuBLASHandle cublas;
  SO2StateBatch state_left(
      cublas, reinterpret_cast<const float *>(left_dev.data()), kN);
  SO2StateBatch state_right(
      cublas, reinterpret_cast<const float *>(right_dev.data()), kN);
  SO2BetweenFactorBatch factor_batch(deltas_dev.data(), kN);

  std::vector<float *> ptrs;
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
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Matrix<2>> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_left.data(), state_left.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<2>), cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_right.data(), state_right.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<2>), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.1f)
          << "rotation " << i << ", element " << j;
    }
  }
}

} // namespace cunls
