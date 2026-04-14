/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file vector_manifold_test.cpp
 * @brief Unified vector manifold tests (Dim=3): state batch dimensions,
 *        prior and between factor LM convergence.
 */

#include <gtest/gtest.h>

#include <random>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/vector_between_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {

namespace {
constexpr int kDim = 3;
}

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(VectorManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<Vector<kDim>> vecs(kN);
  for (auto &v : vecs) {
    v.fill(0.0f);
  }
  dvector<Vector<kDim>> vecs_dev(vecs);

  VectorStateBatch<kDim> states(
      reinterpret_cast<const float *>(vecs_dev.data()), kN);

  EXPECT_EQ(states.TangentSize(), static_cast<size_t>(kDim));
  EXPECT_EQ(states.AmbientSize(), static_cast<size_t>(kDim));
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(VectorManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 5000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> value_dist(-5.0f, 5.0f);
  std::uniform_real_distribution<float> pert_dist(-0.5f, 0.5f);

  hvector<Vector<kDim>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    for (int k = 0; k < kDim; ++k) {
      targets[i][k] = value_dist(rng);
      initials[i][k] = targets[i][k] + pert_dist(rng);
    }
  }

  dvector<Vector<kDim>> targets_dev(targets), initials_dev(initials);

  VectorStateBatch<kDim> state_batch(
      reinterpret_cast<const float *>(initials_dev.data()), kN);
  PriorVectorFactorBatch<kDim> factor_batch(targets_dev.data(), kN);

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

  EXPECT_LT(summary.final_cost, 1e-6f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Vector<kDim>> optimized(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(optimized.data(), state_batch.StateBlockDevicePtr(0),
                 kN * sizeof(Vector<kDim>), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (int j = 0; j < kDim; ++j) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], 1e-4f)
          << "vector " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(VectorManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 5000;
  std::mt19937 rng1(60), rng2(61);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  hvector<Vector<kDim>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    for (int k = 0; k < kDim; ++k) {
      left[i][k] = dist(rng1);
      right[i][k] = dist(rng2);
    }
  }

  dvector<Vector<kDim>> left_dev(left), right_dev(right);
  Vector<kDim> zero;
  zero.fill(0.0f);
  hvector<Vector<kDim>> deltas(kN, zero);
  dvector<Vector<kDim>> deltas_dev(deltas);

  VectorStateBatch<kDim> state_left(
      reinterpret_cast<const float *>(left_dev.data()), kN);
  VectorStateBatch<kDim> state_right(
      reinterpret_cast<const float *>(right_dev.data()), kN);
  VectorBetweenFactorBatch<kDim> factor_batch(deltas_dev.data(), kN);

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

  EXPECT_LT(summary.final_cost, 1e-6f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Vector<kDim>> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_left.data(), state_left.StateBlockDevicePtr(0),
                 kN * sizeof(Vector<kDim>), cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_right.data(), state_right.StateBlockDevicePtr(0),
                 kN * sizeof(Vector<kDim>), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (int j = 0; j < kDim; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 1e-4f)
          << "vector " << i << ", element " << j;
    }
  }
}

} // namespace cunls
