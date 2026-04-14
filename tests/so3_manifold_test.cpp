/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file so3_manifold_test.cpp
 * @brief Unified SO(3) manifold tests: state batch dimensions, prior and
 * between factor LM convergence.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/so3_between_factor_batch.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

namespace {

Matrix<3> MakeSO3Identity() {
  Matrix<3> I{};
  I.fill(0.0f);
  I[0] = I[4] = I[8] = 1.0f;
  return I;
}

dvector<Matrix<3>> GenerateRandomSO3(size_t n, uint32_t seed, float magnitude) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-magnitude, magnitude);

  hvector<Vector<3>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      twists[i][k] = dist(rng);
    }
  }

  dvector<Vector<3>> twists_dev(twists);
  dvector<Matrix<3>> out(n);
  CudaStream stream;
  ComputeExpSO3(stream.GetStream(),
                reinterpret_cast<const float *>(twists_dev.data()), 3, 3, 9, n,
                reinterpret_cast<float *>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

dvector<Matrix<3>> PerturbSO3(const dvector<Matrix<3>> &targets, size_t n,
                              uint32_t seed, float magnitude,
                              cuBLASHandle &cublas) {
  dvector<Matrix<3>> perturbations = GenerateRandomSO3(n, seed, magnitude);
  dvector<Matrix<3>> result(n);

  CudaStream stream;
  auto handle =
      static_cast<cublasHandle_t>(cublas.GetHandle(stream.GetStream()));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &alpha,
      reinterpret_cast<const float *>(perturbations.data()), 3, 9,
      reinterpret_cast<const float *>(targets.data()), 3, 9, &beta,
      reinterpret_cast<float *>(result.data()), 3, 9, n));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return result;
}

} // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(SO3ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<Matrix<3>> rots(kN, MakeSO3Identity());
  dvector<Matrix<3>> rots_dev(rots);

  cuBLASHandle cublas;
  SO3StateBatch states(cublas, reinterpret_cast<const float *>(rots_dev.data()),
                       kN);

  EXPECT_EQ(states.TangentSize(), 3u);
  EXPECT_EQ(states.AmbientSize(), 9u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(SO3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;

  cuBLASHandle cublas;
  dvector<Matrix<3>> targets = GenerateRandomSO3(kN, 42, 0.5f);
  dvector<Matrix<3>> initials = PerturbSO3(targets, kN, 43, 0.3f, cublas);

  SO3StateBatch state_batch(
      cublas, reinterpret_cast<const float *>(initials.data()), kN);
  SO3PriorFactorBatch factor_batch(targets.data(), kN);

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

  hvector<Matrix<3>> optimized(kN), target_host(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(optimized.data(), state_batch.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<3>), cudaMemcpyDeviceToHost));
  targets.CopyToHost(target_host.data(), kN);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 9; ++j) {
      ASSERT_NEAR(optimized[i][j], target_host[i][j], 1e-2f)
          << "rotation " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(SO3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;

  dvector<Matrix<3>> rots_left = GenerateRandomSO3(kN, 60, 0.5f);
  dvector<Matrix<3>> rots_right = GenerateRandomSO3(kN, 61, 0.5f);

  hvector<Matrix<3>> deltas(kN, MakeSO3Identity());
  dvector<Matrix<3>> deltas_dev(deltas);

  cuBLASHandle cublas;
  SO3StateBatch state_left(
      cublas, reinterpret_cast<const float *>(rots_left.data()), kN);
  SO3StateBatch state_right(
      cublas, reinterpret_cast<const float *>(rots_right.data()), kN);
  SO3BetweenFactorBatch factor_batch(deltas_dev.data(), kN);

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

  hvector<Matrix<3>> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_left.data(), state_left.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<3>), cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_right.data(), state_right.StateBlockDevicePtr(0),
                 kN * sizeof(Matrix<3>), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 9; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.1f)
          << "rotation " << i << ", element " << j;
    }
  }
}

} // namespace cunls
