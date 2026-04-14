/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file se3_manifold_test.cpp
 * @brief Unified SE(3) manifold tests: state batch dimensions, prior and
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
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"

namespace cunls {

namespace {

SE3Transform MakeSE3Identity() {
  SE3Transform I{};
  I.fill(0.0f);
  I[0] = I[5] = I[10] = I[15] = 1.0f;
  return I;
}

dvector<SE3Transform> GenerateRandomSE3(size_t n, uint32_t seed, float rot_mag,
                                        float trans_mag) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> rot_dist(-rot_mag, rot_mag);
  std::uniform_real_distribution<float> trans_dist(-trans_mag, trans_mag);

  hvector<Vector<6>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    twists[i][0] = rot_dist(rng);
    twists[i][1] = rot_dist(rng);
    twists[i][2] = rot_dist(rng);
    twists[i][3] = trans_dist(rng);
    twists[i][4] = trans_dist(rng);
    twists[i][5] = trans_dist(rng);
  }

  dvector<Vector<6>> twists_dev(twists);
  dvector<SE3Transform> out(n);
  CudaStream stream;
  ComputeExpSE3(stream.GetStream(),
                reinterpret_cast<const float *>(twists_dev.data()), 6, 4, 16, n,
                reinterpret_cast<float *>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

dvector<SE3Transform> PerturbSE3(const dvector<SE3Transform> &targets, size_t n,
                                 uint32_t seed, float rot_mag, float trans_mag,
                                 cuBLASHandle &cublas) {
  dvector<SE3Transform> perturbations =
      GenerateRandomSE3(n, seed, rot_mag, trans_mag);
  dvector<SE3Transform> result(n);

  CudaStream stream;
  auto handle =
      static_cast<cublasHandle_t>(cublas.GetHandle(stream.GetStream()));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &alpha,
      reinterpret_cast<const float *>(perturbations.data()), 4, 16,
      reinterpret_cast<const float *>(targets.data()), 4, 16, &beta,
      reinterpret_cast<float *>(result.data()), 4, 16, n));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return result;
}

} // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(SE3ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<SE3Transform> transforms(kN, MakeSE3Identity());
  dvector<SE3Transform> transforms_dev(transforms);

  cuBLASHandle cublas;
  SE3StateBatch states(
      cublas, reinterpret_cast<const float *>(transforms_dev.data()), kN);

  EXPECT_EQ(states.TangentSize(), 6u);
  EXPECT_EQ(states.AmbientSize(), 16u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(SE3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;

  cuBLASHandle cublas;
  dvector<SE3Transform> targets = GenerateRandomSE3(kN, 42, 0.5f, 2.0f);
  dvector<SE3Transform> initials =
      PerturbSE3(targets, kN, 43, 0.1f, 0.3f, cublas);

  SE3StateBatch state_batch(
      cublas, reinterpret_cast<const float *>(initials.data()), kN);
  SE3PriorFactorBatch factor_batch(targets.data(), kN);

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

  EXPECT_LT(summary.final_cost, 1e-2f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<SE3Transform> optimized(kN), target_host(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(optimized.data(), state_batch.StateBlockDevicePtr(0),
                 kN * sizeof(SE3Transform), cudaMemcpyDeviceToHost));
  targets.CopyToHost(target_host.data(), kN);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(optimized[i][j], target_host[i][j], 1e-2f)
          << "transform " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(SE3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;

  dvector<SE3Transform> poses_left = GenerateRandomSE3(kN, 60, 0.3f, 1.0f);
  dvector<SE3Transform> poses_right = GenerateRandomSE3(kN, 61, 0.3f, 1.0f);

  hvector<SE3Transform> deltas(kN, MakeSE3Identity());
  dvector<SE3Transform> deltas_dev(deltas);

  cuBLASHandle cublas;
  SE3StateBatch state_left(
      cublas, reinterpret_cast<const float *>(poses_left.data()), kN);
  SE3StateBatch state_right(
      cublas, reinterpret_cast<const float *>(poses_right.data()), kN);
  SE3BetweenFactorBatch factor_batch(deltas_dev.data(), kN);

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

  hvector<SE3Transform> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_left.data(), state_left.StateBlockDevicePtr(0),
                 kN * sizeof(SE3Transform), cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(opt_right.data(), state_right.StateBlockDevicePtr(0),
                 kN * sizeof(SE3Transform), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.1f)
          << "transform " << i << ", element " << j;
    }
  }
}

} // namespace cunls
