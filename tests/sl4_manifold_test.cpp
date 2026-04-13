/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file sl4_manifold_test.cpp
 * @brief Unified SL(4) manifold tests: state batch dimensions, prior and between
 *        factor LM convergence.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sl4_between_factor_batch.h"
#include "cunls/factor/sl4_prior_factor_batch.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/sl4_state_batch.h"

namespace cunls {

namespace {

SL4Transform MakeSL4Identity() {
  SL4Transform I{};
  I.fill(0.0f);
  I[0] = I[5] = I[10] = I[15] = 1.0f;
  return I;
}

dvector<SL4Transform> GenerateRandomSL4(size_t n, uint32_t seed,
                                        float magnitude) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-magnitude, magnitude);

  hvector<Vector<15>> xi(n);
  for (size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 15; ++k) {
      xi[i][k] = dist(rng);
    }
  }

  dvector<Vector<15>> xi_dev(xi);
  dvector<SL4Transform> out(n);
  CudaStream stream;
  ComputeExpSL4(stream.GetStream(),
                reinterpret_cast<const float*>(xi_dev.data()), 15, 4, 16, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

dvector<SL4Transform> PerturbSL4(const dvector<SL4Transform>& targets,
                                 size_t n, uint32_t seed, float magnitude,
                                 cuBLASHandle& cublas) {
  dvector<SL4Transform> perturbations =
      GenerateRandomSL4(n, seed, magnitude);
  dvector<SL4Transform> result(n);

  CudaStream stream;
  auto handle =
      static_cast<cublasHandle_t>(cublas.GetHandle(stream.GetStream()));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &alpha,
      reinterpret_cast<const float*>(perturbations.data()), 4, 16,
      reinterpret_cast<const float*>(targets.data()), 4, 16, &beta,
      reinterpret_cast<float*>(result.data()), 4, 16, n));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return result;
}

}  // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(SL4ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<SL4Transform> transforms(kN, MakeSL4Identity());
  dvector<SL4Transform> transforms_dev(transforms);

  cuBLASHandle cublas;
  SL4StateBatch states(
      cublas, reinterpret_cast<const float*>(transforms_dev.data()), kN);

  EXPECT_EQ(states.TangentSize(), 15u);
  EXPECT_EQ(states.AmbientSize(), 16u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(SL4ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 800;

  cuBLASHandle cublas;
  dvector<SL4Transform> targets = GenerateRandomSL4(kN, 44, 0.08f);
  dvector<SL4Transform> initials =
      PerturbSL4(targets, kN, 45, 0.04f, cublas);

  SL4StateBatch state_batch(
      cublas, reinterpret_cast<const float*>(initials.data()), kN);
  SL4PriorFactorBatch factor_batch(cublas, targets.data(), kN);

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

  hvector<SL4Transform> optimized(kN), target_host(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(),
                                 state_batch.StateBlockDevicePtr(0),
                                 kN * sizeof(SL4Transform),
                                 cudaMemcpyDeviceToHost));
  targets.CopyToHost(target_host.data(), kN);

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(optimized[i][j], target_host[i][j], 0.15f)
          << "transform " << i << ", element " << j;
    }
  }
}

// ============================================================================
// Between factor LM convergence
// ============================================================================

TEST(SL4ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 10000;

  dvector<SL4Transform> poses_left = GenerateRandomSL4(kN, 50, 0.5f);
  dvector<SL4Transform> poses_right = GenerateRandomSL4(kN, 51, 0.5f);

  hvector<SL4Transform> deltas(kN, MakeSL4Identity());
  dvector<SL4Transform> deltas_dev(deltas);

  cuBLASHandle cublas;
  SL4StateBatch state_left(
      cublas, reinterpret_cast<const float*>(poses_left.data()), kN);
  SL4StateBatch state_right(
      cublas, reinterpret_cast<const float*>(poses_right.data()), kN);
  SL4BetweenFactorBatch factor_batch(cublas, deltas_dev.data(), kN);

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

  hvector<SL4Transform> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_left.data(),
                                 state_left.StateBlockDevicePtr(0),
                                 kN * sizeof(SL4Transform),
                                 cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_right.data(),
                                 state_right.StateBlockDevicePtr(0),
                                 kN * sizeof(SL4Transform),
                                 cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.15f)
          << "transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
