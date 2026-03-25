/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file sim3_manifold_test.cpp
 * @brief Unified Sim(3) manifold tests: state batch dimensions, prior and between
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
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/factor/similarity3_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/similarity3_state_batch.h"

namespace cunls {

namespace {

/**
 * @brief Build a Sim(3) 4x4 matrix on host from Lie algebra tangent.
 *
 * Uses Rodrigues for R, V-matrix for translation, exp(lambda) for scale.
 */
Matrix<4> MakeSim3(float w1, float w2, float w3,
                   float u1, float u2, float u3, float lambda) {
  float theta2 = w1 * w1 + w2 * w2 + w3 * w3;
  float theta = std::sqrt(theta2);

  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    float st = std::sin(theta), ct = std::cos(theta);
    A1 = st / theta;
    A2 = (1.0f - ct) / theta2;
    A3 = (1.0f - A1) / theta2;
    A4 = (0.5f - A2) / theta2;
  } else {
    A1 = 1.0f - theta2 / 6.0f;
    A2 = 0.5f - theta2 / 24.0f;
    A3 = 1.0f / 6.0f - theta2 / 120.0f;
    A4 = 1.0f / 24.0f - theta2 / 720.0f;
  }

  float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    float e = std::exp(-lambda);
    P_c = (1.0f - e) / lambda;
    float alpha = lambda2 / (lambda2 + theta2);
    float beta = (e - 1.0f + lambda) / lambda2;
    float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    float one_m_a = 1.0f - alpha;
    Q_c = alpha * beta + one_m_a * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_a * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  float ct_val = 1.0f - A2 * theta2;
  Matrix<4> T;
  T[0] = ct_val + A2 * w1 * w1;
  T[1] = A2 * w1 * w2 - A1 * w3;
  T[2] = A2 * w1 * w3 + A1 * w2;
  T[4] = A2 * w2 * w1 + A1 * w3;
  T[5] = ct_val + A2 * w2 * w2;
  T[6] = A2 * w2 * w3 - A1 * w1;
  T[8] = A2 * w3 * w1 - A1 * w2;
  T[9] = A2 * w3 * w2 + A1 * w1;
  T[10] = ct_val + A2 * w3 * w3;

  float diag = P_c - R_c * theta2;
  float dot_wu = w1 * u1 + w2 * u2 + w3 * u3;
  float cx = w2 * u3 - w3 * u2;
  float cy = w3 * u1 - w1 * u3;
  float cz = w1 * u2 - w2 * u1;
  T[3] = diag * u1 + Q_c * cx + R_c * w1 * dot_wu;
  T[7] = diag * u2 + Q_c * cy + R_c * w2 * dot_wu;
  T[11] = diag * u3 + Q_c * cz + R_c * w3 * dot_wu;

  T[12] = 0.0f;
  T[13] = 0.0f;
  T[14] = 0.0f;
  T[15] = std::exp(-lambda);
  return T;
}

Matrix<4> MakeSim3Identity() {
  Matrix<4> T{};
  T[0] = T[5] = T[10] = T[15] = 1.0f;
  return T;
}

Matrix<4> MatMul4x4(const Matrix<4>& A, const Matrix<4>& B) {
  Matrix<4> C;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) {
        s += A[r * 4 + k] * B[k * 4 + c];
      }
      C[r * 4 + c] = s;
    }
  }
  return C;
}

}  // namespace

// ============================================================================
// State batch dimensions
// ============================================================================

TEST(Sim3ManifoldTest, StateDimensions) {
  constexpr size_t kN = 10;
  hvector<Matrix<4>> transforms(kN, MakeSim3Identity());
  dvector<Matrix<4>> transforms_dev(transforms);

  cuBLASHandle cublas;
  Similarity3StateBatch states(
      cublas, reinterpret_cast<const float*>(transforms_dev.data()), kN);

  EXPECT_EQ(states.TangentSize(), 7u);
  EXPECT_EQ(states.AmbientSize(), 16u);
  EXPECT_EQ(states.NumStateBlocks(), kN);
}

// ============================================================================
// Prior factor LM convergence
// ============================================================================

TEST(Sim3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> rot_dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> trans_dist(-2.0f, 2.0f);
  std::uniform_real_distribution<float> scale_dist(-0.3f, 0.3f);
  std::uniform_real_distribution<float> pert_rot(-0.1f, 0.1f);
  std::uniform_real_distribution<float> pert_trans(-0.3f, 0.3f);
  std::uniform_real_distribution<float> pert_scale(-0.05f, 0.05f);

  hvector<Matrix<4>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    targets[i] = MakeSim3(rot_dist(rng), rot_dist(rng), rot_dist(rng),
                           trans_dist(rng), trans_dist(rng), trans_dist(rng),
                           scale_dist(rng));
    Matrix<4> pert = MakeSim3(pert_rot(rng), pert_rot(rng), pert_rot(rng),
                               pert_trans(rng), pert_trans(rng), pert_trans(rng),
                               pert_scale(rng));
    initials[i] = MatMul4x4(targets[i], pert);
  }

  dvector<Matrix<4>> targets_dev(targets), initials_dev(initials);

  cuBLASHandle cublas;
  Similarity3StateBatch state_batch(
      cublas, reinterpret_cast<const float*>(initials_dev.data()), kN);
  Similarity3PriorFactorBatch factor_batch(cublas, targets_dev.data(), kN);

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
  LevenbergMarquardtMinimizerOptions lm_opts;
  lm_opts.base_options = opts;
  lm_opts.initial_lambda = 1e-3f;

  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_opts);
  MinimizerSummary summary =
      minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-2f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  hvector<Matrix<4>> optimized(kN), target_host(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(),
                                 state_batch.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<4>),
                                 cudaMemcpyDeviceToHost));
  targets_dev.CopyToHost(target_host.data(), kN);

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

TEST(Sim3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(70), rng2(71);
  std::uniform_real_distribution<float> rot_dist(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans_dist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> scale_dist(-0.2f, 0.2f);

  hvector<Matrix<4>> poses_left(kN), poses_right(kN);
  for (size_t i = 0; i < kN; ++i) {
    poses_left[i] = MakeSim3(rot_dist(rng1), rot_dist(rng1), rot_dist(rng1),
                              trans_dist(rng1), trans_dist(rng1),
                              trans_dist(rng1), scale_dist(rng1));
    poses_right[i] = MakeSim3(rot_dist(rng2), rot_dist(rng2), rot_dist(rng2),
                               trans_dist(rng2), trans_dist(rng2),
                               trans_dist(rng2), scale_dist(rng2));
  }

  dvector<Matrix<4>> left_dev(poses_left), right_dev(poses_right);
  hvector<Matrix<4>> deltas(kN, MakeSim3Identity());
  dvector<Matrix<4>> deltas_dev(deltas);

  cuBLASHandle cublas;
  Similarity3StateBatch state_left(
      cublas, reinterpret_cast<const float*>(left_dev.data()), kN);
  Similarity3StateBatch state_right(
      cublas, reinterpret_cast<const float*>(right_dev.data()), kN);
  Similarity3BetweenFactorBatch factor_batch(cublas, deltas_dev.data(), kN);

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

  hvector<Matrix<4>> opt_left(kN), opt_right(kN);
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_left.data(),
                                 state_left.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<4>),
                                 cudaMemcpyDeviceToHost));
  THROW_ON_CUDA_ERROR(cudaMemcpy(opt_right.data(),
                                 state_right.StateBlockDevicePtr(0),
                                 kN * sizeof(Matrix<4>),
                                 cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kN; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(opt_left[i][j], opt_right[i][j], 0.1f)
          << "transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
