/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @file similarity3_prior_factor_test.cpp
 * @brief Tests for Similarity3StateBatch and
 *        Similarity3PriorFactorBatch with LM convergence.
 *
 * Generates random Sim(3) transforms as targets, perturbs them, and verifies
 * that LM optimization with the Sim(3) prior factor converges.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity3_prior_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/similarity3_state_batch.h"

namespace cunls {

/**
 * @brief Helper: build a Sim(3) 4x4 matrix on the host from (R, t, s).
 *
 * Uses Rodrigues' formula for R = Exp(w), V-matrix for t = V*u, s = exp(lambda).
 *
 * @param w  3D rotation (axis-angle)
 * @param u  3D translation tangent component
 * @param lambda  log-scale
 * @return Matrix<4>  row-major 4x4 matrix [R t; 0 1/s]
 */
static Matrix<4> MakeSim3Matrix(float w1, float w2, float w3, float u1,
                                float u2, float u3, float lambda) {
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

  // R via Rodrigues
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

  // t = V*u
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

/// Host-side 4x4 row-major matrix multiply
static Matrix<4> MatMul4x4(const Matrix<4>& A, const Matrix<4>& B) {
  Matrix<4> C;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 4; ++k) s += A[r * 4 + k] * B[k * 4 + c];
      C[r * 4 + c] = s;
    }
  return C;
}

/**
 * @brief Test fixture for Sim(3) prior factor tests.
 */
class Similarity3PriorCostTest : public ::testing::Test {
 public:
  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rot_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> trans_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> scale_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> pert_rot(-0.1f, 0.1f);
    std::uniform_real_distribution<float> pert_trans(-0.3f, 0.3f);
    std::uniform_real_distribution<float> pert_scale(-0.05f, 0.05f);

    hvector<Matrix<4>> targets_host(num_transforms_);
    hvector<Matrix<4>> initials_host(num_transforms_);

    for (size_t i = 0; i < num_transforms_; ++i) {
      // Target
      targets_host[i] = MakeSim3Matrix(
          rot_dist(rng), rot_dist(rng), rot_dist(rng), trans_dist(rng),
          trans_dist(rng), trans_dist(rng), scale_dist(rng));

      // Perturbation
      Matrix<4> pert = MakeSim3Matrix(
          pert_rot(rng), pert_rot(rng), pert_rot(rng), pert_trans(rng),
          pert_trans(rng), pert_trans(rng), pert_scale(rng));

      // Initial = Target * Perturbation
      initials_host[i] = MatMul4x4(targets_host[i], pert);
    }

    target_transforms_device_ = DeviceVector<Matrix<4>>(targets_host);
    initial_transforms_device_ = DeviceVector<Matrix<4>>(initials_host);
  }

  const size_t num_transforms_ = 10000;
  const uint32_t fixed_seed_ = 42;
  DeviceVector<Matrix<4>> target_transforms_device_;
  DeviceVector<Matrix<4>> initial_transforms_device_;

  cuBLASHandle cublas_handle_;

  profiler::Domain profiler_domain_{"Similarity3PriorCostTest"};
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(Similarity3PriorCostTest, TangentSize) {
  const float* ptr =
      reinterpret_cast<const float*>(initial_transforms_device_.data());
  Similarity3StateBatch batch(cublas_handle_, ptr, num_transforms_);
  EXPECT_EQ(batch.TangentSize(), 7u);
}

TEST_F(Similarity3PriorCostTest, AmbientSize) {
  const float* ptr =
      reinterpret_cast<const float*>(initial_transforms_device_.data());
  Similarity3StateBatch batch(cublas_handle_, ptr, num_transforms_);
  EXPECT_EQ(batch.AmbientSize(), 16u);
}

TEST_F(Similarity3PriorCostTest, NumStateBlocks) {
  const float* ptr =
      reinterpret_cast<const float*>(initial_transforms_device_.data());
  Similarity3StateBatch batch(cublas_handle_, ptr, num_transforms_);
  EXPECT_EQ(batch.NumStateBlocks(), num_transforms_);
}

TEST_F(Similarity3PriorCostTest, LMConvergence) {
  auto test_range = profiler_domain_.CreateDomainRange("LMConvergence");

  const float* state_data_ptr =
      reinterpret_cast<const float*>(initial_transforms_device_.data());
  Similarity3StateBatch state_batch(cublas_handle_, state_data_ptr, num_transforms_);

  Similarity3PriorFactorBatch factor_batch(
      cublas_handle_, target_transforms_device_.data(), num_transforms_);

  std::vector<float*> state_pointers;
  state_pointers.reserve(num_transforms_);
  for (size_t i = 0; i < num_transforms_; ++i)
    state_pointers.push_back(state_batch.StateBlockDevicePtr(i));

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-8f;

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;

  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_options);
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-2f);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);

  // Verify element-wise
  hvector<Matrix<4>> optimized(num_transforms_);
  hvector<Matrix<4>> targets(num_transforms_);
  const float* opt_ptr = state_batch.StateBlockDevicePtr(0);
  THROW_ON_CUDA_ERROR(cudaMemcpy(optimized.data(), opt_ptr,
                                  num_transforms_ * sizeof(Matrix<4>),
                                  cudaMemcpyDeviceToHost));
  target_transforms_device_.CopyToHost(targets.data(), num_transforms_);

  constexpr float tolerance = 1e-2f;
  for (size_t i = 0; i < num_transforms_; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_NEAR(optimized[i][j], targets[i][j], tolerance)
          << "Mismatch at transform " << i << ", element " << j;
    }
  }
}

}  // namespace cunls
