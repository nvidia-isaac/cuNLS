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
 * @file constant_velocity_se3_factor_batch_test.cpp
 * @brief Unit tests for ConstantVelocitySE3FactorBatch.
 */

#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

SE3Transform ExpSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_dev({twist});
  dvector<SE3Transform> out_dev(1);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()),
                /*twist_stride=*/6, /*transform_pitch=*/4,
                /*transform_stride=*/16, /*size=*/1, reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  SE3Transform out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

SE3Transform ComposeSE3(const SE3Transform &a, const SE3Transform &b) {
  SE3Transform out{};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 4; k++) acc += a[i * 4 + k] * b[k * 4 + j];
      out[i * 4 + j] = acc;
    }
  return out;
}

Vector<6> LogSE3(const SE3Transform &pose) {
  CudaStream stream;
  dvector<SE3Transform> pose_dev({pose});
  dvector<Vector<6>> out_dev(1);
  ComputeLogSE3(stream.GetStream(), reinterpret_cast<const float *>(pose_dev.data()),
                /*transform_pitch=*/4, /*transform_stride=*/16,
                /*twist_stride=*/6, /*size=*/1, reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Vector<6> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Matrix<6> JacobianLeftInverseSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_dev({twist});
  dvector<Matrix<6>> out_dev(1);
  ComputeJacobianLeftInverseSE3(stream.GetStream(),
                                reinterpret_cast<const float *>(twist_dev.data()),
                                /*twist_stride=*/6, /*jacobian_pitch=*/6, /*jacobian_stride=*/36,
                                /*size=*/1, reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Matrix<6> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Matrix<6> JacobianLeftSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_dev({twist});
  dvector<Matrix<6>> out_dev(1);
  ComputeJacobianLeftSE3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()),
                         /*twist_stride=*/6, /*jacobian_pitch=*/6,
                         /*jacobian_stride=*/36, /*size=*/1,
                         reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Matrix<6> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Vector<6> MatVec6(const Matrix<6> &m, const Vector<6> &v) {
  Vector<6> out{};
  for (int i = 0; i < 6; i++) {
    float acc = 0.0f;
    for (int j = 0; j < 6; j++) acc += m[i * 6 + j] * v[j];
    out[i] = acc;
  }
  return out;
}

/**
 * @brief Evaluates one ConstantVelocitySE3FactorBatch factor and returns
 * residuals (and, if requested, the dense 12x24 Jacobian) on host.
 */
struct SingleFactorResult {
  std::vector<float> residuals;  // 12
  std::vector<float> jacobian;   // 12x24, row-major (optional)
};

SingleFactorResult EvaluateOne(const SE3Transform &pose_k, const SE3Transform &pose_k1,
                               const Vector<6> &vel_k, const Vector<6> &vel_k1, float dt,
                               bool want_jacobian) {
  CudaStream stream;
  dvector<SE3Transform> pose_k_dev({pose_k});
  dvector<SE3Transform> pose_k1_dev({pose_k1});
  dvector<Vector<6>> vel_k_dev({vel_k});
  dvector<Vector<6>> vel_k1_dev({vel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     reinterpret_cast<const float *>(vel_k_dev.data()),
                                     reinterpret_cast<const float *>(vel_k1_dev.data())};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantVelocitySE3FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(12);
  dvector<float> jac_dev(want_jacobian ? 12 * 24 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(12);
  res_dev.CopyToHost(result.residuals.data(), 12);
  if (want_jacobian) {
    result.jacobian.resize(12 * 24);
    jac_dev.CopyToHost(result.jacobian.data(), 12 * 24);
  }
  return result;
}

class ConstantVelocitySE3FactorBatchTest : public ::testing::Test {
 protected:
  std::mt19937 rng_{42};
};

TEST_F(ConstantVelocitySE3FactorBatchTest, ZeroResidualForExactConstantVelocity) {
  std::uniform_real_distribution<float> rot(-0.4f, 0.4f);
  std::uniform_real_distribution<float> trans(-1.0f, 1.0f);

  Vector<6> vel_k{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  const float dt = 0.2f;

  SE3Transform pose_k = ExpSE3({0.1f, -0.2f, 0.05f, 0.5f, -0.3f, 0.2f});
  Vector<6> step_twist;
  for (int i = 0; i < 6; i++) step_twist[i] = dt * vel_k[i];
  SE3Transform pose_k1 = ComposeSE3(pose_k, ExpSE3(step_twist));

  // Constant velocity: vel_k+1 == vel_k, and Jl_inv(step_twist) is applied
  // to it, so we must supply vel_k1 s.t. Jl_inv(step_twist) * vel_k1 == vel_k.
  Matrix<6> jl = JacobianLeftSE3(step_twist);  // Jl_inv^{-1} == Jl
  Vector<6> vel_k1 = MatVec6(jl, vel_k);

  auto result = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, dt, /*want_jacobian=*/false);
  for (int i = 0; i < 12; i++) {
    EXPECT_NEAR(result.residuals[i], 0.0f, 1e-3f) << "residual index " << i;
  }
}

TEST_F(ConstantVelocitySE3FactorBatchTest, NumericalJacobianPoseBlocksAndVelBlocks) {
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans(-1.0f, 1.0f);

  SE3Transform pose_k =
      ExpSE3({rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)});
  SE3Transform pose_k1 =
      ExpSE3({rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)});
  Vector<6> vel_k{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  Vector<6> vel_k1{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  const float dt = 0.37f;

  auto base = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, dt, /*want_jacobian=*/true);

  constexpr float eps = 1e-4f;
  constexpr int kDim = 6;
  constexpr int kCols = 24;

  // --- d(r_pose)/d(pose_k), d(r_pose)/d(pose_k+1): exact, from J_l^{-1}/J_r^{-1}. ---
  for (int block = 0; block < 2; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<6> delta{};
      delta[k] = eps;
      SE3Transform pk_plus = (block == 0) ? ComposeSE3(pose_k, ExpSE3(delta)) : pose_k;
      SE3Transform pk1_plus = (block == 1) ? ComposeSE3(pose_k1, ExpSE3(delta)) : pose_k1;
      delta[k] = -eps;
      SE3Transform pk_minus = (block == 0) ? ComposeSE3(pose_k, ExpSE3(delta)) : pose_k;
      SE3Transform pk1_minus = (block == 1) ? ComposeSE3(pose_k1, ExpSE3(delta)) : pose_k1;

      auto plus = EvaluateOne(pk_plus, pk1_plus, vel_k, vel_k1, dt, false);
      auto minus = EvaluateOne(pk_minus, pk1_minus, vel_k, vel_k1, dt, false);

      for (int row = 0; row < kDim; row++) {
        float numerical = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
        float analytical = base.jacobian[row * kCols + block * kDim + k];
        EXPECT_NEAR(analytical, numerical, 5e-2f)
            << "r_pose/pose block " << block << " row " << row << " col " << k;
      }
    }
  }

  // --- d(r_pose)/d(vel_k) = -dt*I, d(r_pose)/d(vel_k+1) = 0: exact. ---
  for (int block = 0; block < 2; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<6> vk_plus = vel_k, vk1_plus = vel_k1;
      Vector<6> vk_minus = vel_k, vk1_minus = vel_k1;
      (block == 0 ? vk_plus : vk1_plus)[k] += eps;
      (block == 0 ? vk_minus : vk1_minus)[k] -= eps;

      auto plus = EvaluateOne(pose_k, pose_k1, vk_plus, vk1_plus, dt, false);
      auto minus = EvaluateOne(pose_k, pose_k1, vk_minus, vk1_minus, dt, false);

      for (int row = 0; row < kDim; row++) {
        float numerical = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
        float analytical = base.jacobian[row * kCols + (2 + block) * kDim + k];
        EXPECT_NEAR(analytical, numerical, 5e-3f)
            << "r_pose/vel block " << block << " row " << row << " col " << k;
      }
    }
  }

  // --- d(r_vel)/d(vel_k) = -I, d(r_vel)/d(vel_k+1) = J_l^{-1}: exact. ---
  for (int block = 0; block < 2; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<6> vk_plus = vel_k, vk1_plus = vel_k1;
      Vector<6> vk_minus = vel_k, vk1_minus = vel_k1;
      (block == 0 ? vk_plus : vk1_plus)[k] += eps;
      (block == 0 ? vk_minus : vk1_minus)[k] -= eps;

      auto plus = EvaluateOne(pose_k, pose_k1, vk_plus, vk1_plus, dt, false);
      auto minus = EvaluateOne(pose_k, pose_k1, vk_minus, vk1_minus, dt, false);

      for (int row = 0; row < kDim; row++) {
        float numerical = (plus.residuals[kDim + row] - minus.residuals[kDim + row]) / (2.0f * eps);
        float analytical = base.jacobian[(kDim + row) * kCols + (2 + block) * kDim + k];
        EXPECT_NEAR(analytical, numerical, 5e-3f)
            << "r_vel/vel block " << block << " row " << row << " col " << k;
      }
    }
  }

  // NOTE: d(r_vel)/d(pose_k) and d(r_vel)/d(pose_k+1) are intentionally
  // approximated as zero (see docs/design/motion_prior_factors.md and the
  // class docs on ConstantVelocitySE3FactorBatch); the true numerical
  // derivative is generally nonzero, so it is deliberately not checked here.
}

// Fixes pose_k, pose_k+1 as constant and solves for vel_k, vel_k+1 only. With
// poses fixed, the residual is exactly linear in the velocities, so
// Gauss-Newton must recover the unique zero-residual solution.
TEST_F(ConstantVelocitySE3FactorBatchTest, RecoversVelocityWithPosesFixed) {
  const float dt = 0.4f;
  SE3Transform pose_k = ExpSE3({0.1f, -0.15f, 0.05f, 0.4f, 0.2f, -0.3f});
  Vector<6> step_twist{0.2f, -0.1f, 0.05f, 0.6f, -0.4f, 0.1f};
  SE3Transform pose_k1 = ComposeSE3(pose_k, ExpSE3(step_twist));

  Vector<6> expected_vel_k;
  for (int i = 0; i < 6; i++) expected_vel_k[i] = step_twist[i] / dt;
  Matrix<6> jl = JacobianLeftSE3(step_twist);
  Vector<6> expected_vel_k1 = MatVec6(jl, expected_vel_k);

  dvector<SE3Transform> poses_dev(std::vector<SE3Transform>{pose_k, pose_k1});
  std::vector<int> const_ids = {0, 1};
  dvector<int> const_ids_dev(const_ids);
  cuBLASHandle cublas_handle;
  SE3StateBatch pose_batch(cublas_handle, reinterpret_cast<const float *>(poses_dev.data()), 2,
                           const_ids_dev.data(), 2);

  // Initialize velocities away from the expected solution.
  std::vector<Vector<6>> vel_init = {Vector<6>{0, 0, 0, 0, 0, 0}, Vector<6>{0, 0, 0, 0, 0, 0}};
  dvector<Vector<6>> vel_dev(vel_init);
  VectorStateBatch<6> vel_batch(reinterpret_cast<const float *>(vel_dev.data()), 2);

  dvector<float> dt_dev(std::vector<float>{dt});
  ConstantVelocitySE3FactorBatch factor(dt_dev.data(), 1);

  std::vector<float *> state_pointers = {
      pose_batch.StateBlockDevicePtr(0), pose_batch.StateBlockDevicePtr(1),
      vel_batch.StateBlockDevicePtr(0), vel_batch.StateBlockDevicePtr(1)};

  Problem problem;
  problem.AddStateBatch(&pose_batch);
  problem.AddStateBatch(&vel_batch);
  problem.AddFactorBatch(&factor, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 5;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-8f;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
      test_utils::PCGBlockSizeFromEnv(6);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
      test_utils::PCGMaxIterFromEnv(400);
  options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance =
      test_utils::PCGTolFromEnv(1e-6f);

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-4f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  ASSERT_LT(summary.final_cost, 1e-4f);

  std::vector<Vector<6>> recovered(2);
  vel_dev.CopyToHost(recovered.data(), 2);

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(recovered[0][i], expected_vel_k[i], 1e-2f) << "vel_k[" << i << "]";
    EXPECT_NEAR(recovered[1][i], expected_vel_k1[i], 1e-2f) << "vel_k+1[" << i << "]";
  }
}

}  // namespace
}  // namespace cunls
