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
 * @file constant_acceleration_se3_factor_batch_test.cpp
 * @brief Unit tests for ConstantAccelerationSE3FactorBatch.
 */

#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {
namespace {

SE3Transform ExpSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_dev({twist});
  dvector<SE3Transform> out_dev(1);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()), 6, 4, 16, 1,
                reinterpret_cast<float *>(out_dev.data()));
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

Matrix<6> JacobianLeftSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_dev({twist});
  dvector<Matrix<6>> out_dev(1);
  ComputeJacobianLeftSE3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()), 6,
                         6, 36, 1, reinterpret_cast<float *>(out_dev.data()));
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

struct SingleFactorResult {
  std::vector<float> residuals;  // 18
  std::vector<float> jacobian;   // 18x36 (optional)
};

SingleFactorResult EvaluateOne(const SE3Transform &pose_k, const SE3Transform &pose_k1,
                               const Vector<6> &vel_k, const Vector<6> &vel_k1,
                               const Vector<6> &accel_k, const Vector<6> &accel_k1, float dt,
                               bool want_jacobian) {
  CudaStream stream;
  dvector<SE3Transform> pose_k_dev({pose_k});
  dvector<SE3Transform> pose_k1_dev({pose_k1});
  dvector<Vector<6>> vel_k_dev({vel_k});
  dvector<Vector<6>> vel_k1_dev({vel_k1});
  dvector<Vector<6>> accel_k_dev({accel_k});
  dvector<Vector<6>> accel_k1_dev({accel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     reinterpret_cast<const float *>(vel_k_dev.data()),
                                     reinterpret_cast<const float *>(vel_k1_dev.data()),
                                     reinterpret_cast<const float *>(accel_k_dev.data()),
                                     reinterpret_cast<const float *>(accel_k1_dev.data())};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantAccelerationSE3FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(18);
  dvector<float> jac_dev(want_jacobian ? 18 * 36 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(18);
  res_dev.CopyToHost(result.residuals.data(), 18);
  if (want_jacobian) {
    result.jacobian.resize(18 * 36);
    jac_dev.CopyToHost(result.jacobian.data(), 18 * 36);
  }
  return result;
}

class ConstantAccelerationSE3FactorBatchTest : public ::testing::Test {
 protected:
  std::mt19937 rng_{142};
};

TEST_F(ConstantAccelerationSE3FactorBatchTest, ZeroResidualForExactConstantAcceleration) {
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans(-1.0f, 1.0f);

  Vector<6> vel_k{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  Vector<6> accel_k{rot(rng_) * 0.5f,   rot(rng_) * 0.5f,   rot(rng_) * 0.5f,
                    trans(rng_) * 0.5f, trans(rng_) * 0.5f, trans(rng_) * 0.5f};
  const float dt = 0.2f;

  SE3Transform pose_k = ExpSE3({0.1f, -0.2f, 0.05f, 0.5f, -0.3f, 0.2f});
  Vector<6> step_twist;
  for (int i = 0; i < 6; i++) step_twist[i] = dt * vel_k[i] + 0.5f * dt * dt * accel_k[i];
  SE3Transform pose_k1 = ComposeSE3(pose_k, ExpSE3(step_twist));

  Matrix<6> jl = JacobianLeftSE3(step_twist);
  Vector<6> vel_k_plus_dt_accel;
  for (int i = 0; i < 6; i++) vel_k_plus_dt_accel[i] = vel_k[i] + dt * accel_k[i];
  Vector<6> vel_k1 = MatVec6(jl, vel_k_plus_dt_accel);
  Vector<6> accel_k1 = MatVec6(jl, accel_k);

  auto result = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, accel_k, accel_k1, dt, false);
  for (int i = 0; i < 18; i++)
    EXPECT_NEAR(result.residuals[i], 0.0f, 1e-3f) << "residual index " << i;
}

TEST_F(ConstantAccelerationSE3FactorBatchTest, NumericalJacobianVelAccelBlocks) {
  std::uniform_real_distribution<float> rot(-0.25f, 0.25f);
  std::uniform_real_distribution<float> trans(-0.8f, 0.8f);

  SE3Transform pose_k =
      ExpSE3({rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)});
  SE3Transform pose_k1 =
      ExpSE3({rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)});
  Vector<6> vel_k{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  Vector<6> vel_k1{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  Vector<6> accel_k{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  Vector<6> accel_k1{rot(rng_), rot(rng_), rot(rng_), trans(rng_), trans(rng_), trans(rng_)};
  const float dt = 0.29f;

  auto base = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, accel_k, accel_k1, dt, true);

  constexpr float eps = 1e-4f;
  constexpr int kDim = 6;
  constexpr int kCols = 36;

  // Check the four velocity/acceleration blocks (indices 2..5) against all
  // three residual rows (r_pose, r_vel, r_accel). Pose blocks (0,1) are
  // skipped for r_vel/r_accel rows (documented simplification); r_pose's
  // dependence on pose blocks mirrors ConstantVelocitySE3FactorBatch and is
  // covered there, so it is not re-verified here.
  for (int block = 2; block < 6; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<6> v[4] = {vel_k, vel_k1, accel_k, accel_k1};
      Vector<6> v_plus[4] = {vel_k, vel_k1, accel_k, accel_k1};
      Vector<6> v_minus[4] = {vel_k, vel_k1, accel_k, accel_k1};
      v_plus[block - 2][k] += eps;
      v_minus[block - 2][k] -= eps;

      auto plus =
          EvaluateOne(pose_k, pose_k1, v_plus[0], v_plus[1], v_plus[2], v_plus[3], dt, false);
      auto minus =
          EvaluateOne(pose_k, pose_k1, v_minus[0], v_minus[1], v_minus[2], v_minus[3], dt, false);

      for (int row = 0; row < 3 * kDim; row++) {
        float numerical = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
        float analytical = base.jacobian[row * kCols + block * kDim + k];
        EXPECT_NEAR(analytical, numerical, 5e-3f)
            << "block " << block << " row " << row << " col " << k;
      }
    }
  }
}

}  // namespace
}  // namespace cunls
