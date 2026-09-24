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
 * @file constant_acceleration_se2_factor_batch_test.cpp
 * @brief Unit tests for ConstantAccelerationSE2FactorBatch.
 */

#include "cunls/factor/constant_acceleration_se2_factor_batch.h"

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

Matrix<3> ExpSE2(const Vector<3> &twist) {
  CudaStream stream;
  dvector<Vector<3>> twist_dev({twist});
  dvector<Matrix<3>> out_dev(1);
  ComputeExpSE2(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()), 3, 9, 1,
                reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Matrix<3> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Matrix<3> ComposeSE2(const Matrix<3> &a, const Matrix<3> &b) {
  Matrix<3> out{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 3; k++) acc += a[i * 3 + k] * b[k * 3 + j];
      out[i * 3 + j] = acc;
    }
  return out;
}

struct SingleFactorResult {
  std::vector<float> residuals;  // 9
  std::vector<float> jacobian;   // 9x18 (optional)
};

SingleFactorResult EvaluateOne(const Matrix<3> &pose_k, const Matrix<3> &pose_k1,
                               const Vector<3> &vel_k, const Vector<3> &vel_k1,
                               const Vector<3> &accel_k, const Vector<3> &accel_k1, float dt,
                               bool want_jacobian) {
  CudaStream stream;
  dvector<Matrix<3>> pose_k_dev({pose_k});
  dvector<Matrix<3>> pose_k1_dev({pose_k1});
  dvector<Vector<3>> vel_k_dev({vel_k});
  dvector<Vector<3>> vel_k1_dev({vel_k1});
  dvector<Vector<3>> accel_k_dev({accel_k});
  dvector<Vector<3>> accel_k1_dev({accel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     reinterpret_cast<const float *>(vel_k_dev.data()),
                                     reinterpret_cast<const float *>(vel_k1_dev.data()),
                                     reinterpret_cast<const float *>(accel_k_dev.data()),
                                     reinterpret_cast<const float *>(accel_k1_dev.data())};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantAccelerationSE2FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(9);
  dvector<float> jac_dev(want_jacobian ? 9 * 18 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(9);
  res_dev.CopyToHost(result.residuals.data(), 9);
  if (want_jacobian) {
    result.jacobian.resize(9 * 18);
    jac_dev.CopyToHost(result.jacobian.data(), 9 * 18);
  }
  return result;
}

class ConstantAccelerationSE2FactorBatchTest : public ::testing::Test {
 protected:
  std::mt19937 rng_{144};
};

TEST_F(ConstantAccelerationSE2FactorBatchTest, ZeroResidualPoseBlockForExactPrediction) {
  std::uniform_real_distribution<float> w(-0.3f, 0.3f);
  Vector<3> vel_k{w(rng_), w(rng_), w(rng_)};
  Vector<3> accel_k{w(rng_) * 0.5f, w(rng_) * 0.5f, w(rng_) * 0.5f};
  const float dt = 0.24f;

  Matrix<3> pose_k = ExpSE2({0.15f, -0.1f, 0.05f});
  Vector<3> step_twist;
  for (int i = 0; i < 3; i++) step_twist[i] = dt * vel_k[i] + 0.5f * dt * dt * accel_k[i];
  Matrix<3> pose_k1 = ComposeSE2(pose_k, ExpSE2(step_twist));

  // As in ConstantVelocitySE2FactorBatchTest, only the pose-block residual
  // is checked directly (no closed-form SE(2) left-Jacobian helper is used
  // in this test); the velocity/acceleration blocks' exact linear structure
  // is verified by the numerical-Jacobian test below.
  auto result = EvaluateOne(pose_k, pose_k1, vel_k, vel_k, accel_k, accel_k, dt, false);
  for (int i = 0; i < 3; i++) EXPECT_NEAR(result.residuals[i], 0.0f, 1e-3f) << "r_pose index " << i;
}

TEST_F(ConstantAccelerationSE2FactorBatchTest, NumericalJacobianVelAccelBlocks) {
  std::uniform_real_distribution<float> w(-0.25f, 0.25f);
  Matrix<3> pose_k = ExpSE2({w(rng_), w(rng_), w(rng_)});
  Matrix<3> pose_k1 = ExpSE2({w(rng_), w(rng_), w(rng_)});
  Vector<3> vel_k{w(rng_), w(rng_), w(rng_)};
  Vector<3> vel_k1{w(rng_), w(rng_), w(rng_)};
  Vector<3> accel_k{w(rng_), w(rng_), w(rng_)};
  Vector<3> accel_k1{w(rng_), w(rng_), w(rng_)};
  const float dt = 0.26f;

  auto base = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, accel_k, accel_k1, dt, true);

  constexpr float eps = 1e-4f;
  constexpr int kDim = 3;
  constexpr int kCols = 18;

  for (int block = 2; block < 6; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<3> v_plus[4] = {vel_k, vel_k1, accel_k, accel_k1};
      Vector<3> v_minus[4] = {vel_k, vel_k1, accel_k, accel_k1};
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
