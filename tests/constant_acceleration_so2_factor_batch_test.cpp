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
 * @file constant_acceleration_so2_factor_batch_test.cpp
 * @brief Unit tests for ConstantAccelerationSO2FactorBatch.
 */

#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {
namespace {

Matrix<2> RotationSO2(float theta) { return {cosf(theta), -sinf(theta), sinf(theta), cosf(theta)}; }

struct SingleFactorResult {
  std::vector<float> residuals;  // 3
  std::vector<float> jacobian;   // 3x6 (optional)
};

SingleFactorResult EvaluateOne(float theta_k, float theta_k1, float vel_k, float vel_k1,
                               float accel_k, float accel_k1, float dt, bool want_jacobian) {
  CudaStream stream;
  dvector<Matrix<2>> pose_k_dev({RotationSO2(theta_k)});
  dvector<Matrix<2>> pose_k1_dev({RotationSO2(theta_k1)});
  dvector<float> vel_k_dev(std::vector<float>{vel_k});
  dvector<float> vel_k1_dev(std::vector<float>{vel_k1});
  dvector<float> accel_k_dev(std::vector<float>{accel_k});
  dvector<float> accel_k1_dev(std::vector<float>{accel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     vel_k_dev.data(),
                                     vel_k1_dev.data(),
                                     accel_k_dev.data(),
                                     accel_k1_dev.data()};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantAccelerationSO2FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(3);
  dvector<float> jac_dev(want_jacobian ? 3 * 6 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(3);
  res_dev.CopyToHost(result.residuals.data(), 3);
  if (want_jacobian) {
    result.jacobian.resize(3 * 6);
    jac_dev.CopyToHost(result.jacobian.data(), 3 * 6);
  }
  return result;
}

TEST(ConstantAccelerationSO2FactorBatchTest, ZeroResidualForExactConstantAcceleration) {
  const float theta_k = 0.2f;
  const float vel_k = 0.4f;
  const float accel_k = 0.1f;
  const float dt = 0.5f;
  const float theta_k1 = theta_k + dt * vel_k + 0.5f * dt * dt * accel_k;
  const float vel_k1 = vel_k + dt * accel_k;

  auto result = EvaluateOne(theta_k, theta_k1, vel_k, vel_k1, accel_k, accel_k, dt, false);
  EXPECT_NEAR(result.residuals[0], 0.0f, 1e-4f);
  EXPECT_NEAR(result.residuals[1], 0.0f, 1e-4f);
  EXPECT_NEAR(result.residuals[2], 0.0f, 1e-4f);
}

TEST(ConstantAccelerationSO2FactorBatchTest, NumericalJacobianAllBlocks) {
  const float theta_k = 0.1f;
  const float theta_k1 = -0.2f;
  const float vel_k = 0.25f;
  const float vel_k1 = -0.15f;
  const float accel_k = 0.05f;
  const float accel_k1 = -0.08f;
  const float dt = 0.31f;

  auto base = EvaluateOne(theta_k, theta_k1, vel_k, vel_k1, accel_k, accel_k1, dt, true);
  constexpr float eps = 1e-4f;

  auto check = [&](auto perturb, int col) {
    auto plus = perturb(eps);
    auto minus = perturb(-eps);
    for (int row = 0; row < 3; row++) {
      float numerical = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
      float analytical = base.jacobian[row * 6 + col];
      EXPECT_NEAR(analytical, numerical, 5e-3f) << "row " << row << " col " << col;
    }
  };

  check(
      [&](float d) {
        return EvaluateOne(theta_k + d, theta_k1, vel_k, vel_k1, accel_k, accel_k1, dt, false);
      },
      0);
  check(
      [&](float d) {
        return EvaluateOne(theta_k, theta_k1 + d, vel_k, vel_k1, accel_k, accel_k1, dt, false);
      },
      1);
  check(
      [&](float d) {
        return EvaluateOne(theta_k, theta_k1, vel_k + d, vel_k1, accel_k, accel_k1, dt, false);
      },
      2);
  check(
      [&](float d) {
        return EvaluateOne(theta_k, theta_k1, vel_k, vel_k1 + d, accel_k, accel_k1, dt, false);
      },
      3);
  check(
      [&](float d) {
        return EvaluateOne(theta_k, theta_k1, vel_k, vel_k1, accel_k + d, accel_k1, dt, false);
      },
      4);
  check(
      [&](float d) {
        return EvaluateOne(theta_k, theta_k1, vel_k, vel_k1, accel_k, accel_k1 + d, dt, false);
      },
      5);
}

}  // namespace
}  // namespace cunls
