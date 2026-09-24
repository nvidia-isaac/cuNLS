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
 * @file constant_velocity_so2_factor_batch_test.cpp
 * @brief Unit tests for ConstantVelocitySO2FactorBatch.
 */

#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {
namespace {

Matrix<2> RotationSO2(float theta) { return {cosf(theta), -sinf(theta), sinf(theta), cosf(theta)}; }

struct SingleFactorResult {
  std::vector<float> residuals;  // 2
  std::vector<float> jacobian;   // 2x4 (optional)
};

SingleFactorResult EvaluateOne(float theta_k, float theta_k1, float vel_k, float vel_k1, float dt,
                               bool want_jacobian) {
  CudaStream stream;
  dvector<Matrix<2>> pose_k_dev({RotationSO2(theta_k)});
  dvector<Matrix<2>> pose_k1_dev({RotationSO2(theta_k1)});
  dvector<float> vel_k_dev(std::vector<float>{vel_k});
  dvector<float> vel_k1_dev(std::vector<float>{vel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     vel_k_dev.data(), vel_k1_dev.data()};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantVelocitySO2FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(2);
  dvector<float> jac_dev(want_jacobian ? 2 * 4 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(2);
  res_dev.CopyToHost(result.residuals.data(), 2);
  if (want_jacobian) {
    result.jacobian.resize(2 * 4);
    jac_dev.CopyToHost(result.jacobian.data(), 2 * 4);
  }
  return result;
}

TEST(ConstantVelocitySO2FactorBatchTest, ZeroResidualForExactConstantVelocity) {
  const float theta_k = 0.3f;
  const float vel_k = 0.5f;
  const float dt = 0.4f;
  const float theta_k1 = theta_k + dt * vel_k;

  auto result = EvaluateOne(theta_k, theta_k1, vel_k, vel_k, dt, false);
  EXPECT_NEAR(result.residuals[0], 0.0f, 1e-4f);
  EXPECT_NEAR(result.residuals[1], 0.0f, 1e-4f);
}

TEST(ConstantVelocitySO2FactorBatchTest, NumericalJacobianAllBlocks) {
  const float theta_k = 0.2f;
  const float theta_k1 = -0.15f;
  const float vel_k = 0.3f;
  const float vel_k1 = -0.1f;
  const float dt = 0.33f;

  auto base = EvaluateOne(theta_k, theta_k1, vel_k, vel_k1, dt, true);
  constexpr float eps = 1e-4f;

  auto check = [&](auto perturb, int col) {
    auto plus = perturb(eps);
    auto minus = perturb(-eps);
    for (int row = 0; row < 2; row++) {
      float numerical = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
      float analytical = base.jacobian[row * 4 + col];
      EXPECT_NEAR(analytical, numerical, 5e-3f) << "row " << row << " col " << col;
    }
  };

  check([&](float d) { return EvaluateOne(theta_k + d, theta_k1, vel_k, vel_k1, dt, false); }, 0);
  check([&](float d) { return EvaluateOne(theta_k, theta_k1 + d, vel_k, vel_k1, dt, false); }, 1);
  check([&](float d) { return EvaluateOne(theta_k, theta_k1, vel_k + d, vel_k1, dt, false); }, 2);
  check([&](float d) { return EvaluateOne(theta_k, theta_k1, vel_k, vel_k1 + d, dt, false); }, 3);
}

}  // namespace
}  // namespace cunls
