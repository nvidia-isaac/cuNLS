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
 * @file constant_velocity_so3_factor_batch_test.cpp
 * @brief Unit tests for ConstantVelocitySO3FactorBatch.
 */

#include "cunls/factor/constant_velocity_so3_factor_batch.h"

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

Matrix<3> ExpSO3(const Vector<3> &twist) {
  CudaStream stream;
  dvector<Vector<3>> twist_dev({twist});
  dvector<Matrix<3>> out_dev(1);
  ComputeExpSO3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()),
                /*twist_stride=*/3, /*rotation_pitch=*/3, /*rotation_stride=*/9,
                /*size=*/1, reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Matrix<3> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Matrix<3> ComposeSO3(const Matrix<3> &a, const Matrix<3> &b) {
  Matrix<3> out{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      float acc = 0.0f;
      for (int k = 0; k < 3; k++) acc += a[i * 3 + k] * b[k * 3 + j];
      out[i * 3 + j] = acc;
    }
  return out;
}

Matrix<3> JacobianLeftSO3(const Vector<3> &twist) {
  CudaStream stream;
  dvector<Vector<3>> twist_dev({twist});
  dvector<Matrix<3>> out_dev(1);
  ComputeJacobianLeftSO3(stream.GetStream(), reinterpret_cast<const float *>(twist_dev.data()),
                         /*twist_stride=*/3, /*jacobian_pitch=*/3,
                         /*jacobian_stride=*/9, /*size=*/1,
                         reinterpret_cast<float *>(out_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  Matrix<3> out;
  out_dev.CopyToHost(&out, 1);
  return out;
}

Vector<3> MatVec3(const Matrix<3> &m, const Vector<3> &v) {
  Vector<3> out{};
  for (int i = 0; i < 3; i++) {
    float acc = 0.0f;
    for (int j = 0; j < 3; j++) acc += m[i * 3 + j] * v[j];
    out[i] = acc;
  }
  return out;
}

struct SingleFactorResult {
  std::vector<float> residuals;  // 6
  std::vector<float> jacobian;   // 6x12, row-major (optional)
};

SingleFactorResult EvaluateOne(const Matrix<3> &pose_k, const Matrix<3> &pose_k1,
                               const Vector<3> &vel_k, const Vector<3> &vel_k1, float dt,
                               bool want_jacobian) {
  CudaStream stream;
  dvector<Matrix<3>> pose_k_dev({pose_k});
  dvector<Matrix<3>> pose_k1_dev({pose_k1});
  dvector<Vector<3>> vel_k_dev({vel_k});
  dvector<Vector<3>> vel_k1_dev({vel_k1});
  dvector<float> dt_dev(std::vector<float>{dt});

  std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_k_dev.data()),
                                     reinterpret_cast<const float *>(pose_k1_dev.data()),
                                     reinterpret_cast<const float *>(vel_k_dev.data()),
                                     reinterpret_cast<const float *>(vel_k1_dev.data())};
  dvector<const float *> ptrs_dev(ptrs);

  ConstantVelocitySO3FactorBatch fb(dt_dev.data(), 1);
  dvector<float> res_dev(6);
  dvector<float> jac_dev(want_jacobian ? 6 * 12 : 0);

  fb.Evaluate(res_dev.data(), want_jacobian ? jac_dev.data() : nullptr, ptrs_dev.data(),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SingleFactorResult result;
  result.residuals.resize(6);
  res_dev.CopyToHost(result.residuals.data(), 6);
  if (want_jacobian) {
    result.jacobian.resize(6 * 12);
    jac_dev.CopyToHost(result.jacobian.data(), 6 * 12);
  }
  return result;
}

class ConstantVelocitySO3FactorBatchTest : public ::testing::Test {
 protected:
  std::mt19937 rng_{43};
};

TEST_F(ConstantVelocitySO3FactorBatchTest, ZeroResidualForExactConstantVelocity) {
  std::uniform_real_distribution<float> w(-0.4f, 0.4f);
  Vector<3> vel_k{w(rng_), w(rng_), w(rng_)};
  const float dt = 0.25f;

  Matrix<3> pose_k = ExpSO3({0.1f, -0.2f, 0.05f});
  Vector<3> step_twist{dt * vel_k[0], dt * vel_k[1], dt * vel_k[2]};
  Matrix<3> pose_k1 = ComposeSO3(pose_k, ExpSO3(step_twist));

  Matrix<3> jl = JacobianLeftSO3(step_twist);
  Vector<3> vel_k1 = MatVec3(jl, vel_k);

  auto result = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, dt, false);
  for (int i = 0; i < 6; i++)
    EXPECT_NEAR(result.residuals[i], 0.0f, 1e-3f) << "residual index " << i;
}

TEST_F(ConstantVelocitySO3FactorBatchTest, NumericalJacobianVelocityBlocks) {
  std::uniform_real_distribution<float> w(-0.3f, 0.3f);
  Matrix<3> pose_k = ExpSO3({w(rng_), w(rng_), w(rng_)});
  Matrix<3> pose_k1 = ExpSO3({w(rng_), w(rng_), w(rng_)});
  Vector<3> vel_k{w(rng_), w(rng_), w(rng_)};
  Vector<3> vel_k1{w(rng_), w(rng_), w(rng_)};
  const float dt = 0.31f;

  auto base = EvaluateOne(pose_k, pose_k1, vel_k, vel_k1, dt, true);

  constexpr float eps = 1e-4f;
  constexpr int kDim = 3;
  constexpr int kCols = 12;

  // d(r_pose)/d(pose_k), d(r_pose)/d(pose_k+1): exact.
  for (int block = 0; block < 2; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<3> delta{};
      delta[k] = eps;
      Matrix<3> pk_plus = (block == 0) ? ComposeSO3(pose_k, ExpSO3(delta)) : pose_k;
      Matrix<3> pk1_plus = (block == 1) ? ComposeSO3(pose_k1, ExpSO3(delta)) : pose_k1;
      delta[k] = -eps;
      Matrix<3> pk_minus = (block == 0) ? ComposeSO3(pose_k, ExpSO3(delta)) : pose_k;
      Matrix<3> pk1_minus = (block == 1) ? ComposeSO3(pose_k1, ExpSO3(delta)) : pose_k1;

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

  // d(r_pose)/d(vel_k)=-dt*I, d(r_pose)/d(vel_k1)=0; d(r_vel)/d(vel_k)=-I,
  // d(r_vel)/d(vel_k1)=Jl_inv: all exact.
  for (int block = 0; block < 2; block++) {
    for (int k = 0; k < kDim; k++) {
      Vector<3> vk_plus = vel_k, vk1_plus = vel_k1;
      Vector<3> vk_minus = vel_k, vk1_minus = vel_k1;
      (block == 0 ? vk_plus : vk1_plus)[k] += eps;
      (block == 0 ? vk_minus : vk1_minus)[k] -= eps;

      auto plus = EvaluateOne(pose_k, pose_k1, vk_plus, vk1_plus, dt, false);
      auto minus = EvaluateOne(pose_k, pose_k1, vk_minus, vk1_minus, dt, false);
      for (int row = 0; row < kDim; row++) {
        float numerical_pose = (plus.residuals[row] - minus.residuals[row]) / (2.0f * eps);
        float analytical_pose = base.jacobian[row * kCols + (2 + block) * kDim + k];
        EXPECT_NEAR(analytical_pose, numerical_pose, 5e-3f)
            << "r_pose/vel block " << block << " row " << row << " col " << k;

        float numerical_vel =
            (plus.residuals[kDim + row] - minus.residuals[kDim + row]) / (2.0f * eps);
        float analytical_vel = base.jacobian[(kDim + row) * kCols + (2 + block) * kDim + k];
        EXPECT_NEAR(analytical_vel, numerical_vel, 5e-3f)
            << "r_vel/vel block " << block << " row " << row << " col " << k;
      }
    }
  }
}

}  // namespace
}  // namespace cunls
