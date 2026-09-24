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
 * @file motion_prior_information_test.cpp
 * @brief Unit tests for ComputeConstantVelocitySqrtInformation /
 * ComputeConstantAccelerationSqrtInformation, and their use with
 * InformationFactorBatch to fuse the paper's closed-form Q(dt)^{-1} process
 * noise into the constant-velocity/-acceleration factor residual+Jacobian.
 */

#include "cunls/factor/motion_prior_information.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {
namespace {

// Reference (host, not GPU) M(dt) for WNOA (Eq. 15): 2x2, entries scaled by
// 1/qc_i applied per-DOF via Kronecker with diag(qc)^{-1}.
std::vector<float> ReferenceInformationCV(float dt, const std::vector<float> &qc, int dim) {
  const int n = 2 * dim;
  std::vector<float> info(n * n, 0.0f);
  const double m[2][2] = {{12.0 / (dt * dt * dt), -6.0 / (dt * dt)}, {-6.0 / (dt * dt), 4.0 / dt}};
  for (int p = 0; p < 2; p++)
    for (int q = 0; q < 2; q++)
      for (int i = 0; i < dim; i++)
        info[(p * dim + i) * n + (q * dim + i)] = static_cast<float>(m[p][q] / qc[i]);
  return info;
}

// Reference M(dt) for WNOJ (Eq. 32): 3x3.
std::vector<float> ReferenceInformationCA(float dt, const std::vector<float> &qc, int dim) {
  const int n = 3 * dim;
  std::vector<float> info(n * n, 0.0f);
  const double t = dt;
  const double m[3][3] = {
      {720.0 / (t * t * t * t * t), -360.0 / (t * t * t * t), 60.0 / (t * t * t)},
      {-360.0 / (t * t * t * t), 192.0 / (t * t * t), -36.0 / (t * t)},
      {60.0 / (t * t * t), -36.0 / (t * t), 9.0 / t}};
  for (int p = 0; p < 3; p++)
    for (int q = 0; q < 3; q++)
      for (int i = 0; i < dim; i++)
        info[(p * dim + i) * n + (q * dim + i)] = static_cast<float>(m[p][q] / qc[i]);
  return info;
}

// Computes S^T * S for an n x n row-major matrix S.
std::vector<float> GramMatrix(const std::vector<float> &s, int n) {
  std::vector<float> out(n * n, 0.0f);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      float acc = 0.0f;
      for (int k = 0; k < n; k++) acc += s[k * n + i] * s[k * n + j];  // S^T S
      out[i * n + j] = acc;
    }
  return out;
}

TEST(MotionPriorInformationTest, ConstantVelocitySqrtInformationMatchesClosedFormQInverse) {
  const int dim = 6;
  const int n = 2 * dim;
  std::vector<float> dt_values = {0.05f, 0.1f, 0.37f, 1.0f, 2.5f};
  std::vector<float> qc = {0.1f, 0.2f, 0.05f, 1.0f, 0.5f, 0.3f};

  dvector<float> dt_dev(dt_values);
  dvector<float> qc_dev(qc);
  dvector<float> sqrt_info_dev(dt_values.size() * n * n);

  CudaStream stream;
  ComputeConstantVelocitySqrtInformation<6>(stream.GetStream(), dt_dev.data(), qc_dev.data(),
                                            dt_values.size(), sqrt_info_dev.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> sqrt_info(dt_values.size() * n * n);
  sqrt_info_dev.CopyToHost(sqrt_info.data(), sqrt_info.size());

  for (size_t f = 0; f < dt_values.size(); f++) {
    std::vector<float> s(sqrt_info.begin() + f * n * n, sqrt_info.begin() + (f + 1) * n * n);
    auto gram = GramMatrix(s, n);
    auto reference = ReferenceInformationCV(dt_values[f], qc, dim);
    for (int i = 0; i < n * n; i++) {
      EXPECT_NEAR(gram[i], reference[i], std::abs(reference[i]) * 1e-3f + 1e-3f)
          << "factor " << f << " entry " << i;
    }
  }
}

TEST(MotionPriorInformationTest, ConstantAccelerationSqrtInformationMatchesClosedFormQInverse) {
  const int dim = 6;
  const int n = 3 * dim;
  std::vector<float> dt_values = {0.05f, 0.1f, 0.37f, 1.0f, 2.5f};
  std::vector<float> qc = {0.1f, 0.2f, 0.05f, 1.0f, 0.5f, 0.3f};

  dvector<float> dt_dev(dt_values);
  dvector<float> qc_dev(qc);
  dvector<float> sqrt_info_dev(dt_values.size() * n * n);

  CudaStream stream;
  ComputeConstantAccelerationSqrtInformation<6>(stream.GetStream(), dt_dev.data(), qc_dev.data(),
                                                dt_values.size(), sqrt_info_dev.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> sqrt_info(dt_values.size() * n * n);
  sqrt_info_dev.CopyToHost(sqrt_info.data(), sqrt_info.size());

  for (size_t f = 0; f < dt_values.size(); f++) {
    std::vector<float> s(sqrt_info.begin() + f * n * n, sqrt_info.begin() + (f + 1) * n * n);
    auto gram = GramMatrix(s, n);
    auto reference = ReferenceInformationCA(dt_values[f], qc, dim);
    for (int i = 0; i < n * n; i++) {
      EXPECT_NEAR(gram[i], reference[i], std::abs(reference[i]) * 1e-3f + 1e-2f)
          << "factor " << f << " entry " << i;
    }
  }
}

TEST(MotionPriorInformationTest, SmallerDimsMatchClosedFormQInverse) {
  for (int dim : {1, 3}) {
    std::vector<float> dt_values = {0.2f, 0.8f};
    std::vector<float> qc(dim, 0.4f);
    const int n_cv = 2 * dim;

    dvector<float> dt_dev(dt_values);
    dvector<float> qc_dev(qc);
    dvector<float> sqrt_info_dev(dt_values.size() * n_cv * n_cv);

    CudaStream stream;
    if (dim == 1) {
      ComputeConstantVelocitySqrtInformation<1>(stream.GetStream(), dt_dev.data(), qc_dev.data(),
                                                dt_values.size(), sqrt_info_dev.data());
    } else {
      ComputeConstantVelocitySqrtInformation<3>(stream.GetStream(), dt_dev.data(), qc_dev.data(),
                                                dt_values.size(), sqrt_info_dev.data());
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    std::vector<float> sqrt_info(dt_values.size() * n_cv * n_cv);
    sqrt_info_dev.CopyToHost(sqrt_info.data(), sqrt_info.size());

    for (size_t f = 0; f < dt_values.size(); f++) {
      std::vector<float> s(sqrt_info.begin() + f * n_cv * n_cv,
                           sqrt_info.begin() + (f + 1) * n_cv * n_cv);
      auto gram = GramMatrix(s, n_cv);
      auto reference = ReferenceInformationCV(dt_values[f], qc, dim);
      for (int i = 0; i < n_cv * n_cv; i++) {
        EXPECT_NEAR(gram[i], reference[i], std::abs(reference[i]) * 1e-3f + 1e-3f)
            << "dim " << dim << " factor " << f << " entry " << i;
      }
    }
  }
}

// End-to-end: InformationFactorBatch<ConstantVelocitySE3FactorBatch> must
// produce exactly S*r and S*J relative to the unwrapped factor's own output,
// which is precisely "fusing the covariance into the residual/Jacobian so
// the normal equations J'^T J' = J^T Information J come out correct".
TEST(MotionPriorInformationTest, InformationFactorBatchFusesWeightIntoResidualAndJacobian) {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans(-1.0f, 1.0f);

  SE3Transform pose_k, pose_k1;
  {
    Vector<6> tw0{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
    Vector<6> tw1{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
    CudaStream stream;
    dvector<Vector<6>> tw_dev({tw0, tw1});
    dvector<SE3Transform> pose_dev(2);
    ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(tw_dev.data()), 6, 4, 16, 2,
                  reinterpret_cast<float *>(pose_dev.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    std::vector<SE3Transform> poses(2);
    pose_dev.CopyToHost(poses.data(), 2);
    pose_k = poses[0];
    pose_k1 = poses[1];
  }
  Vector<6> vel_k{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
  Vector<6> vel_k1{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
  const float dt = 0.42f;
  std::vector<float> qc = {0.2f, 0.2f, 0.2f, 1.0f, 1.0f, 1.0f};

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

  CudaStream stream;

  // Unweighted evaluation.
  ConstantVelocitySE3FactorBatch unweighted(dt_dev.data(), 1);
  dvector<float> res_dev(12);
  dvector<float> jac_dev(12 * 24);
  unweighted.Evaluate(res_dev.data(), jac_dev.data(), ptrs_dev.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> r(12), j(12 * 24);
  res_dev.CopyToHost(r.data(), 12);
  jac_dev.CopyToHost(j.data(), 12 * 24);

  // sqrt-information for this single factor.
  dvector<float> qc_dev(qc);
  dvector<float> sqrt_info_dev(12 * 12);
  ComputeConstantVelocitySqrtInformation<6>(stream.GetStream(), dt_dev.data(), qc_dev.data(), 1,
                                            sqrt_info_dev.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> s(12 * 12);
  sqrt_info_dev.CopyToHost(s.data(), 12 * 12);

  // Weighted evaluation via InformationFactorBatch.
  cuBLASHandle cublas_handle;
  InformationFactorBatch<ConstantVelocitySE3FactorBatch> weighted(
      cublas_handle, reinterpret_cast<const Matrix<12> *>(sqrt_info_dev.data()), 1, dt_dev.data(),
      1);
  dvector<float> res_w_dev(12);
  dvector<float> jac_w_dev(12 * 24);
  weighted.Evaluate(res_w_dev.data(), jac_w_dev.data(), ptrs_dev.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> r_w(12), j_w(12 * 24);
  res_w_dev.CopyToHost(r_w.data(), 12);
  jac_w_dev.CopyToHost(j_w.data(), 12 * 24);

  // r_w should equal S * r, j_w should equal S * j (S row-major 12x12).
  for (int i = 0; i < 12; i++) {
    float expected_r = 0.0f;
    for (int k = 0; k < 12; k++) expected_r += s[i * 12 + k] * r[k];
    EXPECT_NEAR(r_w[i], expected_r, 1e-3f) << "residual row " << i;

    for (int col = 0; col < 24; col++) {
      float expected_j = 0.0f;
      for (int k = 0; k < 12; k++) expected_j += s[i * 12 + k] * j[k * 24 + col];
      EXPECT_NEAR(j_w[i * 24 + col], expected_j, 1e-2f) << "jacobian row " << i << " col " << col;
    }
  }

  // The convenience wrapper must produce byte-identical output to manually
  // computing sqrt-information and composing it with InformationFactorBatch
  // above: users should never have to do that composition themselves.
  dvector<float> qc_dev2(qc);
  cuBLASHandle cublas_handle2;
  CudaStream stream2;
  ConstantVelocityInformationSE3FactorBatch convenience(cublas_handle2, stream2.GetStream(),
                                                        dt_dev.data(), qc_dev2.data(), 1);
  dvector<float> res_c_dev(12);
  dvector<float> jac_c_dev(12 * 24);
  convenience.Evaluate(res_c_dev.data(), jac_c_dev.data(), ptrs_dev.data(), stream2.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream2.GetStream()));
  std::vector<float> r_c(12), j_c(12 * 24);
  res_c_dev.CopyToHost(r_c.data(), 12);
  jac_c_dev.CopyToHost(j_c.data(), 12 * 24);

  for (int i = 0; i < 12; i++) {
    EXPECT_NEAR(r_c[i], r_w[i], 1e-4f) << "convenience-wrapper residual row " << i;
  }
  for (int i = 0; i < 12 * 24; i++) {
    EXPECT_NEAR(j_c[i], j_w[i], 1e-4f) << "convenience-wrapper jacobian entry " << i;
  }
}

// Smoke test: every named alias constructs and evaluates without error, on
// a single trivial (identity pose, zero velocity/acceleration) factor.
TEST(MotionPriorInformationTest, AllConvenienceAliasesConstructAndEvaluate) {
  CudaStream stream;
  cuBLASHandle cublas_handle;
  dvector<float> dt_dev(std::vector<float>{0.1f});

  {
    std::vector<float> qc = {1, 1, 1, 1, 1, 1};
    dvector<float> qc_dev(qc);
    ConstantVelocityInformationSE3FactorBatch fb(cublas_handle, stream.GetStream(), dt_dev.data(),
                                                 qc_dev.data(), 1);
    dvector<float> res(12), jac(12 * 24);
    SE3Transform identity{};
    identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
    dvector<SE3Transform> pose_dev({identity, identity});
    dvector<Vector<6>> vel_dev({Vector<6>{}, Vector<6>{}});
    std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_dev.data()),
                                       reinterpret_cast<const float *>(pose_dev.data() + 1),
                                       reinterpret_cast<const float *>(vel_dev.data()),
                                       reinterpret_cast<const float *>(vel_dev.data() + 1)};
    dvector<const float *> ptrs_dev(ptrs);
    ASSERT_TRUE(fb.Evaluate(res.data(), jac.data(), ptrs_dev.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
  {
    std::vector<float> qc = {1, 1, 1, 1, 1, 1};
    dvector<float> qc_dev(qc);
    ConstantAccelerationInformationSE3FactorBatch fb(cublas_handle, stream.GetStream(),
                                                     dt_dev.data(), qc_dev.data(), 1);
    dvector<float> res(18), jac(18 * 36);
    SE3Transform identity{};
    identity[0] = identity[5] = identity[10] = identity[15] = 1.0f;
    dvector<SE3Transform> pose_dev({identity, identity});
    dvector<Vector<6>> zero_dev({Vector<6>{}, Vector<6>{}, Vector<6>{}, Vector<6>{}});
    std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_dev.data()),
                                       reinterpret_cast<const float *>(pose_dev.data() + 1),
                                       reinterpret_cast<const float *>(zero_dev.data()),
                                       reinterpret_cast<const float *>(zero_dev.data() + 1),
                                       reinterpret_cast<const float *>(zero_dev.data() + 2),
                                       reinterpret_cast<const float *>(zero_dev.data() + 3)};
    dvector<const float *> ptrs_dev(ptrs);
    ASSERT_TRUE(fb.Evaluate(res.data(), jac.data(), ptrs_dev.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
  {
    std::vector<float> qc = {1};
    dvector<float> qc_dev(qc);
    ConstantVelocityInformationSO2FactorBatch fb(cublas_handle, stream.GetStream(), dt_dev.data(),
                                                 qc_dev.data(), 1);
    dvector<float> res(2), jac(2 * 4);
    Matrix<2> identity{1, 0, 0, 1};
    dvector<Matrix<2>> pose_dev({identity, identity});
    dvector<float> zero_dev(std::vector<float>{0.0f, 0.0f});
    std::vector<const float *> ptrs = {reinterpret_cast<const float *>(pose_dev.data()),
                                       reinterpret_cast<const float *>(pose_dev.data() + 1),
                                       zero_dev.data(), zero_dev.data() + 1};
    dvector<const float *> ptrs_dev(ptrs);
    ASSERT_TRUE(fb.Evaluate(res.data(), jac.data(), ptrs_dev.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }
}

}  // namespace
}  // namespace cunls
