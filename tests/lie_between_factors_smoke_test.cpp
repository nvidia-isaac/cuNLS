/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file lie_between_factors_smoke_test.cpp
 * @brief Smoke tests: Evaluate() for SE(2), SO(2), SO(3), Sim(2), Sim(3), and
 *        vector between factors with identity relative measurements.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

namespace {

static Matrix<3> MakeSim2Matrix(float x, float y, float theta, float scale) {
  float c = cosf(theta);
  float s = sinf(theta);
  float inv_s = 1.0f / scale;
  return {c, -s, x, s, c, y, 0.0f, 0.0f, inv_s};
}

static Matrix<4> MakeSim3Identity() {
  Matrix<4> T{};
  T[0] = T[5] = T[10] = 1.0f;
  T[15] = 1.0f;
  return T;
}

}  // namespace

TEST(LieBetweenFactorsSmoke, SE2BetweenEvaluate) {
  constexpr size_t kN = 128;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> ang(-0.4f, 0.4f);
  std::uniform_real_distribution<float> t(-1.0f, 1.0f);

  std::vector<Matrix<3>> poses(kN + 1);
  for (size_t i = 0; i <= kN; ++i) {
    float th = ang(rng);
    poses[i] = {cosf(th), -sinf(th), t(rng), sinf(th), cosf(th), t(rng), 0.0f, 0.0f, 1.0f};
  }

  DeviceVector<Matrix<3>> poses_dev(poses);
  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Matrix<3> I{};
  I[0] = I[4] = I[8] = 1.0f;
  std::vector<Matrix<3>> deltas(kN, I);
  DeviceVector<Matrix<3>> deltas_dev(deltas);

  DeviceVector<float> res(3 * kN);
  DeviceVector<float> jac(18 * kN);
  SE2BetweenFactorBatch fb(reinterpret_cast<const SE2Transform *>(deltas_dev.data()), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  CudaStream stream;
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

TEST(LieBetweenFactorsSmoke, SO2BetweenEvaluate) {
  constexpr size_t kN = 128;
  std::mt19937 rng(8);
  std::uniform_real_distribution<float> ang(-0.5f, 0.5f);

  std::vector<Matrix<2>> poses(kN + 1);
  for (size_t i = 0; i <= kN; ++i) {
    float th = ang(rng);
    poses[i] = {cosf(th), -sinf(th), sinf(th), cosf(th)};
  }

  DeviceVector<Matrix<2>> poses_dev(poses);
  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Matrix<2> I{};
  I[0] = I[3] = 1.0f;
  std::vector<Matrix<2>> deltas(kN, I);
  DeviceVector<Matrix<2>> deltas_dev(deltas);

  DeviceVector<float> res(kN);
  DeviceVector<float> jac(2 * kN);
  SO2BetweenFactorBatch fb(reinterpret_cast<const SO2Rotation *>(deltas_dev.data()), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  CudaStream stream;
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

TEST(LieBetweenFactorsSmoke, SO3BetweenEvaluate) {
  constexpr size_t kN = 128;
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> w(-0.2f, 0.2f);

  hvector<Vector<3>> twists(kN + 1);
  for (size_t i = 0; i <= kN; ++i) {
    twists[i][0] = w(rng);
    twists[i][1] = w(rng);
    twists[i][2] = w(rng);
  }

  dvector<Vector<3>> twists_dev(twists);
  DeviceVector<Matrix<3>> poses_dev(kN + 1);
  CudaStream stream;
  constexpr size_t twist_stride = 3;
  constexpr size_t rot_pitch = 3;
  constexpr size_t rot_stride = 9;
  ComputeExpSO3(stream.GetStream(), reinterpret_cast<const float *>(twists_dev.data()),
                twist_stride, rot_pitch, rot_stride, kN + 1,
                reinterpret_cast<float *>(poses_dev.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Matrix<3> I{};
  I[0] = I[4] = I[8] = 1.0f;
  std::vector<Matrix<3>> deltas(kN, I);
  DeviceVector<Matrix<3>> deltas_dev(deltas);

  DeviceVector<float> res(3 * kN);
  DeviceVector<float> jac(18 * kN);
  SO3BetweenFactorBatch fb(reinterpret_cast<const SO3Rotation *>(deltas_dev.data()), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

TEST(LieBetweenFactorsSmoke, Similarity2BetweenEvaluate) {
  constexpr size_t kN = 128;
  std::mt19937 rng(10);
  std::uniform_real_distribution<float> ang(-0.3f, 0.3f);
  std::uniform_real_distribution<float> tr(-0.5f, 0.5f);
  std::uniform_real_distribution<float> sc(0.8f, 1.2f);

  std::vector<Matrix<3>> poses(kN + 1);
  for (size_t i = 0; i <= kN; ++i) {
    poses[i] = MakeSim2Matrix(tr(rng), tr(rng), ang(rng), sc(rng));
  }

  DeviceVector<Matrix<3>> poses_dev(poses);
  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Matrix<3> I = MakeSim2Matrix(0.0f, 0.0f, 0.0f, 1.0f);
  std::vector<Matrix<3>> deltas(kN, I);
  DeviceVector<Matrix<3>> deltas_dev(deltas);

  DeviceVector<float> res(4 * kN);
  DeviceVector<float> jac(32 * kN);
  Similarity2BetweenFactorBatch fb(
      reinterpret_cast<const Similarity2Transform *>(deltas_dev.data()), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  CudaStream stream;
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

TEST(LieBetweenFactorsSmoke, Similarity3BetweenEvaluate) {
  constexpr size_t kN = 128;
  std::vector<Matrix<4>> poses(kN + 1, MakeSim3Identity());
  DeviceVector<Matrix<4>> poses_dev(poses);
  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(poses_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(poses_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Matrix<4> I = MakeSim3Identity();
  std::vector<Matrix<4>> deltas(kN, I);
  DeviceVector<Matrix<4>> deltas_dev(deltas);

  DeviceVector<float> res(7 * kN);
  DeviceVector<float> jac(98 * kN);
  Similarity3BetweenFactorBatch fb(
      reinterpret_cast<const Similarity3Transform *>(deltas_dev.data()), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  CudaStream stream;
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

TEST(LieBetweenFactorsSmoke, VectorBetweenEvaluate) {
  constexpr size_t kN = 256;
  std::vector<Vector<3>> left(kN + 1);
  std::vector<Vector<3>> right(kN + 1);
  for (size_t i = 0; i <= kN; ++i) {
    left[i] = {static_cast<float>(i), 0.0f, 0.0f};
    right[i] = {static_cast<float>(i) + 0.1f, 0.0f, 0.0f};
  }
  DeviceVector<Vector<3>> left_dev(left);
  DeviceVector<Vector<3>> right_dev(right);

  std::vector<const float *> ptrs(2 * kN);
  for (size_t i = 0; i < kN; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float *>(left_dev.data() + (i + 1));
    ptrs[2 * i + 1] = reinterpret_cast<const float *>(right_dev.data() + i);
  }
  DeviceVector<const float *> state_ptrs_dev(ptrs);

  Vector<3> delta{};
  std::vector<Vector<3>> deltas(kN, delta);
  DeviceVector<Vector<3>> deltas_dev(deltas);

  DeviceVector<float> res(3 * kN);
  DeviceVector<float> jac(18 * kN);
  VectorBetweenFactorBatch<3> fb(deltas_dev.data(), kN);
  fb.SetNumActiveFactors(fb.Capacity());
  CudaStream stream;
  fb.Evaluate(res.data(), jac.data(), reinterpret_cast<const float *const *>(state_ptrs_dev.data()),
              stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

namespace {

/** Yaw of R_right after solving R_left = Rot(0.5) (constant), Delta = Rot(0.3). */
template <class Factor, class State, class Rotation, int D>
float SolvedYaw(const std::vector<float> &left, const std::vector<float> &delta,
                const std::vector<float> &right0) {
  std::vector<float> h(left);
  h.insert(h.end(), right0.begin(), right0.end());
  dvector<float> x(h), d(delta);
  dvector<int> const_ids(std::vector<int>{0});
  State states(x.data(), 2, const_ids.data(), 1);
  states.SetNumActiveStates(2, 1);
  Factor factor(reinterpret_cast<const Rotation *>(d.data()), 1);
  factor.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&factor, {states.StateDevicePtr(0), states.StateDevicePtr(1)});
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
  GaussNewtonMinimizer minimizer(options);
  CudaStream stream;
  minimizer.Minimize(stream.GetStream(), problem);
  std::vector<float> out(2 * D * D);
  x.CopyToHost(out.data(), out.size());
  const float *r = out.data() + D * D;
  return std::atan2(r[D], r[0]);  // R[1][0], R[0][0]
}

std::vector<float> Rot2(float a) { return {std::cos(a), -std::sin(a), std::sin(a), std::cos(a)}; }
std::vector<float> Rot3z(float a) {
  return {std::cos(a), -std::sin(a), 0.f, std::sin(a), std::cos(a), 0.f, 0.f, 0.f, 1.f};
}

}  // namespace

// Both rotation between factors use residual = Log(R_left^T R_right Delta):
// zero at R_right = R_left Delta^T (yaw 0.5 - 0.3 = 0.2).
TEST(LieBetweenFactorsSmoke, SO2AndSO3ShareTheResidualConvention) {
  const float so2 = SolvedYaw<SO2BetweenFactorBatch, SO2StateBatch, SO2Rotation, 2>(
      Rot2(0.5f), Rot2(0.3f), Rot2(0.f));
  const float so3 = SolvedYaw<SO3BetweenFactorBatch, SO3StateBatch, SO3Rotation, 3>(
      Rot3z(0.5f), Rot3z(0.3f), Rot3z(0.f));
  EXPECT_NEAR(so2, 0.2f, 1e-5f);
  EXPECT_NEAR(so3, 0.2f, 1e-5f);
}

}  // namespace cunls
