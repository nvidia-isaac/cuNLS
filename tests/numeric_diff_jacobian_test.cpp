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
 * @file numeric_diff_jacobian_test.cpp
 * @brief Compares NumericDiffJacobianBuilder's finite-difference Jacobians
 * against each factor's analytic Jacobian, for factors spanning Euclidean,
 * SO(3), and SE(3) manifolds.
 */

#include "cunls/minimizer/numeric_diff_jacobian.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/minimizer_state.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr uint32_t kSeed = 12345u;

/**
 * @brief Compares two dense per-factor Jacobian buffers element-wise.
 *
 * Uses a combined relative/absolute tolerance since numeric-diff Jacobians
 * are computed in float32 and central differencing amplifies rounding error
 * relative to the analytic values.
 */
void ExpectJacobiansClose(const std::vector<float> &analytic, const std::vector<float> &numeric,
                          float rel_tol, float abs_tol) {
  ASSERT_EQ(analytic.size(), numeric.size());
  for (size_t i = 0; i < analytic.size(); i++) {
    float a = analytic[i];
    float n = numeric[i];
    float tol = abs_tol + rel_tol * std::max(std::fabs(a), std::fabs(n));
    EXPECT_NEAR(a, n, tol) << "Mismatch at flattened index " << i << " (analytic=" << a
                           << ", numeric=" << n << ")";
  }
}

/** @brief Central-diff tolerance used by every test in this file. */
constexpr float kRelTol = 5e-2f;
constexpr float kAbsTol = 5e-3f;

TEST(NumericDiffJacobianTest, VectorBetweenMatchesAnalytic) {
  constexpr int kDim = 3;
  constexpr size_t kNumStates = 6;
  constexpr size_t kNumFactors = kNumStates - 1;

  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<Vector<kDim>> states_host(kNumStates);
  for (auto &v : states_host) {
    for (int i = 0; i < kDim; i++) v[i] = dist(rng);
  }
  std::vector<Vector<kDim>> deltas_host(kNumFactors);
  for (auto &v : deltas_host) {
    for (int i = 0; i < kDim; i++) v[i] = dist(rng);
  }

  DeviceVector<Vector<kDim>> states_device(states_host);
  DeviceVector<Vector<kDim>> deltas_device(deltas_host);

  VectorStateBatch<kDim> state_batch(reinterpret_cast<const float *>(states_device.data()),
                                     kNumStates);
  VectorBetweenFactorBatch<kDim> factor_batch(deltas_device.data(), kNumFactors);

  std::vector<float *> state_pointers(kNumFactors * 2);
  for (size_t f = 0; f < kNumFactors; f++) {
    state_pointers[2 * f + 0] = state_batch.StateBlockDevicePtr(f);
    state_pointers[2 * f + 1] = state_batch.StateBlockDevicePtr(f + 1);
  }

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerState ms(stream.GetStream(), problem);

  const size_t residual_size = factor_batch.ResidualsSize();
  const size_t total_cols = kDim + kDim;
  const size_t jac_floats = kNumFactors * residual_size * total_cols;

  dvector<float> residuals_analytic(kNumFactors * residual_size);
  dvector<float> jacobian_analytic(jac_floats);
  auto ptrs = ms.GetStatePointers()[0].data();
  factor_batch.Evaluate(residuals_analytic.data(), jacobian_analytic.data(), ptrs,
                        stream.GetStream());

  dvector<float> residuals_baseline(kNumFactors * residual_size);
  factor_batch.Evaluate(residuals_baseline.data(), nullptr, ptrs, stream.GetStream());

  dvector<float> jacobian_numeric(jac_floats);
  NumericDiffJacobianBuilder builder;
  NumericDiffOptions options;
  builder.Compute(stream.GetStream(), problem, 0, ms, residuals_baseline.data(),
                  jacobian_numeric.data(), options);

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> jac_an_host(jac_floats), jac_num_host(jac_floats);
  jacobian_analytic.CopyToHost(jac_an_host.data(), jac_floats);
  jacobian_numeric.CopyToHost(jac_num_host.data(), jac_floats);

  ExpectJacobiansClose(jac_an_host, jac_num_host, kRelTol, kAbsTol);
}

// SO3BetweenFactorBatch: residual = Log(R_left^T * R_right * Delta^T). Uses
// non-identity deltas (identity deltas make the left/right analytic-vs-Ad
// ordering bug invisible, since Ad(I) = I). This regression-tests the fix to
// so3_between_fused_jacobians_kernel (see so3_between_factor_batch.cu):
// previously the left block was `-Ad(Delta) * J_l^{-1}(r)` and the right
// block was `J_r^{-1}(r)` (no Delta factor at all), which does not match the
// residual's actual dependence on the SO3StateBatch::Plus right-multiplicative
// retraction. Fixed to left = `-J_l^{-1}(r)`, right = `J_r^{-1}(r) * Delta`.
TEST(NumericDiffJacobianTest, SO3BetweenMatchesAnalytic) {
  constexpr size_t kNumFactors = 5;
  constexpr size_t kNumStates = kNumFactors + 1;

  std::mt19937 rng(kSeed + 1);
  std::uniform_real_distribution<float> rot_dist(-0.5f, 0.5f);

  CudaStream stream;
  cuBLASHandle cublas;

  hvector<Vector<3>> state_twists(kNumStates);
  for (auto &t : state_twists) {
    for (int i = 0; i < 3; i++) t[i] = rot_dist(rng);
  }
  dvector<Vector<3>> state_twists_d(state_twists);
  dvector<SO3Rotation> states_d(kNumStates);
  ComputeExpSO3(stream.GetStream(), reinterpret_cast<const float *>(state_twists_d.data()), 3, 3, 9,
                kNumStates, reinterpret_cast<float *>(states_d.data()));

  hvector<Vector<3>> delta_twists(kNumFactors);
  for (auto &t : delta_twists) {
    for (int i = 0; i < 3; i++) t[i] = rot_dist(rng);
  }
  dvector<Vector<3>> delta_twists_d(delta_twists);
  dvector<SO3Rotation> deltas_d(kNumFactors);
  ComputeExpSO3(stream.GetStream(), reinterpret_cast<const float *>(delta_twists_d.data()), 3, 3, 9,
                kNumFactors, reinterpret_cast<float *>(deltas_d.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SO3StateBatch state_batch(cublas, reinterpret_cast<const float *>(states_d.data()), kNumStates);
  SO3BetweenFactorBatch factor_batch(deltas_d.data(), kNumFactors);

  std::vector<float *> state_pointers(kNumFactors * 2);
  for (size_t f = 0; f < kNumFactors; f++) {
    state_pointers[2 * f + 0] = state_batch.StateBlockDevicePtr(f);
    state_pointers[2 * f + 1] = state_batch.StateBlockDevicePtr(f + 1);
  }

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerState ms(stream.GetStream(), problem);

  const size_t residual_size = factor_batch.ResidualsSize();
  const size_t total_cols = 3 + 3;
  const size_t jac_floats = kNumFactors * residual_size * total_cols;

  dvector<float> residuals_analytic(kNumFactors * residual_size);
  dvector<float> jacobian_analytic(jac_floats);
  auto ptrs = ms.GetStatePointers()[0].data();
  factor_batch.Evaluate(residuals_analytic.data(), jacobian_analytic.data(), ptrs,
                        stream.GetStream());

  dvector<float> residuals_baseline(kNumFactors * residual_size);
  factor_batch.Evaluate(residuals_baseline.data(), nullptr, ptrs, stream.GetStream());

  dvector<float> jacobian_numeric(jac_floats);
  NumericDiffJacobianBuilder builder;
  NumericDiffOptions options;
  builder.Compute(stream.GetStream(), problem, 0, ms, residuals_baseline.data(),
                  jacobian_numeric.data(), options);

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> jac_an_host(jac_floats), jac_num_host(jac_floats);
  jacobian_analytic.CopyToHost(jac_an_host.data(), jac_floats);
  jacobian_numeric.CopyToHost(jac_num_host.data(), jac_floats);

  ExpectJacobiansClose(jac_an_host, jac_num_host, kRelTol, kAbsTol);
}

// SE3BetweenFactorBatch: residual = Log(Delta * T_left^{-1} * T_right). Uses
// non-identity deltas (see the comment on SO3BetweenMatchesAnalytic above for
// why identity deltas would hide the bug). This regression-tests the fix to
// se3_between_fused_jacobians_kernel (see se3_between_factor_batch.cu):
// previously the left block multiplied Ad(Delta) and J_l^{-1}(twist) in the
// wrong order (`-Ad(Delta) * J_l^{-1}(twist)`) instead of the correct
// `-J_l^{-1}(twist) * Ad(Delta)`.
TEST(NumericDiffJacobianTest, SE3BetweenMatchesAnalytic) {
  constexpr size_t kNumFactors = 5;
  constexpr size_t kNumStates = kNumFactors + 1;

  std::mt19937 rng(kSeed + 2);
  std::uniform_real_distribution<float> rot_dist(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans_dist(-2.0f, 2.0f);

  CudaStream stream;
  cuBLASHandle cublas;

  auto make_twists = [&](size_t n) {
    hvector<Vector<6>> twists(n);
    for (auto &t : twists) {
      t[0] = rot_dist(rng);
      t[1] = rot_dist(rng);
      t[2] = rot_dist(rng);
      t[3] = trans_dist(rng);
      t[4] = trans_dist(rng);
      t[5] = trans_dist(rng);
    }
    return twists;
  };

  hvector<Vector<6>> state_twists = make_twists(kNumStates);
  dvector<Vector<6>> state_twists_d(state_twists);
  dvector<SE3Transform> states_d(kNumStates);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(state_twists_d.data()), 6, 4,
                16, kNumStates, reinterpret_cast<float *>(states_d.data()));

  hvector<Vector<6>> delta_twists = make_twists(kNumFactors);
  dvector<Vector<6>> delta_twists_d(delta_twists);
  dvector<SE3Transform> deltas_d(kNumFactors);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(delta_twists_d.data()), 6, 4,
                16, kNumFactors, reinterpret_cast<float *>(deltas_d.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  SE3StateBatch state_batch(cublas, reinterpret_cast<const float *>(states_d.data()), kNumStates);
  SE3BetweenFactorBatch factor_batch(deltas_d.data(), kNumFactors);

  std::vector<float *> state_pointers(kNumFactors * 2);
  for (size_t f = 0; f < kNumFactors; f++) {
    state_pointers[2 * f + 0] = state_batch.StateBlockDevicePtr(f);
    state_pointers[2 * f + 1] = state_batch.StateBlockDevicePtr(f + 1);
  }

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerState ms(stream.GetStream(), problem);

  const size_t residual_size = factor_batch.ResidualsSize();
  const size_t total_cols = 6 + 6;
  const size_t jac_floats = kNumFactors * residual_size * total_cols;

  dvector<float> residuals_analytic(kNumFactors * residual_size);
  dvector<float> jacobian_analytic(jac_floats);
  auto ptrs = ms.GetStatePointers()[0].data();
  factor_batch.Evaluate(residuals_analytic.data(), jacobian_analytic.data(), ptrs,
                        stream.GetStream());

  dvector<float> residuals_baseline(kNumFactors * residual_size);
  factor_batch.Evaluate(residuals_baseline.data(), nullptr, ptrs, stream.GetStream());

  dvector<float> jacobian_numeric(jac_floats);
  NumericDiffJacobianBuilder builder;
  NumericDiffOptions options;
  builder.Compute(stream.GetStream(), problem, 0, ms, residuals_baseline.data(),
                  jacobian_numeric.data(), options);

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> jac_an_host(jac_floats), jac_num_host(jac_floats);
  jacobian_analytic.CopyToHost(jac_an_host.data(), jac_floats);
  jacobian_numeric.CopyToHost(jac_num_host.data(), jac_floats);

  ExpectJacobiansClose(jac_an_host, jac_num_host, kRelTol, kAbsTol);
}

TEST(NumericDiffJacobianTest, ForwardDiffMatchesAnalytic) {
  // Sanity check for the forward-difference path (kForward), on the same
  // Euclidean between-factor setup as VectorBetweenMatchesAnalytic, with a
  // looser tolerance since forward differencing is first-order accurate.
  constexpr int kDim = 3;
  constexpr size_t kNumStates = 6;
  constexpr size_t kNumFactors = kNumStates - 1;

  std::mt19937 rng(kSeed + 3);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<Vector<kDim>> states_host(kNumStates);
  for (auto &v : states_host) {
    for (int i = 0; i < kDim; i++) v[i] = dist(rng);
  }
  std::vector<Vector<kDim>> deltas_host(kNumFactors);
  for (auto &v : deltas_host) {
    for (int i = 0; i < kDim; i++) v[i] = dist(rng);
  }

  DeviceVector<Vector<kDim>> states_device(states_host);
  DeviceVector<Vector<kDim>> deltas_device(deltas_host);

  VectorStateBatch<kDim> state_batch(reinterpret_cast<const float *>(states_device.data()),
                                     kNumStates);
  VectorBetweenFactorBatch<kDim> factor_batch(deltas_device.data(), kNumFactors);

  std::vector<float *> state_pointers(kNumFactors * 2);
  for (size_t f = 0; f < kNumFactors; f++) {
    state_pointers[2 * f + 0] = state_batch.StateBlockDevicePtr(f);
    state_pointers[2 * f + 1] = state_batch.StateBlockDevicePtr(f + 1);
  }

  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerState ms(stream.GetStream(), problem);

  const size_t residual_size = factor_batch.ResidualsSize();
  const size_t total_cols = kDim + kDim;
  const size_t jac_floats = kNumFactors * residual_size * total_cols;

  dvector<float> jacobian_analytic(jac_floats);
  dvector<float> residuals_analytic(kNumFactors * residual_size);
  auto ptrs = ms.GetStatePointers()[0].data();
  factor_batch.Evaluate(residuals_analytic.data(), jacobian_analytic.data(), ptrs,
                        stream.GetStream());

  dvector<float> residuals_baseline(kNumFactors * residual_size);
  factor_batch.Evaluate(residuals_baseline.data(), nullptr, ptrs, stream.GetStream());

  dvector<float> jacobian_numeric(jac_floats);
  NumericDiffJacobianBuilder builder;
  NumericDiffOptions options;
  options.method = NumericDiffOptions::Method::kForward;
  builder.Compute(stream.GetStream(), problem, 0, ms, residuals_baseline.data(),
                  jacobian_numeric.data(), options);

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> jac_an_host(jac_floats), jac_num_host(jac_floats);
  jacobian_analytic.CopyToHost(jac_an_host.data(), jac_floats);
  jacobian_numeric.CopyToHost(jac_num_host.data(), jac_floats);

  // Forward diff (linear factors here are exact regardless, since the
  // between-factor residual is affine in the tangent perturbation).
  ExpectJacobiansClose(jac_an_host, jac_num_host, kRelTol, kAbsTol);
}

}  // namespace
}  // namespace cunls
