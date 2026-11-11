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

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

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

/** @brief Central-diff tolerance used by every test in this file. */
constexpr float kRelTol = 5e-2f;
constexpr float kAbsTol = 5e-3f;

/**
 * @brief Computes the Jacobians of factor batch 0 of `problem` analytically
 * and with NumericDiffJacobianBuilder (at the problem's current states) and
 * compares them element-wise.
 *
 * Uses a combined relative/absolute tolerance since numeric-diff Jacobians
 * are computed in float32 and central differencing amplifies rounding error
 * relative to the analytic values.
 */
void ExpectNumericMatchesAnalytic(Problem &problem, const FactorBatch &factor,
                                  const NumericDiffOptions &options = {}) {
  ASSERT_TRUE(problem.CheckConsistency());
  CudaStream stream;
  MinimizerState ms(stream.GetStream(), problem);
  const size_t num_factors = factor.NumActiveFactors();
  size_t cols = 0;
  for (size_t s : factor.StateSizes()) cols += s;
  const size_t jac_floats = num_factors * factor.ResidualsSize() * cols;

  dvector<float> residuals(num_factors * factor.ResidualsSize()), jacobian_analytic(jac_floats),
      jacobian_numeric(jac_floats);
  auto ptrs = ms.GetStatePointers()[0].data();
  factor.Evaluate(residuals.data(), jacobian_analytic.data(), ptrs, stream.GetStream());
  factor.Evaluate(residuals.data(), nullptr, ptrs, stream.GetStream());
  NumericDiffJacobianBuilder builder;
  builder.Compute(stream.GetStream(), problem, 0, ms, residuals.data(), jacobian_numeric.data(),
                  options);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> analytic(jac_floats), numeric(jac_floats);
  jacobian_analytic.CopyToHost(analytic.data(), jac_floats);
  jacobian_numeric.CopyToHost(numeric.data(), jac_floats);
  for (size_t i = 0; i < jac_floats; i++) {
    const float a = analytic[i], n = numeric[i];
    const float tol = kAbsTol + kRelTol * std::max(std::fabs(a), std::fabs(n));
    EXPECT_NEAR(a, n, tol) << "Mismatch at flattened index " << i << " (analytic=" << a
                           << ", numeric=" << n << ")";
  }
}

/** Between-factor state pointers chaining states (f, f + 1) of one batch. */
template <typename StateBatchT>
std::vector<float *> ChainPointers(StateBatchT &batch, size_t num_factors) {
  std::vector<float *> pointers;
  for (size_t f = 0; f < num_factors; f++) {
    pointers.push_back(batch.StateDevicePtr(f));
    pointers.push_back(batch.StateDevicePtr(f + 1));
  }
  return pointers;
}

/**
 * Exp (library kernel) of n tangents drawn uniformly: the first 3 (rotation)
 * components in ±rot_mag, the others in ±trans_mag.
 */
template <int T, typename Element>
dvector<Element> RandomExp(size_t n, std::mt19937 &rng, float rot_mag, float trans_mag) {
  std::uniform_real_distribution<float> rot_dist(-rot_mag, rot_mag);
  std::uniform_real_distribution<float> trans_dist(-trans_mag, trans_mag);
  hvector<Vector<T>> twists(n);
  for (auto &t : twists)
    for (int i = 0; i < T; i++) t[i] = i < 3 ? rot_dist(rng) : trans_dist(rng);
  dvector<Vector<T>> twists_d(twists);
  dvector<Element> out(n);
  CudaStream stream;
  const float *in = reinterpret_cast<const float *>(twists_d.data());
  float *o = reinterpret_cast<float *>(out.data());
  if constexpr (T == 3) {
    ComputeExpSO3(stream.GetStream(), in, 3, 3, 9, n, o);
  } else {
    ComputeExpSE3(stream.GetStream(), in, 6, 4, 16, n, o);
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

/** Euclidean between factors chaining 6 random R^3 states. */
void CheckVectorBetween(uint32_t seed, const NumericDiffOptions &options) {
  constexpr int kDim = 3;
  constexpr size_t kNumStates = 6;
  constexpr size_t kNumFactors = kNumStates - 1;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<Vector<kDim>> states_host(kNumStates), deltas_host(kNumFactors);
  for (auto *vs : {&states_host, &deltas_host})
    for (auto &v : *vs)
      for (int i = 0; i < kDim; i++) v[i] = dist(rng);
  DeviceVector<Vector<kDim>> states_device(states_host);
  DeviceVector<Vector<kDim>> deltas_device(deltas_host);

  VectorStateBatch<kDim> state_batch(reinterpret_cast<const float *>(states_device.data()),
                                     kNumStates);
  state_batch.SetNumActiveStates(kNumStates);
  VectorBetweenFactorBatch<kDim> factor_batch(deltas_device.data(), kNumFactors);
  factor_batch.SetNumActiveFactors(kNumFactors);
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, ChainPointers(state_batch, kNumFactors));
  ExpectNumericMatchesAnalytic(problem, factor_batch, options);
}

TEST(NumericDiffJacobianTest, VectorBetweenMatchesAnalytic) { CheckVectorBetween(kSeed, {}); }

// Sanity check for the forward-difference path (kForward), on the same
// Euclidean between-factor setup. The between-factor residual is affine in the
// tangent perturbation, so forward differences are exact up to round-off.
TEST(NumericDiffJacobianTest, ForwardDiffMatchesAnalytic) {
  NumericDiffOptions options;
  options.method = NumericDiffOptions::Method::kForward;
  CheckVectorBetween(kSeed + 3, options);
}

// SO3BetweenFactorBatch: residual = Log(R_left^T * R_right * Delta). Uses
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
  auto states_d = RandomExp<3, SO3Rotation>(kNumStates, rng, 0.5f, 0.0f);
  auto deltas_d = RandomExp<3, SO3Rotation>(kNumFactors, rng, 0.5f, 0.0f);

  SO3StateBatch state_batch(reinterpret_cast<const float *>(states_d.data()), kNumStates);
  state_batch.SetNumActiveStates(kNumStates);
  SO3BetweenFactorBatch factor_batch(deltas_d.data(), kNumFactors);
  factor_batch.SetNumActiveFactors(kNumFactors);
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, ChainPointers(state_batch, kNumFactors));
  ExpectNumericMatchesAnalytic(problem, factor_batch);
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
  auto states_d = RandomExp<6, SE3Transform>(kNumStates, rng, 0.3f, 2.0f);
  auto deltas_d = RandomExp<6, SE3Transform>(kNumFactors, rng, 0.3f, 2.0f);

  SE3StateBatch state_batch(reinterpret_cast<const float *>(states_d.data()), kNumStates);
  state_batch.SetNumActiveStates(kNumStates);
  SE3BetweenFactorBatch factor_batch(deltas_d.data(), kNumFactors);
  factor_batch.SetNumActiveFactors(kNumFactors);
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, ChainPointers(state_batch, kNumFactors));
  ExpectNumericMatchesAnalytic(problem, factor_batch);
}

// A factor slot may read from different state batches for different factors
// (e.g. free and fixed poses kept in separate batches). Regression test: the
// builder used to take each slot's owning batch from factor 0 only, so the
// other factors were differenced at the wrong state (here also past the end of
// factor 0's two-state batch).
TEST(NumericDiffJacobianTest, SlotsReadingDifferentStateBatches) {
  constexpr size_t kNumA = 2, kNumB = 5, kNumFactors = 5;
  std::mt19937 rng(kSeed + 4);
  auto states_a = RandomExp<6, SE3Transform>(kNumA, rng, 0.3f, 2.0f);
  auto states_b = RandomExp<6, SE3Transform>(kNumB, rng, 0.3f, 2.0f);
  auto deltas = RandomExp<6, SE3Transform>(kNumFactors, rng, 0.3f, 2.0f);

  SE3StateBatch batch_a(reinterpret_cast<const float *>(states_a.data()), kNumA);
  batch_a.SetNumActiveStates(kNumA);
  SE3StateBatch batch_b(reinterpret_cast<const float *>(states_b.data()), kNumB);
  batch_b.SetNumActiveStates(kNumB);
  SE3BetweenFactorBatch factor_batch(deltas.data(), kNumFactors);
  factor_batch.SetNumActiveFactors(kNumFactors);

  // (left, right): both slots mix the two batches across factors.
  float *a0 = batch_a.StateDevicePtr(0), *a1 = batch_a.StateDevicePtr(1);
  auto b = [&](size_t i) { return batch_b.StateDevicePtr(i); };
  Problem problem;
  problem.AddStateBatch(&batch_a);
  problem.AddStateBatch(&batch_b);
  problem.AddFactorBatch(&factor_batch, {a0, b(0), b(1), a1, b(4), b(2), a1, b(3), b(0), a0});
  ExpectNumericMatchesAnalytic(problem, factor_batch);
}

}  // namespace
}  // namespace cunls
