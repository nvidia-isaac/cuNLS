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
 * @file capacity_test.cpp
 * @brief Capacity and active sizes of factor and state batches
 * (docs/design/reusable_buffers.md):
 *  - a factor batch built for N factors and resized to n < N evaluates bitwise
 *    like a batch built for exactly n factors; resizing above N throws;
 *  - a state batch resized to n blocks updates exactly its first n blocks, also
 *    with replicas, and honors the active constant-id count;
 *  - factors cache nothing derived from their measurements: rewriting the
 *    measurements in place changes the next evaluation (the factors that used
 *    to precompute inverses / adjoints in their constructors).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/evaluate_items_check.h"

namespace cunls {
namespace {

using evaluate_items_test::ToDevice;
using evaluate_items_test::ToHost;
using FactorPtr = std::unique_ptr<FactorBatch>;

cuBLASHandle &Cublas() {
  static cuBLASHandle handle;
  return handle;
}

std::vector<float> RandomVector(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float &x : v) x = dist(rng);
  return v;
}

/** `count` valid elements of a matrix Lie group: identity (+) random tangent via Plus. */
template <class State>
dvector<float> LieElements(size_t count, int ambient, int tangent, float scale, uint32_t seed) {
  const int dim = static_cast<int>(std::lround(std::sqrt(ambient)));
  std::vector<float> identity(count * ambient, 0.f);
  for (size_t b = 0; b < count; ++b) {
    for (int i = 0; i < dim; ++i) identity[b * ambient + i * dim + i] = 1.f;
  }
  auto base = ToDevice(identity);
  auto delta = ToDevice(RandomVector(count * tangent, -scale, scale, seed));
  dvector<float> out(identity.size());
  CudaStream stream;
  State states(Cublas(), base.data(), count);
  states.SetNumStateBlocks(states.Capacity(), states.ConstCapacity());
  states.Plus(base.data(), delta.data(), out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

/** Residuals and Jacobians of the first `n` factors (pointer table of n * B entries). */
std::pair<std::vector<float>, std::vector<float>> EvaluateFirst(const FactorBatch &factor, size_t n,
                                                                const dvector<float *> &pointers) {
  const auto sizes = factor.StateBlockSizes();
  size_t cols = 0;
  for (size_t s : sizes) cols += s;
  const size_t m = factor.ResidualsSize();
  dvector<float> res(n * m), jac(n * m * cols);
  CudaStream stream;
  EXPECT_TRUE(factor.Evaluate(res.data(), jac.data(), pointers.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return {ToHost(res), ToHost(jac)};
}

bool SameBits(const std::vector<float> &a, const std::vector<float> &b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/**
 * Builds a factor batch over `capacity` measurements, resizes it to `n`, and
 * compares it bitwise with a batch built over exactly `n` measurements.
 * make(count) builds a batch reading the first `count` measurements;
 * `pointers` holds capacity * B state pointers.
 */
void CheckActiveSize(const std::function<FactorPtr(size_t)> &make, size_t capacity, size_t n,
                     const std::vector<float *> &pointers) {
  FactorPtr resized = make(capacity);
  ASSERT_EQ(resized->Capacity(), capacity);
  ASSERT_EQ(resized->NumFactors(), 0u);  // zero until set
  resized->SetNumFactors(n);
  EXPECT_EQ(resized->NumFactors(), n);
  EXPECT_EQ(resized->Capacity(), capacity);

  const size_t b = resized->StateBlockSizes().size();
  dvector<float *> d_pointers(std::vector<float *>(pointers.begin(), pointers.begin() + n * b));
  FactorPtr exact = make(n);
  exact->SetNumFactors(n);
  const auto got = EvaluateFirst(*resized, n, d_pointers);
  const auto want = EvaluateFirst(*exact, n, d_pointers);
  EXPECT_TRUE(SameBits(got.first, want.first)) << "residuals";
  EXPECT_TRUE(SameBits(got.second, want.second)) << "Jacobians";

  EXPECT_THROW(resized->SetNumFactors(capacity + 1), std::invalid_argument);
  EXPECT_EQ(resized->NumFactors(), n);  // unchanged by the failed call
  resized->SetNumFactors(capacity);
  EXPECT_EQ(resized->NumFactors(), capacity);
}

constexpr size_t kCapacity = 300;
constexpr size_t kActive = 117;

// ============================================================================
// Factors: active size behaves like an exact-size batch
// ============================================================================

TEST(FactorCapacity, SE3Prior) {
  auto targets = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 1);
  auto states = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 2);
  std::vector<float *> pointers;
  for (size_t i = 0; i < kCapacity; ++i) pointers.push_back(states.data() + i * 16);
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<SE3PriorFactorBatch>(
            reinterpret_cast<const SE3Transform *>(targets.data()), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, SE3Between) {
  auto deltas = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 3);
  auto states = LieElements<SE3StateBatch>(2 * kCapacity, 16, 6, 0.3f, 4);
  std::vector<float *> pointers;
  for (size_t i = 0; i < 2 * kCapacity; ++i) pointers.push_back(states.data() + i * 16);
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<SE3BetweenFactorBatch>(
            reinterpret_cast<const SE3Transform *>(deltas.data()), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, PnP) {
  auto observations = ToDevice(RandomVector(2 * kCapacity, -0.4f, 0.4f, 5));
  std::vector<float> points = RandomVector(3 * kCapacity, -1.f, 1.f, 6);
  for (size_t i = 0; i < kCapacity; ++i) points[3 * i + 2] += 4.f;  // in front of the camera
  auto d_points = ToDevice(points);
  auto pose = LieElements<SE3StateBatch>(1, 16, 6, 0.1f, 7);
  const std::vector<float *> pointers(kCapacity, pose.data());
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<PnPFactorBatch>(
            reinterpret_cast<const Vector<2> *>(observations.data()),
            reinterpret_cast<const Vector<3> *>(d_points.data()), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, Reprojection) {
  auto observations = ToDevice(RandomVector(2 * kCapacity, -0.4f, 0.4f, 8));
  auto poses = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.1f, 9);
  std::vector<float> points = RandomVector(3 * kCapacity, -1.f, 1.f, 10);
  for (size_t i = 0; i < kCapacity; ++i) points[3 * i + 2] += 4.f;
  auto d_points = ToDevice(points);
  std::vector<float *> pointers;
  for (size_t i = 0; i < kCapacity; ++i) {
    pointers.push_back(poses.data() + i * 16);
    pointers.push_back(d_points.data() + i * 3);
  }
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<ReprojectionFactorBatch>(
            reinterpret_cast<const Vector<2> *>(observations.data()), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, ConstantVelocitySE3) {
  auto dt = ToDevice(RandomVector(kCapacity, 0.05f, 1.f, 11));
  auto poses = LieElements<SE3StateBatch>(2 * kCapacity, 16, 6, 0.3f, 12);
  auto velocities = ToDevice(RandomVector(2 * kCapacity * 6, -1.f, 1.f, 13));
  std::vector<float *> pointers;
  for (size_t i = 0; i < kCapacity; ++i) {
    pointers.push_back(poses.data() + (2 * i) * 16);
    pointers.push_back(poses.data() + (2 * i + 1) * 16);
    pointers.push_back(velocities.data() + (2 * i) * 6);
    pointers.push_back(velocities.data() + (2 * i + 1) * 6);
  }
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<ConstantVelocitySE3FactorBatch>(dt.data(), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, InformationWrapperForwardsTheSize) {
  auto targets = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 14);
  auto states = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 15);
  auto matrices = ToDevice(RandomVector(kCapacity * 36, -1.f, 1.f, 16));
  std::vector<float *> pointers;
  for (size_t i = 0; i < kCapacity; ++i) pointers.push_back(states.data() + i * 16);
  using Wrapped = InformationFactorBatch<SE3PriorFactorBatch>;
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<Wrapped>(
            Cublas(), reinterpret_cast<const Wrapped::InformationMatrix *>(matrices.data()), count,
            reinterpret_cast<const SE3Transform *>(targets.data()), count);
      },
      kCapacity, kActive, pointers);
}

TEST(FactorCapacity, WeightedWrapperForwardsTheSize) {
  auto targets = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 17);
  auto states = LieElements<SE3StateBatch>(kCapacity, 16, 6, 0.3f, 18);
  auto weights = ToDevice(RandomVector(kCapacity, 0.5f, 2.f, 19));
  std::vector<float *> pointers;
  for (size_t i = 0; i < kCapacity; ++i) pointers.push_back(states.data() + i * 16);
  CheckActiveSize(
      [&](size_t count) {
        return std::make_unique<WeightedFactorBatch<SE3PriorFactorBatch>>(
            weights.data(), count, reinterpret_cast<const SE3Transform *>(targets.data()), count);
      },
      kCapacity, kActive, pointers);
}

/** A custom factor that overrides NumFactors() without passing a capacity to its base. */
class FixedSizeFactor : public FactorBatch {
 public:
  bool Evaluate(float *, float *, float const *const *, cudaStream_t, const int *,
                size_t) const override {
    return true;
  }
  size_t ResidualsSize() const override { return 1; }
  std::vector<size_t> StateBlockSizes() const override { return {1}; }
  size_t NumFactors() const override { return 5; }
};

TEST(FactorCapacity, OverridingNumFactorsWithoutCapacityIsRejected) {
  FixedSizeFactor factor;
  EXPECT_EQ(factor.NumFactors(), 5u);
  EXPECT_EQ(factor.Capacity(), 0u);  // capacity is what the base was constructed with
  EXPECT_THROW(factor.SetNumFactors(3), std::invalid_argument);

  // The problem reports NumFactors() > Capacity().
  std::vector<float> zeros(5, 0.f);
  auto x = ToDevice(zeros);
  VectorStateBatch<1> states(x.data(), 5);
  states.SetNumStateBlocks(5);
  std::vector<float *> pointers;
  for (size_t i = 0; i < 5; ++i) pointers.push_back(states.StateBlockDevicePtr(i));
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&factor, pointers);
  EXPECT_FALSE(problem.CheckConsistency());
}

TEST(FactorCapacity, ActiveCountStartsAtZero) {
  auto targets = ToDevice(std::vector<float>(16 * 4, 0.f));
  SE3PriorFactorBatch prior(reinterpret_cast<const SE3Transform *>(targets.data()), 4);
  EXPECT_EQ(prior.Capacity(), 4u);
  EXPECT_EQ(prior.NumFactors(), 0u);
  prior.SetNumFactors(4);
  EXPECT_EQ(prior.NumFactors(), 4u);
  VectorStateBatch<3> states(targets.data(), 8);
  EXPECT_EQ(states.Capacity(), 8u);
  EXPECT_EQ(states.NumStateBlocks(), 0u);
}

// ============================================================================
// States: Plus touches exactly the active blocks
// ============================================================================

/**
 * Plus over a batch resized to `n` blocks (and `replicas` replicas) must equal
 * Plus over an exact-size batch and leave everything past the active blocks
 * untouched.
 */
template <class State, class Make>
void CheckStateActiveSize(Make make, int ambient, int tangent, size_t capacity, size_t n,
                          size_t replicas, const dvector<float> &x) {
  auto delta = ToDevice(RandomVector(capacity * replicas * tangent, -0.2f, 0.2f, 21));
  const float sentinel = 7.25f;
  dvector<float> got = ToDevice(std::vector<float>(capacity * replicas * ambient, sentinel));
  dvector<float> want(capacity * replicas * ambient);
  std::unique_ptr<State> resized = make(capacity);
  ASSERT_EQ(resized->Capacity(), capacity);
  ASSERT_EQ(resized->NumStateBlocks(), 0u);  // zero until set
  resized->SetNumStateBlocks(n);
  EXPECT_EQ(resized->NumStateBlocks(), n);
  std::unique_ptr<State> exact = make(n);
  exact->SetNumStateBlocks(n);
  CudaStream stream;
  resized->Plus(x.data(), delta.data(), got.data(), stream.GetStream(), replicas);
  exact->Plus(x.data(), delta.data(), want.data(), stream.GetStream(), replicas);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto g = ToHost(got);
  const auto w = ToHost(want);
  const size_t active = n * replicas * ambient;
  EXPECT_EQ(0, std::memcmp(g.data(), w.data(), active * sizeof(float)));
  for (size_t i = active; i < g.size(); ++i) {
    ASSERT_EQ(g[i], sentinel) << "Plus wrote past the active blocks at " << i;
  }
  EXPECT_THROW(resized->SetNumStateBlocks(capacity + 1), std::invalid_argument);
}

TEST(StateCapacity, VectorStateBatch) {
  for (size_t replicas : {1, 3}) {
    auto x = ToDevice(RandomVector(kCapacity * replicas * 3, -1.f, 1.f, 22));
    CheckStateActiveSize<VectorStateBatch<3>>(
        [&](size_t count) { return std::make_unique<VectorStateBatch<3>>(x.data(), count); }, 3, 3,
        kCapacity, kActive, replicas, x);
  }
}

TEST(StateCapacity, SE3StateBatch) {
  for (size_t replicas : {1, 2}) {
    auto x = LieElements<SE3StateBatch>(kCapacity * replicas, 16, 6, 0.3f, 23);
    CheckStateActiveSize<SE3StateBatch>(
        [&](size_t count) { return std::make_unique<SE3StateBatch>(Cublas(), x.data(), count); },
        16, 6, kCapacity, kActive, replicas, x);
  }
}

TEST(StateCapacity, SL4StateBatch) {
  for (size_t replicas : {1, 2}) {
    auto x = LieElements<SL4StateBatch>(kCapacity * replicas, 16, 15, 0.05f, 24);
    CheckStateActiveSize<SL4StateBatch>(
        [&](size_t count) { return std::make_unique<SL4StateBatch>(Cublas(), x.data(), count); },
        16, 15, kCapacity, kActive, replicas, x);
  }
}

TEST(StateCapacity, ConstantIdCount) {
  auto x = LieElements<SE3StateBatch>(10, 16, 6, 0.3f, 25);
  auto ids = ToDevice(std::vector<int>{1, 3, 5});
  SE3StateBatch states(Cublas(), x.data(), 10, ids.data(), 3);
  EXPECT_EQ(states.Capacity(), 10u);
  EXPECT_EQ(states.ConstCapacity(), 3u);
  EXPECT_EQ(states.NumStateBlocks(), 0u);
  EXPECT_EQ(states.NumConstStateBlocks(), 0u);
  states.SetNumStateBlocks(4, 2);
  EXPECT_EQ(states.NumStateBlocks(), 4u);
  EXPECT_EQ(states.NumConstStateBlocks(), 2u);
  EXPECT_EQ(states.ConstStateIds(), ids.data());
  // Inactive blocks stay addressable up to the capacity, so connectivity for a
  // coming solve can be built before resizing.
  EXPECT_EQ(states.StateBlockDevicePtr(4), states.StateBlockDevicePtr(0) + 4 * 16);
  EXPECT_NE(states.StateBlockDevicePtr(9), nullptr);
  EXPECT_EQ(states.StateBlockDevicePtr(10), nullptr);  // past the capacity
  EXPECT_THROW(states.SetNumStateBlocks(4, 4), std::invalid_argument);
  EXPECT_THROW(states.SetNumStateBlocks(11, 0), std::invalid_argument);
  states.SetNumStateBlocks(10, 3);
  EXPECT_EQ(states.NumConstStateBlocks(), 3u);
}

// ============================================================================
// Factors cache nothing derived from their measurements
// ============================================================================

template <class Factor, class Obs>
FactorPtr MakeLieFactor(const float *measurements, size_t count) {
  FactorPtr factor;
  if constexpr (std::is_constructible_v<Factor, cuBLASHandle &, const Obs *, size_t>) {
    factor = std::make_unique<Factor>(Cublas(), reinterpret_cast<const Obs *>(measurements), count);
  } else {
    factor = std::make_unique<Factor>(reinterpret_cast<const Obs *>(measurements), count);
  }
  factor->SetNumFactors(count);
  return factor;
}

/**
 * Evaluates a factor, rewrites its measurement buffer in place with other
 * elements, evaluates again: the second result must differ from the first and
 * match a factor freshly built over the new measurements.
 */
template <class Factor, class State, class Obs>
void CheckMeasurementRewrite(int ambient, int tangent, int blocks, float scale) {
  const size_t n = 64;
  auto measurements = LieElements<State>(n, ambient, tangent, scale, 31);
  auto replacement = LieElements<State>(n, ambient, tangent, scale, 32);
  auto states = LieElements<State>(n * blocks, ambient, tangent, scale, 33);
  std::vector<float *> pointers;
  for (size_t i = 0; i < n * blocks; ++i) pointers.push_back(states.data() + i * ambient);
  dvector<float *> d_pointers(pointers);

  FactorPtr factor = MakeLieFactor<Factor, Obs>(measurements.data(), n);
  const auto before = EvaluateFirst(*factor, n, d_pointers);
  THROW_ON_CUDA_ERROR(cudaMemcpy(measurements.data(), replacement.data(),
                                 replacement.size() * sizeof(float), cudaMemcpyDeviceToDevice));
  const auto after = EvaluateFirst(*factor, n, d_pointers);
  FactorPtr fresh = MakeLieFactor<Factor, Obs>(measurements.data(), n);
  const auto want = EvaluateFirst(*fresh, n, d_pointers);

  EXPECT_FALSE(SameBits(before.first, after.first)) << "residuals did not follow the new data";
  EXPECT_TRUE(SameBits(after.first, want.first)) << "residuals";
  EXPECT_TRUE(SameBits(after.second, want.second)) << "Jacobians";
}

TEST(MeasurementRewrite, SE2Prior) {
  CheckMeasurementRewrite<SE2PriorFactorBatch, SE2StateBatch, SE2Transform>(9, 3, 1, 0.3f);
}
TEST(MeasurementRewrite, SE3Prior) {
  CheckMeasurementRewrite<SE3PriorFactorBatch, SE3StateBatch, SE3Transform>(16, 6, 1, 0.3f);
}
TEST(MeasurementRewrite, Sim2Prior) {
  CheckMeasurementRewrite<Similarity2PriorFactorBatch, Similarity2StateBatch, Similarity2Transform>(
      9, 4, 1, 0.3f);
}
TEST(MeasurementRewrite, Sim3Prior) {
  CheckMeasurementRewrite<Similarity3PriorFactorBatch, Similarity3StateBatch, Similarity3Transform>(
      16, 7, 1, 0.3f);
}
TEST(MeasurementRewrite, SL4Prior) {
  CheckMeasurementRewrite<SL4PriorFactorBatch, SL4StateBatch, SL4Transform>(16, 15, 1, 0.05f);
}
TEST(MeasurementRewrite, SO3Between) {
  CheckMeasurementRewrite<SO3BetweenFactorBatch, SO3StateBatch, SO3Rotation>(9, 3, 2, 0.3f);
}
TEST(MeasurementRewrite, SE3Between) {
  CheckMeasurementRewrite<SE3BetweenFactorBatch, SE3StateBatch, SE3Transform>(16, 6, 2, 0.3f);
}
TEST(MeasurementRewrite, Sim3Between) {
  CheckMeasurementRewrite<Similarity3BetweenFactorBatch, Similarity3StateBatch,
                          Similarity3Transform>(16, 7, 2, 0.3f);
}
TEST(MeasurementRewrite, SL4Between) {
  CheckMeasurementRewrite<SL4BetweenFactorBatch, SL4StateBatch, SL4Transform>(16, 15, 2, 0.05f);
}

}  // namespace
}  // namespace cunls
