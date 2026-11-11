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
 * @file manifold_test.cpp
 * @brief Per manifold (vector, SO(2), SE(2), Sim(2), SO(3), SE(3), Sim(3),
 * SL(4)): state batch dimensions, and Levenberg-Marquardt convergence of
 * independent prior factors (states reach their targets) and of identity
 * between factors (left and right states meet).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/prior/so2_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr float kNoCostBound = std::numeric_limits<float>::infinity();
constexpr float kPi = static_cast<float>(M_PI);

/** Settings and expectations of one LM convergence check. */
struct Convergence {
  int max_iterations;
  float tolerance;       ///< state and cost tolerance
  float max_final_cost;  ///< kNoCostBound to skip
  float element_tolerance;
};

/** Prior checks: 50 iterations, tolerances 1e-8. */
Convergence PriorSettings(float max_final_cost, float element_tolerance) {
  return {50, 1e-8f, max_final_cost, element_tolerance};
}

/** Between checks: 80 iterations, tolerances 1e-7, no cost bound. */
Convergence BetweenSettings(float element_tolerance) {
  return {80, 1e-7f, kNoCostBound, element_tolerance};
}

template <typename Element>
hvector<Element> ToHost(const dvector<Element> &d) {
  hvector<Element> h(d.size());
  d.CopyToHost(h.data(), h.size());
  return h;
}

template <typename Element>
void ExpectElementsNear(const hvector<Element> &a, const hvector<Element> &b, float tolerance) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < a[i].size(); ++j) {
      ASSERT_NEAR(a[i][j], b[i][j], tolerance) << "element " << i << ", entry " << j;
    }
  }
}

/** Row-major N x N product. */
template <typename Element>
Element MatMul(const Element &a, const Element &b) {
  const int n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.size()))));
  Element c{};
  for (int r = 0; r < n; ++r)
    for (int col = 0; col < n; ++col)
      for (int k = 0; k < n; ++k) c[r * n + col] += a[r * n + k] * b[k * n + col];
  return c;
}

template <typename Element>
Element Identity() {
  const int n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(Element{}.size()))));
  Element e{};
  for (int i = 0; i < n; ++i) e[i * n + i] = 1.0f;
  return e;
}

/** Group types of one manifold. */
template <typename StatesT, typename ElementT, typename PriorT, typename BetweenT,
          typename MeasurementT = ElementT>
struct Manifold {
  using States = StatesT;
  using Element = ElementT;
  using Prior = PriorT;
  using Between = BetweenT;
  using Measurement = MeasurementT;
};

using VectorManifold = Manifold<VectorStateBatch<3>, Vector<3>, PriorVectorFactorBatch<3>,
                                VectorBetweenFactorBatch<3>>;
using SO2Manifold =
    Manifold<SO2StateBatch, Matrix<2>, SO2PriorFactorBatch, SO2BetweenFactorBatch, SO2Rotation>;
using SE2Manifold =
    Manifold<SE2StateBatch, Matrix<3>, SE2PriorFactorBatch, SE2BetweenFactorBatch, SE2Transform>;
using Sim2Manifold = Manifold<Similarity2StateBatch, Matrix<3>, Similarity2PriorFactorBatch,
                              Similarity2BetweenFactorBatch, Similarity2Transform>;
using SO3Manifold =
    Manifold<SO3StateBatch, Matrix<3>, SO3PriorFactorBatch, SO3BetweenFactorBatch, SO3Rotation>;
using SE3Manifold =
    Manifold<SE3StateBatch, SE3Transform, SE3PriorFactorBatch, SE3BetweenFactorBatch>;
using Sim3Manifold = Manifold<Similarity3StateBatch, Matrix<4>, Similarity3PriorFactorBatch,
                              Similarity3BetweenFactorBatch, Similarity3Transform>;
using SL4Manifold =
    Manifold<SL4StateBatch, SL4Transform, SL4PriorFactorBatch, SL4BetweenFactorBatch>;

template <typename M>
void ExpectStateDimensions(size_t tangent, size_t ambient) {
  constexpr size_t kN = 10;
  dvector<typename M::Element> dev{hvector<typename M::Element>(kN)};
  typename M::States states(reinterpret_cast<const float *>(dev.data()), kN);
  states.SetNumActiveStates(states.Capacity(), states.ConstCapacity());
  EXPECT_EQ(states.TangentSize(), tangent);
  EXPECT_EQ(states.AmbientSize(), ambient);
  EXPECT_EQ(states.NumActiveStates(), kN);
}

/**
 * Minimizes `factor` with LM (initial lambda 1e-3) over one state batch per
 * entry of `arrays` (item i of the factor reads element i of each, in order),
 * updating the arrays in place, and checks the cost.
 */
template <typename M>
void MinimizeInPlace(FactorBatch &factor, const std::vector<dvector<typename M::Element> *> &arrays,
                     const Convergence &c) {
  const size_t n = arrays[0]->size();
  std::vector<std::unique_ptr<typename M::States>> batches;
  Problem problem;
  for (auto *array : arrays) {
    batches.push_back(
        std::make_unique<typename M::States>(reinterpret_cast<const float *>(array->data()), n));
    batches.back()->SetNumActiveStates(n, 0);
    problem.AddStateBatch(batches.back().get());
  }
  std::vector<float *> ptrs;
  for (size_t i = 0; i < n; ++i)
    for (auto &batch : batches) ptrs.push_back(batch->StateDevicePtr(i));
  factor.SetNumActiveFactors(n);
  problem.AddFactorBatch(&factor, ptrs);
  ASSERT_TRUE(problem.CheckConsistency());

  LevenbergMarquardtMinimizerOptions lm_opts;
  lm_opts.base_options.max_num_iterations = c.max_iterations;
  lm_opts.base_options.state_tolerance = c.tolerance;
  lm_opts.base_options.cost_tolerance = c.tolerance;
  lm_opts.base_options.disable_safety_checks = false;
  lm_opts.initial_lambda = 1e-3f;
  CudaStream stream;
  LevenbergMarquardtMinimizer minimizer(lm_opts);
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, c.max_final_cost);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_GT(summary.num_iterations, 0u);
}

/** Priors pull states from `initials` to `targets`. */
template <typename M>
void ExpectPriorsConverge(const hvector<typename M::Element> &targets,
                          const hvector<typename M::Element> &initials, const Convergence &c) {
  dvector<typename M::Element> targets_dev(targets), states(initials);
  typename M::Prior factor(reinterpret_cast<const typename M::Measurement *>(targets_dev.data()),
                           targets.size());
  MinimizeInPlace<M>(factor, {&states}, c);
  ExpectElementsNear(ToHost(states), targets, c.element_tolerance);
}

/** Identity between factors pull left[i] and right[i] together. */
template <typename M>
void ExpectBetweensAlign(const hvector<typename M::Element> &left,
                         const hvector<typename M::Element> &right,
                         const typename M::Element &identity, const Convergence &c) {
  dvector<typename M::Element> left_dev(left), right_dev(right),
      deltas(hvector<typename M::Element>(left.size(), identity));
  typename M::Between factor(reinterpret_cast<const typename M::Measurement *>(deltas.data()),
                             left.size());
  MinimizeInPlace<M>(factor, {&left_dev, &right_dev}, c);
  ExpectElementsNear(ToHost(left_dev), ToHost(right_dev), c.element_tolerance);
}

/**
 * Exp (on the device, with the library's Exp kernel) of tangents drawn
 * uniformly: the first `num_rot` components in ±rot_mag, the others in
 * ±other_mag.
 */
template <int T, typename Element, typename ExpFn>
hvector<Element> RandomExp(size_t n, uint32_t seed, int num_rot, float rot_mag, float other_mag,
                           ExpFn exp, size_t pitch) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> rot(-rot_mag, rot_mag), other(-other_mag, other_mag);
  hvector<Vector<T>> tangents(n);
  for (auto &x : tangents)
    for (int k = 0; k < T; ++k) x[k] = k < num_rot ? rot(rng) : other(rng);
  dvector<Vector<T>> tangents_dev(tangents);
  dvector<Element> out(n);
  CudaStream stream;
  exp(stream.GetStream(), reinterpret_cast<const float *>(tangents_dev.data()), T, pitch,
      Element{}.size(), n, reinterpret_cast<float *>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return ToHost(out);
}

/** targets[i] * perturbations[i]. */
template <typename Element>
hvector<Element> PerturbRight(const hvector<Element> &targets,
                              const hvector<Element> &perturbations) {
  hvector<Element> out(targets.size());
  for (size_t i = 0; i < targets.size(); ++i) out[i] = MatMul(targets[i], perturbations[i]);
  return out;
}

// ============================================================================
// Vector (R^3)
// ============================================================================

TEST(VectorManifoldTest, StateDimensions) { ExpectStateDimensions<VectorManifold>(3, 3); }

TEST(VectorManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 5000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> value_dist(-5.0f, 5.0f), pert_dist(-0.5f, 0.5f);
  hvector<Vector<3>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    for (int k = 0; k < 3; ++k) {
      targets[i][k] = value_dist(rng);
      initials[i][k] = targets[i][k] + pert_dist(rng);
    }
  }
  ExpectPriorsConverge<VectorManifold>(targets, initials, PriorSettings(1e-6f, 1e-4f));
}

TEST(VectorManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 5000;
  std::mt19937 rng1(60), rng2(61);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  hvector<Vector<3>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    for (int k = 0; k < 3; ++k) {
      left[i][k] = dist(rng1);
      right[i][k] = dist(rng2);
    }
  }
  ExpectBetweensAlign<VectorManifold>(left, right, Vector<3>{}, PriorSettings(1e-6f, 1e-4f));
}

// ============================================================================
// SO(2)
// ============================================================================

Matrix<2> MakeSO2(float theta) { return {cosf(theta), -sinf(theta), sinf(theta), cosf(theta)}; }

TEST(SO2ManifoldTest, StateDimensions) { ExpectStateDimensions<SO2Manifold>(1, 4); }

TEST(SO2ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(-kPi, kPi), pert_dist(-0.3f, 0.3f);
  hvector<Matrix<2>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    const float a = angle_dist(rng);
    targets[i] = MakeSO2(a);
    initials[i] = MakeSO2(a + pert_dist(rng));
  }
  ExpectPriorsConverge<SO2Manifold>(targets, initials, PriorSettings(1e-4f, 1e-3f));
}

TEST(SO2ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(60), rng2(61);
  std::uniform_real_distribution<float> angle_dist(-kPi, kPi);
  hvector<Matrix<2>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    left[i] = MakeSO2(angle_dist(rng1));
    right[i] = MakeSO2(angle_dist(rng2));
  }
  ExpectBetweensAlign<SO2Manifold>(left, right, Identity<Matrix<2>>(), BetweenSettings(0.1f));
}

// ============================================================================
// SE(2)
// ============================================================================

Matrix<3> MakeSE2(float theta, float tx, float ty) {
  const float c = cosf(theta), s = sinf(theta);
  return {c, -s, tx, s, c, ty, 0.0f, 0.0f, 1.0f};
}

TEST(SE2ManifoldTest, StateDimensions) { ExpectStateDimensions<SE2Manifold>(3, 9); }

TEST(SE2ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(-kPi, kPi), trans_dist(-2.0f, 2.0f);
  std::uniform_real_distribution<float> pert_angle(-0.3f, 0.3f), pert_trans(-0.3f, 0.3f);
  hvector<Matrix<3>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    const float theta = angle_dist(rng);
    const float tx = trans_dist(rng), ty = trans_dist(rng);
    targets[i] = MakeSE2(theta, tx, ty);
    const float pa = pert_angle(rng), px = pert_trans(rng), py = pert_trans(rng);
    initials[i] = MakeSE2(theta + pa, tx + px, ty + py);
  }
  ExpectPriorsConverge<SE2Manifold>(targets, initials, PriorSettings(1e-4f, 1e-2f));
}

TEST(SE2ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(50), rng2(51);
  std::uniform_real_distribution<float> angle_dist(-0.5f, 0.5f), trans_dist(-1.0f, 1.0f);
  hvector<Matrix<3>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    const float a1 = angle_dist(rng1), x1 = trans_dist(rng1), y1 = trans_dist(rng1);
    const float a2 = angle_dist(rng2), x2 = trans_dist(rng2), y2 = trans_dist(rng2);
    left[i] = MakeSE2(a1, x1, y1);
    right[i] = MakeSE2(a2, x2, y2);
  }
  ExpectBetweensAlign<SE2Manifold>(left, right, Identity<Matrix<3>>(), BetweenSettings(0.1f));
}

// ============================================================================
// Sim(2)
// ============================================================================

Matrix<3> MakeSim2(float x, float y, float theta, float scale) {
  const float c = cosf(theta), s = sinf(theta);
  return {c, -s, x, s, c, y, 0.0f, 0.0f, 1.0f / scale};
}

TEST(Sim2ManifoldTest, StateDimensions) { ExpectStateDimensions<Sim2Manifold>(4, 9); }

TEST(Sim2ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> angle_dist(-kPi, kPi), trans_dist(-2.0f, 2.0f),
      scale_dist(0.5f, 2.0f);
  std::uniform_real_distribution<float> pert_angle(-0.3f, 0.3f), pert_trans(-0.3f, 0.3f),
      pert_scale(0.85f, 1.15f);
  hvector<Matrix<3>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    const float theta = angle_dist(rng);
    const float tx = trans_dist(rng), ty = trans_dist(rng);
    const float s = scale_dist(rng);
    targets[i] = MakeSim2(tx, ty, theta, s);
    const float px = pert_trans(rng), py = pert_trans(rng), pa = pert_angle(rng),
                ps = pert_scale(rng);
    initials[i] = MakeSim2(tx + px, ty + py, theta + pa, s * ps);
  }
  ExpectPriorsConverge<Sim2Manifold>(targets, initials, PriorSettings(1e-4f, 1e-2f));
}

TEST(Sim2ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(60), rng2(61);
  std::uniform_real_distribution<float> angle_dist(-0.3f, 0.3f), trans_dist(-0.5f, 0.5f),
      scale_dist(0.8f, 1.2f);
  auto random_sim2 = [&](std::mt19937 &rng) {
    const float x = trans_dist(rng), y = trans_dist(rng), a = angle_dist(rng), s = scale_dist(rng);
    return MakeSim2(x, y, a, s);
  };
  hvector<Matrix<3>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    left[i] = random_sim2(rng1);
    right[i] = random_sim2(rng2);
  }
  ExpectBetweensAlign<Sim2Manifold>(left, right, Identity<Matrix<3>>(), BetweenSettings(0.1f));
}

// ============================================================================
// SO(3)
// ============================================================================

hvector<Matrix<3>> RandomSO3(size_t n, uint32_t seed, float magnitude) {
  return RandomExp<3, Matrix<3>>(n, seed, 3, magnitude, magnitude, ComputeExpSO3, 3);
}

TEST(SO3ManifoldTest, StateDimensions) { ExpectStateDimensions<SO3Manifold>(3, 9); }

TEST(SO3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  const auto targets = RandomSO3(kN, 42, 0.5f);
  ExpectPriorsConverge<SO3Manifold>(targets, PerturbRight(targets, RandomSO3(kN, 43, 0.3f)),
                                    PriorSettings(1e-4f, 1e-2f));
}

TEST(SO3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  ExpectBetweensAlign<SO3Manifold>(RandomSO3(kN, 60, 0.5f), RandomSO3(kN, 61, 0.5f),
                                   Identity<Matrix<3>>(), BetweenSettings(0.1f));
}

// ============================================================================
// SE(3)
// ============================================================================

hvector<SE3Transform> RandomSE3(size_t n, uint32_t seed, float rot_mag, float trans_mag) {
  return RandomExp<6, SE3Transform>(n, seed, 3, rot_mag, trans_mag, ComputeExpSE3, 4);
}

TEST(SE3ManifoldTest, StateDimensions) { ExpectStateDimensions<SE3Manifold>(6, 16); }

TEST(SE3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  const auto targets = RandomSE3(kN, 42, 0.5f, 2.0f);
  ExpectPriorsConverge<SE3Manifold>(targets, PerturbRight(targets, RandomSE3(kN, 43, 0.1f, 0.3f)),
                                    PriorSettings(1e-2f, 1e-2f));
}

TEST(SE3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  ExpectBetweensAlign<SE3Manifold>(RandomSE3(kN, 60, 0.3f, 1.0f), RandomSE3(kN, 61, 0.3f, 1.0f),
                                   Identity<SE3Transform>(), BetweenSettings(0.1f));
}

// ============================================================================
// Sim(3)
// ============================================================================

/**
 * Sim(3) 4x4 matrix from the tangent [w, u, lambda]: Rodrigues for R, the
 * V-matrix for the translation, exp(-lambda) as the bottom-right entry.
 */
Matrix<4> MakeSim3(float w1, float w2, float w3, float u1, float u2, float u3, float lambda) {
  const float theta2 = w1 * w1 + w2 * w2 + w3 * w3;
  const float theta = std::sqrt(theta2);

  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    const float st = std::sin(theta), ct = std::cos(theta);
    A1 = st / theta;
    A2 = (1.0f - ct) / theta2;
    A3 = (1.0f - A1) / theta2;
    A4 = (0.5f - A2) / theta2;
  } else {
    A1 = 1.0f - theta2 / 6.0f;
    A2 = 0.5f - theta2 / 24.0f;
    A3 = 1.0f / 6.0f - theta2 / 120.0f;
    A4 = 1.0f / 24.0f - theta2 / 720.0f;
  }

  const float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    const float e = std::exp(-lambda);
    P_c = (1.0f - e) / lambda;
    const float alpha = lambda2 / (lambda2 + theta2);
    const float beta = (e - 1.0f + lambda) / lambda2;
    const float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    const float one_m_a = 1.0f - alpha;
    Q_c = alpha * beta + one_m_a * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_a * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  const float ct_val = 1.0f - A2 * theta2;
  Matrix<4> T;
  T[0] = ct_val + A2 * w1 * w1;
  T[1] = A2 * w1 * w2 - A1 * w3;
  T[2] = A2 * w1 * w3 + A1 * w2;
  T[4] = A2 * w2 * w1 + A1 * w3;
  T[5] = ct_val + A2 * w2 * w2;
  T[6] = A2 * w2 * w3 - A1 * w1;
  T[8] = A2 * w3 * w1 - A1 * w2;
  T[9] = A2 * w3 * w2 + A1 * w1;
  T[10] = ct_val + A2 * w3 * w3;

  const float diag = P_c - R_c * theta2;
  const float dot_wu = w1 * u1 + w2 * u2 + w3 * u3;
  const float cx = w2 * u3 - w3 * u2;
  const float cy = w3 * u1 - w1 * u3;
  const float cz = w1 * u2 - w2 * u1;
  T[3] = diag * u1 + Q_c * cx + R_c * w1 * dot_wu;
  T[7] = diag * u2 + Q_c * cy + R_c * w2 * dot_wu;
  T[11] = diag * u3 + Q_c * cz + R_c * w3 * dot_wu;

  T[12] = 0.0f;
  T[13] = 0.0f;
  T[14] = 0.0f;
  T[15] = std::exp(-lambda);
  return T;
}

/** MakeSim3 of a tangent drawn uniformly in ±rot, ±trans, ±scale. */
Matrix<4> RandomSim3(std::mt19937 &rng, float rot, float trans, float scale) {
  std::uniform_real_distribution<float> r(-rot, rot), t(-trans, trans), s(-scale, scale);
  const float w1 = r(rng), w2 = r(rng), w3 = r(rng);
  const float u1 = t(rng), u2 = t(rng), u3 = t(rng);
  return MakeSim3(w1, w2, w3, u1, u2, u3, s(rng));
}

TEST(Sim3ManifoldTest, StateDimensions) { ExpectStateDimensions<Sim3Manifold>(7, 16); }

TEST(Sim3ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 10000;
  std::mt19937 rng(42);
  hvector<Matrix<4>> targets(kN), initials(kN);
  for (size_t i = 0; i < kN; ++i) {
    targets[i] = RandomSim3(rng, 0.5f, 2.0f, 0.3f);
    initials[i] = MatMul(targets[i], RandomSim3(rng, 0.1f, 0.3f, 0.05f));
  }
  ExpectPriorsConverge<Sim3Manifold>(targets, initials, PriorSettings(1e-2f, 1e-2f));
}

TEST(Sim3ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 2000;
  std::mt19937 rng1(70), rng2(71);
  hvector<Matrix<4>> left(kN), right(kN);
  for (size_t i = 0; i < kN; ++i) {
    left[i] = RandomSim3(rng1, 0.3f, 0.5f, 0.2f);
    right[i] = RandomSim3(rng2, 0.3f, 0.5f, 0.2f);
  }
  ExpectBetweensAlign<Sim3Manifold>(left, right, Identity<Matrix<4>>(), BetweenSettings(1e-4f));
}

// ============================================================================
// SL(4)
// ============================================================================

hvector<SL4Transform> RandomSL4(size_t n, uint32_t seed, float magnitude) {
  return RandomExp<15, SL4Transform>(n, seed, 15, magnitude, magnitude, ComputeExpSL4, 4);
}

TEST(SL4ManifoldTest, StateDimensions) { ExpectStateDimensions<SL4Manifold>(15, 16); }

TEST(SL4ManifoldTest, PriorLMConvergence) {
  constexpr size_t kN = 800;
  const auto targets = RandomSL4(kN, 44, 0.08f);
  ExpectPriorsConverge<SL4Manifold>(targets, PerturbRight(targets, RandomSL4(kN, 45, 0.04f)),
                                    {80, 1e-7f, kNoCostBound, 0.15f});
}

TEST(SL4ManifoldTest, BetweenLMConvergence) {
  constexpr size_t kN = 10000;
  ExpectBetweensAlign<SL4Manifold>(RandomSL4(kN, 50, 0.5f), RandomSL4(kN, 51, 0.5f),
                                   Identity<SL4Transform>(), BetweenSettings(0.15f));
}

}  // namespace
}  // namespace cunls
