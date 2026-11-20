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

// Accuracy of the exp / log maps in float32 across scales, from near-identity
// elements (where series and closed forms switch) to large ones; the
// exponentials of SE2, Sim2 and Sim3 are the state batches' Plus from the
// identity: SE2 exp and log against a double-precision closed form, and
// log(exp(xi)) round trips for SE2, Sim3 and SL4; SE3 exp and log against a
// double-precision closed form; Sim2 log and the fused Sim2 Plus against a
// double-precision closed form; the fused Sim3 Plus against a double-precision
// matrix exponential; the Sim2 and Sim3 logs at and near a half-turn.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"

namespace cunls {
namespace {

constexpr size_t kN = 512;

std::vector<float> RandomTangents(size_t dim, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.f, scale);
  std::vector<float> xi(kN * dim);
  for (auto &v : xi) v = normal(rng);
  return xi;
}

/**
 * Scales rotation parts to angles below 2.8 rad, where log is unique (beyond
 * pi it wraps by 2 pi). `offset`/`count`: the rotation components of each
 * tangent of size `dim`.
 */
void ClampRotations(std::vector<float> &xi, size_t dim, size_t offset, size_t count) {
  for (size_t i = 0; i < xi.size() / dim; ++i) {
    float *w = xi.data() + i * dim + offset;
    float norm = 0.f;
    for (size_t k = 0; k < count; ++k) norm += w[k] * w[k];
    norm = std::sqrt(norm);
    if (norm > 2.8f) {
      for (size_t k = 0; k < count; ++k) w[k] *= 2.8f / norm;
    }
  }
}

std::vector<float> ToHost(const dvector<float> &d) {
  std::vector<float> h(d.size());
  d.CopyToHost(h.data(), h.size());
  return h;
}

/** Max |a - b| over the arrays. */
double MaxError(const std::vector<float> &a, const std::vector<float> &b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

/**
 * Exp of `count` tangents through a state batch's Plus from the identity (the
 * library's exponential map for that group), `dim` x `dim` matrices.
 */
template <class States>
dvector<float> ExpThroughPlus(const dvector<float> &xi, int dim, cudaStream_t stream,
                              size_t count = kN) {
  const size_t ambient = static_cast<size_t>(dim) * dim;
  std::vector<float> identity(count * ambient, 0.f);
  for (size_t i = 0; i < count; ++i) {
    for (int k = 0; k < dim; ++k) identity[i * ambient + k * (dim + 1)] = 1.f;
  }
  dvector<float> d_x(identity), out(count * ambient);
  States states(d_x.data(), count);
  states.SetNumActiveStates(count);
  states.Plus(d_x.data(), xi.data(), out.data(), stream);
  return out;
}

class LieMapAccuracy : public ::testing::TestWithParam<float> {};

TEST_P(LieMapAccuracy, SE2ExpAndLogMatchDoublePrecision) {
  const float scale = GetParam();
  CudaStream stream;
  std::vector<float> xi = RandomTangents(3, scale, 1);
  ClampRotations(xi, 3, 2, 1);
  // Reference in double: T = [R(w), V(w) v; 0 1], V = [[s/w, -(1-c)/w], [(1-c)/w, s/w]].
  std::vector<float> ref(kN * 9);
  for (size_t i = 0; i < kN; ++i) {
    const double vx = xi[3 * i], vy = xi[3 * i + 1], w = xi[3 * i + 2];
    const double c = std::cos(w), s = std::sin(w);
    const double a = std::fabs(w) < 1e-8 ? 1.0 - w * w / 6.0 : s / w;
    const double b = std::fabs(w) < 1e-8 ? w / 2.0 : (1.0 - c) / w;
    const double t[9] = {c, -s, a * vx - b * vy, s, c, b * vx + a * vy, 0.0, 0.0, 1.0};
    std::copy(t, t + 9, ref.begin() + 9 * i);
  }
  dvector<float> d_xi(xi), d_log(kN * 3);
  const dvector<float> d_t = ExpThroughPlus<SE2StateBatch>(d_xi, 3, stream.GetStream());
  ComputeLogSE2(stream.GetStream(), d_t.data(), 9, 3, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const float tol = 2e-6f * std::max(1.f, scale);
  EXPECT_LT(MaxError(ToHost(d_t), ref), tol) << "exp, scale " << scale;
  EXPECT_LT(MaxError(ToHost(d_log), xi), 5 * tol) << "log(exp), scale " << scale;
}

// SE2StateBatch::Plus from the identity is Exp(delta). With unit translations
// and rotations below 1e-3 the translation needs the first-order term
// (1 - cos w) / w ~ w / 2: dropping it errs by |v| |w| / 2 (up to 5e-4 here).
TEST(LieMapAccuracy, SE2StatePlusKeepsTheFirstOrderTermAtSmallAngles) {
  CudaStream stream;
  std::mt19937 rng(7);
  std::normal_distribution<float> unit(0.f, 1.f);
  std::uniform_real_distribution<float> small(-1e-3f, 1e-3f);
  std::vector<float> xi(kN * 3), identity(kN * 9, 0.f), ref(kN * 9);
  for (size_t i = 0; i < kN; ++i) {
    xi[3 * i] = unit(rng);
    xi[3 * i + 1] = unit(rng);
    xi[3 * i + 2] = small(rng);
    identity[9 * i] = identity[9 * i + 4] = identity[9 * i + 8] = 1.f;
    const double vx = xi[3 * i], vy = xi[3 * i + 1], w = xi[3 * i + 2];
    const double c = std::cos(w), s = std::sin(w);
    const double a = std::fabs(w) < 1e-8 ? 1.0 - w * w / 6.0 : s / w;
    const double b = std::fabs(w) < 1e-8 ? w / 2.0 : (1.0 - c) / w;
    const double t[9] = {c, -s, a * vx - b * vy, s, c, b * vx + a * vy, 0.0, 0.0, 1.0};
    std::copy(t, t + 9, ref.begin() + 9 * i);
  }
  dvector<float> d_x(identity), d_xi(xi), d_out(kN * 9);
  SE2StateBatch states(d_x.data(), kN);
  states.SetNumActiveStates(kN);
  states.Plus(d_x.data(), d_xi.data(), d_out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_out), ref), 2e-6);
}

/** exp(X) of a row-major 4x4 matrix in double: scaling and squaring of the Taylor series. */
void MatrixExp4(const double *X, double *E) {
  double norm = 0.0;
  for (int i = 0; i < 16; ++i) norm = std::max(norm, std::fabs(X[i]));
  int squarings = 0;
  while (norm > 0.25) {
    norm *= 0.5;
    ++squarings;
  }
  double A[16], term[16], next[16];
  for (int i = 0; i < 16; ++i) {
    A[i] = std::ldexp(X[i], -squarings);
    term[i] = E[i] = (i % 5 == 0) ? 1.0 : 0.0;
  }
  const auto multiply = [](const double *a, const double *b, double *c) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) sum += a[4 * i + k] * b[4 * k + j];
        c[4 * i + j] = sum;
      }
    }
  };
  for (int n = 1; n <= 20; ++n) {
    multiply(term, A, next);
    for (int i = 0; i < 16; ++i) {
      term[i] = next[i] / n;
      E[i] += term[i];
    }
  }
  for (int k = 0; k < squarings; ++k) {
    multiply(E, E, next);
    std::copy(next, next + 16, E);
  }
}

/**
 * T = Exp(xi) of Sim(3) for xi = [w, u, lambda], computed in double: the
 * matrix exponential of [[w]_x, u; 0, -lambda] (row-major 4x4).
 */
void Sim3Exp(const float *xi, float *T) {
  const double w1 = xi[0], w2 = xi[1], w3 = xi[2], lambda = xi[6];
  const double X[16] = {0.0, -w3, w2,  xi[3], w3,  0.0, -w1, xi[4],
                        -w2, w1,  0.0, xi[5], 0.0, 0.0, 0.0, -lambda};
  double E[16];
  MatrixExp4(X, E);
  std::copy(E, E + 16, T);
}

// Similarity3StateBatch::Plus from the identity is Exp(delta), the matrix
// exponential of [[w]_x, u; 0, -lambda]. With unit translations and log-scales
// between 1e-4 and 1e-2 the coefficients of V = P I + Q [w]_x + R [w]_x^2 must
// not be computed by their closed forms, which cancel in float32:
// (1 - e^-lambda) / lambda alone carries the rounding of e^-lambda, up to
// 6e-8 / |lambda| (6e-4 at 1e-4).
TEST(LieMapAccuracy, Sim3StatePlusStaysAccurateAtSmallLogScales) {
  CudaStream stream;
  std::mt19937 rng(11);
  std::normal_distribution<float> unit(0.f, 1.f), rotation(0.f, 0.3f);
  std::uniform_real_distribution<float> log_scale(-4.f, -2.f);
  std::vector<float> xi(kN * 7), identity(kN * 16, 0.f), ref(kN * 16);
  for (size_t i = 0; i < kN; ++i) {
    float *d = xi.data() + 7 * i;
    for (int k = 0; k < 3; ++k) d[k] = rotation(rng);
    for (int k = 3; k < 6; ++k) d[k] = unit(rng);
    d[6] = (i % 2 == 0 ? 1.f : -1.f) * std::pow(10.f, log_scale(rng));
    for (int k = 0; k < 4; ++k) identity[16 * i + 5 * k] = 1.f;
    Sim3Exp(d, ref.data() + 16 * i);
  }
  dvector<float> d_x(identity), d_xi(xi), d_out(kN * 16);
  Similarity3StateBatch states(d_x.data(), kN);
  states.SetNumActiveStates(kN);
  states.Plus(d_x.data(), d_xi.data(), d_out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_out), ref), 2e-6);
}

// SE(3) with small rotations and unit translations: the left Jacobian that
// maps rho to the translation must stay accurate where its coefficients
// cancel in float32 ((1 - cos θ)/θ² was 100% wrong at θ = 1e-4).
TEST_P(LieMapAccuracy, SE3ExpAndLogMatchDoublePrecision) {
  const float scale = GetParam();
  CudaStream stream;
  std::vector<float> xi = RandomTangents(6, 1.f, 4);
  for (size_t i = 0; i < kN; ++i)
    for (int k = 0; k < 3; ++k) xi[6 * i + k] *= scale;  // rotation at this scale
  ClampRotations(xi, 6, 0, 3);
  std::vector<float> ref(kN * 16);
  for (size_t i = 0; i < kN; ++i) {
    const double w[3] = {xi[6 * i], xi[6 * i + 1], xi[6 * i + 2]};
    const double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    const double a = th < 1e-6 ? 1 - th * th / 6 : std::sin(th) / th;
    const double b = th < 1e-6 ? 0.5 - th * th / 24 : (1 - std::cos(th)) / (th * th);
    const double c = th < 1e-4 ? 1.0 / 6 - th * th / 120 : (th - std::sin(th)) / (th * th * th);
    const double K[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
    double K2[9] = {};
    for (int r = 0; r < 3; ++r)
      for (int col = 0; col < 3; ++col)
        for (int k = 0; k < 3; ++k) K2[r * 3 + col] += K[r * 3 + k] * K[k * 3 + col];
    double T[16] = {};
    for (int r = 0; r < 3; ++r) {
      for (int col = 0; col < 3; ++col) {
        const double id = r == col ? 1.0 : 0.0;
        T[r * 4 + col] = id + a * K[r * 3 + col] + b * K2[r * 3 + col];
        const double J = id + b * K[r * 3 + col] + c * K2[r * 3 + col];
        T[r * 4 + 3] += J * xi[6 * i + 3 + col];
      }
    }
    T[15] = 1;
    std::copy(T, T + 16, ref.begin() + 16 * i);
  }
  dvector<float> d_xi(xi), d_t(kN * 16), d_log(kN * 6);
  ComputeExpSE3(stream.GetStream(), d_xi.data(), 6, 4, 16, kN, d_t.data());
  ComputeLogSE3(stream.GetStream(), d_t.data(), 4, 16, 6, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_t), ref), 2e-6) << "exp, rotation scale " << scale;
  EXPECT_LT(MaxError(ToHost(d_log), xi), 1e-5) << "log(exp), rotation scale " << scale;
}

/**
 * T = Exp(xi) of Sim(2) for xi = [u, theta, lambda] (lambda != 0 or theta !=
 * 0), computed in double: T = [R(theta), V u; 0, e^-lambda], where V u is the
 * complex product (e^{i theta} - e^-lambda) / (lambda + i theta) * (u_x + i u_y).
 */
void Sim2Exp(const float *xi, float *T) {
  const double theta = xi[2], lambda = xi[3];
  const double c = std::cos(theta), s = std::sin(theta);
  const std::complex<double> z(lambda, theta);
  const std::complex<double> v = (std::polar(1.0, theta) - std::exp(-lambda)) / z;
  const std::complex<double> t = v * std::complex<double>(xi[0], xi[1]);
  const double E[9] = {c, -s, t.real(), s, c, t.imag(), 0.0, 0.0, std::exp(-lambda)};
  std::copy(E, E + 9, T);
}

/**
 * Sim(2) tangents [u, theta, lambda] with unit translations, rotations of 0.3
 * rad and log-scales of either sign between 1e-5 and 1e-2 in magnitude, and
 * Exp of each computed in double (Sim2Exp).
 */
void SmallLogScaleSim2(std::vector<float> &xi, std::vector<float> &ref) {
  std::mt19937 rng(13);
  std::normal_distribution<float> unit(0.f, 1.f), rotation(0.f, 0.3f);
  std::uniform_real_distribution<float> log_scale(-5.f, -2.f);
  xi.resize(kN * 4);
  ref.resize(kN * 9);
  for (size_t i = 0; i < kN; ++i) {
    float *d = xi.data() + 4 * i;
    d[0] = unit(rng);
    d[1] = unit(rng);
    d[2] = rotation(rng);
    d[3] = (i % 2 == 0 ? 1.f : -1.f) * std::pow(10.f, log_scale(rng));
    Sim2Exp(d, ref.data() + 9 * i);
  }
}

// With unit translations and small log-scales the Sim(2) translation V u needs
// the first-order term in lambda (dropping it below 1e-4 errs by |u| |lambda| / 2)
// and coefficients that do not cancel in float32: (1 - e^-lambda) / lambda
// carries the rounding of e^-lambda, up to 6e-8 / |lambda| (6e-4 at 1e-4).
// The log inverts the double-precision exponentials.
TEST(LieMapAccuracy, Sim2LogStaysAccurateAtSmallLogScales) {
  CudaStream stream;
  std::vector<float> xi, ref;
  SmallLogScaleSim2(xi, ref);
  dvector<float> d_ref(ref), d_log(kN * 4);
  ComputeLogSim2(stream.GetStream(), d_ref.data(), 9, 4, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_log), xi), 1e-5) << "log of the reference transforms";
}

// Similarity2StateBatch::Plus from the identity is Exp(delta).
TEST(LieMapAccuracy, Sim2StatePlusStaysAccurateAtSmallLogScales) {
  CudaStream stream;
  std::vector<float> xi, ref, identity(kN * 9, 0.f);
  SmallLogScaleSim2(xi, ref);
  for (size_t i = 0; i < kN; ++i) identity[9 * i] = identity[9 * i + 4] = identity[9 * i + 8] = 1.f;
  dvector<float> d_x(identity), d_xi(xi), d_out(kN * 9);
  Similarity2StateBatch states(d_x.data(), kN);
  states.SetNumActiveStates(kN);
  states.Plus(d_x.data(), d_xi.data(), d_out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_out), ref), 4e-6);
}

TEST_P(LieMapAccuracy, Sim3LogInvertsExp) {
  const float scale = GetParam();
  CudaStream stream;
  std::vector<float> xi = RandomTangents(7, scale, 2);
  ClampRotations(xi, 7, 0, 3);
  dvector<float> d_xi(xi), d_log(kN * 7);
  const dvector<float> d_t = ExpThroughPlus<Similarity3StateBatch>(d_xi, 4, stream.GetStream());
  ComputeLogSim3(stream.GetStream(), d_t.data(), 16, 7, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_LT(MaxError(ToHost(d_log), xi), 2e-5 * std::max(1.f, scale)) << "scale " << scale;
}

/** Max |a - b| over row `row` of arrays with `stride` floats per row. */
double MaxRowError(const std::vector<float> &a, const std::vector<float> &b, size_t row,
                   size_t stride) {
  double worst = 0.0;
  for (size_t k = row * stride; k < (row + 1) * stride; ++k) {
    worst = std::max(worst, std::fabs(double(a[k]) - b[k]));
  }
  return worst;
}

// Sim(3) elements at and near a half-turn: rotations of pi and pi - 1e-3 about
// the x axis and an oblique axis, with scales e^{+-0.4} and unit-size
// translations. At pi, R - R^T vanishes, so an angle-axis log that takes the
// axis from the skew part returns w = 0 (Exp(Log(T)) was off by 33), and near
// pi it divides by a sin(theta) that carries the rounding of acos (the log was
// off by up to 0.13 at pi - 1e-3). At pi the axis sign is free: Exp(Log(T))
// must give T back; below pi the log is unique and must match the tangent.
TEST(LieMapAccuracy, Sim3LogHandlesHalfTurns) {
  const double axes[2][3] = {{1.0, 0.0, 0.0}, {1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0}};
  const double angles[2] = {M_PI, M_PI - 1e-3};
  const float u[3] = {0.7f, -1.2f, 2.5f};
  constexpr size_t kCases = 4;
  std::vector<float> xi(kCases * 7), ref(kCases * 16, 0.f);
  for (size_t i = 0; i < kCases; ++i) {
    const double *n = axes[i / 2];
    const double angle = angles[i % 2];
    float *x = xi.data() + 7 * i;
    float *T = ref.data() + 16 * i;
    for (int k = 0; k < 3; ++k) x[k] = static_cast<float>(angle * n[k]);
    std::copy(u, u + 3, x + 3);
    x[6] = i % 2 == 0 ? 0.4f : -0.4f;
    if (i % 2 == 0) {
      // Exact half-turn R = 2 n n^T - I (symmetric: R - R^T = 0), translation u.
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T[4 * r + c] = static_cast<float>(2 * n[r] * n[c] - (r == c));
        T[4 * r + 3] = u[r];
      }
      T[15] = std::exp(-x[6]);
    } else {
      Sim3Exp(x, T);
    }
  }
  CudaStream stream;
  dvector<float> d_ref(ref), d_log(kCases * 7);
  ComputeLogSim3(stream.GetStream(), d_ref.data(), 16, 7, kCases, d_log.data());
  const dvector<float> d_exp =
      ExpThroughPlus<Similarity3StateBatch>(d_log, 4, stream.GetStream(), kCases);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto log = ToHost(d_log);
  const auto exp = ToHost(d_exp);
  for (size_t i = 0; i < kCases; ++i) {
    SCOPED_TRACE("case " + std::to_string(i));
    const float *l = log.data() + 7 * i;  // [w, u, lambda]
    EXPECT_NEAR(std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]), angles[i % 2], 1e-6);
    EXPECT_NEAR(l[6], xi[7 * i + 6], 1e-6);
    EXPECT_LT(MaxRowError(exp, ref, i, 16), 1e-5) << "Exp(Log(T)) != T";
    if (i % 2 == 1) {
      EXPECT_LT(MaxRowError(log, xi, i, 7), 1e-5) << "Log(Exp(xi)) != xi";
    }
  }
}

// Sim(2) at and near a half-turn: theta = atan2(R10, R00) stays exact there
// (pi or -pi at a half-turn, both valid), so Exp(Log(T)) = T at pi and
// Log(Exp(xi)) = xi just below.
TEST(LieMapAccuracy, Sim2LogHandlesHalfTurns) {
  const float u[2] = {0.7f, -1.2f};
  constexpr size_t kCases = 2;
  const double angles[kCases] = {M_PI, M_PI - 1e-3};
  std::vector<float> xi = {u[0], u[1], static_cast<float>(angles[0]), 0.4f,
                           u[0], u[1], static_cast<float>(angles[1]), -0.4f};
  // Exact half-turn R = -I with translation u; below it Exp(xi) in double.
  std::vector<float> ref = {-1.f, 0.f, u[0], 0.f, -1.f, u[1], 0.f, 0.f, std::exp(-0.4f)};
  ref.resize(kCases * 9);
  Sim2Exp(xi.data() + 4, ref.data() + 9);
  CudaStream stream;
  dvector<float> d_ref(ref), d_log(kCases * 4);
  ComputeLogSim2(stream.GetStream(), d_ref.data(), 9, 4, kCases, d_log.data());
  const dvector<float> d_exp =
      ExpThroughPlus<Similarity2StateBatch>(d_log, 3, stream.GetStream(), kCases);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto log = ToHost(d_log);
  const auto exp = ToHost(d_exp);
  for (size_t i = 0; i < kCases; ++i) {
    SCOPED_TRACE("case " + std::to_string(i));
    EXPECT_NEAR(std::fabs(log[4 * i + 2]), angles[i], 1e-6);
    EXPECT_LT(MaxRowError(exp, ref, i, 9), 1e-5) << "Exp(Log(T)) != T";
  }
  EXPECT_LT(MaxRowError(log, xi, 1, 4), 1e-5) << "Log(Exp(xi)) != xi";
}

TEST_P(LieMapAccuracy, SL4LogInvertsExp) {
  const float scale = std::min(GetParam(), 0.5f);  // keep exp(xi) well inside the log's domain
  CudaStream stream;
  std::vector<float> xi = RandomTangents(15, scale, 3);
  dvector<float> d_xi(xi), d_t(kN * 16), d_log(kN * 15);
  ComputeExpSL4(stream.GetStream(), d_xi.data(), 15, 4, 16, kN, d_t.data());
  ComputeLogSL4(stream.GetStream(), d_t.data(), 4, 16, 15, kN, d_log.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  // Before the Denman-Beavers fix the matrix square root (plain Newton) was
  // unstable and the log was off by up to 5 for ~2% of elements at scale 0.5.
  EXPECT_LT(MaxError(ToHost(d_log), xi), 5e-4) << "scale " << scale;
}

INSTANTIATE_TEST_SUITE_P(Scales, LieMapAccuracy, ::testing::Values(1e-4f, 1e-2f, 0.3f, 1.0f));

}  // namespace
}  // namespace cunls
