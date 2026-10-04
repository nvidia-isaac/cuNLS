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

// ImuFactorBatch: against a float64 reference of its residual (Jacobians by
// central differences, several chain lengths, an IMU-body extrinsic), against
// the Schur complement of the explicit chain with every intermediate state as
// a variable (float64, dense), zero residual on a noiseless chain, and a
// keyframe trajectory recovered by LevenbergMarquardtMinimizer.

#include "cunls/factor/imu_factor_batch.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "dynamics_test_support.h"

namespace cunls {
namespace {

using namespace dynamics_test;

// --- Dense float64 helpers (row-major) ------------------------------------------

Vec MatMul(const Vec &A, const Vec &B, int n, int k, int m) {
  Vec C(static_cast<size_t>(n) * m, 0.0);
  for (int i = 0; i < n; ++i)
    for (int l = 0; l < k; ++l) {
      const double a = A[static_cast<size_t>(i) * k + l];
      if (a == 0.0) continue;
      for (int j = 0; j < m; ++j)
        C[static_cast<size_t>(i) * m + j] += a * B[static_cast<size_t>(l) * m + j];
    }
  return C;
}

Vec Transpose(const Vec &A, int n, int m) {
  Vec T(A.size());
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      T[static_cast<size_t>(j) * n + i] = A[static_cast<size_t>(i) * m + j];
  return T;
}

/** Lower Cholesky factor of an SPD n x n matrix. */
Vec Cholesky(const Vec &A, int n) {
  Vec L(static_cast<size_t>(n) * n, 0.0);
  for (int j = 0; j < n; ++j) {
    double d = A[static_cast<size_t>(j) * n + j];
    for (int k = 0; k < j; ++k) d -= L[j * n + k] * L[j * n + k];
    EXPECT_GT(d, 0.0) << "not positive definite";
    L[static_cast<size_t>(j) * n + j] = std::sqrt(d);
    for (int i = j + 1; i < n; ++i) {
      double v = A[static_cast<size_t>(i) * n + j];
      for (int k = 0; k < j; ++k)
        v -= L[static_cast<size_t>(i) * n + k] * L[static_cast<size_t>(j) * n + k];
      L[static_cast<size_t>(i) * n + j] = v / L[static_cast<size_t>(j) * n + j];
    }
  }
  return L;
}

Vec LowerInverse(const Vec &L, int n) {
  Vec X(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    X[i * n + i] = 1.0 / L[i * n + i];
    for (int j = 0; j < i; ++j) {
      double v = 0.0;
      for (int k = j; k < i; ++k) v += L[i * n + k] * X[k * n + j];
      X[i * n + j] = -v / L[i * n + i];
    }
  }
  return X;
}

/** Solves A X = B for SPD A (n x n), B n x m. */
Vec SolveSpd(const Vec &A, const Vec &B, int n, int m) {
  const Vec L = Cholesky(A, n);
  Vec X = B;
  for (int c = 0; c < m; ++c) {
    for (int i = 0; i < n; ++i) {
      double v = X[static_cast<size_t>(i) * m + c];
      for (int k = 0; k < i; ++k)
        v -= L[static_cast<size_t>(i) * n + k] * X[static_cast<size_t>(k) * m + c];
      X[static_cast<size_t>(i) * m + c] = v / L[static_cast<size_t>(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      double v = X[static_cast<size_t>(i) * m + c];
      for (int k = i + 1; k < n; ++k)
        v -= L[static_cast<size_t>(k) * n + i] * X[static_cast<size_t>(k) * m + c];
      X[static_cast<size_t>(i) * m + c] = v / L[static_cast<size_t>(i) * n + i];
    }
  }
  return X;
}

Vec Inverse3(const Vec &J) {
  const double det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
                     J[2] * (J[3] * J[7] - J[4] * J[6]);
  return {(J[4] * J[8] - J[5] * J[7]) / det, (J[2] * J[7] - J[1] * J[8]) / det,
          (J[1] * J[5] - J[2] * J[4]) / det, (J[5] * J[6] - J[3] * J[8]) / det,
          (J[0] * J[8] - J[2] * J[6]) / det, (J[2] * J[3] - J[0] * J[5]) / det,
          (J[3] * J[7] - J[4] * J[6]) / det, (J[1] * J[6] - J[0] * J[7]) / det,
          (J[0] * J[4] - J[1] * J[3]) / det};
}

Vec Skew(const double *v) { return {0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}; }

Vec Rot(const Vec &T) { return {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]}; }

Vec MulVec3(const Vec &R, const double *x) {
  return {R[0] * x[0] + R[1] * x[1] + R[2] * x[2], R[3] * x[0] + R[4] * x[1] + R[5] * x[2],
          R[6] * x[0] + R[7] * x[1] + R[8] * x[2]};
}

Vec RoundToFloat(const Vec &v) {
  Vec r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = static_cast<float>(v[i]);
  return r;
}

Vec Pose(const Vec &R, const double *p) {
  return {R[0], R[1], R[2], p[0], R[3], R[4], R[5], p[1], R[6], R[7], R[8], p[2], 0, 0, 0, 1};
}

// --- Float64 model (independent of the kernel) ------------------------------------

struct Sample {
  double w[3], a[3], dt;
};
using Samples = std::vector<Sample>;

struct Model {
  double g[3], gyro_var, accel_var, integration_var, gyro_bias_var, accel_bias_var;
  Vec body_from_imu;
};

Model ToModel(const ImuParameters &p) {
  Model m;
  for (int i = 0; i < 3; ++i) m.g[i] = p.gravity[i];
  m.gyro_var = double(p.gyro_noise_density) * p.gyro_noise_density;
  m.accel_var = double(p.accel_noise_density) * p.accel_noise_density;
  m.integration_var = double(p.integration_noise_density) * p.integration_noise_density;
  m.gyro_bias_var = double(p.gyro_bias_random_walk) * p.gyro_bias_random_walk;
  m.accel_bias_var = double(p.accel_bias_random_walk) * p.accel_bias_random_walk;
  m.body_from_imu.assign(p.body_from_imu.begin(), p.body_from_imu.end());
  return m;
}

struct NavState {
  Vec R;
  double v[3], p[3];
};

/** One Euler step of the IMU state with bias b = [b_g; b_a]. */
NavState Step(const Model &m, const NavState &x, const Sample &s, const double *b) {
  const double h = s.dt;
  const double th[3] = {(s.w[0] - b[0]) * h, (s.w[1] - b[1]) * h, (s.w[2] - b[2]) * h};
  const double a[3] = {s.a[0] - b[3], s.a[1] - b[4], s.a[2] - b[5]};
  const Vec Ra = MulVec3(x.R, a);
  NavState y;
  y.R = Mul3(x.R, ExpSO3(th));
  for (int i = 0; i < 3; ++i) {
    const double acc = Ra[i] + m.g[i];
    y.v[i] = x.v[i] + h * acc;
    y.p[i] = x.p[i] + h * x.v[i] + 0.5 * h * h * acc;
  }
  return y;
}

/** Covariance of the predicted state (tangent [δφ, δv, δp]) by dense Φ Σ Φᵀ + Q. */
Vec PredictedCovariance(const Model &m, const Samples &samples, Vec R, const double *b) {
  Vec S(81, 0.0);
  for (const Sample &s : samples) {
    const double h = s.dt;
    const double th[3] = {(s.w[0] - b[0]) * h, (s.w[1] - b[1]) * h, (s.w[2] - b[2]) * h};
    const double mth[3] = {-th[0], -th[1], -th[2]};
    const double a[3] = {s.a[0] - b[3], s.a[1] - b[4], s.a[2] - b[5]};
    const Vec E = ExpSO3(th);
    const auto Jr_arr = Rodrigues(mth, true);  // J_r(θ) = J_l(-θ)
    const Vec Jr(Jr_arr.begin(), Jr_arr.end());
    const Vec M = MatMul(R, Skew(a), 3, 3, 3);
    Vec Phi(81, 0.0), Q(81, 0.0);
    for (int i = 0; i < 9; ++i) Phi[i * 9 + i] = 1.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        Phi[i * 9 + j] = E[j * 3 + i];
        Phi[(3 + i) * 9 + j] = -h * M[i * 3 + j];
        Phi[(6 + i) * 9 + j] = -0.5 * h * h * M[i * 3 + j];
      }
    for (int i = 0; i < 3; ++i) Phi[(6 + i) * 9 + 3 + i] = h;
    const Vec JJ = MatMul(Jr, Transpose(Jr, 3, 3), 3, 3, 3);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) Q[i * 9 + j] = m.gyro_var * h * JJ[i * 3 + j];
      Q[(3 + i) * 9 + 3 + i] = m.accel_var * h;
      Q[(3 + i) * 9 + 6 + i] = Q[(6 + i) * 9 + 3 + i] = 0.5 * m.accel_var * h * h;
      Q[(6 + i) * 9 + 6 + i] = 0.25 * m.accel_var * h * h * h + m.integration_var * h;
    }
    S = MatMul(MatMul(Phi, S, 9, 9, 9), Transpose(Phi, 9, 9), 9, 9, 9);
    for (int i = 0; i < 81; ++i) S[i] += Q[i];
    R = Mul3(R, E);
  }
  return S;
}

/** IMU state at a keyframe from the pose slot (body_from_world) and the velocity slot. */
NavState ImuState(const Model &m, const Vec &X, const Vec &v) {
  const Vec T = Mul4(Inv4(X), m.body_from_imu);  // world_from_imu
  return {Rot(T), {v[0], v[1], v[2]}, {T[3], T[7], T[11]}};
}

NavState Integrate(const Model &m, const Samples &samples, NavState x, const double *b) {
  for (const Sample &s : samples) x = Step(m, x, s, b);
  return x;
}

double Duration(const Samples &samples) {
  double T = 0;
  for (const Sample &s : samples) T += s.dt;
  return T;
}

/** e = [Log(R̂ᵀ R_b); v_b - v̂; p_b - p̂] for slots (T_a, v_a, b_a, T_b, v_b, b_b). */
Vec Defect(const Model &m, const Samples &samples, const std::vector<Vec> &s) {
  const NavState xb = ImuState(m, s[3], s[4]);
  const NavState pred = Integrate(m, samples, ImuState(m, s[0], s[1]), s[2].data());
  Vec e = LogSO3(Mul3(TransposeSO3(pred.R), xb.R));
  for (int i = 0; i < 3; ++i) e.push_back(xb.v[i] - pred.v[i]);
  for (int i = 0; i < 3; ++i) e.push_back(xb.p[i] - pred.p[i]);
  return e;
}

/** Whitening of the base point, held fixed under perturbations (as in Gauss-Newton). */
struct Whitening {
  Vec W;          // L⁻¹ D⁻¹, 9x9
  double inv[6];  // bias random walk 1 / σ
};

Whitening BaseWhitening(const Model &m, const Samples &samples, const std::vector<Vec> &s) {
  const NavState xa = ImuState(m, s[0], s[1]);
  const Vec Sigma = PredictedCovariance(m, samples, xa.R, s[2].data());
  const Vec e = Defect(m, samples, s);
  const auto Jl = Rodrigues(e.data(), true);
  Vec Dinv(81, 0.0);
  for (int i = 3; i < 9; ++i) Dinv[i * 9 + i] = 1.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) Dinv[i * 9 + j] = Jl[i * 3 + j];
  Whitening w;
  w.W = MatMul(LowerInverse(Cholesky(Sigma, 9), 9), Dinv, 9, 9, 9);
  const double T = Duration(samples);
  for (int i = 0; i < 3; ++i) {
    w.inv[i] = 1.0 / std::sqrt(m.gyro_bias_var * T);
    w.inv[3 + i] = 1.0 / std::sqrt(m.accel_bias_var * T);
  }
  return w;
}

Vec Reference(const Model &m, const Samples &samples, const Whitening &w,
              const std::vector<Vec> &s) {
  Vec r = MatMul(w.W, Defect(m, samples, s), 9, 9, 1);
  for (int i = 0; i < 6; ++i) r.push_back(w.inv[i] * (s[5][i] - s[2][i]));
  return r;
}

// --- Test data --------------------------------------------------------------------

ImuParameters TestParameters() {
  ImuParameters p;
  p.gyro_noise_density = 1e-2f;
  p.accel_noise_density = 1e-1f;
  p.integration_noise_density = 1e-2f;
  p.gyro_bias_random_walk = 1e-3f;
  p.accel_bias_random_walk = 1e-2f;
  const double xi[6] = {0.1, -0.2, 0.3, 0.05, -0.02, 0.1};
  const Vec T = Exp6(xi);
  for (int i = 0; i < 16; ++i) p.body_from_imu[i] = static_cast<float>(T[i]);
  return p;
}

Samples RandomSamples(std::mt19937 &rng, int n) {
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.8, 1.2);
  Samples samples(n);
  for (Sample &s : samples) {
    for (int i = 0; i < 3; ++i) {
      s.w[i] = 0.7 * normal(rng);
      s.a[i] = normal(rng) + (i == 2 ? 9.8 : 0.0);
    }
    s.dt = 0.005 * uniform(rng);
  }
  return samples;
}

/**
 * States of one factor: a random keyframe a, and keyframe b at the
 * prediction perturbed by `noise` (rotation, velocity and position).
 */
std::vector<Slot> RandomItem(std::mt19937 &rng, const Model &m, const Samples &samples,
                             double noise) {
  std::normal_distribution<double> normal(0.0, 1.0);
  double x[6];
  for (double &c : x) c = normal(rng);
  const Vec Xa = Exp6(x);  // body_from_world
  const Vec va = {normal(rng), normal(rng), normal(rng)};
  Vec ba(6);
  for (int i = 0; i < 6; ++i) ba[i] = (i < 3 ? 0.02 : 0.2) * normal(rng);
  // Round the keyframe a states to float first: the prediction must start from them.
  const Vec Xa_f = RoundToFloat(Xa), va_f = RoundToFloat(va), ba_f = RoundToFloat(ba);
  const NavState pred = Integrate(m, samples, ImuState(m, Xa_f, va_f), ba_f.data());
  double d[6];
  for (int i = 0; i < 6; ++i) d[i] = noise * normal(rng);
  const Vec Tb_imu = Mul4(Pose(pred.R, pred.p), Exp6(d));
  const Vec Xb = Mul4(m.body_from_imu, Inv4(Tb_imu));  // body_from_world
  Vec vb(3), bb(6);
  for (int i = 0; i < 3; ++i) vb[i] = pred.v[i] + noise * normal(rng);
  for (int i = 0; i < 6; ++i) bb[i] = ba_f[i] + 0.01 * normal(rng);
  return {{SlotKind::kSE3, ToFloat(Xa_f)},    {SlotKind::kVector, ToFloat(va_f)},
          {SlotKind::kVector, ToFloat(ba_f)}, {SlotKind::kSE3, ToFloat(Xb)},
          {SlotKind::kVector, ToFloat(vb)},   {SlotKind::kVector, ToFloat(bb)}};
}

std::vector<Vec> SlotValues(const std::vector<Slot> &slots) {
  return Perturbed(slots, Vec(30, 0.0));
}

/** Device copies of the samples (7 floats each) and CSR offsets. */
struct DeviceSamples {
  dvector<float> samples;
  dvector<int> offsets;
  size_t count = 0;  // samples
  /** One factor per entry of per_factor. */
  explicit DeviceSamples(const std::vector<Samples> &per_factor)
      : DeviceSamples(per_factor, Indices(per_factor.size()), per_factor.size()) {}
  /**
   * `factors` factors, factor f using base[which[f % which.size()]], flattened
   * directly (large replicated batches never hold Samples copies).
   */
  DeviceSamples(const std::vector<Samples> &base, const std::vector<int> &which, size_t factors) {
    size_t total = 0;
    for (size_t f = 0; f < factors; ++f) total += base[which[f % which.size()]].size();
    std::vector<float> flat;
    flat.reserve(7 * total);
    std::vector<int> off = {0};
    off.reserve(factors + 1);
    for (size_t f = 0; f < factors; ++f) {
      for (const Sample &s : base[which[f % which.size()]]) {
        for (double w : s.w) flat.push_back(static_cast<float>(w));
        for (double a : s.a) flat.push_back(static_cast<float>(a));
        flat.push_back(static_cast<float>(s.dt));
      }
      off.push_back(static_cast<int>(flat.size() / 7));
    }
    samples = dvector<float>(flat);
    offsets = dvector<int>(off);
    count = total;
  }
  static std::vector<int> Indices(size_t n) {
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int>(i);
    return v;
  }
};

/** Rounds the samples to float, as the factor sees them. */
Samples Rounded(Samples samples) {
  for (Sample &s : samples) {
    for (int i = 0; i < 3; ++i) {
      s.w[i] = static_cast<float>(s.w[i]);
      s.a[i] = static_cast<float>(s.a[i]);
    }
    s.dt = static_cast<float>(s.dt);
  }
  return samples;
}

/** Evaluates `factor` on one item per entry; returns residuals and Jacobians (item-major). */
void EvaluateItems(ImuFactorBatch &factor, const std::vector<std::vector<Slot>> &items,
                   std::vector<float> *res, std::vector<float> *jac) {
  std::vector<float> storage;
  std::vector<size_t> offsets;
  for (const auto &item : items)
    for (const Slot &s : item) {
      offsets.push_back(storage.size());
      storage.insert(storage.end(), s.storage.begin(), s.storage.end());
    }
  CudaStream stream;
  dvector<float> d_storage(storage);
  std::vector<float *> ptrs;
  for (size_t o : offsets) ptrs.push_back(d_storage.data() + o);
  dvector<float *> d_ptrs(ptrs);
  const size_t n = items.size();
  dvector<float> d_res(n * 15), d_jac(n * 15 * 30);
  factor.SetNumActiveFactors(n);
  ASSERT_TRUE(factor.Evaluate(d_res.data(), d_jac.data(), d_ptrs.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  res->resize(n * 15);
  jac->resize(n * 15 * 30);
  d_res.CopyToHost(res->data(), res->size());
  d_jac.CopyToHost(jac->data(), jac->size());
}

// --- Tests ------------------------------------------------------------------------

TEST(ImuFactorBatch, MatchesReference) {
  const int kLengths[] = {1, 2, 3, 7, 40, 200};
  constexpr int kItems = 70;  // several warps, a partial warp at the end
  const ImuParameters params = TestParameters();
  const Model m = ToModel(params);
  std::mt19937 rng(7);
  std::vector<Samples> samples;
  std::vector<std::vector<Slot>> items;
  std::vector<Whitening> whitening;
  for (int t = 0; t < kItems; ++t) {
    samples.push_back(Rounded(RandomSamples(rng, kLengths[t % 6])));
    items.push_back(RandomItem(rng, m, samples.back(), t % 2 == 0 ? 1e-3 : 1e-2));
    whitening.push_back(BaseWhitening(m, samples.back(), SlotValues(items.back())));
  }
  DeviceSamples d(samples);
  // The sample count only sizes the work split: one thread per chain, and up
  // to 32 (a typical chain of 512 samples).
  for (size_t hint : {d.count, static_cast<size_t>(512 * kItems)}) {
    ImuFactorBatch factor(d.samples.data(), d.offsets.data(), hint, params, kItems);
    EXPECT_EQ(factor.ResidualsSize(), 15u);
    EXPECT_EQ(factor.StateSizes(), (std::vector<size_t>{6, 3, 6, 6, 3, 6}));
    CheckFactor(
        factor, items,
        [&](int t, const std::vector<Vec> &s) { return Reference(m, samples[t], whitening[t], s); },
        "imu, num_samples " + std::to_string(hint), 5e-4, 5e-4);
  }
}

/**
 * Explicit chain: the keyframe states plus N - 1 intermediate IMU states as
 * variables, one whitened 9-row defect per sample (noise of the step at the
 * linearization point) and the bias random walk rows. Interior states at the
 * forward integration from keyframe a. Returns the stacked residual at the
 * perturbation `delta` (30 keyframe tangent entries, then 9 per interior state).
 */
struct ExplicitChain {
  const Model &m;
  const Samples &samples;
  std::vector<Slot> slots;
  std::vector<NavState> interior;  // k = 1 .. N - 1
  std::vector<Vec> weights;        // per step, 9x9

  ExplicitChain(const Model &model, const Samples &s, const std::vector<Slot> &item)
      : m(model), samples(s), slots(item) {
    const std::vector<Vec> base = SlotValues(slots);
    NavState x = ImuState(m, base[0], base[1]);
    for (size_t k = 0; k + 1 < samples.size(); ++k) {
      x = Step(m, x, samples[k], base[2].data());
      interior.push_back(x);
    }
    // Step weights at the base point: noise Jacobian of each defect.
    const std::vector<NavState> chain = Chain(Vec(Columns(), 0.0));
    for (size_t k = 0; k < samples.size(); ++k) {
      const Sample &s = samples[k];
      const double h = s.dt;
      const double *b = base[2].data();
      const double mth[3] = {-(s.w[0] - b[0]) * h, -(s.w[1] - b[1]) * h, -(s.w[2] - b[2]) * h};
      const auto Jr_arr = Rodrigues(mth, true);
      const Vec eps = DefectRotation(chain[k], chain[k + 1], s, b);
      const auto Jl_arr = Rodrigues(eps.data(), true);
      const Vec G = MatMul(Inverse3(Vec(Jl_arr.begin(), Jl_arr.end())),
                           Vec(Jr_arr.begin(), Jr_arr.end()), 3, 3, 3);
      Vec N(54, 0.0);  // ∂defect/∂[η_g, η_a]
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          N[i * 6 + j] = h * G[i * 3 + j];
          N[(3 + i) * 6 + 3 + j] = h * chain[k].R[i * 3 + j];
          N[(6 + i) * 6 + 3 + j] = 0.5 * h * h * chain[k].R[i * 3 + j];
        }
      Vec Sn(36, 0.0);
      for (int i = 0; i < 3; ++i) {
        Sn[i * 6 + i] = m.gyro_var / h;
        Sn[(3 + i) * 6 + 3 + i] = m.accel_var / h;
      }
      Vec Q = MatMul(MatMul(N, Sn, 9, 6, 6), Transpose(N, 9, 6), 9, 6, 9);
      for (int i = 0; i < 3; ++i) Q[(6 + i) * 9 + 6 + i] += m.integration_var * h;
      weights.push_back(LowerInverse(Cholesky(Q, 9), 9));
    }
  }

  int Columns() const { return 30 + 9 * static_cast<int>(interior.size()); }

  Vec DefectRotation(const NavState &x, const NavState &y, const Sample &s, const double *b) const {
    const NavState f = Step(m, x, s, b);
    return LogSO3(Mul3(TransposeSO3(f.R), y.R));
  }

  std::vector<NavState> Chain(const Vec &delta) const {
    const std::vector<Vec> st = Perturbed(slots, Vec(delta.begin(), delta.begin() + 30));
    std::vector<NavState> chain = {ImuState(m, st[0], st[1])};
    for (size_t k = 0; k < interior.size(); ++k) {
      const double *d = delta.data() + 30 + 9 * k;
      NavState x = interior[k];
      x.R = Mul3(x.R, ExpSO3(d));
      for (int i = 0; i < 3; ++i) {
        x.v[i] += d[3 + i];
        x.p[i] += d[6 + i];
      }
      chain.push_back(x);
    }
    chain.push_back(ImuState(m, st[3], st[4]));
    return chain;
  }

  Vec Residual(const Vec &delta) const {
    const std::vector<Vec> st = Perturbed(slots, Vec(delta.begin(), delta.begin() + 30));
    const std::vector<NavState> chain = Chain(delta);
    Vec r;
    for (size_t k = 0; k < samples.size(); ++k) {
      const NavState f = Step(m, chain[k], samples[k], st[2].data());
      Vec e = LogSO3(Mul3(TransposeSO3(f.R), chain[k + 1].R));
      for (int i = 0; i < 3; ++i) e.push_back(chain[k + 1].v[i] - f.v[i]);
      for (int i = 0; i < 3; ++i) e.push_back(chain[k + 1].p[i] - f.p[i]);
      const Vec w = MatMul(weights[k], e, 9, 9, 1);
      r.insert(r.end(), w.begin(), w.end());
    }
    const double T = Duration(samples);
    for (int i = 0; i < 6; ++i) {
      const double var = i < 3 ? m.gyro_bias_var : m.accel_bias_var;
      r.push_back((st[5][i] - st[2][i]) / std::sqrt(var * T));
    }
    return r;
  }
};

TEST(ImuFactorBatch, IsTheSchurComplementOfTheExplicitChain) {
  const ImuParameters params = TestParameters();
  const Model m = ToModel(params);
  std::mt19937 rng(11);
  for (int n : {1, 3, 40}) {
    const Samples samples = Rounded(RandomSamples(rng, n));
    const std::vector<Slot> item = RandomItem(rng, m, samples, 1e-2);
    DeviceSamples d({samples});
    ImuFactorBatch factor(d.samples.data(), d.offsets.data(), d.count, params, 1);
    std::vector<float> res, jac;
    EvaluateItems(factor, {item}, &res, &jac);

    // Explicit chain: Jacobian by central differences, then eliminate the interior.
    const ExplicitChain chain(m, samples, item);
    const int cols = chain.Columns(), ni = cols - 30;
    const Vec r0 = chain.Residual(Vec(cols, 0.0));
    const int rows = static_cast<int>(r0.size());
    Vec J(static_cast<size_t>(rows) * cols);
    for (int c = 0; c < cols; ++c) {
      Vec dlt(cols, 0.0);
      dlt[c] = 1e-6;
      const Vec rp = chain.Residual(dlt);
      dlt[c] = -1e-6;
      const Vec rm = chain.Residual(dlt);
      for (int i = 0; i < rows; ++i) J[static_cast<size_t>(i) * cols + c] = (rp[i] - rm[i]) / 2e-6;
    }
    const Vec H = MatMul(Transpose(J, rows, cols), J, cols, rows, cols);
    const Vec g = MatMul(Transpose(J, rows, cols), r0, cols, rows, 1);
    Vec Hs(900), gs(30);
    double cost = 0;
    for (double v : r0) cost += v * v;
    for (int i = 0; i < 30; ++i) {
      gs[i] = g[i];
      for (int j = 0; j < 30; ++j) Hs[i * 30 + j] = H[static_cast<size_t>(i) * cols + j];
    }
    if (ni > 0) {
      Vec Hii(static_cast<size_t>(ni) * ni), Hib(static_cast<size_t>(ni) * 31);
      for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < ni; ++j)
          Hii[static_cast<size_t>(i) * ni + j] = H[static_cast<size_t>(30 + i) * cols + 30 + j];
        for (int j = 0; j < 30; ++j)
          Hib[static_cast<size_t>(i) * 31 + j] = H[static_cast<size_t>(30 + i) * cols + j];
        Hib[static_cast<size_t>(i) * 31 + 30] = g[30 + i];
      }
      const Vec X = SolveSpd(Hii, Hib, ni, 31);  // H_II⁻¹ [H_IB, g_I]
      for (int i = 0; i < 30; ++i) {
        for (int k = 0; k < ni; ++k) {
          const double hbi = H[static_cast<size_t>(i) * cols + 30 + k];
          for (int j = 0; j < 30; ++j) Hs[i * 30 + j] -= hbi * X[static_cast<size_t>(k) * 31 + j];
          gs[i] -= hbi * X[static_cast<size_t>(k) * 31 + 30];
        }
      }
      for (int k = 0; k < ni; ++k) cost -= g[30 + k] * X[static_cast<size_t>(k) * 31 + 30];
    }

    // The factor: Jᵀ J, Jᵀ r and |r|².
    double h_max = 0, h_err = 0, g_max = 0, g_err = 0, f_cost = 0;
    for (int i = 0; i < 15; ++i) f_cost += double(res[i]) * res[i];
    for (int i = 0; i < 30; ++i) {
      double gi = 0;
      for (int k = 0; k < 15; ++k) gi += double(jac[k * 30 + i]) * res[k];
      g_max = std::max(g_max, std::fabs(gs[i]));
      g_err = std::max(g_err, std::fabs(gi - gs[i]));
      for (int j = 0; j < 30; ++j) {
        double hij = 0;
        for (int k = 0; k < 15; ++k) hij += double(jac[k * 30 + i]) * jac[k * 30 + j];
        h_max = std::max(h_max, std::fabs(Hs[i * 30 + j]));
        h_err = std::max(h_err, std::fabs(hij - Hs[i * 30 + j]));
      }
    }
    EXPECT_LT(h_err, 2e-5 * h_max) << "N = " << n;
    EXPECT_LT(g_err, 5e-4 * g_max) << "N = " << n;
    EXPECT_NEAR(f_cost, cost, 2e-3 * cost) << "N = " << n;
  }
}

/**
 * Evaluates `copies` factors, factor f being item which[f % which.size()];
 * returns residuals and Jacobians.
 */
void EvaluateReplicated(const ImuParameters &params, const std::vector<Samples> &samples,
                        const std::vector<std::vector<Slot>> &items, const std::vector<int> &which,
                        int copies, size_t samples_per_factor_hint, std::vector<float> *res,
                        std::vector<float> *jac) {
  const int n = static_cast<int>(which.size());
  DeviceSamples d(samples, which, copies);
  std::vector<float> storage;
  std::vector<size_t> offsets;
  for (const auto &item : items)
    for (const Slot &sl : item) {
      offsets.push_back(storage.size());
      storage.insert(storage.end(), sl.storage.begin(), sl.storage.end());
    }
  dvector<float> d_storage(storage);
  std::vector<float *> ptrs;
  for (int c = 0; c < copies; ++c)
    for (int b = 0; b < 6; ++b) ptrs.push_back(d_storage.data() + offsets[which[c % n] * 6 + b]);
  dvector<float *> d_ptrs(ptrs);
  dvector<float> d_res(static_cast<size_t>(copies) * 15), d_jac(static_cast<size_t>(copies) * 450);
  const size_t hint = samples_per_factor_hint > 0 ? samples_per_factor_hint * copies : d.count;
  ImuFactorBatch factor(d.samples.data(), d.offsets.data(), hint, params, copies);
  factor.SetNumActiveFactors(copies);
  CudaStream stream;
  ASSERT_TRUE(factor.Evaluate(d_res.data(), d_jac.data(), d_ptrs.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  res->resize(d_res.size());
  jac->resize(d_jac.size());
  d_res.CopyToHost(res->data(), res->size());
  d_jac.CopyToHost(jac->data(), jac->size());
}

TEST(ImuFactorBatch, LanesPerFactorAgree) {
  // Chains run on 1 to 32 lanes, by batch size and typical chain length (the
  // num_samples hint), each factor using at most one lane per 4 of its own
  // samples. Every split gives the same factor up to round-off (the order of
  // summation differs): hints for 1, 2, ..., 32 lanes, and a batch large
  // enough for one lane per chain whatever the hint (short chains only, up to
  // 40 samples, which the hint alone would split over 8 lanes).
  const int kLengths[] = {1, 2, 3, 7, 40, 200, 1000};
  constexpr int kItems = 42;
  const ImuParameters params = TestParameters();
  const Model m = ToModel(params);
  std::mt19937 rng(23);
  std::vector<Samples> samples;
  std::vector<std::vector<Slot>> items;
  for (int t = 0; t < kItems; ++t) {
    samples.push_back(Rounded(RandomSamples(rng, kLengths[t % 7])));
    items.push_back(RandomItem(rng, m, samples.back(), 1e-2));
  }
  const std::vector<int> all = DeviceSamples::Indices(kItems);
  std::vector<int> short_chains;
  for (int t = 0; t < kItems; ++t)
    if (samples[t].size() <= 40) short_chains.push_back(t);
  std::vector<float> res0, jac0;
  EvaluateReplicated(params, samples, items, all, kItems, 1, &res0, &jac0);  // one lane per chain
  struct Case {
    const std::vector<int> *which;
    int copies;
    size_t hint;
  };
  for (const Case c : {Case{&all, kItems, 16}, Case{&all, kItems, 32}, Case{&all, kItems, 64},
                       Case{&all, kItems, 128}, Case{&all, kItems, 256}, Case{&all, kItems, 512},
                       Case{&short_chains, 65536, 512}}) {
    std::vector<float> res, jac;
    EvaluateReplicated(params, samples, items, *c.which, c.copies, c.hint, &res, &jac);
    double res_err = 0, jac_err = 0;
    for (int f = 0; f < c.copies; ++f) {
      const int t = (*c.which)[f % c.which->size()];
      for (int i = 0; i < 15; ++i) {
        const double r0 = res0[t * 15 + i];
        res_err = std::max(
            res_err, std::fabs(res[static_cast<size_t>(f) * 15 + i] - r0) / (1 + std::fabs(r0)));
      }
      // Relative to the row's largest entry: with poses rotated about the
      // world origin, small entries are differences of large lever-arm terms
      // and carry the round-off of the whole row.
      for (int r = 0; r < 15; ++r) {
        double row_max = 0;
        for (int c = 0; c < 30; ++c)
          row_max = std::max(row_max, std::fabs(double(jac0[t * 450 + r * 30 + c])));
        for (int c = 0; c < 30; ++c) {
          const double d = std::fabs(jac[static_cast<size_t>(f) * 450 + r * 30 + c] -
                                     jac0[t * 450 + r * 30 + c]);
          jac_err = std::max(jac_err, d / (1 + row_max));
        }
      }
    }
    // 1000-sample chains summed in different orders.
    EXPECT_LT(res_err, 1e-3) << c.copies << " factors, hint " << c.hint;
    EXPECT_LT(jac_err, 5e-4) << c.copies << " factors, hint " << c.hint;
  }
}

TEST(ImuFactorBatch, NoiselessChainHasZeroResidual) {
  ImuParameters params;  // defaults: realistic noise densities
  const Model m = ToModel(params);
  std::mt19937 rng(3);
  std::vector<Samples> samples;
  std::vector<std::vector<Slot>> items;
  for (int n : {1, 10, 200, 400}) {
    samples.push_back(Rounded(RandomSamples(rng, n)));
    items.push_back(RandomItem(rng, m, samples.back(), 0.0));
    // b_b = b_a: no bias change.
    items.back()[5].storage = items.back()[2].storage;
  }
  DeviceSamples d(samples);
  ImuFactorBatch factor(d.samples.data(), d.offsets.data(), d.count, params, samples.size());
  std::vector<float> res, jac;
  EvaluateItems(factor, items, &res, &jac);
  for (size_t t = 0; t < items.size(); ++t)
    for (int i = 0; i < 15; ++i)
      EXPECT_NEAR(res[t * 15 + i], 0.f, 0.05f) << "N = " << samples[t].size() << " row " << i;
}

TEST(ImuFactorBatch, EmptyChainStaysFinite) {
  // A factor without samples (offsets[f] == offsets[f + 1]) or of zero
  // duration is invalid input, but must not put inf / NaN into the solve.
  const ImuParameters params = TestParameters();
  const Model m = ToModel(params);
  std::mt19937 rng(29);
  Samples zero_dt = Rounded(RandomSamples(rng, 3));
  for (Sample &s : zero_dt) s.dt = 0.0;
  const std::vector<Samples> samples = {Samples{}, zero_dt};
  std::vector<std::vector<Slot>> items;
  for (const Samples &s : samples) items.push_back(RandomItem(rng, m, s, 1e-2));
  DeviceSamples d(samples);
  dvector<float> d_samples(std::vector<float>(7, 0.f));  // non-null buffer for the empty batch
  ImuFactorBatch factor(d.count > 0 ? d.samples.data() : d_samples.data(), d.offsets.data(),
                        d.count, params, samples.size());
  std::vector<float> res, jac;
  EvaluateItems(factor, items, &res, &jac);
  for (float v : res) EXPECT_TRUE(std::isfinite(v));
  for (float v : jac) EXPECT_TRUE(std::isfinite(v));
}

TEST(ImuFactorBatch, LevenbergMarquardtRecoversKeyframes) {
  // Keyframes along a noiseless IMU trajectory with a constant bias; pose
  // priors on every keyframe stand in for vision. Velocities start at zero,
  // biases at zero, poses perturbed: the solve recovers velocities and bias.
  constexpr int kKeyframes = 6, kSamples = 40;
  // Default (EuRoC) chain noise. The bias random walk is loosened: with the
  // default values its rows are about 1e5 times stiffer than the weakest
  // directions and the float32 linear solve stalls (see ImuParameters).
  ImuParameters params;
  params.gyro_bias_random_walk = 1e-2f;
  params.accel_bias_random_walk = 1e-1f;
  const double xi[6] = {0.05, 0.1, -0.2, 0.1, 0.05, -0.03};
  const Vec Tbi = Exp6(xi);
  for (int i = 0; i < 16; ++i) params.body_from_imu[i] = static_cast<float>(Tbi[i]);
  const Model m = ToModel(params);
  std::mt19937 rng(5);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double bias[6] = {0.01, -0.02, 0.015, 0.1, -0.05, 0.08};

  std::vector<Samples> samples;
  std::vector<Vec> poses, vels;
  NavState x{ExpSO3(xi), {1.0, -0.5, 0.2}, {0.3, 0.1, -0.2}};
  for (int k = 0; k < kKeyframes; ++k) {
    poses.push_back(Mul4(Tbi, Inv4(Pose(x.R, x.p))));  // body_from_world
    vels.push_back({x.v[0], x.v[1], x.v[2]});
    if (k + 1 < kKeyframes) {
      samples.push_back(Rounded(RandomSamples(rng, kSamples)));
      x = Integrate(m, samples.back(), x, bias);
    }
  }
  std::vector<float> h_poses, h_init_poses, h_vels(3 * kKeyframes, 0.f),
      h_biases(6 * kKeyframes, 0.f);
  for (int k = 0; k < kKeyframes; ++k) {
    const auto p = ToFloat(poses[k]);
    h_poses.insert(h_poses.end(), p.begin(), p.end());
    double d[6];
    for (double &c : d) c = 0.02 * normal(rng);
    const auto q = ToFloat(Mul4(poses[k], Exp6(d)));
    h_init_poses.insert(h_init_poses.end(), q.begin(), q.end());
  }
  dvector<float> d_poses(h_init_poses), d_vels(h_vels), d_biases(h_biases), d_prior(h_poses);
  DeviceSamples d(samples);

  SE3StateBatch pose_states(d_poses.data(), kKeyframes);
  pose_states.SetNumActiveStates(kKeyframes);
  VectorStateBatch<3> vel_states(d_vels.data(), kKeyframes);
  vel_states.SetNumActiveStates(kKeyframes);
  VectorStateBatch<6> bias_states(d_biases.data(), kKeyframes);
  bias_states.SetNumActiveStates(kKeyframes);

  ImuFactorBatch imu(d.samples.data(), d.offsets.data(), d.count, params, kKeyframes - 1);
  imu.SetNumActiveFactors(kKeyframes - 1);
  WeightedFactorBatch<SE3PriorFactorBatch> priors(
      1e3f, reinterpret_cast<const SE3Transform *>(d_prior.data()), kKeyframes);
  priors.SetNumActiveFactors(kKeyframes);
  std::vector<float *> imu_ptrs, prior_ptrs;
  for (int k = 0; k + 1 < kKeyframes; ++k) {
    for (int j : {k, k + 1}) {
      imu_ptrs.push_back(pose_states.StateDevicePtr(j));
      imu_ptrs.push_back(vel_states.StateDevicePtr(j));
      imu_ptrs.push_back(bias_states.StateDevicePtr(j));
    }
  }
  for (int k = 0; k < kKeyframes; ++k) prior_ptrs.push_back(pose_states.StateDevicePtr(k));

  Problem problem;
  problem.AddStateBatch(&pose_states);
  problem.AddStateBatch(&vel_states);
  problem.AddStateBatch(&bias_states);
  problem.AddFactorBatch(&imu, imu_ptrs);
  problem.AddFactorBatch(&priors, prior_ptrs);

  LevenbergMarquardtMinimizerOptions options;
  options.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  options.base_options.max_num_iterations = 50;
  LevenbergMarquardtMinimizer minimizer(options);
  CudaStream stream;
  const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  EXPECT_LT(summary.final_cost, 1e-6f * summary.initial_cost);

  std::vector<float> sol_poses(h_poses.size()), sol_vels(h_vels.size()),
      sol_biases(h_biases.size());
  d_poses.CopyToHost(sol_poses.data(), sol_poses.size());
  d_vels.CopyToHost(sol_vels.data(), sol_vels.size());
  d_biases.CopyToHost(sol_biases.data(), sol_biases.size());
  for (int k = 0; k < kKeyframes; ++k) {
    for (int i = 0; i < 12; ++i) EXPECT_NEAR(sol_poses[16 * k + i], h_poses[16 * k + i], 1e-4);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(sol_vels[3 * k + i], vels[k][i], 2e-3) << k;
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(sol_biases[6 * k + i], bias[i], 2e-3) << k;
  }
}

TEST(ImuFactorBatch, RejectsInvalidArguments) {
  dvector<float> samples(std::vector<float>(7, 0.f));
  dvector<int> offsets(std::vector<int>{0, 1});
  ImuParameters p;
  EXPECT_THROW(ImuFactorBatch(nullptr, offsets.data(), 1, p, 1), std::invalid_argument);
  EXPECT_THROW(ImuFactorBatch(samples.data(), nullptr, 1, p, 1), std::invalid_argument);
  p.integration_noise_density = 0.f;
  EXPECT_THROW(ImuFactorBatch(samples.data(), offsets.data(), 1, p, 1), std::invalid_argument);
}

int EnvInt(const char *name, int fallback) {
  const char *v = std::getenv(name);
  return v != nullptr ? std::atoi(v) : fallback;
}

/**
 * VIO-shaped problem for profiling under nsys: K keyframes joined by IMU
 * factors (N samples each), 10 K landmarks each seen by up to 7 nearby
 * keyframes (reprojection factors), keyframe 0 fixed. Both factor types
 * read the same rig_from_world pose states (camera = rig = IMU); truth is
 * consistent, the initial values are perturbed. Sizes: CUNLS_IMU_PROFILE_KEYFRAMES (101),
 * CUNLS_IMU_PROFILE_SAMPLES (200), CUNLS_IMU_PROFILE_LANDMARKS (10 K);
 * CUNLS_IMU_PROFILE_SOLVER=cudss selects
 * cuDSS instead of the default PCG. Run with --gtest_also_run_disabled_tests.
 */
TEST(ImuFactorBatch, DISABLED_VioProfile) {
  const int K = EnvInt("CUNLS_IMU_PROFILE_KEYFRAMES", 101);
  const int N = EnvInt("CUNLS_IMU_PROFILE_SAMPLES", 200);
  const char *solver_env = std::getenv("CUNLS_IMU_PROFILE_SOLVER");
  const bool cudss = solver_env != nullptr && std::string(solver_env) == "cudss";
  const int L = EnvInt("CUNLS_IMU_PROFILE_LANDMARKS", 10 * K);
  // Moderate noise so that the float32 solve converges (see ImuParameters);
  // the kernels' cost does not depend on the values.
  ImuParameters params = TestParameters();
  params.gyro_bias_random_walk = 1e-2f;
  params.accel_bias_random_walk = 1e-1f;
  for (int i = 0; i < 16; ++i) params.body_from_imu[i] = i % 5 == 0 ? 1.f : 0.f;  // IMU = body
  const Model m = ToModel(params);
  std::mt19937 rng(17);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const double bias[6] = {0.01, -0.02, 0.015, 0.1, -0.05, 0.08};

  // Truth: a smooth-ish random IMU trajectory (sample-to-sample correlated).
  std::vector<Samples> samples;
  std::vector<Vec> poses, vels;
  NavState x{ExpSO3(std::vector<double>{0.1, 0.2, -0.1}.data()), {1.0, 0.0, 0.0}, {0, 0, 0}};
  Sample cur{{0, 0, 0}, {0, 0, 9.8}, 0.005};
  for (int k = 0; k < K; ++k) {
    poses.push_back(Inv4(Pose(x.R, x.p)));  // rig_from_world
    vels.push_back({x.v[0], x.v[1], x.v[2]});
    if (k + 1 == K) break;
    Samples ss(N);
    for (Sample &s : ss) {
      for (int i = 0; i < 3; ++i) {
        cur.w[i] = 0.98 * cur.w[i] + 0.01 * normal(rng);
        cur.a[i] = 0.98 * cur.a[i] + 0.02 * (i == 2 ? 9.8 : 0.0) + 0.1 * normal(rng);
      }
      s = cur;
    }
    samples.push_back(Rounded(ss));
    x = Integrate(m, samples.back(), x, bias);
  }
  // Landmarks in front of an anchor keyframe (camera_from_world = pose), seen
  // by the keyframes within ±3 of the anchor where they have depth.
  std::vector<float> h_points, h_obs;
  std::vector<int> obs_kf, obs_pt;
  for (int l = 0; l < L; ++l) {
    const int anchor = l % K;
    const double pc[3] = {2 * uniform(rng), 2 * uniform(rng), 4 + 2 * uniform(rng)};
    const Vec Ti = Inv4(poses[anchor]);
    const Vec Pw = {Ti[0] * pc[0] + Ti[1] * pc[1] + Ti[2] * pc[2] + Ti[3],
                    Ti[4] * pc[0] + Ti[5] * pc[1] + Ti[6] * pc[2] + Ti[7],
                    Ti[8] * pc[0] + Ti[9] * pc[1] + Ti[10] * pc[2] + Ti[11]};
    for (int i = 0; i < 3; ++i) h_points.push_back(static_cast<float>(Pw[i] + 0.05 * normal(rng)));
    for (int k = std::max(0, anchor - 3); k <= std::min(K - 1, anchor + 3); ++k) {
      const Vec &T = poses[k];
      const double c[3] = {T[0] * Pw[0] + T[1] * Pw[1] + T[2] * Pw[2] + T[3],
                           T[4] * Pw[0] + T[5] * Pw[1] + T[6] * Pw[2] + T[7],
                           T[8] * Pw[0] + T[9] * Pw[1] + T[10] * Pw[2] + T[11]};
      if (c[2] < 0.5) continue;
      h_obs.push_back(static_cast<float>(c[0] / c[2]));
      h_obs.push_back(static_cast<float>(c[1] / c[2]));
      obs_kf.push_back(k);
      obs_pt.push_back(l);
    }
  }
  const int num_obs = static_cast<int>(obs_kf.size());
  std::vector<float> h_poses, h_vels, h_biases(6 * K, 0.f);
  for (int k = 0; k < K; ++k) {
    double d[6];
    // Translation only: a rotation of rig_from_world turns the world about
    // its origin, which moves distant points by meters.
    for (int i = 0; i < 6; ++i) d[i] = (k == 0 || i < 3 ? 0.0 : 0.01) * normal(rng);
    const auto q = ToFloat(Mul4(poses[k], Exp6(d)));
    h_poses.insert(h_poses.end(), q.begin(), q.end());
    for (int i = 0; i < 3; ++i)
      h_vels.push_back(static_cast<float>(vels[k][i] + 0.05 * normal(rng)));
  }
  dvector<float> d_poses(h_poses), d_vels(h_vels), d_biases(h_biases), d_points(h_points),
      d_obs(h_obs);
  dvector<int> fixed(std::vector<int>{0});
  DeviceSamples d(samples);

  SE3StateBatch pose_states(d_poses.data(), K, fixed.data(), 1);
  pose_states.SetNumActiveStates(K, 1);
  VectorStateBatch<3> vel_states(d_vels.data(), K);
  vel_states.SetNumActiveStates(K);
  VectorStateBatch<6> bias_states(d_biases.data(), K);
  bias_states.SetNumActiveStates(K);
  VectorStateBatch<3> point_states(d_points.data(), L);
  point_states.SetNumActiveStates(L);

  ImuFactorBatch imu(d.samples.data(), d.offsets.data(), d.count, params, K - 1);
  imu.SetNumActiveFactors(K - 1);
  WeightedFactorBatch<ReprojectionFactorBatch> reproj(
      500.f, reinterpret_cast<const Vector<2> *>(d_obs.data()), num_obs);
  reproj.SetNumActiveFactors(num_obs);
  std::vector<float *> imu_ptrs, reproj_ptrs;
  for (int k = 0; k + 1 < K; ++k)
    for (int j : {k, k + 1}) {
      imu_ptrs.push_back(pose_states.StateDevicePtr(j));
      imu_ptrs.push_back(vel_states.StateDevicePtr(j));
      imu_ptrs.push_back(bias_states.StateDevicePtr(j));
    }
  for (int o = 0; o < num_obs; ++o) {
    reproj_ptrs.push_back(pose_states.StateDevicePtr(obs_kf[o]));
    reproj_ptrs.push_back(point_states.StateDevicePtr(obs_pt[o]));
  }
  Problem problem;
  problem.AddStateBatch(&pose_states);
  problem.AddStateBatch(&vel_states);
  problem.AddStateBatch(&bias_states);
  problem.AddStateBatch(&point_states);
  problem.AddFactorBatch(&imu, imu_ptrs);
  problem.AddFactorBatch(&reproj, reproj_ptrs);

  LevenbergMarquardtMinimizerOptions options;
  options.base_options.max_num_iterations = 20;
  if (cudss) options.base_options.sparse_linear_solver_type = SparseLinearSolverType::cuDSS;
  LevenbergMarquardtMinimizer minimizer(options);
  CudaStream stream;
  // Solve twice from the same initial values: the first call includes the
  // one-time setup (structure analysis, allocations), the second is warm.
  MinimizerSummary summary;
  double ms = 0;
  for (int run = 0; run < 2; ++run) {
    d_poses.CopyFromHost(h_poses.data(), h_poses.size());
    d_vels.CopyFromHost(h_vels.data(), h_vels.size());
    d_biases.CopyFromHost(h_biases.data(), h_biases.size());
    d_points.CopyFromHost(h_points.data(), h_points.size());
    THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    summary = minimizer.Minimize(stream.GetStream(), problem);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %s solve: %.2f ms\n", run == 0 ? "cold" : "warm", ms);
  }
  std::printf(
      "VIO profile: %d keyframes, %d samples per pair, %d landmarks, %d observations, %s\n"
      "  %zu iterations, cost %g -> %g, %.2f ms (%.3f ms per iteration)\n",
      K, N, L, num_obs, cudss ? "cuDSS" : "PCG", summary.num_iterations, summary.initial_cost,
      summary.final_cost, ms, ms / std::max<size_t>(1, summary.num_iterations));
}

// Throughput: run with --gtest_also_run_disabled_tests.
TEST(ImuFactorBatch, DISABLED_Throughput) {
  const ImuParameters params;
  const Model m = ToModel(params);
  std::mt19937 rng(1);
  // CUNLS_IMU_BENCH_SAMPLES / CUNLS_IMU_BENCH_FACTORS select one configuration.
  // The sweep runs 65536 factors with short chains only (20 samples); an
  // explicit configuration runs as given.
  std::vector<int> lengths = {20, 200, 1000}, counts = {16, 128, 1024, 4096, 16384, 65536};
  const bool explicit_config =
      EnvInt("CUNLS_IMU_BENCH_SAMPLES", 0) > 0 || EnvInt("CUNLS_IMU_BENCH_FACTORS", 0) > 0;
  if (EnvInt("CUNLS_IMU_BENCH_SAMPLES", 0) > 0) lengths = {EnvInt("CUNLS_IMU_BENCH_SAMPLES", 0)};
  if (EnvInt("CUNLS_IMU_BENCH_FACTORS", 0) > 0) counts = {EnvInt("CUNLS_IMU_BENCH_FACTORS", 0)};
  for (int n : lengths) {
    for (int factors : counts) {
      if (!explicit_config && factors >= 65536 && n > 20) continue;
      const Samples s = Rounded(RandomSamples(rng, n));
      const std::vector<Slot> item = RandomItem(rng, m, s, 1e-3);
      DeviceSamples d({s}, {0}, factors);
      std::vector<float> storage;
      for (const Slot &sl : item)
        storage.insert(storage.end(), sl.storage.begin(), sl.storage.end());
      dvector<float> d_storage(storage);
      std::vector<float *> ptrs;
      for (int f = 0; f < factors; ++f) {
        size_t o = 0;
        for (const Slot &sl : item) {
          ptrs.push_back(d_storage.data() + o);
          o += sl.storage.size();
        }
      }
      dvector<float *> d_ptrs(ptrs);
      dvector<float> res(static_cast<size_t>(factors) * 15),
          jac(static_cast<size_t>(factors) * 450);
      ImuFactorBatch factor(d.samples.data(), d.offsets.data(), d.count, params, factors);
      factor.SetNumActiveFactors(factors);
      CudaStream stream;
      for (bool with_jac : {false, true}) {
        float *j = with_jac ? jac.data() : nullptr;
        factor.Evaluate(res.data(), j, d_ptrs.data(), stream.GetStream());
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        constexpr int kReps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kReps; ++r)
          factor.Evaluate(res.data(), j, d_ptrs.data(), stream.GetStream());
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                .count() /
            kReps;
        std::printf("N = %3d, %5d factors, %s: %8.1f us (%.2f ns per sample)\n", n, factors,
                    with_jac ? "jacobians" : "residuals", us, 1e3 * us / (double(n) * factors));
      }
    }
  }
}

}  // namespace
}  // namespace cunls
