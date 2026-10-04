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

#pragma once

// Test support for the dynamics factors: float64 SE(2) / SE(3) math written
// independently of the library (storage layouts of SE2StateBatch /
// SE3StateBatch, right perturbations), and a checker that compares a factor's
// residuals and Jacobians with a float64 reference residual and its central
// differences.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/factor_batch.h"
#include "evaluate_items_check.h"

namespace cunls {
namespace dynamics_test {

using Vec = std::vector<double>;

// --- SE(3): row-major 4x4, tangent [φ, ρ] with t = J_l(φ) ρ ------------------

inline Vec Mul4(const Vec &a, const Vec &b) {
  Vec c(16, 0.0);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) c[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return c;
}

inline Vec Inv4(const Vec &T) {
  Vec I(16, 0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) I[i * 4 + j] = T[j * 4 + i];
    for (int k = 0; k < 3; ++k) I[i * 4 + 3] -= T[k * 4 + i] * T[k * 4 + 3];
  }
  I[15] = 1;
  return I;
}

/** I + a [w]x + b [w]x² with (a, b) = (sin θ/θ, (1-cos θ)/θ²) or, for J_l, ((1-cos θ)/θ², (θ-sin
 * θ)/θ³). */
inline std::array<double, 9> Rodrigues(const double *w, bool left_jacobian) {
  const double th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2], th = std::sqrt(th2);
  double a, b;
  if (th < 1e-4) {  // series (no cancellation)
    a = left_jacobian ? 0.5 - th2 / 24 : 1 - th2 / 6;
    b = left_jacobian ? 1.0 / 6 - th2 / 120 : 0.5 - th2 / 24;
  } else {
    const double h = std::sin(0.5 * th);
    const double one_minus_cos = 2 * h * h;
    a = left_jacobian ? one_minus_cos / th2 : std::sin(th) / th;
    b = left_jacobian ? (th - std::sin(th)) / (th2 * th) : one_minus_cos / th2;
  }
  const double K[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
  std::array<double, 9> M{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double k2 = 0;
      for (int k = 0; k < 3; ++k) k2 += K[i * 3 + k] * K[k * 3 + j];
      M[i * 3 + j] = (i == j ? 1.0 : 0.0) + a * K[i * 3 + j] + b * k2;
    }
  return M;
}

inline Vec Exp6(const double *x) {
  const auto R = Rodrigues(x, false), J = Rodrigues(x, true);
  Vec T(16, 0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) T[i * 4 + j] = R[i * 3 + j];
    for (int k = 0; k < 3; ++k) T[i * 4 + 3] += J[i * 3 + k] * x[3 + k];
  }
  T[15] = 1;
  return T;
}

inline Vec Log6(const Vec &T) {
  const double v[3] = {(T[9] - T[6]) / 2, (T[2] - T[8]) / 2, (T[4] - T[1]) / 2};
  const double s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const double c = (T[0] + T[5] + T[10] - 1) / 2;
  const double th = std::atan2(s, c);
  const double k = s < 1e-12 ? 1.0 : th / s;
  Vec x = {k * v[0], k * v[1], k * v[2]};
  // ρ = J_l(φ)⁻¹ t, by solving the 3x3 system.
  const auto J = Rodrigues(x.data(), true);
  const double det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
                     J[2] * (J[3] * J[7] - J[4] * J[6]);
  const double inv[9] = {(J[4] * J[8] - J[5] * J[7]) / det, (J[2] * J[7] - J[1] * J[8]) / det,
                         (J[1] * J[5] - J[2] * J[4]) / det, (J[5] * J[6] - J[3] * J[8]) / det,
                         (J[0] * J[8] - J[2] * J[6]) / det, (J[2] * J[3] - J[0] * J[5]) / det,
                         (J[3] * J[7] - J[4] * J[6]) / det, (J[1] * J[6] - J[0] * J[7]) / det,
                         (J[0] * J[4] - J[1] * J[3]) / det};
  const double t[3] = {T[3], T[7], T[11]};
  for (int i = 0; i < 3; ++i)
    x.push_back(inv[i * 3] * t[0] + inv[i * 3 + 1] * t[1] + inv[i * 3 + 2] * t[2]);
  return x;
}

// --- SO(3): row-major 3x3 -------------------------------------------------------

inline Vec ExpSO3(const double *w) {
  const auto R = Rodrigues(w, false);
  return Vec(R.begin(), R.end());
}

inline Vec LogSO3(const Vec &R) {
  Vec T(16, 0.0);  // embed in SE(3) with zero translation
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) T[i * 4 + j] = R[i * 3 + j];
  T[15] = 1;
  const Vec x = Log6(T);
  return {x[0], x[1], x[2]};
}

inline Vec TransposeSO3(const Vec &R) {
  return {R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]};
}

// --- SE(2): row-major 3x3, tangent [v_x, v_y, θ] --------------------------------

inline Vec Mul3(const Vec &a, const Vec &b) {
  Vec c(9, 0.0);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return c;
}

inline Vec Inv3(const Vec &t) {
  return {t[0], t[3], -(t[0] * t[2] + t[3] * t[5]), t[1], t[4], -(t[1] * t[2] + t[4] * t[5]), 0,
          0,    1};
}

inline void SE2Coefficients(double th, double *a, double *b) {  // sin θ/θ, (1-cos θ)/θ
  if (th == 0.0) {
    *a = 1;
    *b = 0;
  } else {
    const double h = std::sin(0.5 * th);
    *a = std::sin(th) / th;
    *b = 2 * h * h / th;
  }
}

inline Vec Exp3(const double *x) {
  double a, b;
  SE2Coefficients(x[2], &a, &b);
  return {std::cos(x[2]),
          -std::sin(x[2]),
          a * x[0] - b * x[1],
          std::sin(x[2]),
          std::cos(x[2]),
          b * x[0] + a * x[1],
          0,
          0,
          1};
}

inline Vec Log3(const Vec &t) {
  const double th = std::atan2(t[3], t[0]);
  double a, b;
  SE2Coefficients(th, &a, &b);
  const double d = a * a + b * b;
  return {(a * t[2] + b * t[5]) / d, (-b * t[2] + a * t[5]) / d, th};
}

// --- Factor checker -----------------------------------------------------------

/** Kind of a state slot. Poses are perturbed on the right (X Exp(δ)). */
enum class SlotKind { kSE2, kSO3, kSE3, kVector };

struct Slot {
  SlotKind kind;
  std::vector<float> storage;  ///< SE(2): 9, SE(3): 16, vector: its size.
  int Tangent() const {
    if (kind == SlotKind::kSE2 || kind == SlotKind::kSO3) return 3;
    return kind == SlotKind::kSE3 ? 6 : static_cast<int>(storage.size());
  }
};

/** Reference residual of one item from the (perturbed) slot values, in float64. */
using ReferenceResidual = std::function<Vec(int item, const std::vector<Vec> &slots)>;

inline std::vector<Vec> Perturbed(const std::vector<Slot> &slots, const Vec &delta) {
  std::vector<Vec> out;
  int c = 0;
  for (const Slot &s : slots) {
    Vec v(s.storage.begin(), s.storage.end());
    if (s.kind == SlotKind::kSE2) {
      v = Mul3(v, Exp3(&delta[c]));
    } else if (s.kind == SlotKind::kSO3) {
      v = Mul3(v, ExpSO3(&delta[c]));
    } else if (s.kind == SlotKind::kSE3) {
      v = Mul4(v, Exp6(&delta[c]));
    } else {
      for (size_t i = 0; i < v.size(); ++i) v[i] += delta[c + i];
    }
    c += s.Tangent();
    out.push_back(v);
  }
  return out;
}

/**
 * Evaluates `factor` (capacity = items.size()) on the items and compares
 * residuals (tolerance `res_tol` relative) and Jacobians (`jac_tol` relative,
 * per entry) with `reference` and its central differences; checks that the
 * residual-only path matches the Jacobian path (to round-off) and the item
 * contract.
 */
inline void CheckFactor(FactorBatch &factor, const std::vector<std::vector<Slot>> &items,
                        const ReferenceResidual &reference, const std::string &label,
                        double res_tol = 2e-5, double jac_tol = 1e-4) {
  const int n_items = static_cast<int>(items.size());
  const int m = static_cast<int>(factor.ResidualsSize());
  int cols = 0;
  for (size_t s : factor.StateSizes()) cols += static_cast<int>(s);
  std::vector<float> storage;
  std::vector<size_t> offsets;
  for (const auto &item : items) {
    ASSERT_EQ(item.size(), factor.StateSizes().size());
    for (const Slot &s : item) {
      offsets.push_back(storage.size());
      storage.insert(storage.end(), s.storage.begin(), s.storage.end());
    }
  }
  CudaStream stream;
  dvector<float> d_storage(storage);
  std::vector<float *> ptrs;
  for (size_t o : offsets) ptrs.push_back(d_storage.data() + o);
  dvector<float *> d_ptrs(ptrs);
  dvector<float> d_res(n_items * m), d_jac(static_cast<size_t>(n_items) * m * cols),
      d_res_only(n_items * m);
  factor.SetNumActiveFactors(n_items);
  ASSERT_TRUE(factor.Evaluate(d_res.data(), d_jac.data(), d_ptrs.data(), stream.GetStream()));
  ASSERT_TRUE(factor.Evaluate(d_res_only.data(), nullptr, d_ptrs.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> res(d_res.size()), jac(d_jac.size()), res_only(d_res_only.size());
  d_res.CopyToHost(res.data(), res.size());
  d_jac.CopyToHost(jac.data(), jac.size());
  d_res_only.CopyToHost(res_only.data(), res_only.size());
  evaluate_items_test::CheckEvaluateItems(factor, 3, [&](int) { return ptrs; });

  for (int t = 0; t < n_items; ++t) {
    const Vec r0 = reference(t, Perturbed(items[t], Vec(cols, 0.0)));
    ASSERT_EQ(static_cast<int>(r0.size()), m);
    for (int i = 0; i < m; ++i) {
      EXPECT_NEAR(res[t * m + i], r0[i], res_tol * (1 + std::fabs(r0[i])))
          << label << " item " << t << " row " << i;
      // Separate kernel instantiations (residual-only / with Jacobians): the
      // compiler may contract multiply-adds differently in each (CUDA 13 on
      // sm_110 does), so they agree to round-off only.
      EXPECT_NEAR(res_only[t * m + i], res[t * m + i], 2e-6 * (1 + std::fabs(res[t * m + i])))
          << label << " residual-only path";
    }
    const double h = 1e-6;
    for (int c = 0; c < cols; ++c) {
      Vec d(cols, 0.0);
      d[c] = h;
      const Vec rp = reference(t, Perturbed(items[t], d));
      d[c] = -h;
      const Vec rm = reference(t, Perturbed(items[t], d));
      for (int i = 0; i < m; ++i) {
        const double fd = (rp[i] - rm[i]) / (2 * h);
        EXPECT_NEAR(jac[(static_cast<size_t>(t) * m + i) * cols + c], fd,
                    jac_tol * (1 + std::fabs(fd)))
            << label << " item " << t << " row " << i << " col " << c;
      }
    }
  }
}

inline std::vector<float> ToFloat(const Vec &v) { return std::vector<float>(v.begin(), v.end()); }

}  // namespace dynamics_test
}  // namespace cunls
