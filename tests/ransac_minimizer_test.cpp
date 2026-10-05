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
 * @file ransac_minimizer_test.cpp
 * @brief Tests for RansacGaussNewtonMinimizer / RansacLevenbergMarquardtMinimizer:
 * kernel-level checks against CPU references, end-to-end outlier rejection on
 * built-in and custom factors, mixed state types, configuration errors,
 * determinism, scoring chunks and waves.
 */

#include "cunls/minimizer/ransac_minimizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac/ransac_kernels.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace {

using namespace ransac_internal;  // NOLINT
using ransac_test::Compose;
using ransac_test::ExpSE3;
using ransac_test::PnPScene;
using ransac_test::RandomTwist;
using ransac_test::RotationErrorDeg;
using ransac_test::TranslationError;

template <typename T>
std::vector<T> ToHost(const dvector<T> &d) {
  std::vector<T> h(d.size());
  if (!h.empty()) {
    d.CopyToHost(h.data(), h.size());
  }
  return h;
}

template <typename T>
dvector<T> ToDevice(const std::vector<T> &h) {
  dvector<T> d(h.size());
  if (!h.empty()) {
    d.CopyFromHost(h.data(), h.size());
  }
  return d;
}

// ============================================================================
// Kernel-level tests
// ============================================================================

TEST(RansacKernels, PermuteIndexIsABijection) {
  for (uint32_t n : {1u, 2u, 3u, 7u, 16u, 17u, 100u, 1000u, 4097u, 65536u, 100003u}) {
    for (uint64_t key : {0ull, 1ull, 0xDEADBEEFull}) {
      std::vector<uint8_t> seen(n, 0);
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t p = PermuteIndex(i, n, key);
        ASSERT_LT(p, n);
        ASSERT_EQ(seen[p], 0) << "n=" << n << " key=" << key << " i=" << i;
        seen[p] = 1;
      }
    }
  }
}

TEST(RansacKernels, PermutationDependsOnKey) {
  const uint32_t n = 1000;
  int same = 0;
  for (uint32_t i = 0; i < n; ++i) {
    same +=
        PermuteIndex(i, n, PermutationKey(1, 0, 0)) == PermuteIndex(i, n, PermutationKey(1, 1, 0));
  }
  EXPECT_LT(same, 20);  // ~1 expected for independent permutations
}

TEST(RansacKernels, DrawSamplesGivesDistinctKeyedIndicesPerSlot) {
  CudaStream stream;
  const int n = 103;
  const int s = 3;
  const int k = 70;
  dvector<int> samples(static_cast<size_t>(k) * s);
  LaunchDrawSamples(stream.GetStream(), n, s, k, 7, 2, samples.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto smp = ToHost(samples);
  for (int slot = 0; slot < k; ++slot) {
    std::set<int> distinct;
    for (int t = 0; t < s; ++t) {
      const int u = smp[slot * s + t];
      ASSERT_GE(u, 0);
      ASSERT_LT(u, n);
      EXPECT_EQ(static_cast<uint32_t>(u), PermuteIndex(t, n, PermutationKey(7, 2, slot)));
      distinct.insert(u);
    }
    EXPECT_EQ(distinct.size(), static_cast<size_t>(s));
  }
}

/** CPU reference of the normal-equation kernel for one slot. */
struct HostView {
  BatchView view;
  std::vector<int> local_col;
  std::vector<float> res, jac, cost;
};

void ReferenceNormalEquations(const std::vector<HostView> &views, const std::vector<int> &samples,
                              int sample_size, const std::vector<uint8_t> &mask, int mask_stride,
                              int slot, int dim, std::vector<double> &h, std::vector<double> &g,
                              double &cost) {
  h.assign(dim * dim, 0.0);
  g.assign(dim, 0.0);
  cost = 0.0;
  // Adds factor f, stored in row `row` of the slot's part of the view's buffers.
  auto add = [&](const HostView &hv, int f, int row_index) {
    const size_t buf = static_cast<size_t>(slot);
    const BatchView &v = hv.view;
    std::vector<double> jl(v.m * dim, 0.0);
    for (int row = 0; row < v.m; ++row) {
      for (int col = 0; col < v.n; ++col) {
        int blk = 0;
        while (blk + 1 < v.nb && col >= v.block_col_off[blk + 1]) {
          ++blk;
        }
        const int lc = hv.local_col[f * v.nb + blk];
        if (lc >= 0) {
          jl[row * dim + lc + col - v.block_col_off[blk]] +=
              hv.jac[buf * v.stride_jac + (static_cast<size_t>(row_index) * v.m + row) * v.n + col];
        }
      }
    }
    for (int row = 0; row < v.m; ++row) {
      const double r = hv.res[buf * v.stride_res + static_cast<size_t>(row_index) * v.m + row];
      for (int a = 0; a < dim; ++a) {
        g[a] -= jl[row * dim + a] * r;
        for (int b = 0; b < dim; ++b) {
          h[a * dim + b] += jl[row * dim + a] * jl[row * dim + b];
        }
      }
    }
    cost += hv.cost[buf * v.stride_cost + row_index];
  };
  for (int t = 0; t < sample_size; ++t) {
    const int u = samples[slot * sample_size + t];
    for (const HostView &hv : views) {
      if (hv.view.kind == kViewSamples && u >= hv.view.u_offset &&
          u < hv.view.u_offset + hv.view.num_factors) {
        add(hv, u - hv.view.u_offset, t);
      }
    }
  }
  for (const HostView &hv : views) {
    if (hv.view.kind == kViewSamples) {
      continue;
    }
    for (int f = 0; f < hv.view.num_factors; ++f) {
      if (hv.view.kind == kViewPerSlotMasked &&
          mask[slot * mask_stride + hv.view.u_offset + f] == 0) {
        continue;
      }
      add(hv, f, f);
    }
  }
}

struct NormalEquationCase {
  int dim;
  int sampled_factors;  ///< Factors of the minimal-sample view.
  int masked_factors;
  int plain_factors;
  SlotGroup group;
};

class RansacNormalEquationsTest : public ::testing::TestWithParam<NormalEquationCase> {};

TEST_P(RansacNormalEquationsTest, MatchesCpuReferenceAndIsDeterministic) {
  const NormalEquationCase c = GetParam();
  const int dim = c.dim;
  const int slots = 5;
  const int sample_size = std::min(3, c.sampled_factors);
  std::mt19937 rng(1234 + dim);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);

  // Block layouts: sample view has two blocks, masked view one, plain view one.
  // Every view holds `rows` rows per slot.
  auto make_view = [&](int kind, int m, std::vector<int> sizes, int nf, int u_offset, int rows) {
    HostView hv;
    BatchView &v = hv.view;
    v.kind = kind;
    v.m = m;
    v.nb = static_cast<int>(sizes.size());
    v.num_factors = nf;
    v.u_offset = u_offset;
    v.n = 0;
    for (int k = 0; k < v.nb; ++k) {
      v.block_col_off[k] = v.n;
      v.block_size[k] = sizes[k];
      v.n += sizes[k];
    }
    hv.local_col.resize(nf * v.nb);
    for (int f = 0; f < nf; ++f) {
      for (int k = 0; k < v.nb; ++k) {
        const int span = dim - sizes[k];
        const int choice = static_cast<int>(rng() % (span + 2));
        hv.local_col[f * v.nb + k] = choice > span ? -1 : choice;  // some constant states
      }
    }
    v.stride_res = static_cast<size_t>(rows) * m;
    v.stride_jac = static_cast<size_t>(rows) * m * v.n;
    v.stride_cost = rows;
    hv.res.resize(slots * v.stride_res);
    hv.jac.resize(slots * v.stride_jac);
    hv.cost.resize(slots * v.stride_cost);
    for (float &x : hv.res) x = uni(rng);
    for (float &x : hv.jac) x = uni(rng);
    for (float &x : hv.cost) x = std::fabs(uni(rng));
    return hv;
  };
  std::vector<HostView> views;
  const int b0 = std::max(1, std::min(dim, 2));
  const int b1 = std::max(1, std::min(dim, 3));
  views.push_back(make_view(kViewSamples, 2, {b0, b1}, c.sampled_factors, 0, sample_size));
  views.push_back(make_view(kViewPerSlotMasked, 3, {std::min(dim, 2)}, c.masked_factors,
                            c.sampled_factors, c.masked_factors));
  views.push_back(make_view(kViewPerSlot, 1, {dim}, c.plain_factors, -1, c.plain_factors));

  const int total_sampled = c.sampled_factors + c.masked_factors;
  std::vector<int> samples(slots * sample_size);
  for (int p = 0; p < slots; ++p) {
    std::vector<int> pool(c.sampled_factors);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int t = 0; t < sample_size; ++t) {
      samples[p * sample_size + t] = pool[t];
    }
  }
  std::vector<uint8_t> mask(slots * total_sampled);
  for (auto &m : mask) m = (rng() % 3) != 0;

  // Upload.
  std::vector<dvector<int>> d_cols;
  std::vector<dvector<float>> d_res, d_jac, d_cost;
  std::vector<BatchView> dev_views;
  for (HostView &hv : views) {
    d_cols.push_back(ToDevice(hv.local_col));
    d_res.push_back(ToDevice(hv.res));
    d_jac.push_back(ToDevice(hv.jac));
    d_cost.push_back(ToDevice(hv.cost));
  }
  for (size_t i = 0; i < views.size(); ++i) {
    BatchView v = views[i].view;
    v.local_col = d_cols[i].data();
    v.res = d_res[i].data();
    v.jac = d_jac[i].data();
    v.cost = d_cost[i].data();
    dev_views.push_back(v);
  }
  auto d_views = ToDevice(dev_views);
  auto d_samples = ToDevice(samples);
  auto d_mask = ToDevice(mask);

  SlotItems items;
  items.views = d_views.data();
  items.num_views = static_cast<int>(dev_views.size());
  items.samples = d_samples.data();
  items.sample_size = sample_size;
  items.mask = d_mask.data();
  items.mask_stride = total_sampled;
  items.items_per_slot = sample_size + c.masked_factors + c.plain_factors;
  items.m_max = 3;

  CudaStream stream;
  dvector<float> h(slots * dim * dim), g(slots * dim), cost(slots), cost2(slots);
  dvector<float> scratch(NormalEquationsScratchFloats(items, slots, dim, c.group));
  LaunchNormalEquations(stream.GetStream(), items, slots, dim, h.data(), g.data(), cost.data(),
                        scratch.data(), c.group);
  dvector<float> cost_scratch(SlotCostScratchFloats(items, slots));
  LaunchSlotCost(stream.GetStream(), items, slots, cost2.data(), cost_scratch.data(), c.group);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hh = ToHost(h);
  const auto gh = ToHost(g);
  const auto ch = ToHost(cost);
  const auto ch2 = ToHost(cost2);

  for (int p = 0; p < slots; ++p) {
    std::vector<double> rh, rg;
    double rc;
    ReferenceNormalEquations(views, samples, sample_size, mask, total_sampled, p, dim, rh, rg, rc);
    double scale = 1.0;
    for (double x : rh) scale = std::max(scale, std::fabs(x));
    for (int i = 0; i < dim * dim; ++i) {
      ASSERT_NEAR(hh[p * dim * dim + i], rh[i], 2e-5 * scale) << "slot " << p << " entry " << i;
    }
    for (int i = 0; i < dim; ++i) {
      ASSERT_NEAR(gh[p * dim + i], rg[i], 2e-5 * scale) << "slot " << p << " g " << i;
    }
    EXPECT_NEAR(ch[p], rc, 1e-4 * std::max(1.0, rc));
    EXPECT_NEAR(ch2[p], rc, 1e-4 * std::max(1.0, rc));
  }

  // Bitwise determinism across launches.
  dvector<float> h2(slots * dim * dim), g2(slots * dim), c3(slots);
  LaunchNormalEquations(stream.GetStream(), items, slots, dim, h2.data(), g2.data(), c3.data(),
                        scratch.data(), c.group);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hh2 = ToHost(h2);
  const auto gh2 = ToHost(g2);
  EXPECT_EQ(0, std::memcmp(hh.data(), hh2.data(), hh.size() * sizeof(float)));
  EXPECT_EQ(0, std::memcmp(gh.data(), gh2.data(), gh.size() * sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(Shapes, RansacNormalEquationsTest,
                         ::testing::Values(NormalEquationCase{1, 4, 3, 2, SlotGroup::kAuto},
                                           NormalEquationCase{1, 4, 3, 2, SlotGroup::kBlock},
                                           NormalEquationCase{5, 10, 7, 3, SlotGroup::kWarp},
                                           NormalEquationCase{5, 10, 7, 3, SlotGroup::kBlock},
                                           NormalEquationCase{6, 12, 20, 5, SlotGroup::kWarp},
                                           NormalEquationCase{6, 40, 3000, 5, SlotGroup::kAuto},
                                           NormalEquationCase{6, 40, 3000, 5, SlotGroup::kWarp},
                                           NormalEquationCase{14, 8, 30, 2, SlotGroup::kWarp},
                                           NormalEquationCase{17, 12, 500, 2, SlotGroup::kAuto},
                                           NormalEquationCase{33, 8, 200, 1, SlotGroup::kBlock},
                                           NormalEquationCase{64, 20, 900, 3, SlotGroup::kAuto},
                                           NormalEquationCase{3, 6, 20000, 2, SlotGroup::kBlock},
                                           NormalEquationCase{24, 6, 30, 2, SlotGroup::kWarp}));

struct SolveCase {
  int dim;
  SolverKind solver;
  bool damped;
};

class RansacSolveTest : public ::testing::TestWithParam<SolveCase> {};

TEST_P(RansacSolveTest, MatchesCpuSolveOnSpdSystems) {
  const SolveCase c = GetParam();
  const int dim = c.dim;
  const int slots = 7;
  std::mt19937 rng(99 + dim);
  std::normal_distribution<double> n01(0, 1);
  std::vector<float> h(slots * dim * dim), g(slots * dim), lam(slots);
  std::vector<int> active(slots, 1);
  active[3] = 0;
  for (int p = 0; p < slots; ++p) {
    std::vector<double> a((dim + 3) * dim);
    for (double &x : a) x = n01(rng);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        double s = (i == j) ? 0.5 : 0.0;
        for (int k = 0; k < dim + 3; ++k) s += a[k * dim + i] * a[k * dim + j];
        h[p * dim * dim + i * dim + j] = static_cast<float>(s);
      }
      g[p * dim + i] = static_cast<float>(n01(rng));
    }
    lam[p] = c.damped ? 0.1f * (p + 1) : 0.f;
  }
  auto dh = ToDevice(h);
  auto dg = ToDevice(g);
  auto dl = ToDevice(lam);
  auto da = ToDevice(active);
  dvector<float> delta(slots * dim), pred(slots), sq(slots);
  dvector<int> ok(slots);
  CudaStream stream;
  LaunchSolve(stream.GetStream(), slots, dim, c.solver, dh.data(), dg.data(),
              c.damped ? dl.data() : nullptr, da.data(), delta.data(), pred.data(), sq.data(),
              ok.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto x = ToHost(delta);
  const auto pr = ToHost(pred);
  const auto okh = ToHost(ok);
  for (int p = 0; p < slots; ++p) {
    if (!active[p]) {
      for (int i = 0; i < dim; ++i) EXPECT_EQ(x[p * dim + i], 0.f);
      continue;
    }
    ASSERT_EQ(okh[p], 1) << "slot " << p;
    // Residual of the damped system in double.
    double res = 0, bn = 0;
    for (int i = 0; i < dim; ++i) {
      double s = 0;
      for (int j = 0; j < dim; ++j) {
        double hij = h[p * dim * dim + i * dim + j];
        if (i == j) {
          hij += c.damped ? lam[p] * std::max<double>(hij, 1e-6) : hij * 1e-6;
        }
        s += hij * x[p * dim + j];
      }
      res += (s - g[p * dim + i]) * (s - g[p * dim + i]);
      bn += g[p * dim + i] * g[p * dim + i];
    }
    EXPECT_LT(std::sqrt(res / bn), 5e-3) << "slot " << p;
    // Predicted decrease g^T x - 0.5 x^T H x with the undamped H.
    double gx = 0, xhx = 0;
    for (int i = 0; i < dim; ++i) {
      gx += g[p * dim + i] * x[p * dim + i];
      for (int j = 0; j < dim; ++j) {
        xhx += x[p * dim + i] * h[p * dim * dim + i * dim + j] * x[p * dim + j];
      }
    }
    EXPECT_NEAR(pr[p], gx - 0.5 * xhx, 1e-3 * std::max(1.0, std::fabs(gx)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Solvers, RansacSolveTest,
    ::testing::Values(SolveCase{1, kSolveCholesky, false}, SolveCase{6, kSolveCholesky, false},
                      SolveCase{6, kSolveLDLT, false}, SolveCase{6, kSolveLDLT, true},
                      SolveCase{13, kSolveCholesky, true}, SolveCase{32, kSolveLDLT, false},
                      SolveCase{33, kSolveCholesky, false}, SolveCase{64, kSolveLDLT, true},
                      SolveCase{64, kSolveCholesky, false}));

TEST(RansacKernels, RankDeficientSystems) {
  // H = v v^T (rank 1), g = 2 v: LDLT must return a finite solution of H x = g,
  // Cholesky must report failure.
  const int dim = 6;
  std::vector<float> v = {1.f, -2.f, 0.5f, 0.f, 3.f, 1.f};
  std::vector<float> h(dim * dim), g(dim);
  for (int i = 0; i < dim; ++i) {
    g[i] = 2.f * v[i];
    for (int j = 0; j < dim; ++j) h[i * dim + j] = v[i] * v[j];
  }
  auto dh = ToDevice(h);
  auto dg = ToDevice(g);
  dvector<float> delta(dim), pred(1), sq(1);
  dvector<int> ok(1);
  CudaStream stream;
  LaunchSolve(stream.GetStream(), 1, dim, kSolveLDLT, dh.data(), dg.data(), nullptr, nullptr,
              delta.data(), pred.data(), sq.data(), ok.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  auto x = ToHost(delta);
  ASSERT_EQ(ToHost(ok)[0], 1);
  for (int i = 0; i < dim; ++i) {
    double s = 0;
    for (int j = 0; j < dim; ++j) s += h[i * dim + j] * x[j];
    EXPECT_NEAR(s, g[i], 1e-3);
    EXPECT_TRUE(std::isfinite(x[i]));
  }
  LaunchSolve(stream.GetStream(), 1, dim, kSolveCholesky, dh.data(), dg.data(), nullptr, nullptr,
              delta.data(), pred.data(), sq.data(), ok.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(ToHost(ok)[0], 0);
  for (float xi : ToHost(delta)) EXPECT_EQ(xi, 0.f);

  // Non-finite input fails for both solvers.
  h[0] = NAN;
  dh = ToDevice(h);
  for (SolverKind s : {kSolveLDLT, kSolveCholesky}) {
    LaunchSolve(stream.GetStream(), 1, dim, s, dh.data(), dg.data(), nullptr, nullptr, delta.data(),
                pred.data(), sq.data(), ok.data());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    EXPECT_EQ(ToHost(ok)[0], 0);
  }
}

class RansacScoreTest : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(RansacScoreTest, MatchesCpuReference) {
  const int slots = 4;
  const int n0 = GetParam().first, n1 = GetParam().second;
  const float tau0 = 0.3f, tau1 = 1.5f;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::vector<float> r0(slots * n0 * 2), r1(slots * n1 * 3), ao(slots * 4);
  for (float &x : r0) x = uni(rng) * 0.4f;
  for (float &x : r1) x = uni(rng) * 1.5f;
  for (float &x : ao) x = std::fabs(uni(rng));
  r0[7] = NAN;  // slot 0, factor 3
  auto d0 = ToDevice(r0), d1 = ToDevice(r1), da = ToDevice(ao);
  std::vector<BatchView> sv(2), av(1);
  sv[0].m = 2;
  sv[0].num_factors = n0;
  sv[0].u_offset = 0;
  sv[0].res = d0.data();
  sv[0].stride_res = n0 * 2;
  sv[0].tau_sq = tau0 * tau0;
  sv[1].m = 3;
  sv[1].num_factors = n1;
  sv[1].u_offset = n0;
  sv[1].res = d1.data();
  sv[1].stride_res = n1 * 3;
  sv[1].tau_sq = tau1 * tau1;
  av[0].num_factors = 4;
  av[0].cost = da.data();
  av[0].stride_cost = 4;
  auto dsv = ToDevice(sv), dav = ToDevice(av);
  std::vector<int> valid = {1, 1, 0, 1};
  auto dvalid = ToDevice(valid);
  for (int rule : {kScoreMSAC, kScoreInlierCount}) {
    ScoreInputs in;
    in.sampled = dsv.data();
    in.num_sampled = 2;
    in.always_on = dav.data();
    in.num_always_on = 1;
    in.total_sampled = n0 + n1;
    in.rule = rule;
    in.add_always_on = 1;
    dvector<float> score(slots);
    dvector<int> inl(slots);
    dvector<uint8_t> mask(slots * (n0 + n1));
    CudaStream stream;
    dvector<float> score_scratch(ScoreScratchFloats(in.total_sampled, slots));
    LaunchScore(stream.GetStream(), in, slots, 0, dvalid.data(), score.data(), inl.data(),
                mask.data(), score_scratch.data());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const auto sh = ToHost(score);
    const auto ih = ToHost(inl);
    const auto mh = ToHost(mask);
    for (int p = 0; p < slots; ++p) {
      double msac = 0, bound = 0, aos = 0;
      int count = 0;
      for (int u = 0; u < n0 + n1; ++u) {
        const bool first = u < n0;
        const int f = first ? u : u - n0;
        const int m = first ? 2 : 3;
        const float t2 = first ? tau0 * tau0 : tau1 * tau1;
        const float *r = first ? &r0[p * n0 * 2 + f * 2] : &r1[p * n1 * 3 + f * 3];
        double e = 0;
        for (int k = 0; k < m; ++k) e += static_cast<double>(r[k]) * r[k];
        const bool inl_ref = std::isfinite(e) && e <= t2;
        msac += (std::isfinite(e) && e < t2) ? e : t2;
        bound += t2;
        count += inl_ref;
        ASSERT_EQ(mh[p * (n0 + n1) + u], inl_ref ? 1 : 0) << "slot " << p << " u " << u;
      }
      for (int f = 0; f < 4; ++f) aos += ao[p * 4 + f];
      EXPECT_EQ(ih[p], count);
      if (!valid[p]) {
        EXPECT_TRUE(std::isinf(sh[p]));
        continue;
      }
      const double expected = rule == kScoreMSAC ? msac + 2.0 * aos : -count + msac / (bound + 1.0);
      EXPECT_NEAR(sh[p], expected, 2e-4 * std::max(1.0, std::fabs(expected)));
    }
  }
}

// Small (one block per slot) and large (several blocks per slot) factor counts.
INSTANTIATE_TEST_SUITE_P(Sizes, RansacScoreTest,
                         ::testing::Values(std::make_pair(50, 30), std::make_pair(9000, 5000)));

TEST(RansacKernels, SelectPicksLowestScoresWithIndexTieBreak) {
  std::vector<float> score = {3.f, 1.f, INFINITY, 1.f, 0.5f, NAN, 2.f};
  std::vector<int> inl = {10, 20, 0, 21, 30, 0, 15};
  std::vector<int> valid = {1, 1, 0, 1, 1, 1, 1};
  auto ds = ToDevice(score);
  auto di = ToDevice(inl);
  auto dv = ToDevice(valid);
  dvector<int> top(4);
  dvector<DeviceStats> stats(1);
  CudaStream stream;
  LaunchInitStats(stream.GetStream(), stats.data());
  LaunchSelect(stream.GetStream(), ds.data(), di.data(), dv.data(), 7, 4, top.data(), stats.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto t = ToHost(top);
  EXPECT_EQ(t, (std::vector<int>{4, 1, 3, 6}));
  DeviceStats st = ToHost(stats)[0];
  EXPECT_EQ(st.improved, 1);
  EXPECT_EQ(st.best_slot, 4);
  EXPECT_EQ(st.best_inliers, 30);
  EXPECT_FLOAT_EQ(st.best_score, 0.5f);
  EXPECT_EQ(st.valid_total, 6);

  // A second selection that does not beat the best keeps it.
  std::vector<float> worse = {1.f, 2.f};
  auto dw = ToDevice(worse);
  LaunchSelect(stream.GetStream(), dw.data(), di.data(), nullptr, 2, 1, top.data(), stats.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  st = ToHost(stats)[0];
  EXPECT_EQ(st.improved, 0);
  EXPECT_FLOAT_EQ(st.best_score, 0.5f);
  EXPECT_EQ(st.valid_total, 6);
}

// ============================================================================
// End-to-end helpers
// ============================================================================

constexpr float kNoise = 1e-3f;       // ~0.5 px at f = 500
constexpr float kTau = 5e-3f;         // 5 sigma
constexpr float kMinOutlier = 2e-2f;  // gross outliers

struct Classification {
  size_t true_inliers = 0;
  size_t true_outliers = 0;
  size_t tp = 0;  // inlier kept
  size_t fp = 0;  // outlier kept
  double precision() const { return tp + fp == 0 ? 1.0 : static_cast<double>(tp) / (tp + fp); }
  double recall() const { return true_inliers == 0 ? 1.0 : static_cast<double>(tp) / true_inliers; }
};

Classification Classify(const uint8_t *device_mask, const std::vector<uint8_t> &is_outlier) {
  std::vector<uint8_t> mask(is_outlier.size());
  THROW_ON_CUDA_ERROR(cudaMemcpy(mask.data(), device_mask, mask.size(), cudaMemcpyDeviceToHost));
  Classification c;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (is_outlier[i]) {
      ++c.true_outliers;
      c.fp += mask[i];
    } else {
      ++c.true_inliers;
      c.tp += mask[i];
    }
  }
  return c;
}

/** Single SE3 pose observing points through PnP factors (+ optional prior). */
struct PnPSetup {
  dvector<SE3Transform> pose{1};
  std::unique_ptr<SE3StateBatch> state;
  dvector<Vector<2>> obs;
  dvector<Vector<3>> pts;
  std::unique_ptr<PnPFactorBatch> pnp;
  dvector<SE3Transform> prior_target{1};
  std::unique_ptr<SE3PriorFactorBatch> prior;
  std::unique_ptr<CauchyLossFunctionBatch> loss;
  Problem problem;

  PnPSetup(const PnPScene &scene, const SE3Transform &init, bool with_prior = false,
           const SE3Transform *prior_pose = nullptr, float cauchy_scale = 0.f) {
    pose.CopyFromHost(&init, 1);
    state = std::make_unique<SE3StateBatch>(reinterpret_cast<float *>(pose.data()), 1);
    state->SetNumActiveStates(state->Capacity(), state->ConstCapacity());
    obs = ToDevice(scene.observations);
    pts = ToDevice(scene.points_world);
    pnp = std::make_unique<PnPFactorBatch>(obs.data(), pts.data(), scene.observations.size());
    pnp->SetNumActiveFactors(pnp->Capacity());
    problem.AddStateBatch(state.get());
    std::vector<float *> ptrs(scene.observations.size(), state->StateDevicePtr(0));
    if (cauchy_scale > 0.f) {
      loss = std::make_unique<CauchyLossFunctionBatch>(cauchy_scale * cauchy_scale,
                                                       1.f / (cauchy_scale * cauchy_scale));
      problem.AddFactorBatch(pnp.get(), loss.get(), ptrs);
    } else {
      problem.AddFactorBatch(pnp.get(), ptrs);
    }
    if (with_prior) {
      prior_target.CopyFromHost(prior_pose, 1);
      prior = std::make_unique<SE3PriorFactorBatch>(prior_target.data(), 1);
      prior->SetNumActiveFactors(prior->Capacity());
      problem.AddFactorBatch(prior.get(), {state->StateDevicePtr(0)});
    }
  }

  SE3Transform Pose() const {
    SE3Transform p;
    pose.CopyToHost(&p, 1);
    return p;
  }
};

/** 0.5 * sum |proj(pose * X) - obs|^2 in double over a scene's inliers. */
/** Inlier cost of the world_from_camera pose state `pose`. */
double HostInlierCost(const PnPScene &s, const SE3Transform &pose) {
  const SE3Transform cam_from_world = ransac_test::Inverse(pose);
  double c = 0;
  for (size_t i = 0; i < s.observations.size(); ++i) {
    if (s.is_outlier[i]) continue;
    const auto p = ransac_test::Transform(
        cam_from_world, {s.points_world[i][0], s.points_world[i][1], s.points_world[i][2]});
    const double du = p[0] / p[2] - s.observations[i][0];
    const double dv = p[1] / p[2] - s.observations[i][1];
    c += 0.5 * (du * du + dv * dv);
  }
  return c;
}

PnPScene InliersOnly(const PnPScene &s) {
  PnPScene out = s;
  out.points_world.clear();
  out.observations.clear();
  out.is_outlier.clear();
  for (size_t i = 0; i < s.observations.size(); ++i) {
    if (!s.is_outlier[i]) {
      out.points_world.push_back(s.points_world[i]);
      out.observations.push_back(s.observations[i]);
      out.is_outlier.push_back(0);
    }
  }
  return out;
}

SE3Transform Perturb(const SE3Transform &pose, uint32_t seed, double rot, double trans) {
  std::mt19937 rng(seed);
  return Compose(pose, ExpSE3(RandomTwist(rng, rot, trans)));
}

RansacMinimizerOptions PnPOptions(size_t num_batches_sampled = 1, bool prior = false) {
  RansacMinimizerOptions o;
  o.hypotheses_per_round = 256;
  o.max_rounds = 8;
  o.seed = 11;
  for (size_t i = 0; i < num_batches_sampled; ++i) {
    o.factor_batches.push_back({RansacRole::kSampled, kTau});
  }
  if (prior) {
    o.factor_batches.push_back({RansacRole::kAlwaysOn, 0.f});
  }
  return o;
}

// ============================================================================
// End-to-end: PnP with outliers
// ============================================================================

struct OutlierCase {
  double outlier_ratio;
  bool levenberg_marquardt;
};

class RansacPnPTest : public ::testing::TestWithParam<OutlierCase> {};

TEST_P(RansacPnPTest, RejectsOutliersAndMatchesInlierOnlySolution) {
  const OutlierCase c = GetParam();
  const PnPScene scene = ransac_test::MakePnPScene(600, c.outlier_ratio, kNoise, kMinOutlier, 21);
  const SE3Transform init = Perturb(scene.world_from_cam, 3, 0.1, 0.3);
  CudaStream stream;

  PnPSetup setup(scene, init);
  RansacSummary summary;
  std::unique_ptr<RansacGaussNewtonMinimizer> ransac;
  if (c.levenberg_marquardt) {
    RansacLevenbergMarquardtMinimizerOptions lm;
    lm.base_options = PnPOptions();
    ransac = std::make_unique<RansacLevenbergMarquardtMinimizer>(lm);
  } else {
    ransac = std::make_unique<RansacGaussNewtonMinimizer>(PnPOptions());
  }
  summary = ransac->Minimize(stream.GetStream(), setup.problem);
  const SE3Transform est = setup.Pose();

  // Reference: plain GN on the true inliers only.
  const PnPScene clean = InliersOnly(scene);
  PnPSetup ref(clean, init);
  GaussNewtonMinimizer gn;
  gn.Minimize(stream.GetStream(), ref.problem);
  const SE3Transform ref_pose = ref.Pose();

  const Classification cls = Classify(ransac->InlierMask(0), scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.995) << "outliers " << c.outlier_ratio;
  EXPECT_GE(cls.precision(), 0.999) << "outliers " << c.outlier_ratio;
  EXPECT_EQ(summary.num_inliers, cls.tp + cls.fp);
  EXPECT_LT(RotationErrorDeg(est, scene.world_from_cam), 0.2);
  EXPECT_LT(TranslationError(est, scene.world_from_cam), 0.02);
  // Same optimum as GN on the true inliers: poses agree to well below the
  // noise level, and the inlier cost (in double) is no worse than GN's. The
  // cost surface is flat in float near the optimum, so compare costs rather
  // than demanding bitwise-close poses.
  EXPECT_LT(RotationErrorDeg(est, ref_pose), 0.05);
  EXPECT_LT(TranslationError(est, ref_pose), 5e-3);
  EXPECT_LE(HostInlierCost(scene, est), HostInlierCost(scene, ref_pose) * (1.0 + 1e-5));
  EXPECT_GE(summary.num_rounds, 1u);
  EXPECT_GT(summary.num_valid_hypotheses, 0u);
  EXPECT_LE(summary.final_cost, summary.initial_cost);
}

INSTANTIATE_TEST_SUITE_P(OutlierRatios, RansacPnPTest,
                         ::testing::Values(OutlierCase{0.0, false}, OutlierCase{0.1, false},
                                           OutlierCase{0.3, false}, OutlierCase{0.5, false},
                                           OutlierCase{0.7, false}, OutlierCase{0.0, true},
                                           OutlierCase{0.3, true}, OutlierCase{0.7, true}));

TEST(RansacMinimizer, PlainGaussNewtonFailsWhereRansacSucceeds) {
  // Sanity check of the test setup itself: outliers ruin the non-robust solve.
  const PnPScene scene = ransac_test::MakePnPScene(600, 0.4, kNoise, kMinOutlier, 22);
  const SE3Transform init = Perturb(scene.world_from_cam, 4, 0.1, 0.3);
  CudaStream stream;
  PnPSetup plain(scene, init);
  GaussNewtonMinimizer gn;
  gn.Minimize(stream.GetStream(), plain.problem);
  EXPECT_GT(RotationErrorDeg(plain.Pose(), scene.world_from_cam), 0.5);

  PnPSetup robust(scene, init);
  RansacGaussNewtonMinimizer ransac(PnPOptions());
  ransac.Minimize(stream.GetStream(), robust.problem);
  EXPECT_LT(RotationErrorDeg(robust.Pose(), scene.world_from_cam), 0.2);
}

TEST(RansacMinimizer, AlwaysOnPriorIsHonoredAndNeverClassified) {
  const PnPScene scene = ransac_test::MakePnPScene(400, 0.5, kNoise, kMinOutlier, 23);
  const SE3Transform init = Perturb(scene.world_from_cam, 5, 0.1, 0.3);
  CudaStream stream;
  PnPSetup setup(scene, init, /*with_prior=*/true, &scene.world_from_cam);
  RansacGaussNewtonMinimizer ransac(PnPOptions(1, /*prior=*/true));
  const RansacSummary s = ransac.Minimize(stream.GetStream(), setup.problem);
  EXPECT_EQ(ransac.InlierMask(1), nullptr);
  const Classification cls = Classify(ransac.InlierMask(0), scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.995);
  EXPECT_GE(cls.precision(), 0.999);
  EXPECT_LT(RotationErrorDeg(setup.Pose(), scene.world_from_cam), 0.2);
  EXPECT_EQ(s.num_inliers, cls.tp + cls.fp);
}

TEST(RansacMinimizer, LargeInitialErrorWithLevenbergMarquardt) {
  const PnPScene scene = ransac_test::MakePnPScene(500, 0.3, kNoise, kMinOutlier, 24);
  const SE3Transform init = Perturb(scene.world_from_cam, 6, 0.35, 0.8);
  CudaStream stream;
  PnPSetup setup(scene, init);
  RansacLevenbergMarquardtMinimizerOptions lm;
  lm.base_options = PnPOptions();
  lm.base_options.hypothesis_iterations = 10;
  RansacLevenbergMarquardtMinimizer ransac(lm);
  ransac.Minimize(stream.GetStream(), setup.problem);
  EXPECT_LT(RotationErrorDeg(setup.Pose(), scene.world_from_cam), 0.2);
  EXPECT_LT(TranslationError(setup.Pose(), scene.world_from_cam), 0.02);
}

// ============================================================================
// Mixed state types, several state / factor batches, custom factors
// ============================================================================

TEST(RansacMinimizer, TwoCameraRigWithBetweenFactor) {
  // Two free SE3 poses (D = 12, s = 6), one PnP batch each, and an always-on
  // between factor tying them together.
  std::mt19937 rng(31);
  const SE3Transform a = ExpSE3(RandomTwist(rng, 0.4, 1.0));
  const SE3Transform rel = ExpSE3(RandomTwist(rng, 0.2, 0.5));
  const SE3Transform b = Compose(rel, a);
  const PnPScene sa = ransac_test::MakePnPScene(400, 0.4, kNoise, kMinOutlier, 32, &a);
  const PnPScene sb = ransac_test::MakePnPScene(300, 0.4, kNoise, kMinOutlier, 33, &b);

  std::vector<SE3Transform> init = {Perturb(a, 1, 0.05, 0.15), Perturb(b, 2, 0.05, 0.15)};
  auto poses = ToDevice(init);
  SE3StateBatch state(reinterpret_cast<float *>(poses.data()), 2);
  state.SetNumActiveStates(state.Capacity(), state.ConstCapacity());
  auto oa = ToDevice(sa.observations);
  auto pa = ToDevice(sa.points_world);
  auto ob = ToDevice(sb.observations);
  auto pb = ToDevice(sb.points_world);
  PnPFactorBatch fa(oa.data(), pa.data(), sa.observations.size());
  fa.SetNumActiveFactors(fa.Capacity());
  PnPFactorBatch fb(ob.data(), pb.data(), sb.observations.size());
  fb.SetNumActiveFactors(fb.Capacity());
  std::vector<SE3Transform> delta = {Compose(ransac_test::Inverse(b), a)};
  auto ddelta = ToDevice(delta);
  SE3BetweenFactorBatch between(ddelta.data(), 1);
  between.SetNumActiveFactors(between.Capacity());

  Problem problem;
  problem.AddStateBatch(&state);
  problem.AddFactorBatch(&fa,
                         std::vector<float *>(sa.observations.size(), state.StateDevicePtr(0)));
  problem.AddFactorBatch(&fb,
                         std::vector<float *>(sb.observations.size(), state.StateDevicePtr(1)));
  problem.AddFactorBatch(&between, {state.StateDevicePtr(0), state.StateDevicePtr(1)});

  RansacMinimizerOptions o = PnPOptions(2, true);
  o.hypotheses_per_round = 512;
  RansacLevenbergMarquardtMinimizerOptions lm;
  lm.base_options = o;
  RansacLevenbergMarquardtMinimizer ransac(lm);
  CudaStream stream;
  const RansacSummary s = ransac.Minimize(stream.GetStream(), problem);
  std::vector<SE3Transform> est(2);
  poses.CopyToHost(est.data(), 2);
  EXPECT_LT(RotationErrorDeg(est[0], a), 0.2);
  EXPECT_LT(RotationErrorDeg(est[1], b), 0.2);
  EXPECT_LT(TranslationError(est[0], a), 0.02);
  EXPECT_LT(TranslationError(est[1], b), 0.02);
  const Classification ca = Classify(ransac.InlierMask(0), sa.is_outlier);
  const Classification cb = Classify(ransac.InlierMask(1), sb.is_outlier);
  EXPECT_GE(ca.recall(), 0.99);
  EXPECT_GE(cb.recall(), 0.99);
  EXPECT_GE(ca.precision(), 0.999);
  EXPECT_GE(cb.precision(), 0.999);
  EXPECT_EQ(ransac.InlierMask(2), nullptr);
  EXPECT_EQ(s.num_inliers, ca.tp + ca.fp + cb.tp + cb.fp);
}

TEST(RansacMinimizer, ReprojectionWithConstantLandmarkBatch) {
  // Free SE3 pose + a fully constant Vector<3> landmark batch: the landmark
  // batch is shared (not replicated) and its states contribute no columns.
  const PnPScene scene = ransac_test::MakePnPScene(500, 0.5, kNoise, kMinOutlier, 41);
  const SE3Transform init = Perturb(scene.world_from_cam, 7, 0.1, 0.3);
  dvector<SE3Transform> pose(1);
  pose.CopyFromHost(&init, 1);
  SE3StateBatch pose_state(reinterpret_cast<float *>(pose.data()), 1);
  pose_state.SetNumActiveStates(pose_state.Capacity(), pose_state.ConstCapacity());
  auto points = ToDevice(scene.points_world);
  std::vector<int> all(scene.points_world.size());
  std::iota(all.begin(), all.end(), 0);
  auto const_ids = ToDevice(all);
  VectorStateBatch<3> landmarks(reinterpret_cast<float *>(points.data()), all.size(),
                                const_ids.data(), all.size());
  landmarks.SetNumActiveStates(landmarks.Capacity(), landmarks.ConstCapacity());
  auto obs = ToDevice(scene.observations);
  ReprojectionFactorBatch reproj(obs.data(), scene.observations.size());
  reproj.SetNumActiveFactors(reproj.Capacity());
  std::vector<float *> ptrs;
  for (size_t i = 0; i < scene.observations.size(); ++i) {
    ptrs.push_back(pose_state.StateDevicePtr(0));
    ptrs.push_back(landmarks.StateDevicePtr(i));
  }
  Problem problem;
  problem.AddStateBatch(&pose_state);
  problem.AddStateBatch(&landmarks);
  problem.AddFactorBatch(&reproj, ptrs);

  RansacGaussNewtonMinimizer ransac(PnPOptions());
  CudaStream stream;
  ransac.Minimize(stream.GetStream(), problem);
  SE3Transform est;
  pose.CopyToHost(&est, 1);
  EXPECT_LT(RotationErrorDeg(est, scene.world_from_cam), 0.2);
  const Classification cls = Classify(ransac.InlierMask(0), scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.995);
  EXPECT_GE(cls.precision(), 0.999);
  // Constant landmarks are untouched.
  std::vector<Vector<3>> after(scene.points_world.size());
  points.CopyToHost(after.data(), after.size());
  EXPECT_EQ(0,
            std::memcmp(after.data(), scene.points_world.data(), after.size() * sizeof(Vector<3>)));
}

TEST(RansacMinimizer, CustomFocalFactorConvergesWithRegularMinimizer) {
  // Validates the custom factor itself (Jacobians, conventions) on clean data.
  const PnPScene scene = ransac_test::MakePnPScene(300, 0.0, kNoise, kMinOutlier, 52);
  const float focal = 480.f;
  std::vector<Vector<2>> pixels(scene.observations.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i][0] = scene.observations[i][0] * focal;
    pixels[i][1] = scene.observations[i][1] * focal;
  }
  const SE3Transform init = Perturb(scene.world_from_cam, 8, 0.05, 0.15);
  dvector<SE3Transform> pose(1);
  pose.CopyFromHost(&init, 1);
  std::vector<float> f0 = {focal * 1.08f};
  auto fdev = ToDevice(f0);
  SE3StateBatch pose_state(reinterpret_cast<float *>(pose.data()), 1);
  pose_state.SetNumActiveStates(pose_state.Capacity(), pose_state.ConstCapacity());
  VectorStateBatch<1> focal_state(fdev.data(), 1);
  focal_state.SetNumActiveStates(focal_state.Capacity(), focal_state.ConstCapacity());
  auto obs = ToDevice(pixels);
  auto pts = ToDevice(scene.points_world);
  ransac_test::FocalPnPFactorBatch factor(obs.data(), pts.data(), pixels.size());
  factor.SetNumActiveFactors(factor.Capacity());
  std::vector<float *> ptrs;
  for (size_t i = 0; i < pixels.size(); ++i) {
    ptrs.push_back(pose_state.StateDevicePtr(0));
    ptrs.push_back(focal_state.StateDevicePtr(0));
  }
  Problem problem;
  problem.AddStateBatch(&pose_state);
  problem.AddStateBatch(&focal_state);
  problem.AddFactorBatch(&factor, ptrs);
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  LevenbergMarquardtMinimizer minimizer(lm);
  CudaStream stream;
  minimizer.Minimize(stream.GetStream(), problem);
  SE3Transform est;
  pose.CopyToHost(&est, 1);
  EXPECT_NEAR(ToHost(fdev)[0], focal, 2.0);
  EXPECT_LT(RotationErrorDeg(est, scene.world_from_cam), 0.3);
}

TEST(RansacMinimizer, CustomFactorMixedStateTypes) {
  // SE3 pose + Vector<1> focal length (D = 7), custom factor reading two states
  // from two state batches of different manifold types.
  const PnPScene scene = ransac_test::MakePnPScene(500, 0.4, kNoise, kMinOutlier, 51);
  const float focal = 480.f;
  std::vector<Vector<2>> pixels(scene.observations.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i][0] = scene.observations[i][0] * focal;
    pixels[i][1] = scene.observations[i][1] * focal;
  }
  const SE3Transform init = Perturb(scene.world_from_cam, 8, 0.05, 0.15);
  dvector<SE3Transform> pose(1);
  pose.CopyFromHost(&init, 1);
  std::vector<float> f0 = {focal * 1.08f};
  auto fdev = ToDevice(f0);
  SE3StateBatch pose_state(reinterpret_cast<float *>(pose.data()), 1);
  pose_state.SetNumActiveStates(pose_state.Capacity(), pose_state.ConstCapacity());
  VectorStateBatch<1> focal_state(fdev.data(), 1);
  focal_state.SetNumActiveStates(focal_state.Capacity(), focal_state.ConstCapacity());
  auto obs = ToDevice(pixels);
  auto pts = ToDevice(scene.points_world);
  ransac_test::FocalPnPFactorBatch factor(obs.data(), pts.data(), pixels.size());
  factor.SetNumActiveFactors(factor.Capacity());
  std::vector<float *> ptrs;
  for (size_t i = 0; i < pixels.size(); ++i) {
    ptrs.push_back(pose_state.StateDevicePtr(0));
    ptrs.push_back(focal_state.StateDevicePtr(0));
  }
  Problem problem;
  problem.AddStateBatch(&pose_state);
  problem.AddStateBatch(&focal_state);
  problem.AddFactorBatch(&factor, ptrs);

  RansacLevenbergMarquardtMinimizerOptions lm;
  lm.base_options.factor_batches = {{RansacRole::kSampled, kTau * focal}};
  lm.base_options.seed = 3;
  lm.base_options.hypotheses_per_round = 512;
  lm.base_options.hypothesis_iterations = 8;
  RansacLevenbergMarquardtMinimizer ransac(lm);
  CudaStream stream;
  ransac.Minimize(stream.GetStream(), problem);
  SE3Transform est;
  pose.CopyToHost(&est, 1);
  const float f_est = ToHost(fdev)[0];
  EXPECT_NEAR(f_est, focal, 2.0);
  EXPECT_LT(RotationErrorDeg(est, scene.world_from_cam), 0.3);
  const Classification cls = Classify(ransac.InlierMask(0), scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.99);
  EXPECT_GE(cls.precision(), 0.999);
}

template <int Dim>
struct LinearProblem {
  ransac_test::LinearScene scene;
  dvector<float> a, y, x;
  std::unique_ptr<VectorStateBatch<Dim>> state;
  std::unique_ptr<ransac_test::LinearRegressionFactorBatch<Dim>> factor;
  Problem problem;
  LinearProblem(size_t n, double outliers, uint32_t seed) {
    scene = ransac_test::MakeLinearScene(Dim, n, outliers, 0.01, 1.0, seed);
    a = ToDevice(scene.a);
    y = ToDevice(scene.y);
    std::vector<float> x0(Dim, 0.f);
    x = ToDevice(x0);
    state = std::make_unique<VectorStateBatch<Dim>>(x.data(), 1);
    state->SetNumActiveStates(state->Capacity(), state->ConstCapacity());
    factor = std::make_unique<ransac_test::LinearRegressionFactorBatch<Dim>>(a.data(), y.data(), n);
    factor->SetNumActiveFactors(factor->Capacity());
    problem.AddStateBatch(state.get());
    problem.AddFactorBatch(factor.get(), std::vector<float *>(n, state->StateDevicePtr(0)));
  }
  double MaxError() const {
    const auto est = ToHost(x);
    double e = 0;
    for (int k = 0; k < Dim; ++k) {
      e = std::max(e, static_cast<double>(std::fabs(est[k] - scene.x_true[k])));
    }
    return e;
  }
};

TEST(RansacMinimizer, CustomLineFit2D) {
  LinearProblem<2> lp(1000, 0.6, 61);
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.05f;
  o.hypotheses_per_round = 128;
  RansacGaussNewtonMinimizer ransac(o);
  CudaStream stream;
  const RansacSummary s = ransac.Minimize(stream.GetStream(), lp.problem);
  EXPECT_LT(lp.MaxError(), 5e-3);
  const Classification cls = Classify(ransac.InlierMask(0), lp.scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.999);
  EXPECT_EQ(cls.fp, 0u);
  EXPECT_NEAR(s.inlier_ratio, 0.4, 0.01);
}

TEST(RansacMinimizer, BoundaryDimension64) {
  LinearProblem<64> lp(3000, 0.02, 62);  // 0.98^64 = 27% all-inlier samples
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.05f;
  o.hypotheses_per_round = 128;
  o.hypothesis_iterations = 2;
  RansacGaussNewtonMinimizer ransac(o);
  CudaStream stream;
  ransac.Minimize(stream.GetStream(), lp.problem);
  EXPECT_LT(lp.MaxError(), 5e-3);
  const Classification cls = Classify(ransac.InlierMask(0), lp.scene.is_outlier);
  EXPECT_GE(cls.recall(), 0.999);
  EXPECT_EQ(cls.fp, 0u);
}

TEST(RansacMinimizer, Dimension33UsesBlockPath) {
  LinearProblem<33> lp(2000, 0.03, 63);  // 0.97^33 = 37% all-inlier samples
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.05f;
  o.hypotheses_per_round = 128;
  o.linear_solver = RansacLinearSolverType::kCholesky;
  RansacGaussNewtonMinimizer ransac(o);
  CudaStream stream;
  ransac.Minimize(stream.GetStream(), lp.problem);
  EXPECT_LT(lp.MaxError(), 5e-3);
}

// ============================================================================
// Configuration errors
// ============================================================================

template <typename F>
void ExpectInvalid(F &&f, const std::string &needle) {
  try {
    f();
    FAIL() << "expected std::invalid_argument containing '" << needle << "'";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
  }
}

TEST(RansacMinimizer, RejectsDimensionAboveLimit) {
  LinearProblem<65> lp(200, 0.0, 71);
  RansacGaussNewtonMinimizer ransac;
  CudaStream stream;
  ExpectInvalid([&] { ransac.Minimize(stream.GetStream(), lp.problem); }, "D = 65");
}

// Features the RANSAC minimizers do not implement are rejected, not ignored.
TEST(RansacMinimizer, RejectsUnsupportedFeatures) {
  CudaStream stream;
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.05f;
  {
    LinearProblem<2> lp(200, 0.0, 81);
    dvector<float> lo(std::vector<float>{-1.f, -1.f}), hi(std::vector<float>{1.f, 1.f});
    lp.state->SetBounds(lo.data(), hi.data());
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), lp.problem); }, "box bounds");
  }
  {
    LinearProblem<2> lp(200, 0.0, 82);
    dvector<float> lo(std::vector<float>{-1.f, -1.f}), hi(std::vector<float>{1.f, 1.f});
    BoundFactorBatch<2> bound(lo.data(), hi.data(), 1);
    bound.SetNumActiveFactors(1);
    lp.problem.AddFactorBatch(&bound, {lp.state->StateDevicePtr(0)});
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), lp.problem); }, "constraint factor batch");
  }
  {
    LinearProblem<2> lp(200, 0.0, 83);
    dvector<int> stages(std::vector<int>{0});
    lp.problem.SetStateStages({stages.data()});
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), lp.problem); }, "state stages");
  }
}

TEST(RansacMinimizer, RejectsBadConfigurations) {
  const PnPScene scene = ransac_test::MakePnPScene(50, 0.0, kNoise, kMinOutlier, 72);
  CudaStream stream;
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o;
    o.factor_batches = {{RansacRole::kAlwaysOn, 0.f}};
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "no kSampled");
  }
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o = PnPOptions();
    o.factor_batches.push_back({RansacRole::kSampled, 1.f});
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "residual batches");
  }
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o = PnPOptions();
    o.sample_size = 51;
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "sample size");
  }
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o = PnPOptions();
    o.max_rounds = 0;
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "max_rounds");
  }
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o = PnPOptions();
    o.hypothesis_iterations = 0;
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "hypothesis_iterations");
  }
  {
    PnPSetup s(scene, scene.world_from_cam);
    RansacMinimizerOptions o = PnPOptions();
    o.factor_batches[0].inlier_threshold = 0.f;
    RansacGaussNewtonMinimizer r(o);
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), s.problem); }, "inlier_threshold");
  }
  {
    // Numeric Jacobians are not supported yet.
    dvector<SE3Transform> pose(1);
    pose.CopyFromHost(&scene.world_from_cam, 1);
    SE3StateBatch st(reinterpret_cast<float *>(pose.data()), 1);
    st.SetNumActiveStates(st.Capacity(), st.ConstCapacity());
    auto obs = ToDevice(scene.observations);
    auto pts = ToDevice(scene.points_world);
    PnPFactorBatch f(obs.data(), pts.data(), scene.observations.size());
    f.SetNumActiveFactors(f.Capacity());
    Problem p;
    p.AddStateBatch(&st);
    p.AddFactorBatch(&f, std::vector<float *>(scene.observations.size(), st.StateDevicePtr(0)),
                     JacobianMode::kNumeric);
    RansacGaussNewtonMinimizer r(PnPOptions());
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), p); }, "numeric");
  }
  {
    // All states constant: D = 0.
    dvector<SE3Transform> pose(1);
    pose.CopyFromHost(&scene.world_from_cam, 1);
    std::vector<int> ids = {0};
    auto did = ToDevice(ids);
    SE3StateBatch st(reinterpret_cast<float *>(pose.data()), 1, did.data(), 1);
    st.SetNumActiveStates(st.Capacity(), st.ConstCapacity());
    auto obs = ToDevice(scene.observations);
    auto pts = ToDevice(scene.points_world);
    PnPFactorBatch f(obs.data(), pts.data(), scene.observations.size());
    f.SetNumActiveFactors(f.Capacity());
    Problem p;
    p.AddStateBatch(&st);
    p.AddFactorBatch(&f, std::vector<float *>(scene.observations.size(), st.StateDevicePtr(0)));
    RansacGaussNewtonMinimizer r(PnPOptions());
    ExpectInvalid([&] { r.Minimize(stream.GetStream(), p); }, "D = 0");
  }
}

TEST(RansacMinimizer, InlierMaskIsNullBeforeAnyRun) {
  RansacGaussNewtonMinimizer r;
  EXPECT_EQ(r.InlierMask(0), nullptr);
  EXPECT_EQ(r.InlierMaskSize(0), 0u);
}

TEST(RansacMinimizer, InlierMaskSizeIsTheBatchSizeOfTheLastRun) {
  const PnPScene scene = ransac_test::MakePnPScene(300, 0.3, kNoise, kMinOutlier, 77);
  PnPSetup s(scene, scene.world_from_cam);
  RansacGaussNewtonMinimizer r(PnPOptions());
  CudaStream stream;
  r.Minimize(stream.GetStream(), s.problem);
  EXPECT_NE(r.InlierMask(0), nullptr);
  EXPECT_EQ(r.InlierMaskSize(0), 300u);
  EXPECT_EQ(r.InlierMask(1), nullptr);  // no such residual batch
  EXPECT_EQ(r.InlierMaskSize(1), 0u);
}

TEST(RansacMinimizer, NoValidHypothesisFallsBackToTheInitialGuess) {
  // An all-zero Jacobian makes every normal-equation matrix zero, so Cholesky
  // rejects every hypothesis and the refinement cannot move either: the result
  // must be the initial guess, not uninitialized memory.
  const size_t n = 50;
  std::vector<float> a(2 * n, 0.f), y(n, 1.f);
  const std::vector<float> x0 = {0.75f, -1.25f};
  auto d_a = ToDevice(a);
  auto d_y = ToDevice(y);
  auto d_x = ToDevice(x0);
  VectorStateBatch<2> state(d_x.data(), 1);
  state.SetNumActiveStates(state.Capacity(), state.ConstCapacity());
  ransac_test::LinearRegressionFactorBatch<2> factor(d_a.data(), d_y.data(), n);
  factor.SetNumActiveFactors(factor.Capacity());
  Problem problem;
  problem.AddStateBatch(&state);
  problem.AddFactorBatch(&factor, std::vector<float *>(n, state.StateDevicePtr(0)));

  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.1f;
  o.linear_solver = RansacLinearSolverType::kCholesky;
  o.max_rounds = 1;
  RansacGaussNewtonMinimizer ransac(o);
  CudaStream stream;
  const RansacSummary summary = ransac.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.num_valid_hypotheses, 0u);
  EXPECT_EQ(ToHost(d_x), x0);
}

// ============================================================================
// Determinism, chunking, waves, reuse
// ============================================================================

struct RunResult {
  SE3Transform pose;
  std::vector<uint8_t> mask;
  RansacSummary summary;
};

RunResult RunPnP(const PnPScene &scene, const SE3Transform &init, RansacMinimizerOptions o,
                 bool lm = false, bool syncing_factor = false) {
  CudaStream stream;
  PnPSetup setup(scene, init);
  std::unique_ptr<ransac_test::SyncingFactorBatch> syncing;
  Problem syncing_problem;
  Problem *problem = &setup.problem;
  if (syncing_factor) {
    syncing = std::make_unique<ransac_test::SyncingFactorBatch>(setup.pnp.get());
    syncing->SetNumActiveFactors(syncing->Capacity());
    syncing_problem.AddStateBatch(setup.state.get());
    syncing_problem.AddFactorBatch(
        syncing.get(),
        std::vector<float *>(scene.observations.size(), setup.state->StateDevicePtr(0)));
    problem = &syncing_problem;
  }
  RunResult out;
  std::unique_ptr<RansacGaussNewtonMinimizer> r;
  if (lm) {
    RansacLevenbergMarquardtMinimizerOptions lo;
    lo.base_options = o;
    r = std::make_unique<RansacLevenbergMarquardtMinimizer>(lo);
  } else {
    r = std::make_unique<RansacGaussNewtonMinimizer>(o);
  }
  out.summary = r->Minimize(stream.GetStream(), *problem);
  out.pose = setup.Pose();
  out.mask.resize(scene.observations.size());
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(out.mask.data(), r->InlierMask(0), out.mask.size(), cudaMemcpyDeviceToHost));
  return out;
}

void ExpectIdentical(const RunResult &a, const RunResult &b) {
  EXPECT_EQ(0, std::memcmp(&a.pose, &b.pose, sizeof(SE3Transform)));
  EXPECT_EQ(a.mask, b.mask);
  EXPECT_EQ(a.summary.num_inliers, b.summary.num_inliers);
  EXPECT_EQ(a.summary.num_rounds, b.summary.num_rounds);
  EXPECT_EQ(a.summary.final_cost, b.summary.final_cost);
}

TEST(RansacMinimizer, CoherentOutliersFromACompetingPose) {
  // 40% of the observations agree with a second, wrong pose; the majority wins.
  const std::array<double, 6> twist = {0.1, -0.08, 0.06, 0.25, -0.2, 0.2};
  const PnPScene scene =
      ransac_test::MakeCoherentPnPScene(800, 0.4, kNoise, kMinOutlier, 25, twist);
  const SE3Transform init = Perturb(scene.world_from_cam, 26, 0.05, 0.15);
  for (bool lm : {false, true}) {
    const RunResult r = RunPnP(scene, init, PnPOptions(), lm);
    EXPECT_LT(RotationErrorDeg(r.pose, scene.world_from_cam), 0.2) << "lm " << lm;
    size_t kept_outliers = 0, kept_inliers = 0;
    for (size_t i = 0; i < r.mask.size(); ++i) {
      (scene.is_outlier[i] ? kept_outliers : kept_inliers) += r.mask[i];
    }
    EXPECT_EQ(kept_outliers, 0u) << "lm " << lm;
    EXPECT_GE(kept_inliers, 475u) << "lm " << lm;  // 480 true inliers
  }
}

TEST(RansacMinimizer, LargeProblemTwoStageScoringAndEarlyExit) {
  // 150k factors: two-stage scoring (> 2 x 16384 sampled factors) and the
  // per-iteration convergence checks of the refinement are active.
  const PnPScene scene = ransac_test::MakePnPScene(150000, 0.5, kNoise, kMinOutlier, 88);
  const SE3Transform init = Perturb(scene.world_from_cam, 16, 0.1, 0.3);
  RansacMinimizerOptions exhaustive = PnPOptions();
  exhaustive.scoring_subset_size = 0;
  const RunResult two_stage = RunPnP(scene, init, PnPOptions());
  const RunResult full = RunPnP(scene, init, exhaustive);
  for (const RunResult *r : {&two_stage, &full}) {
    EXPECT_LT(RotationErrorDeg(r->pose, scene.world_from_cam), 0.05);
    size_t kept_outliers = 0, missed_inliers = 0;
    for (size_t i = 0; i < r->mask.size(); ++i) {
      kept_outliers += scene.is_outlier[i] && r->mask[i];
      missed_inliers += !scene.is_outlier[i] && !r->mask[i];
    }
    EXPECT_EQ(kept_outliers, 0u);
    EXPECT_LE(missed_inliers, 10u);  // of 75000
  }
  // Both scoring schemes lead to the same optimum.
  EXPECT_LT(RotationErrorDeg(two_stage.pose, full.pose), 0.01);
  EXPECT_EQ(two_stage.summary.num_inliers, full.summary.num_inliers);
  // Deterministic for a fixed seed.
  ExpectIdentical(two_stage, RunPnP(scene, init, PnPOptions()));
}

TEST(RansacMinimizer, TwoStageScoringWithWrappedCustomFactorIsIdentical) {
  // The two-stage subset evaluates items with explicit factor ids; a custom
  // wrapper that forwards them must reproduce the built-in factor exactly.
  const PnPScene scene = ransac_test::MakePnPScene(40000, 0.4, kNoise, kMinOutlier, 89);
  const SE3Transform init = Perturb(scene.world_from_cam, 17, 0.1, 0.3);
  RansacMinimizerOptions two_stage = PnPOptions();
  two_stage.hypotheses_per_round = 64;
  two_stage.scoring_subset_size = 4096;
  const RunResult direct = RunPnP(scene, init, two_stage);
  const RunResult wrapped = RunPnP(scene, init, two_stage, false, /*syncing_factor=*/true);
  ExpectIdentical(direct, wrapped);
}

TEST(RansacMinimizer, SameSeedIsBitwiseReproducible) {
  const PnPScene scene = ransac_test::MakePnPScene(700, 0.5, kNoise, kMinOutlier, 81);
  const SE3Transform init = Perturb(scene.world_from_cam, 9, 0.1, 0.3);
  for (bool lm : {false, true}) {
    const RunResult a = RunPnP(scene, init, PnPOptions(), lm);
    const RunResult b = RunPnP(scene, init, PnPOptions(), lm);
    ExpectIdentical(a, b);
  }
}

TEST(RansacMinimizer, CustomFactorThatSynchronizesGivesIdenticalResults) {
  // A custom factor may do anything inside Evaluate (here: synchronize).
  const PnPScene scene = ransac_test::MakePnPScene(300, 0.3, kNoise, kMinOutlier, 83);
  const SE3Transform init = Perturb(scene.world_from_cam, 11, 0.1, 0.3);
  const RunResult direct = RunPnP(scene, init, PnPOptions());
  const RunResult syncing = RunPnP(scene, init, PnPOptions(), false, /*syncing_factor=*/true);
  ExpectIdentical(direct, syncing);
}

TEST(RansacMinimizer, ScoringChunkSizeDoesNotChangeResults) {
  const PnPScene scene = ransac_test::MakePnPScene(500, 0.5, kNoise, kMinOutlier, 84);
  const SE3Transform init = Perturb(scene.world_from_cam, 12, 0.1, 0.3);
  RansacMinimizerOptions tiny = PnPOptions();
  tiny.scoring_memory_budget_bytes = 1;  // one hypothesis per chunk
  ExpectIdentical(RunPnP(scene, init, PnPOptions()), RunPnP(scene, init, tiny));
}

TEST(RansacMinimizer, MoreHypothesesThanFactors) {
  // 64 minimal samples of 3 out of 30 factors: samples overlap across slots.
  const PnPScene scene = ransac_test::MakePnPScene(30, 0.2, kNoise, kMinOutlier, 85);
  const SE3Transform init = Perturb(scene.world_from_cam, 13, 0.05, 0.15);
  RansacMinimizerOptions o = PnPOptions();
  o.hypotheses_per_round = 64;
  const RunResult r = RunPnP(scene, init, o);
  EXPECT_LT(RotationErrorDeg(r.pose, scene.world_from_cam), 0.5);
  size_t fp = 0, tp = 0;
  for (size_t i = 0; i < r.mask.size(); ++i) {
    (scene.is_outlier[i] ? fp : tp) += r.mask[i];
  }
  EXPECT_EQ(fp, 0u);
  EXPECT_GE(tp, 23u);
}

TEST(RansacMinimizer, InlierCountScoringAndCholesky) {
  const PnPScene scene = ransac_test::MakePnPScene(500, 0.4, kNoise, kMinOutlier, 86);
  const SE3Transform init = Perturb(scene.world_from_cam, 14, 0.1, 0.3);
  RansacMinimizerOptions o = PnPOptions();
  o.scoring = RansacScoring::kInlierCount;
  o.linear_solver = RansacLinearSolverType::kCholesky;
  const RunResult r = RunPnP(scene, init, o);
  EXPECT_LT(RotationErrorDeg(r.pose, scene.world_from_cam), 0.2);
}

TEST(RansacMinimizer, MinimizerIsReusableAcrossProblems) {
  RansacGaussNewtonMinimizer ransac(PnPOptions());
  CudaStream stream;
  for (uint32_t seed : {91u, 92u, 93u}) {
    const PnPScene scene =
        ransac_test::MakePnPScene(200 + 150 * (seed - 90), 0.4, kNoise, kMinOutlier, seed);
    PnPSetup setup(scene, Perturb(scene.world_from_cam, seed, 0.1, 0.3));
    ransac.Minimize(stream.GetStream(), setup.problem);
    EXPECT_LT(RotationErrorDeg(setup.Pose(), scene.world_from_cam), 0.2) << seed;
    const Classification cls = Classify(ransac.InlierMask(0), scene.is_outlier);
    EXPECT_GE(cls.recall(), 0.995);
    EXPECT_GE(cls.precision(), 0.999);
  }
}

}  // namespace
}  // namespace cunls
