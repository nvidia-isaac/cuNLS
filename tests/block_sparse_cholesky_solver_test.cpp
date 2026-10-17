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
 * @file block_sparse_cholesky_solver_test.cpp
 * @brief BlockSparseCholeskySolver on block-sparse SPD matrices shaped like
 * the reduced systems of bundle adjustment: banded chains (visual-inertial),
 * random covisibility, dense blocks wider than a panel, disconnected parts,
 * tiny and empty systems; refactorization with new values; non-SPD input.
 * Checked by the double-precision residual.
 */

#include "cunls/linear_solver/block_sparse_cholesky_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/problem.h"

namespace cunls {
namespace {

/** A symmetric block pattern: block sizes and the coupled block pairs (i < j). */
struct BlockPattern {
  std::vector<int> sizes;
  std::set<std::pair<int, int>> pairs;
};

/** A host SPD matrix (full, double) on a block pattern, diagonally dominant. */
struct HostSystem {
  int n = 0;
  std::vector<double> a;  // row-major n x n
  std::vector<int> row_offsets, col_ids;
};

HostSystem MakeSystem(const BlockPattern &pattern, unsigned seed) {
  HostSystem s;
  std::vector<int> start{0};
  for (int size : pattern.sizes) start.push_back(start.back() + size);
  s.n = start.back();
  const int n = s.n;
  s.a.assign(static_cast<size_t>(n) * n, 0.0);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto fill = [&](int bi, int bj) {
    for (int i = start[bi]; i < start[bi + 1]; ++i) {
      for (int j = start[bj]; j < start[bj + 1]; ++j) {
        if (bi == bj && j > i) continue;
        const double v = u(rng);
        s.a[static_cast<size_t>(i) * n + j] = v;
        s.a[static_cast<size_t>(j) * n + i] = v;
      }
    }
  };
  for (size_t b = 0; b < pattern.sizes.size(); ++b) fill(static_cast<int>(b), static_cast<int>(b));
  for (const auto &[bi, bj] : pattern.pairs) fill(bi, bj);
  for (int i = 0; i < n; ++i) {
    double off = 0.0;
    for (int j = 0; j < n; ++j) {
      if (j != i) off += std::fabs(s.a[static_cast<size_t>(i) * n + j]);
    }
    s.a[static_cast<size_t>(i) * n + i] = off + 1.0 + std::fabs(u(rng));
  }
  // Pattern CSR: the blocks, not the values (explicit zeros allowed).
  std::vector<std::vector<char>> coupled(pattern.sizes.size(),
                                         std::vector<char>(pattern.sizes.size(), 0));
  for (size_t b = 0; b < pattern.sizes.size(); ++b) coupled[b][b] = 1;
  for (const auto &[bi, bj] : pattern.pairs) coupled[bi][bj] = coupled[bj][bi] = 1;
  s.row_offsets.push_back(0);
  for (size_t bi = 0; bi < pattern.sizes.size(); ++bi) {
    for (int i = start[bi]; i < start[bi + 1]; ++i) {
      for (size_t bj = 0; bj < pattern.sizes.size(); ++bj) {
        if (!coupled[bi][bj]) continue;
        for (int j = start[bj]; j < start[bj + 1]; ++j) s.col_ids.push_back(j);
      }
      s.row_offsets.push_back(static_cast<int>(s.col_ids.size()));
    }
  }
  return s;
}

std::vector<float> CsrValues(const HostSystem &s) {
  std::vector<float> values;
  for (int i = 0; i < s.n; ++i) {
    for (int e = s.row_offsets[i]; e < s.row_offsets[i + 1]; ++e) {
      values.push_back(static_cast<float>(s.a[static_cast<size_t>(i) * s.n + s.col_ids[e]]));
    }
  }
  return values;
}

/** Device copy of a host system plus a right-hand side. */
struct DeviceSystem {
  explicit DeviceSystem(const HostSystem &s) : rhs(s.n), result(s.n) {
    matrix.row_offsets = dvector<int>(s.row_offsets);
    if (!s.col_ids.empty()) matrix.col_ids = dvector<int>(s.col_ids);
    SetValues(s);
  }
  void SetValues(const HostSystem &s) {
    const std::vector<float> values = CsrValues(s);
    matrix.values.resize(values.size());
    if (!values.empty()) matrix.values.CopyFromHost(values.data(), values.size());
  }
  CSRSparseMatrix matrix;
  dvector<float> rhs, result;
};

/**
 * Relative residual ‖A x - b‖ / ‖b‖ (double) of the GPU solution; the test
 * matrices are diagonally dominant, so it bounds the relative error too.
 */
double SolveError(BlockSparseCholeskySolver &solver, cudaStream_t stream, const HostSystem &s,
                  DeviceSystem &d, unsigned seed, bool *ok) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> b(s.n);
  std::vector<float> bf(s.n);
  for (int i = 0; i < s.n; ++i) bf[i] = static_cast<float>(b[i] = u(rng));
  if (s.n > 0) d.rhs.CopyFromHost(bf.data(), bf.size());
  *ok = solver.Solve(stream, d.matrix, d.rhs, d.result);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  if (s.n == 0) return 0.0;
  std::vector<float> x(s.n);
  d.result.CopyToHost(x.data(), x.size());
  double num = 0.0, den = 0.0;
  for (int i = 0; i < s.n; ++i) {
    double r = -b[i];
    for (int e = s.row_offsets[i]; e < s.row_offsets[i + 1]; ++e) {
      r += s.a[static_cast<size_t>(i) * s.n + s.col_ids[e]] * x[s.col_ids[e]];
    }
    num += r * r;
    den += b[i] * b[i];
  }
  return std::sqrt(num / den);
}

/** Keyframes of state blocks {6, 3, 6} (pose, velocity, bias), each coupled to the next `band`. */
BlockPattern Chain(int keyframes, int band) {
  BlockPattern p;
  for (int k = 0; k < keyframes; ++k) {
    for (int size : {6, 3, 6}) p.sizes.push_back(size);
  }
  for (int k = 0; k < keyframes; ++k) {
    for (int a = 0; a < 3; ++a) {
      for (int b = a + 1; b < 3; ++b) p.pairs.insert({3 * k + a, 3 * k + b});
    }
    for (int d = 1; d <= band && k + d < keyframes; ++d) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) p.pairs.insert({3 * k + a, 3 * (k + d) + b});
      }
    }
  }
  return p;
}

/** Poses (6) along a trajectory, coupled to neighbors and to random others with probability `fill`.
 */
BlockPattern Covisibility(int poses, double fill, unsigned seed) {
  BlockPattern p;
  p.sizes.assign(poses, 6);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int i = 0; i < poses; ++i) {
    for (int j = i + 1; j < poses; ++j) {
      if (j - i <= 3 || u(rng) < fill) p.pairs.insert({i, j});
    }
  }
  return p;
}

BlockPattern Dense(int blocks, int size) {
  BlockPattern p;
  p.sizes.assign(blocks, size);
  for (int i = 0; i < blocks; ++i) {
    for (int j = i + 1; j < blocks; ++j) p.pairs.insert({i, j});
  }
  return p;
}

/** A w x h grid of blocks of the given size, each coupled to its right and lower neighbors. */
BlockPattern Grid(int w, int h, int size) {
  BlockPattern p;
  p.sizes.assign(w * h, size);
  for (int i = 0; i < w * h; ++i) {
    if (i % w + 1 < w) p.pairs.insert({i, i + 1});
    if (i + w < w * h) p.pairs.insert({i, i + w});
  }
  return p;
}

/**
 * The full system of bundle adjustment: a chain of poses (6) and landmarks
 * (3) each seen by 3 to 5 random poses, so that every pose reaches far more
 * vertices than the average (the lazily updated vertices of minimum degree).
 */
BlockPattern BundleAdjustment(int poses, int landmarks, unsigned seed) {
  BlockPattern p;
  p.sizes.assign(poses, 6);
  p.sizes.insert(p.sizes.end(), landmarks, 3);
  for (int i = 0; i + 1 < poses; ++i) p.pairs.insert({i, i + 1});
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> pose(0, poses - 1), views(3, 5);
  for (int l = 0; l < landmarks; ++l) {
    for (int v = views(rng); v > 0; --v) p.pairs.insert({pose(rng), poses + l});
  }
  return p;
}

struct Case {
  const char *name;
  BlockPattern pattern;
};

std::vector<Case> Cases() {
  std::vector<Case> cases;
  cases.push_back({"Scalar", {{1}, {}}});
  cases.push_back({"TwoBlocks", {{3, 6}, {{0, 1}}}});
  cases.push_back({"Chain", Chain(120, 6)});
  cases.push_back({"LongChain", Chain(400, 3)});
  cases.push_back({"Covisibility", Covisibility(250, 0.15, 7)});
  cases.push_back({"SparseCovisibility", Covisibility(300, 0.02, 11)});
  cases.push_back({"Dense", Dense(50, 6)});  // 300 columns: several panels per supernode
  cases.push_back({"WideBlock", {{150, 6, 70}, {{0, 1}, {1, 2}}}});
  cases.push_back({"Grid", Grid(30, 30, 3)});
  cases.push_back({"ScalarGrid", Grid(40, 40, 1)});
  cases.push_back({"BundleAdjustment", BundleAdjustment(12, 500, 5)});
  BlockPattern two = Chain(60, 4);
  const BlockPattern other = Covisibility(40, 0.1, 3);
  const int offset = static_cast<int>(two.sizes.size());
  two.sizes.insert(two.sizes.end(), other.sizes.begin(), other.sizes.end());
  for (const auto &[i, j] : other.pairs) two.pairs.insert({i + offset, j + offset});
  cases.push_back({"Disconnected", two});
  return cases;
}

class BlockSparseCholeskyTest : public ::testing::TestWithParam<Case> {};

TEST_P(BlockSparseCholeskyTest, SolvesAccurately) {
  const HostSystem s = MakeSystem(GetParam().pattern, 1);
  DeviceSystem d(s);
  CudaStream stream;
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  bool ok = false;
  EXPECT_LT(SolveError(solver, stream.GetStream(), s, d, 2, &ok), 1e-5);
  EXPECT_TRUE(ok);
  // A second right-hand side, and new values on the same pattern.
  EXPECT_LT(SolveError(solver, stream.GetStream(), s, d, 3, &ok), 1e-5);
  const HostSystem s2 = MakeSystem(GetParam().pattern, 4);
  d.SetValues(s2);
  EXPECT_LT(SolveError(solver, stream.GetStream(), s2, d, 5, &ok), 1e-5);
  EXPECT_TRUE(ok);
}

INSTANTIATE_TEST_SUITE_P(Structures, BlockSparseCholeskyTest, ::testing::ValuesIn(Cases()),
                         [](const ::testing::TestParamInfo<Case> &info) {
                           return std::string(info.param.name);
                         });

/** Panels never exceed the width limit, and every column belongs to exactly one panel. */
TEST(BlockSparseCholeskyPlanTest, PanelsCoverTheColumns) {
  const HostSystem s = MakeSystem(Dense(50, 6), 1);
  DeviceSystem d(s);
  CudaStream stream;
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  const internal::CholeskyPlan &plan = solver.Plan();
  int next = 0;
  for (const auto &panel : plan.panels) {
    EXPECT_LE(panel.width, internal::kCholeskyPanelWidth);
    EXPECT_GE(panel.col, next);
    next = panel.col + panel.width;
  }
  EXPECT_EQ(next, s.n);
  EXPECT_GT(plan.num_levels, 1);
}

TEST(BlockSparseCholeskySolverTest, EmptySystem) {
  HostSystem s;
  s.row_offsets = {0};
  DeviceSystem d(s);
  CudaStream stream;
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  EXPECT_TRUE(solver.Solve(stream.GetStream(), d.matrix, d.rhs, d.result));
}

/** A negative pivot is reported with safety checks; without them the result is non-finite. */
TEST(BlockSparseCholeskySolverTest, ReportsNonPositiveDefinite) {
  HostSystem s = MakeSystem(Chain(30, 3), 1);
  s.a[static_cast<size_t>(200) * s.n + 200] = -1.0;
  for (bool checks : {true, false}) {
    DeviceSystem d(s);
    CudaStream stream;
    BlockSparseCholeskySolver solver;
    if (!checks) solver.DisableSafetyChecks();
    ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
    bool ok = true;
    SolveError(solver, stream.GetStream(), s, d, 2, &ok);
    std::vector<float> x(s.n);
    d.result.CopyToHost(x.data(), x.size());
    bool finite = true;
    for (float v : x) finite = finite && std::isfinite(v);
    if (checks) {
      EXPECT_FALSE(ok);
    } else {
      EXPECT_TRUE(ok);
      EXPECT_FALSE(finite);
    }
  }
}

/**
 * A consistent semidefinite system, the normal equations of relative
 * measurements between poses without an anchor (Gauss-Newton with a free
 * gauge): the vanishing pivots fix their unknowns, and the solution
 * satisfies the system.
 */
TEST(BlockSparseCholeskySolverTest, SolvesConsistentSemidefiniteSystems) {
  const BlockPattern pattern = Covisibility(40, 0.1, 3);
  HostSystem s = MakeSystem(pattern, 1);
  const int n = s.n;
  std::fill(s.a.begin(), s.a.end(), 0.0);
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (const auto &[bi, bj] : pattern.pairs) {
    // Residual R (x_i - x_j): blocks RᵀR on the diagonal, -RᵀR off it.
    double r[6][6], rtr[6][6] = {};
    for (auto &row : r) {
      for (double &v : row) v = u(rng);
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        for (int k = 0; k < 6; ++k) rtr[i][j] += r[k][i] * r[k][j];
      }
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        s.a[static_cast<size_t>(6 * bi + i) * n + 6 * bi + j] += rtr[i][j];
        s.a[static_cast<size_t>(6 * bj + i) * n + 6 * bj + j] += rtr[i][j];
        s.a[static_cast<size_t>(6 * bi + i) * n + 6 * bj + j] -= rtr[i][j];
        s.a[static_cast<size_t>(6 * bj + i) * n + 6 * bi + j] -= rtr[i][j];
      }
    }
  }
  // b = A y: consistent (orthogonal to the null space, a common 6-vector).
  std::vector<double> y(n), b(n, 0.0);
  for (double &v : y) v = u(rng);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) b[i] += s.a[static_cast<size_t>(i) * n + j] * y[j];
  }
  std::vector<float> bf(b.begin(), b.end());
  DeviceSystem d(s);
  d.rhs.CopyFromHost(bf.data(), n);
  CudaStream stream;
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  EXPECT_TRUE(solver.Solve(stream.GetStream(), d.matrix, d.rhs, d.result));
  std::vector<float> x(n);
  d.result.CopyToHost(x.data(), n);
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; ++i) {
    double r = -b[i];
    for (int j = 0; j < n; ++j) r += s.a[static_cast<size_t>(i) * n + j] * x[j];
    num += r * r;
    den += b[i] * b[i];
  }
  EXPECT_LT(std::sqrt(num / den), 1e-4);
}

/**
 * Strongly coupled unknowns (pivot 1e-4 of its diagonal entry, as with stiff
 * motion priors) are regular, not a rank deficiency: solved accurately.
 */
TEST(BlockSparseCholeskySolverTest, KeepsSmallRegularPivots) {
  HostSystem s = MakeSystem(Dense(2, 1), 1);
  const double c = 1.0 - 5e-5;  // second pivot 1 - c^2 = 1e-4
  s.a = {1.0, c, c, 1.0};
  DeviceSystem d(s);
  const std::vector<float> b = {1.f, -1.f};
  d.rhs.CopyFromHost(b.data(), 2);
  CudaStream stream;
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  EXPECT_TRUE(solver.Solve(stream.GetStream(), d.matrix, d.rhs, d.result));
  std::vector<float> x(2);
  d.result.CopyToHost(x.data(), 2);
  // x = (1, -1) / (1 - c) = ±2e4; float32 keeps about 4 digits at this conditioning.
  EXPECT_NEAR(x[0], 1.0 / (1.0 - c), 2e2);
  EXPECT_NEAR(x[1], -1.0 / (1.0 - c), 2e2);
}

TEST(BlockSparseCholeskySolverTest, CreatedByTheFactory) {
  const HostSystem s = MakeSystem(Covisibility(40, 0.2, 5), 1);
  DeviceSystem d(s);
  CudaStream stream;
  SparseLinearSolverPtr solver =
      CreateSparseLinearSolver(SparseLinearSolverType::BlockSparseCholesky, {});
  ASSERT_NE(dynamic_cast<BlockSparseCholeskySolver *>(solver.get()), nullptr);
  ASSERT_TRUE(solver->Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  bool ok = false;
  EXPECT_LT(SolveError(*dynamic_cast<BlockSparseCholeskySolver *>(solver.get()), stream.GetStream(),
                       s, d, 2, &ok),
            1e-5);
  EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace cunls
