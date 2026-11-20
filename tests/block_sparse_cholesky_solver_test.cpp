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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
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

/**
 * A host SPD matrix (double) on a block pattern, diagonally dominant. Sparse: batches of
 * many subproblems have far too many columns for a dense copy.
 */
struct HostSystem {
  int n = 0;
  std::unordered_map<int64_t, double> a;  // (i, j) at i * n + j; absent entries are 0
  std::vector<int> row_offsets, col_ids;

  double &At(int i, int j) { return a[static_cast<int64_t>(i) * n + j]; }
  double Get(int i, int j) const {
    const auto it = a.find(static_cast<int64_t>(i) * n + j);
    return it == a.end() ? 0.0 : it->second;
  }
};

HostSystem MakeSystem(const BlockPattern &pattern, unsigned seed) {
  HostSystem s;
  std::vector<int> start{0};
  for (int size : pattern.sizes) start.push_back(start.back() + size);
  s.n = start.back();
  const int n = s.n;
  std::vector<double> off(n, 0.0);  // per row: sum of |off-diagonal entries|
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto fill = [&](int bi, int bj) {
    for (int i = start[bi]; i < start[bi + 1]; ++i) {
      for (int j = start[bj]; j < start[bj + 1]; ++j) {
        if (bi == bj && j > i) continue;
        const double v = u(rng);
        s.At(i, j) = v;
        s.At(j, i) = v;
        if (i != j) {
          off[i] += std::fabs(v);
          off[j] += std::fabs(v);
        }
      }
    }
  };
  for (size_t b = 0; b < pattern.sizes.size(); ++b) fill(static_cast<int>(b), static_cast<int>(b));
  for (const auto &[bi, bj] : pattern.pairs) fill(bi, bj);
  for (int i = 0; i < n; ++i) s.At(i, i) = off[i] + 1.0 + std::fabs(u(rng));
  // Pattern CSR: the blocks, not the values (explicit zeros allowed).
  std::vector<std::set<int>> coupled(pattern.sizes.size());
  for (size_t b = 0; b < pattern.sizes.size(); ++b) coupled[b].insert(static_cast<int>(b));
  for (const auto &[bi, bj] : pattern.pairs) {
    coupled[bi].insert(bj);
    coupled[bj].insert(bi);
  }
  s.row_offsets.push_back(0);
  for (size_t bi = 0; bi < pattern.sizes.size(); ++bi) {
    for (int i = start[bi]; i < start[bi + 1]; ++i) {
      for (int bj : coupled[bi]) {
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
      values.push_back(static_cast<float>(s.Get(i, s.col_ids[e])));
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
      r += s.Get(i, s.col_ids[e]) * x[s.col_ids[e]];
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

/** `blocks` blocks of `size` columns, each coupled to the next. */
BlockPattern BlockChain(int blocks, int size) {
  BlockPattern p;
  p.sizes.assign(blocks, size);
  for (int i = 0; i + 1 < blocks; ++i) p.pairs.insert({i, i + 1});
  return p;
}

/** `copies` disconnected copies of a pattern (a batch of independent subproblems). */
BlockPattern Replicate(const BlockPattern &one, int copies) {
  BlockPattern p;
  const int blocks = static_cast<int>(one.sizes.size());
  for (int c = 0; c < copies; ++c) {
    p.sizes.insert(p.sizes.end(), one.sizes.begin(), one.sizes.end());
    for (const auto &[i, j] : one.pairs) p.pairs.insert({c * blocks + i, c * blocks + j});
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
  // Batches: at least kCholeskyMinSmallPanelsPerLevel small panels per level take the warp
  // kernels (a single dense block, panels of exactly 32 rows, chains and grids with updates
  // between small panels, some of them to a shared target, small and tall panels together).
  cases.push_back({"BatchOfBlocks", Replicate({{9}, {}}, 400)});
  cases.push_back({"BatchOf32Blocks", Replicate({{32}, {}}, 300)});
  cases.push_back(
      {"BatchOfMixedBlocks", Replicate({{1, 5, 16, 31}, {{0, 1}, {1, 2}, {2, 3}}}, 260)});
  cases.push_back({"BatchOfChains", Replicate(BlockChain(20, 3), 300)});
  cases.push_back({"BatchOfGrids", Replicate(Grid(4, 4, 2), 260)});
  cases.push_back({"BatchOfSmallAndTall", Replicate({{30, 2, 40}, {{0, 1}, {1, 2}}}, 300)});
  cases.push_back({"BatchOfBundleAdjustments", Replicate(BundleAdjustment(3, 10, 5), 260)});
  cases.push_back({"BatchOfMixedChains", Replicate(BlockChain(40, 6), 260)});
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

/**
 * Per panel (factor, backward) or run (update): the tasks of the program's launches of that
 * phase that start it (a warp, or the CTA of its first tile).
 */
std::vector<int> Started(const internal::CholeskyPlan &plan, internal::CholeskyPlan::Phase phase) {
  using Phase = internal::CholeskyPlan::Phase;
  std::vector<int> started(phase == Phase::kUpdate ? plan.runs.size() : plan.panels.size(), 0);
  for (const auto &launch : plan.program) {
    if (launch.phase != phase) continue;
    for (int k = 0; k < launch.num_tasks; ++k) {
      const auto &task = plan.tasks[launch.first_task + k];
      const int first = phase == Phase::kUpdate ? plan.runs[task.index].begin : 0;
      if (launch.warp_width != 0 || task.row_begin == first) ++started[task.index];
    }
  }
  return started;
}

TEST(BlockSparseCholeskyPlanTest, SmallPanelsOfBatchesTakeWarps) {
  using Phase = internal::CholeskyPlan::Phase;
  auto plan_of = [](const BlockPattern &pattern) {
    const HostSystem s = MakeSystem(pattern, 1);
    DeviceSystem d(s);
    CudaStream stream;
    BlockSparseCholeskySolver solver;
    EXPECT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
    return solver.Plan();
  };
  constexpr int kBatch = internal::kCholeskyMinSmallPanelsPerLevel;
  // A batch of 32-row blocks: every panel a warp; roots, so one factor launch is the program.
  const internal::CholeskyPlan blocks = plan_of(Replicate({{32}, {}}, kBatch));
  EXPECT_EQ(blocks.num_small_panels, static_cast<int>(blocks.panels.size()));
  ASSERT_EQ(blocks.program.size(), 1u);
  EXPECT_EQ(blocks.program[0].phase, Phase::kFactor);
  EXPECT_EQ(blocks.program[0].warp_width, 32);
  EXPECT_EQ(plan_of(Replicate({{6}, {}}, kBatch)).program[0].warp_width, 8);
  // 33 rows, or too few panels in the level: the CTA kernels, the roots again in one launch.
  const internal::CholeskyPlan tall = plan_of(Replicate({{33}, {}}, kBatch));
  EXPECT_EQ(tall.num_small_panels, 0);
  ASSERT_EQ(tall.program.size(), 1u);
  EXPECT_EQ(tall.program[0].warp_width, 0);
  EXPECT_EQ(plan_of(Replicate({{9}, {}}, kBatch - 1)).num_small_panels, 0);
  // Chains and grids: every panel factored once, every run applied once, every panel but a
  // root (solved backward by its factor task) solved backward once.
  for (const BlockPattern &one : {BlockChain(40, 6), Grid(4, 4, 2)}) {
    const internal::CholeskyPlan plan = plan_of(Replicate(one, kBatch));
    EXPECT_GT(plan.num_small_panels, 0);
    const std::vector<int> factored = Started(plan, Phase::kFactor);
    const std::vector<int> backward = Started(plan, Phase::kBackward);
    const std::vector<int> updated = Started(plan, Phase::kUpdate);
    for (size_t p = 0; p < plan.panels.size(); ++p) {
      EXPECT_EQ(factored[p], 1) << "panel " << p;
      EXPECT_EQ(backward[p], plan.panels[p].height > plan.panels[p].width ? 1 : 0) << "panel " << p;
    }
    for (size_t r = 0; r < updated.size(); ++r) EXPECT_EQ(updated[r], 1) << "run " << r;
  }
}

/**
 * SetRowActivity: a block-diagonal system of three independent chains with
 * the middle chain declared idle. Its panels are skipped (an indefinite
 * matrix there does not fail the solve), its solution entries are 0, and the
 * other chains get the solution of the full system.
 */
class BlockSparseCholeskyBatchTest : public ::testing::TestWithParam<int> {};

/** The tests below run on one system (CTA kernels) and on a batch of copies (warp kernels). */
INSTANTIATE_TEST_SUITE_P(Copies, BlockSparseCholeskyBatchTest,
                         ::testing::Values(1, internal::kCholeskyMinSmallPanelsPerLevel));

TEST_P(BlockSparseCholeskyBatchTest, SkipsIdleComponents) {
  const int kChains = 3 * GetParam(), kBlocks = 12, kSize = 6;
  BlockPattern pattern;
  for (int b = 0; b < kChains * kBlocks; ++b) {
    pattern.sizes.push_back(kSize);
    if ((b + 1) % kBlocks != 0) pattern.pairs.insert({b, b + 1});
  }
  const HostSystem healthy = MakeSystem(pattern, 3);
  const int n = healthy.n, chain = kBlocks * kSize;
  // The middle chain made indefinite: a full solve of it fails.
  HostSystem broken = healthy;
  broken.At(chain + 5, chain + 5) = -10.0;

  CudaStream stream;
  DeviceSystem d(healthy);
  BlockSparseCholeskySolver solver;
  ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
  bool ok = false;
  EXPECT_LT(SolveError(solver, stream.GetStream(), healthy, d, 7, &ok), 1e-5);
  ASSERT_TRUE(ok);
  std::vector<float> full(n);
  d.result.CopyToHost(full.data(), n);

  std::vector<int> active(n, 1);
  for (int i = chain; i < 2 * chain; ++i) active[i] = 0;  // the second chain
  dvector<int> row_active(active);
  d.SetValues(broken);
  SolveError(solver, stream.GetStream(), broken, d, 7, &ok);
  EXPECT_FALSE(ok) << "the indefinite chain fails a full solve";
  solver.SetRowActivity(row_active.data());
  SolveError(solver, stream.GetStream(), broken, d, 7, &ok);
  EXPECT_TRUE(ok) << "the idle chain is not factored";
  std::vector<float> x(n);
  d.result.CopyToHost(x.data(), n);
  for (int i = 0; i < n; ++i) {
    if (active[i] != 0) {
      EXPECT_NEAR(x[i], full[i], 1e-5f * (1.f + std::fabs(full[i]))) << "row " << i;
    } else {
      EXPECT_EQ(x[i], 0.f) << "row " << i;
    }
  }
  // Every row active again: the same solution as without row activity.
  solver.SetRowActivity(nullptr);
  d.SetValues(healthy);
  EXPECT_LT(SolveError(solver, stream.GetStream(), healthy, d, 7, &ok), 1e-5);
  EXPECT_TRUE(ok);
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
TEST_P(BlockSparseCholeskyBatchTest, ReportsNonPositiveDefinite) {
  HostSystem s =
      MakeSystem(GetParam() == 1 ? Chain(30, 3) : Replicate(BlockChain(20, 3), GetParam()), 1);
  s.At(200, 200) = -1.0;
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
/**
 * An infinite diagonal entry (an overflowed assembly) fails the checked solve: the pivot is not
 * finite, so it is not mistaken for a zero pivot (a free gauge) whatever its threshold, also on a
 * leaf panel, which reads its threshold from the same entry.
 */
TEST_P(BlockSparseCholeskyBatchTest, ReportsInfiniteDiagonal) {
  const HostSystem healthy =
      MakeSystem(GetParam() == 1 ? Chain(30, 3) : Replicate(BlockChain(20, 3), GetParam()), 1);
  for (int i : {0, 200, healthy.n - 1}) {
    HostSystem s = healthy;
    s.At(i, i) = std::numeric_limits<double>::infinity();
    DeviceSystem d(s);
    CudaStream stream;
    BlockSparseCholeskySolver solver;
    ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
    bool ok = true;
    SolveError(solver, stream.GetStream(), s, d, 2, &ok);
    EXPECT_FALSE(ok) << "infinite diagonal at " << i;
  }
}

TEST_P(BlockSparseCholeskyBatchTest, SolvesConsistentSemidefiniteSystems) {
  const BlockPattern pattern =
      GetParam() == 1 ? Covisibility(40, 0.1, 3) : Replicate(Covisibility(4, 0.0, 3), GetParam());
  HostSystem s = MakeSystem(pattern, 1);
  const int n = s.n;
  s.a.clear();
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
        s.At(6 * bi + i, 6 * bi + j) += rtr[i][j];
        s.At(6 * bj + i, 6 * bj + j) += rtr[i][j];
        s.At(6 * bi + i, 6 * bj + j) -= rtr[i][j];
        s.At(6 * bj + i, 6 * bi + j) -= rtr[i][j];
      }
    }
  }
  // b = A y: consistent (orthogonal to the null space, a common 6-vector).
  std::vector<double> y(n), b(n, 0.0);
  for (double &v : y) v = u(rng);
  for (int i = 0; i < n; ++i) {
    for (int e = s.row_offsets[i]; e < s.row_offsets[i + 1]; ++e) {
      b[i] += s.Get(i, s.col_ids[e]) * y[s.col_ids[e]];
    }
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
    for (int e = s.row_offsets[i]; e < s.row_offsets[i + 1]; ++e) {
      r += s.Get(i, s.col_ids[e]) * x[s.col_ids[e]];
    }
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
  s.At(0, 0) = s.At(1, 1) = 1.0;
  s.At(0, 1) = s.At(1, 0) = c;
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

// Its accuracy is SolvesAccurately's; here only the factory's choice of backend.
TEST(BlockSparseCholeskySolverTest, CreatedByTheFactory) {
  SparseLinearSolverPtr solver =
      CreateSparseLinearSolver(SparseLinearSolverType::BlockSparseCholesky, {});
  EXPECT_NE(dynamic_cast<BlockSparseCholeskySolver *>(solver.get()), nullptr);
}

/**
 * Solve time per structure (clear, assembly through the entry map, factorization and solve,
 * without safety checks as in real-time use): minimum and median over repeated solves of one
 * system, by CUDA events. The structures span the solver's workload classes: deep trees
 * (chains, windows, grids), large factors, dense reduced systems, batches of small
 * subproblems. Run: nls_tests --gtest_also_run_disabled_tests --gtest_filter='*SolveTime'.
 */
TEST(BlockSparseCholeskyBenchmark, DISABLED_SolveTime) {
  const std::vector<Case> cases = {
      {"chain_6x10000", BlockChain(10000, 6)},
      {"vio_window_300", Chain(300, 3)},
      {"pgo_window_300", Covisibility(300, 0.02, 11)},
      {"pgo_random_5000", Covisibility(5000, 1.6e-4, 5)},
      {"grid_6_100x100", Grid(100, 100, 6)},
      {"grid_3_200x200", Grid(200, 200, 3)},
      {"grid_1_300x300", Grid(300, 300, 1)},
      {"covisibility_720", Covisibility(720, 0.15, 7)},
      {"dense_6x300", Dense(300, 6)},
      {"bundle_adjustment_200x10000", BundleAdjustment(200, 10000, 5)},
      {"batch_7x32768", Replicate({{7}, {}}, 32768)},
      {"batch_36x1024", Replicate({{36}, {}}, 1024)},
      {"batch_grid_4x4x3x512", Replicate(Grid(4, 4, 3), 512)},
      {"batch_chain_12x3x256", Replicate(BlockChain(12, 3), 256)},
      {"batch_chain_16x6x1024", Replicate(BlockChain(16, 6), 1024)},
  };
  CudaStream stream;
  cudaEvent_t start, stop;
  THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
  THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));
  for (const Case &c : cases) {
    const HostSystem s = MakeSystem(c.pattern, 1);
    DeviceSystem d(s);
    const std::vector<float> ones(s.n, 1.f);
    d.rhs.CopyFromHost(ones.data(), ones.size());
    BlockSparseCholeskySolver solver;
    solver.DisableSafetyChecks();
    ASSERT_TRUE(solver.Initialize(stream.GetStream(), Problem(), d.matrix, d.rhs, d.result));
    std::vector<float> times;
    constexpr int kWarmUp = 3;
    for (int rep = 0, reps = kWarmUp + 1; rep < reps; ++rep) {
      THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream.GetStream()));
      solver.Solve(stream.GetStream(), d.matrix, d.rhs, d.result);
      THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream.GetStream()));
      THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));
      float ms = 0.f;
      THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
      if (rep < kWarmUp) continue;
      times.push_back(1e3f * ms);
      // About 0.3 s per structure, 20 to 300 solves.
      if (rep == kWarmUp) reps += std::clamp(static_cast<int>(3e5f / times[0]), 20, 300);
    }
    std::sort(times.begin(), times.end());
    std::printf("BENCH chol_solve %-28s min_us=%10.1f median_us=%10.1f n=%d levels=%d\n", c.name,
                times.front(), times[times.size() / 2], s.n, solver.Plan().num_levels);
  }
  THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
  THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));
}

}  // namespace
}  // namespace cunls
