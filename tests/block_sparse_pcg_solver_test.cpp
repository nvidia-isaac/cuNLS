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
 * @file block_sparse_pcg_solver_test.cpp
 * @brief Tests of BlockSparsePCGSolver on systems larger than one wave of CTAs.
 */

#include "cunls/linear_solver/block_sparse_pcg_solver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/** Host CSR copy of a symmetric positive definite system. */
struct System {
  int n = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_ids;
  std::vector<float> values;
};

/**
 * @brief Builds a diagonally dominant SPD matrix with a bundle-adjustment
 *        pattern: a chain of 6-dof bodies, and 3-dof points each coupled to
 *        1..max_bodies_per_point random bodies.
 */
System MakeBundleAdjustmentSystem(int num_bodies, int num_points, int max_bodies_per_point) {
  System s;
  const int body_rows = 6 * num_bodies;
  s.n = body_rows + 3 * num_points;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> val(-1.f, 1.f);
  std::uniform_int_distribution<int> body(0, num_bodies - 1);
  std::uniform_int_distribution<int> bodies_per_point(1, max_bodies_per_point);
  std::vector<std::vector<std::pair<int, float>>> rows(s.n);
  auto add = [&](int r, int c) {
    const float v = val(rng);
    rows[r].emplace_back(c, v);
    rows[c].emplace_back(r, v);
  };
  for (int b = 0; b < num_bodies; ++b) {
    for (int u = 0; u < 6; ++u) {
      for (int w = u + 1; w < 6; ++w) add(6 * b + u, 6 * b + w);
      if (b + 1 < num_bodies) {
        for (int w = 0; w < 6; ++w) add(6 * b + u, 6 * (b + 1) + w);
      }
    }
  }
  for (int p = 0; p < num_points; ++p) {
    const int row = body_rows + 3 * p;
    std::vector<int> bodies(bodies_per_point(rng));
    for (int &b : bodies) b = body(rng);
    std::sort(bodies.begin(), bodies.end());
    bodies.erase(std::unique(bodies.begin(), bodies.end()), bodies.end());
    for (int b : bodies) {
      for (int t = 0; t < 3; ++t) {
        for (int u = 0; u < 6; ++u) add(row + t, 6 * b + u);
      }
    }
    add(row, row + 1);
    add(row, row + 2);
    add(row + 1, row + 2);
  }
  s.row_ptr.push_back(0);
  for (int r = 0; r < s.n; ++r) {
    float abs_sum = 0.1f;
    for (const auto &e : rows[r]) abs_sum += std::fabs(e.second);
    rows[r].emplace_back(r, abs_sum);
    std::sort(rows[r].begin(), rows[r].end());
    for (const auto &e : rows[r]) {
      s.col_ids.push_back(e.first);
      s.values.push_back(e.second);
    }
    s.row_ptr.push_back(static_cast<int>(s.col_ids.size()));
  }
  return s;
}

/** Copies a host CSR system into BSR storage with square tiles of edge @p edge. */
void ToBsr(const System &s, int edge, BSRSparseMatrix &m) {
  const int num_block_rows = s.n / edge;
  std::vector<int> row_offsets = {0};
  std::vector<int> col_ids;
  std::vector<float> values;
  int max_tiles = 0;
  for (int br = 0; br < num_block_rows; ++br) {
    std::map<int, std::vector<float>> tiles;
    for (int k = 0; k < edge; ++k) {
      const int r = br * edge + k;
      for (int e = s.row_ptr[r]; e < s.row_ptr[r + 1]; ++e) {
        auto &tile = tiles[s.col_ids[e] / edge];
        tile.resize(edge * edge);
        tile[k * edge + s.col_ids[e] % edge] = s.values[e];
      }
    }
    for (const auto &[bc, tile] : tiles) {
      col_ids.push_back(bc);
      values.insert(values.end(), tile.begin(), tile.end());
    }
    row_offsets.push_back(static_cast<int>(col_ids.size()));
    max_tiles = std::max(max_tiles, static_cast<int>(tiles.size()));
  }
  m.row_offsets = dvector<int>(row_offsets);
  m.col_ids = dvector<int>(col_ids);
  m.values = dvector<float>(values);
  m.block_size = edge;
  m.max_tiles_per_row = max_tiles;
  m.num_block_rows = num_block_rows;
}

/** Runs the solver once from x_0 = 0 and returns x and the iteration count. */
template <typename Matrix>
std::pair<std::vector<float>, int> SolveOnce(const Matrix &m, const std::vector<float> &b,
                                             const BlockSparsePCGOptions &options) {
  const int n = static_cast<int>(b.size());
  dvector<float> rhs(b);
  dvector<float> result(n);
  CudaStream stream;
  BlockSparsePCGSolver solver(options);
  EXPECT_TRUE(solver.Initialize(stream.GetStream(), Problem(), m, rhs, result));
  EXPECT_TRUE(solver.Solve(stream.GetStream(), m, rhs, result));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> x(n);
  result.CopyToHost(x.data(), n);
  return {x, solver.LastIterations()};
}

/** Runs the solver once on the CSR form of @p s; see the template above. */
std::pair<std::vector<float>, int> SolveOnce(const System &s, const std::vector<float> &b,
                                             const BlockSparsePCGOptions &options) {
  CSRSparseMatrix m;
  test_utils::CreateCSRSparseMatrix(s.row_ptr, s.col_ids, s.values, m);
  return SolveOnce(m, b, options);
}

/** Jacobi-preconditioned CG in double from x_0 = 0 for a fixed number of iterations. */
std::vector<double> ReferencePcg(const System &s, const std::vector<float> &b, int iterations) {
  const int n = s.n;
  std::vector<double> x(n, 0.0), r(b.begin(), b.end()), z(n), p(n), ap(n), inv_diag(n);
  for (int i = 0; i < n; ++i) {
    for (int e = s.row_ptr[i]; e < s.row_ptr[i + 1]; ++e) {
      if (s.col_ids[e] == i) inv_diag[i] = 1.0 / s.values[e];
    }
    z[i] = inv_diag[i] * r[i];
  }
  p = z;
  double rz = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
  for (int it = 0; it < iterations; ++it) {
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int e = s.row_ptr[i]; e < s.row_ptr[i + 1]; ++e) sum += s.values[e] * p[s.col_ids[e]];
      ap[i] = sum;
    }
    const double alpha = rz / std::inner_product(p.begin(), p.end(), ap.begin(), 0.0);
    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      z[i] = inv_diag[i] * r[i];
    }
    const double rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
    for (int i = 0; i < n; ++i) p[i] = z[i] + rz_new / rz * p[i];
    rz = rz_new;
  }
  return x;
}

}  // namespace

// About 1.4 M rows: the PCG kernels launch more CTAs than fit on a GPU at once,
// and every CTA of the direction update must use the same beta.
TEST(BlockSparsePCGSolverTest, MatchesReferencePcgIteratesOnLargeSystem) {
  constexpr int kIterations = 8;
  const System s = MakeBundleAdjustmentSystem(40000, 400000, 1);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions options;
  options.block_size = 1;
  options.relative_tolerance = 0.f;
  options.absolute_tolerance = 0.f;
  options.max_iterations = kIterations;
  const auto [x, iterations] = SolveOnce(s, b, options);
  ASSERT_EQ(iterations, kIterations);
  const std::vector<double> ref = ReferencePcg(s, b, kIterations);
  double err = 0.0;
  double norm = 0.0;
  for (int i = 0; i < s.n; ++i) {
    err += (x[i] - ref[i]) * (x[i] - ref[i]);
    norm += ref[i] * ref[i];
  }
  EXPECT_LT(std::sqrt(err / norm), 1e-4);
}

// About 1.4 M rows in BSR storage: H p runs the in-repo BSR kernels, which add
// in a fixed order, so the PCG dot products decide whether solves repeat.
TEST(BlockSparsePCGSolverTest, RepeatedSolvesAreBitIdentical) {
  const System s = MakeBundleAdjustmentSystem(40000, 400000, 1);
  BSRSparseMatrix m;
  ToBsr(s, 3, m);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions options;
  options.block_layout = {{40000, 6}, {400000, 3}};
  options.relative_tolerance = 0.f;
  options.absolute_tolerance = 0.f;
  options.max_iterations = 50;
  options.deterministic_mode = true;
  const auto [x0, iterations0] = SolveOnce(m, b, options);
  for (int k = 0; k < 3; ++k) {
    const auto [x, iterations] = SolveOnce(m, b, options);
    EXPECT_EQ(iterations, iterations0);
    EXPECT_EQ(std::memcmp(x.data(), x0.data(), x.size() * sizeof(float)), 0);
  }
}

// About 1.2 M rows, with body rows of up to 27 k non-zeros, on which the
// default cuSPARSE SpMV is not bit-reproducible.
TEST(BlockSparsePCGSolverTest, DeterministicModeRepeatsSolvesWithLongRows) {
  const System s = MakeBundleAdjustmentSystem(200, 400000, 8);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions options;
  options.block_layout = {{200, 6}, {400000, 3}};
  options.relative_tolerance = 0.f;
  options.absolute_tolerance = 0.f;
  options.max_iterations = 50;
  options.deterministic_mode = true;
  const auto [x0, iterations0] = SolveOnce(s, b, options);
  for (int k = 0; k < 3; ++k) {
    const auto [x, iterations] = SolveOnce(s, b, options);
    EXPECT_EQ(iterations, iterations0);
    EXPECT_EQ(std::memcmp(x.data(), x0.data(), x.size() * sizeof(float)), 0);
  }
}

}  // namespace cunls
