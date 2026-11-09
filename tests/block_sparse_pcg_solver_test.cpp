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
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/ba_hessian_vector_product.h"
#include "cunls/minimizer/bsr_matrix.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"
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

/** Which points of MakeBundleAdjustmentSystem have no bodies. */
enum class Unobserved {
  kNone,        ///< every point sees at least one body
  kSomePoints,  ///< every 7th point and points 1000..1063
  kAllPoints,   ///< no point sees a body
};

/**
 * @brief Builds a diagonally dominant SPD matrix with a bundle-adjustment
 *        pattern: a chain of 6-dof bodies, and 3-dof points each coupled to
 *        1..max_bodies_per_point random bodies (none for the points selected
 *        by @p unobserved).
 */
System MakeBundleAdjustmentSystem(int num_bodies, int num_points, int max_bodies_per_point,
                                  unsigned seed = 11, Unobserved unobserved = Unobserved::kNone) {
  System s;
  const int body_rows = 6 * num_bodies;
  s.n = body_rows + 3 * num_points;
  std::mt19937 rng(seed);
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
    if (unobserved == Unobserved::kAllPoints ||
        (unobserved == Unobserved::kSomePoints && (p % 7 == 0 || (p >= 1000 && p < 1064)))) {
      bodies.clear();
    }
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

/** Uploads a host System as scalar CSR. */
void ToCsr(const System &s, CSRSparseMatrix &m) {
  test_utils::CreateCSRSparseMatrix(s.row_ptr, s.col_ids, s.values, m);
}

/** Uploads a host System as 3x3 block storage (rows and columns tiled by 3). */
void ToBsr3(const System &s, BSRSparseMatrix &m) {
  std::vector<int> row_ptr = {0};
  std::vector<int> col_ids;
  std::vector<float> values;
  int max_tiles = 0;
  for (int br = 0; br < s.n / 3; ++br) {
    std::map<int, std::array<float, 9>> tiles;
    for (int r = 0; r < 3; ++r) {
      for (int e = s.row_ptr[3 * br + r]; e < s.row_ptr[3 * br + r + 1]; ++e) {
        tiles[s.col_ids[e] / 3][3 * r + s.col_ids[e] % 3] = s.values[e];
      }
    }
    for (const auto &[c, tile] : tiles) {
      col_ids.push_back(c);
      values.insert(values.end(), tile.begin(), tile.end());
    }
    max_tiles = std::max(max_tiles, static_cast<int>(tiles.size()));
    row_ptr.push_back(static_cast<int>(col_ids.size()));
  }
  m.row_offsets = dvector<int>(row_ptr);
  m.col_ids = dvector<int>(col_ids);
  m.values = dvector<float>(values);
  m.block_size = 3;
  m.num_block_rows = s.n / 3;
  m.max_tiles_per_row = max_tiles;
}

/** Multiplies every value in the 3x6 block of (point, its first body) and its transpose by f. */
void ScalePointBlock(System &s, int num_bodies, int point, float f) {
  const int row = 6 * num_bodies + 3 * point;
  const int body_col = s.col_ids[s.row_ptr[row]];
  for (int r = 0; r < s.n; ++r) {
    for (int e = s.row_ptr[r]; e < s.row_ptr[r + 1]; ++e) {
      const int c = s.col_ids[e];
      const bool in_point_rows = r >= row && r < row + 3 && c >= body_col && c < body_col + 6;
      const bool in_body_rows = c >= row && c < row + 3 && r >= body_col && r < body_col + 6;
      if (in_point_rows || in_body_rows) s.values[e] *= f;
    }
  }
}

/** Multiplies every value in the rows and columns of body b (its pose-pose block once) by f. */
void ScaleBody(System &s, int b, float f) {
  for (int r = 0; r < s.n; ++r) {
    for (int e = s.row_ptr[r]; e < s.row_ptr[r + 1]; ++e) {
      const int c = s.col_ids[e];
      if (r / 6 == b || c / 6 == b) s.values[e] *= f;
    }
  }
}

/** Runs the BA product on the device and checks it against a double host product. */
template <typename Matrix>
std::vector<float> BaProduct(const Matrix &m, int block_size, int num_bodies, int num_points,
                             const std::vector<float> &x, bool *fallback) {
  CudaStream stream;
  BaHessianProduct product;
  EXPECT_TRUE(product.Build(stream.GetStream(), m.row_offsets.data(), m.col_ids.data(), block_size,
                            num_bodies, num_points));
  dvector<float> d_x(x);
  dvector<float> d_y(x.size());
  product.Multiply(stream.GetStream(), m.row_offsets.data(), m.col_ids.data(), m.values.data(),
                   d_x.data(), d_y.data());
  *fallback = product.LastMultiplyUsedFallback();
  std::vector<float> y(x.size());
  d_y.CopyToHost(y.data(), y.size());
  return y;
}

/** Max over rows of |y - H x| / sum |H_ij x_j|, with H x in double. */
double MaxRowError(const System &s, const std::vector<float> &x, const std::vector<float> &y) {
  double worst = 0.0;
  for (int i = 0; i < s.n; ++i) {
    double sum = 0.0;
    double abs_sum = 1e-30;
    for (int e = s.row_ptr[i]; e < s.row_ptr[i + 1]; ++e) {
      const double t = static_cast<double>(s.values[e]) * x[s.col_ids[e]];
      sum += t;
      abs_sum += std::fabs(t);
    }
    worst = std::max(worst, std::fabs(y[i] - sum) / abs_sum);
  }
  return worst;
}

/** Bundle-adjustment layout of MakeBundleAdjustmentSystem's output. */
BlockSparsePCGOptions BaOptions(int num_bodies, int num_points, bool ba_product) {
  BlockSparsePCGOptions options;
  options.block_layout = {{num_bodies, 6}, {num_points, 3}};
  if (ba_product) {
    options.hessian_product = BlockSparsePCGOptions::HessianProduct::kBundleAdjustment;
  }
  return options;
}

/** Exposes which `H p` the solver uses. */
class InspectablePCGSolver : public BlockSparsePCGSolver {
 public:
  using BlockSparsePCGSolver::BlockSparsePCGSolver;
  using BlockSparsePCGSolver::UsesBundleAdjustmentProduct;
};

/**
 * Initializes with @p init, solves once with @p m and returns x; *uses_ba
 * reports which H p the solver took.
 */
template <typename InitMatrix, typename Matrix>
std::vector<float> SolveAfterInitialize(const InitMatrix &init, const Matrix &m,
                                        const std::vector<float> &b,
                                        const BlockSparsePCGOptions &options, bool *uses_ba) {
  dvector<float> rhs(b);
  dvector<float> result(b.size());
  CudaStream stream;
  InspectablePCGSolver solver(options);
  EXPECT_TRUE(solver.Initialize(stream.GetStream(), Problem(), init, rhs, result));
  EXPECT_TRUE(solver.Solve(stream.GetStream(), m, rhs, result));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  *uses_ba = solver.UsesBundleAdjustmentProduct();
  std::vector<float> x(b.size());
  result.CopyToHost(x.data(), x.size());
  return x;
}

/** Solves once and returns x; *uses_ba reports which H p the solver took. */
template <typename Matrix>
std::vector<float> SolveWith(const Matrix &m, const std::vector<float> &b,
                             const BlockSparsePCGOptions &options, bool *uses_ba) {
  return SolveAfterInitialize(m, m, b, options, uses_ba);
}

/** Copy of @p s with rows a and a + 3 coupled: no longer a bundle-adjustment pattern. */
System CoupleRows(const System &s, int a) {
  System coupled = s;
  coupled.row_ptr.assign(1, 0);
  coupled.col_ids.clear();
  coupled.values.clear();
  for (int r = 0; r < s.n; ++r) {
    std::vector<std::pair<int, float>> row;
    for (int e = s.row_ptr[r]; e < s.row_ptr[r + 1]; ++e)
      row.emplace_back(s.col_ids[e], s.values[e]);
    if (r == a) row.emplace_back(a + 3, 0.01f);
    if (r == a + 3) row.emplace_back(a, 0.01f);
    std::sort(row.begin(), row.end());
    for (const auto &e : row) {
      coupled.col_ids.push_back(e.first);
      coupled.values.push_back(e.second);
    }
    coupled.row_ptr.push_back(static_cast<int>(coupled.col_ids.size()));
  }
  return coupled;
}

/** Whether the GPU can run the bundle-adjustment product (sm_80 or newer). */
bool BaProductSupported() {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  return cc_major >= 8;
}

/** y = H x on the device with an already built product. */
template <typename Matrix>
std::vector<float> MultiplyWith(BaHessianProduct &product, const Matrix &m,
                                const std::vector<float> &x) {
  CudaStream stream;
  dvector<float> d_x(x);
  dvector<float> d_y(x.size());
  EXPECT_TRUE(product.Multiply(stream.GetStream(), m.row_offsets.data(), m.col_ids.data(),
                               m.values.data(), d_x.data(), d_y.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> y(x.size());
  d_y.CopyToHost(y.data(), y.size());
  return y;
}

/** Relative L2 distance of a from b. */
double RelativeDistance(const std::vector<float> &a, const std::vector<float> &b) {
  double err = 0.0;
  double norm = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    err += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    norm += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(err / norm);
}

// Bodies and points of the BA tests: more points than one wave of warps
// handles (main, heavy and light windows all occur) and points seen by up to
// 24 bodies (heavy windows hold points with 16 or more).
constexpr int kBaBodies = 200;
constexpr int kBaPoints = 30000;
constexpr int kBaMaxBodiesPerPoint = 24;

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

/**
 * Scalar CSR and 3x3 block storage give the same H x, bit for bit, and on
 * every repeat; both match a double product.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductMatchesReferenceAndRepeats) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  const System s = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
  CSRSparseMatrix csr;
  BSRSparseMatrix bsr;
  ToCsr(s, csr);
  ToBsr3(s, bsr);
  std::vector<float> x;
  test_utils::GenerateRandomVector(s.n, x);
  bool fallback = true;
  const std::vector<float> y = BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
  EXPECT_FALSE(fallback);
  EXPECT_LT(MaxRowError(s, x, y), 1e-5);
  for (int repeat = 0; repeat < 3; ++repeat) {
    const std::vector<float> y_csr = BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
    const std::vector<float> y_bsr = BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_EQ(std::memcmp(y_csr.data(), y.data(), y.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(y_bsr.data(), y.data(), y.size() * sizeof(float)), 0);
  }
}

/**
 * One point's camera block is scaled far beyond the sampled range: by 2^30
 * the pose rows stay in fixed point (spill accumulator), by 2^60 they take
 * the scalar fallback.  Both stay accurate and repeatable.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductHandlesOutliers) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  std::vector<float> x;
  for (const auto &[scale, expect_fallback] : {std::pair{0x1p30f, false}, {0x1p60f, true}}) {
    System s = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
    // Point 1 is not among the points the kernel samples for its scale.
    ScalePointBlock(s, kBaBodies, 1, scale);
    test_utils::GenerateRandomVector(s.n, x);
    CSRSparseMatrix csr;
    BSRSparseMatrix bsr;
    ToCsr(s, csr);
    ToBsr3(s, bsr);
    bool fallback = !expect_fallback;
    const std::vector<float> y_csr = BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_EQ(fallback, expect_fallback);
    EXPECT_LT(MaxRowError(s, x, y_csr), 1e-5);
    const std::vector<float> y_bsr = BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_EQ(fallback, expect_fallback);
    EXPECT_LT(MaxRowError(s, x, y_bsr), 1e-5);
    const std::vector<float> y_again = BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_EQ(std::memcmp(y_again.data(), y_bsr.data(), y_bsr.size() * sizeof(float)), 0);
  }
}

/**
 * One body's rows and columns are scaled far below the sampled range, so its
 * pose-row terms fall below the fixed-point resolution; those rows are
 * recomputed with the scalar product and stay accurate and repeatable.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductHandlesSmallCameraBlocks) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  std::vector<float> x;
  test_utils::GenerateRandomVector(6 * kBaBodies + 3 * kBaPoints, x);
  for (const float scale : {0x1p-10f, 0x1p-20f, 0x1p-30f}) {
    System s = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
    ScaleBody(s, 5, scale);
    CSRSparseMatrix csr;
    BSRSparseMatrix bsr;
    ToCsr(s, csr);
    ToBsr3(s, bsr);
    bool fallback = true;
    const std::vector<float> y_csr = BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_FALSE(fallback);
    EXPECT_LT(MaxRowError(s, x, y_csr), 1e-5) << scale;
    const std::vector<float> y_bsr = BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_LT(MaxRowError(s, x, y_bsr), 1e-5) << scale;
    const std::vector<float> y_again = BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback);
    EXPECT_EQ(std::memcmp(y_again.data(), y_bsr.data(), y_bsr.size() * sizeof(float)), 0);
  }
}

/**
 * With a zero pose direction, body 5's pose rows are only its point terms;
 * scaled by 2^-30 or 2^-40 (same sign), each rounds to zero in fixed point,
 * and the rows are still recomputed with the scalar product.  Body 7 has zero
 * couplings, so its pose rows stay exactly zero.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductHandlesTermsRoundedToZero) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  std::vector<float> x;
  test_utils::GenerateRandomVector(6 * kBaBodies + 3 * kBaPoints, x);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = i < 6 * kBaBodies ? 0.f : std::fabs(x[i]);
  }
  for (const float scale : {0x1p-30f, 0x1p-40f}) {
    System s = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
    for (int r = 0; r < s.n; ++r) {
      for (int e = s.row_ptr[r]; e < s.row_ptr[r + 1]; ++e) {
        if (r / 6 == 5 || s.col_ids[e] / 6 == 5) s.values[e] = std::fabs(s.values[e]);
      }
    }
    ScaleBody(s, 5, scale);
    ScaleBody(s, 7, 0.f);
    CSRSparseMatrix csr;
    BSRSparseMatrix bsr;
    ToCsr(s, csr);
    ToBsr3(s, bsr);
    bool fallback = true;
    for (const bool tiles : {false, true}) {
      const std::vector<float> y = tiles ? BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback)
                                         : BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
      EXPECT_FALSE(fallback);
      EXPECT_LT(MaxRowError(s, x, y), 1e-5) << scale << " " << tiles;
      for (int i = 42; i < 48; ++i) EXPECT_EQ(y[i], 0.f) << i;
      const std::vector<float> y_again =
          tiles ? BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback)
                : BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
      EXPECT_EQ(std::memcmp(y_again.data(), y.data(), y.size() * sizeof(float)), 0);
    }
  }
}

/**
 * The solver takes the BA product for CSR and block input and converges to
 * the same iterate as the generic product.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductMatchesGenericSolve) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  const System s = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
  CSRSparseMatrix csr;
  BSRSparseMatrix bsr;
  ToCsr(s, csr);
  ToBsr3(s, bsr);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions generic = BaOptions(kBaBodies, kBaPoints, false);
  BlockSparsePCGOptions ba = BaOptions(kBaBodies, kBaPoints, true);
  for (auto *options : {&generic, &ba}) {
    options->relative_tolerance = 0.f;
    options->absolute_tolerance = 0.f;
    options->max_iterations = 30;
  }
  bool uses_ba = true;
  const std::vector<float> x_ref = SolveWith(csr, b, generic, &uses_ba);
  EXPECT_FALSE(uses_ba);
  const std::vector<float> x_csr = SolveWith(csr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  EXPECT_LT(RelativeDistance(x_csr, x_ref), 1e-4);
  const std::vector<float> x_bsr = SolveWith(bsr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  EXPECT_LT(RelativeDistance(x_bsr, x_ref), 1e-4);
  const std::vector<float> x_bsr_generic = SolveWith(bsr, b, generic, &uses_ba);
  EXPECT_FALSE(uses_ba);
  EXPECT_LT(RelativeDistance(x_bsr, x_bsr_generic), 1e-4);
}

/**
 * A layout or pattern that is not bundle-adjustment shaped keeps the generic
 * product and still solves.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductFallsBackToGeneric) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  System s = MakeBundleAdjustmentSystem(kBaBodies, 2000, 4);
  CSRSparseMatrix csr;
  ToCsr(s, csr);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions options = BaOptions(kBaBodies, 2000, true);
  bool uses_ba = false;
  const std::vector<float> x_ba = SolveWith(csr, b, options, &uses_ba);
  EXPECT_TRUE(uses_ba);
  options.block_layout.clear();
  options.block_size = 3;
  const std::vector<float> x_uniform = SolveWith(csr, b, options, &uses_ba);
  EXPECT_FALSE(uses_ba);
  // Couple the last two points: no longer a bundle-adjustment pattern.
  const System coupled = CoupleRows(s, 6 * kBaBodies + 3 * 1998);
  CSRSparseMatrix coupled_csr;
  ToCsr(coupled, coupled_csr);
  options = BaOptions(kBaBodies, 2000, true);
  const std::vector<float> x_coupled = SolveWith(coupled_csr, b, options, &uses_ba);
  EXPECT_FALSE(uses_ba);
  EXPECT_LT(RelativeDistance(x_coupled, x_ba), 1e-2);
  EXPECT_LT(RelativeDistance(x_uniform, x_ba), 1e-2);
}

/** Scalar CSR where row 6b + 1 has as many pose columns as row 6b, but not the same ones. */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductRejectsMismatchedPoseColumns) {
  int device = 0;
  int cc_major = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  if (cc_major < 8) GTEST_SKIP() << "needs sm_80 or newer";
  System s = MakeBundleAdjustmentSystem(kBaBodies, 2000, 4);
  // Row 7 holds the pose columns of bodies 0..2; swap column 0 for column 18 (body 3).
  const int e0 = s.row_ptr[7];
  ASSERT_EQ(s.col_ids[e0], 0);
  s.col_ids[e0] = 18;
  std::sort(s.col_ids.begin() + e0, s.col_ids.begin() + e0 + 18);
  CSRSparseMatrix csr;
  ToCsr(s, csr);
  CudaStream stream;
  BaHessianProduct product;
  EXPECT_FALSE(product.Build(stream.GetStream(), csr.row_offsets.data(), csr.col_ids.data(), 1,
                             kBaBodies, 2000));
}

/**
 * Two products on two streams, called alternately without synchronization,
 * each give their own bit-exact result, and the scalar-fallback flag of one
 * instance does not leak into the other.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductConcurrentInstances) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  constexpr int kBodiesB = 150;
  constexpr int kPointsB = 20000;
  constexpr int kRounds = 20;
  System sa = MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint);
  const System sb = MakeBundleAdjustmentSystem(kBodiesB, kPointsB, kBaMaxBodiesPerPoint, 23);
  CSRSparseMatrix a;
  BSRSparseMatrix b;
  ToCsr(sa, a);
  ToBsr3(sb, b);
  std::vector<float> xa;
  std::vector<float> xb;
  test_utils::GenerateRandomVector(sa.n, xa);
  test_utils::GenerateRandomVector(sb.n, xb);
  bool fallback = true;
  const std::vector<float> ref_a = BaProduct(a, 1, kBaBodies, kBaPoints, xa, &fallback);
  const std::vector<float> ref_b = BaProduct(b, 3, kBodiesB, kPointsB, xb, &fallback);

  CudaStream stream_a;
  CudaStream stream_b;
  BaHessianProduct pa;
  BaHessianProduct pb;
  ASSERT_TRUE(pa.Build(stream_a.GetStream(), a.row_offsets.data(), a.col_ids.data(), 1, kBaBodies,
                       kBaPoints));
  ASSERT_TRUE(pb.Build(stream_b.GetStream(), b.row_offsets.data(), b.col_ids.data(), 3, kBodiesB,
                       kPointsB));
  dvector<float> d_xa(xa);
  dvector<float> d_xb(xb);
  std::vector<dvector<float>> ya;
  std::vector<dvector<float>> yb;
  for (int r = 0; r < kRounds; ++r) {
    ya.emplace_back(sa.n);
    yb.emplace_back(sb.n);
  }
  for (int r = 0; r < kRounds; ++r) {
    ASSERT_TRUE(pa.Multiply(stream_a.GetStream(), a.row_offsets.data(), a.col_ids.data(),
                            a.values.data(), d_xa.data(), ya[r].data()));
    ASSERT_TRUE(pb.Multiply(stream_b.GetStream(), b.row_offsets.data(), b.col_ids.data(),
                            b.values.data(), d_xb.data(), yb[r].data()));
  }
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
  std::vector<float> y(std::max(sa.n, sb.n));
  for (int r = 0; r < kRounds; ++r) {
    ya[r].CopyToHost(y.data(), sa.n);
    EXPECT_EQ(std::memcmp(y.data(), ref_a.data(), sa.n * sizeof(float)), 0) << r;
    yb[r].CopyToHost(y.data(), sb.n);
    EXPECT_EQ(std::memcmp(y.data(), ref_b.data(), sb.n * sizeof(float)), 0) << r;
  }

  // A takes the scalar fallback (same pattern, one block scaled by 2^60), then B runs clean.
  ScalePointBlock(sa, kBaBodies, 1, 0x1p60f);
  CSRSparseMatrix a_outlier;
  ToCsr(sa, a_outlier);
  ASSERT_TRUE(pa.Multiply(stream_a.GetStream(), a_outlier.row_offsets.data(),
                          a_outlier.col_ids.data(), a_outlier.values.data(), d_xa.data(),
                          ya[0].data()));
  ASSERT_TRUE(pb.Multiply(stream_b.GetStream(), b.row_offsets.data(), b.col_ids.data(),
                          b.values.data(), d_xb.data(), yb[0].data()));
  EXPECT_TRUE(pa.LastMultiplyUsedFallback());
  EXPECT_FALSE(pb.LastMultiplyUsedFallback());
}

/** 1379 cameras and 524287 points are the limits; one more is rejected with the reason. */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductSizeLimits) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  constexpr int kMaxCameras = 1379;
  constexpr int kPoints = 5000;
  const System s = MakeBundleAdjustmentSystem(kMaxCameras, kPoints, 8);
  CSRSparseMatrix csr;
  ToCsr(s, csr);
  std::vector<float> x;
  test_utils::GenerateRandomVector(s.n, x);
  bool fallback = true;
  const std::vector<float> y = BaProduct(csr, 1, kMaxCameras, kPoints, x, &fallback);
  EXPECT_FALSE(fallback);
  EXPECT_LT(MaxRowError(s, x, y), 1e-5);

  CudaStream stream;
  BaHessianProduct product;
  EXPECT_FALSE(product.Build(stream.GetStream(), csr.row_offsets.data(), csr.col_ids.data(), 1,
                             kMaxCameras + 1, kPoints));
  EXPECT_FALSE(product.Ready());
  EXPECT_NE(product.FailureReason().find("1380 cameras > 1379"), std::string::npos)
      << product.FailureReason();
  // The size checks run before the pattern is read, so this small matrix can stand in for one
  // with 524288 points.
  EXPECT_FALSE(product.Build(stream.GetStream(), csr.row_offsets.data(), csr.col_ids.data(), 1,
                             kMaxCameras, 524288));
  EXPECT_NE(product.FailureReason().find("524288 points > 524287"), std::string::npos)
      << product.FailureReason();
}

/**
 * NaN in a pose entry of x, Inf in a point entry of x, and Inf in one
 * camera-point value give non-finite rows exactly where a double product
 * does and accurate finite rows; the next clean call on the same instance is
 * accurate and takes no fallback.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductNonFiniteInputs) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  constexpr int kPoints = 5000;
  const System s = MakeBundleAdjustmentSystem(kBaBodies, kPoints, 8);
  CSRSparseMatrix csr;
  BSRSparseMatrix bsr;
  ToCsr(s, csr);
  ToBsr3(s, bsr);
  std::vector<float> x;
  test_utils::GenerateRandomVector(s.n, x);
  const float inf = std::numeric_limits<float>::infinity();
  const int point_row = 6 * kBaBodies + 3 * 100;
  for (int c = 0; c < 3; ++c) {
    System bad = s;
    std::vector<float> x_bad = x;
    if (c == 0) {
      x_bad[6 * 3 + 2] = std::numeric_limits<float>::quiet_NaN();
    } else if (c == 1) {
      x_bad[point_row + 1] = inf;
    } else {
      // First camera-point value of the point's row and its transposed copy.
      const int col = bad.col_ids[bad.row_ptr[point_row]];
      bad.values[bad.row_ptr[point_row]] = inf;
      for (int e = bad.row_ptr[col]; e < bad.row_ptr[col + 1]; ++e) {
        if (bad.col_ids[e] == point_row) bad.values[e] = inf;
      }
    }
    CSRSparseMatrix bad_csr;
    BSRSparseMatrix bad_bsr;
    ToCsr(bad, bad_csr);
    ToBsr3(bad, bad_bsr);
    for (const int block_size : {1, 3}) {
      SCOPED_TRACE("case " + std::to_string(c) + ", block size " + std::to_string(block_size));
      CudaStream stream;
      BaHessianProduct product;
      const int *row_offsets = block_size == 1 ? csr.row_offsets.data() : bsr.row_offsets.data();
      const int *col_ids = block_size == 1 ? csr.col_ids.data() : bsr.col_ids.data();
      ASSERT_TRUE(
          product.Build(stream.GetStream(), row_offsets, col_ids, block_size, kBaBodies, kPoints));
      const std::vector<float> y = block_size == 1 ? MultiplyWith(product, bad_csr, x_bad)
                                                   : MultiplyWith(product, bad_bsr, x_bad);
      // A non-finite camera-point term sends the pose rows to the scalar product; NaN in x's pose
      // part only reaches pose-pose terms, whose rows are recomputed one by one.
      EXPECT_EQ(product.LastMultiplyUsedFallback(), c != 0);
      int non_finite = 0;
      double worst = 0.0;
      for (int i = 0; i < bad.n; ++i) {
        double sum = 0.0;
        double abs_sum = 1e-30;
        for (int e = bad.row_ptr[i]; e < bad.row_ptr[i + 1]; ++e) {
          const double t = static_cast<double>(bad.values[e]) * x_bad[bad.col_ids[e]];
          sum += t;
          abs_sum += std::fabs(t);
        }
        ASSERT_EQ(std::isfinite(sum), std::isfinite(y[i])) << "row " << i;
        if (std::isfinite(sum)) {
          worst = std::max(worst, std::fabs(y[i] - sum) / abs_sum);
        } else {
          ++non_finite;
        }
      }
      EXPECT_GT(non_finite, 0);
      EXPECT_LT(worst, 1e-5);
      const std::vector<float> y_clean =
          block_size == 1 ? MultiplyWith(product, csr, x) : MultiplyWith(product, bsr, x);
      EXPECT_FALSE(product.LastMultiplyUsedFallback());
      EXPECT_LT(MaxRowError(s, x, y_clean), 1e-5);
    }
  }
}

/** Points that see no camera (some of them, or all) keep the product accurate and repeatable. */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductPointsWithoutCameras) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  for (const Unobserved unobserved : {Unobserved::kSomePoints, Unobserved::kAllPoints}) {
    const System s =
        MakeBundleAdjustmentSystem(kBaBodies, kBaPoints, kBaMaxBodiesPerPoint, 11, unobserved);
    CSRSparseMatrix csr;
    BSRSparseMatrix bsr;
    ToCsr(s, csr);
    ToBsr3(s, bsr);
    std::vector<float> x;
    test_utils::GenerateRandomVector(s.n, x);
    bool fallback = true;
    for (const bool tiles : {false, true}) {
      SCOPED_TRACE(std::string(unobserved == Unobserved::kAllPoints ? "all" : "some") +
                   (tiles ? ", 3x3 blocks" : ", CSR"));
      const std::vector<float> y = tiles ? BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback)
                                         : BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
      EXPECT_FALSE(fallback);
      EXPECT_LT(MaxRowError(s, x, y), 1e-5);
      const std::vector<float> y_again =
          tiles ? BaProduct(bsr, 3, kBaBodies, kBaPoints, x, &fallback)
                : BaProduct(csr, 1, kBaBodies, kBaPoints, x, &fallback);
      EXPECT_EQ(std::memcmp(y_again.data(), y.data(), y.size() * sizeof(float)), 0);
    }
  }
}

/**
 * Solve with another storage or pattern than Initialize re-checks the
 * matrix: it rebuilds the product when the new matrix fits and falls back
 * to the generic product when it does not.
 */
TEST(BlockSparsePCGSolverTest, BundleAdjustmentProductPatternChangeAfterInitialize) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  constexpr int kPoints = 2000;
  const System s = MakeBundleAdjustmentSystem(kBaBodies, kPoints, 4);
  const System other = MakeBundleAdjustmentSystem(kBaBodies, kPoints, 4, 23);
  const System coupled = CoupleRows(s, 6 * kBaBodies + 3 * (kPoints - 2));
  CSRSparseMatrix csr;
  CSRSparseMatrix other_csr;
  CSRSparseMatrix coupled_csr;
  BSRSparseMatrix bsr;
  ToCsr(s, csr);
  ToCsr(other, other_csr);
  ToCsr(coupled, coupled_csr);
  ToBsr3(s, bsr);
  std::vector<float> b;
  test_utils::GenerateRandomVector(s.n, b);
  BlockSparsePCGOptions ba = BaOptions(kBaBodies, kPoints, true);
  BlockSparsePCGOptions generic = BaOptions(kBaBodies, kPoints, false);
  for (auto *options : {&generic, &ba}) {
    options->relative_tolerance = 0.f;
    options->absolute_tolerance = 0.f;
    options->max_iterations = 30;
  }
  // The PCG dot products sum in arbitrary order, so solves are compared to 1e-5, not bitwise; a
  // product with stale tables is off by far more.
  bool uses_ba = false;
  const std::vector<float> x_fresh = SolveWith(csr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  const std::vector<float> x_bsr_then_csr = SolveAfterInitialize(bsr, csr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  EXPECT_LT(RelativeDistance(x_bsr_then_csr, x_fresh), 1e-5);

  const std::vector<float> x_other_fresh = SolveWith(other_csr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  const std::vector<float> x_other = SolveAfterInitialize(csr, other_csr, b, ba, &uses_ba);
  EXPECT_TRUE(uses_ba);
  EXPECT_LT(RelativeDistance(x_other, x_other_fresh), 1e-5);
  EXPECT_GT(RelativeDistance(x_other, x_fresh), 1e-2);

  const std::vector<float> x_coupled_generic = SolveWith(coupled_csr, b, generic, &uses_ba);
  const std::vector<float> x_coupled = SolveAfterInitialize(csr, coupled_csr, b, ba, &uses_ba);
  EXPECT_FALSE(uses_ba);
  EXPECT_LT(RelativeDistance(x_coupled, x_coupled_generic), 1e-5);
}

/**
 * Prints the time of one `H x` with the bundle-adjustment product and with
 * the generic CSR (cuSPARSE) and 3x3 block products, for a small and a large
 * system.  Run with --gtest_also_run_disabled_tests.
 */
TEST(BlockSparsePCGSolverTest, DISABLED_BundleAdjustmentProductTiming) {
  if (!BaProductSupported()) GTEST_SKIP() << "needs sm_80 or newer";
  constexpr int kCalls = 50;
  for (const auto &[bodies, points, per_point] :
       {std::array<int, 3>{kBaBodies, kBaPoints, kBaMaxBodiesPerPoint},
        std::array<int, 3>{1000, 400000, 8}}) {
    const System s = MakeBundleAdjustmentSystem(bodies, points, per_point);
    CSRSparseMatrix csr;
    BSRSparseMatrix bsr;
    ToCsr(s, csr);
    ToBsr3(s, bsr);
    std::vector<float> x;
    test_utils::GenerateRandomVector(s.n, x);
    dvector<float> d_x(x);
    dvector<float> d_y(s.n);
    dvector<uint8_t> buffer;
    CudaStream stream;
    cuSPARSEHandle handle;
    BaHessianProduct ba_csr;
    BaHessianProduct ba_bsr;
    ASSERT_TRUE(ba_csr.Build(stream.GetStream(), csr.row_offsets.data(), csr.col_ids.data(), 1,
                             bodies, points));
    ASSERT_TRUE(ba_bsr.Build(stream.GetStream(), bsr.row_offsets.data(), bsr.col_ids.data(), 3,
                             bodies, points));
    const std::array<std::pair<const char *, std::function<void()>>, 4> products = {{
        {"BA, CSR input",
         [&] {
           ba_csr.Multiply(stream.GetStream(), csr.row_offsets.data(), csr.col_ids.data(),
                           csr.values.data(), d_x.data(), d_y.data());
         }},
        {"BA, 3x3 block input",
         [&] {
           ba_bsr.Multiply(stream.GetStream(), bsr.row_offsets.data(), bsr.col_ids.data(),
                           bsr.values.data(), d_x.data(), d_y.data());
         }},
        {"generic CSR (cuSPARSE)",
         [&] {
           MultiplyCSRByDenseVector(stream.GetStream(), handle.GetHandle(stream.GetStream()), csr,
                                    s.n, s.n, static_cast<int>(csr.NumNonZeros()), d_x, d_y,
                                    buffer);
         }},
        {"generic 3x3 block", [&] { MultiplyBSRByDenseVector(stream.GetStream(), bsr, d_x, d_y); }},
    }};
    cudaEvent_t start;
    cudaEvent_t stop;
    THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
    THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));
    for (const auto &[name, multiply] : products) {
      multiply();  // warm-up
      THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream.GetStream()));
      for (int i = 0; i < kCalls; ++i) multiply();
      THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream.GetStream()));
      THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));
      float ms = 0.f;
      THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
      std::printf("[BA H x] %d cameras, %d points, %zu nonzeros, %s: %.1f us per call\n", bodies,
                  points, csr.NumNonZeros(), name, 1000.f * ms / kCalls);
    }
    THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
    THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));
  }
}

}  // namespace cunls
