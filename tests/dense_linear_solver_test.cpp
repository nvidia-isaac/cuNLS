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

/** @file dense_linear_solver_test.cpp
 *  @brief Tests of the dense backends (DenseLDLT, DenseCholesky, DenseQR): the
 *  same systems run through each, with per-backend expectations where they
 *  differ (only DenseCholesky rejects a symmetric indefinite matrix).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

constexpr float kSolutionTolerance = 1e-3f;

/** A x A^T + n I with entries of x uniform in [-bound, bound]: SPD. */
std::vector<float> RandomSPDMatrix(int n, int seed, float value_abs_bound = 0.5f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-value_abs_bound, value_abs_bound);
  std::vector<float> m(static_cast<size_t>(n) * n);
  for (float &v : m) {
    v = dist(rng);
  }
  std::vector<float> matrix(static_cast<size_t>(n) * n, 0.0f);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float value = row == col ? static_cast<float>(n) : 0.0f;
      for (int k = 0; k < n; ++k) {
        value += m[row * n + k] * m[col * n + k];
      }
      matrix[row * n + col] = value;
    }
  }
  return matrix;
}

std::vector<float> MultiplyMatVec(const std::vector<float> &A, const std::vector<float> &x, int n) {
  std::vector<float> out(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[i] += A[i * n + j] * x[j];
    }
  }
  return out;
}

/** 4x4 symmetric indefinite matrix: y^T A y < 0 for y = [1, -1, 0, 0]. */
const std::vector<float> kIndefiniteMatrix = {
    0.0f, 2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 0.0f, 1.0f, 2.0f,
};

/** Rank-1 3x3 matrix: every row is a multiple of [1, 2, 3]. */
const std::vector<float> kRankOneMatrix = {
    1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f, 3.0f, 6.0f, 9.0f,
};

class DenseSolverTest : public ::testing::TestWithParam<SparseLinearSolverType> {
 protected:
  void SetUp() override { solver_ = CreateSparseLinearSolver(GetParam(), {}); }

  /**
   * Uploads the dense n x n matrix @p A (as CSR, zeros dropped) and @p rhs,
   * initializes and solves with the one solver instance of the test, and
   * returns Solve's result; @p solution receives x when it succeeded.
   */
  bool Solve(const std::vector<float> &A, const std::vector<float> &rhs_host,
             std::vector<float> *solution = nullptr) {
    const int n = static_cast<int>(rhs_host.size());
    std::vector<int> row_ptr{0};
    std::vector<int> col_idx;
    std::vector<float> values;
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < n; ++col) {
        if (A[row * n + col] != 0.0f) {
          col_idx.push_back(col);
          values.push_back(A[row * n + col]);
        }
      }
      row_ptr.push_back(static_cast<int>(col_idx.size()));
    }
    CSRSparseMatrix matrix;
    test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
    dvector<float> rhs(rhs_host);
    dvector<float> result(n);
    EXPECT_TRUE(solver_->Initialize(stream_.GetStream(), Problem(), matrix, rhs, result));
    const bool ok = solver_->Solve(stream_.GetStream(), matrix, rhs, result);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));
    if (ok && solution != nullptr) {
      solution->assign(n, 0.0f);
      result.CopyToHost(solution->data(), n);
    }
    return ok;
  }

  /** Solves A x = A x_true and checks x against x_true. */
  void ExpectSolves(const std::vector<float> &A, const std::vector<float> &x_true) {
    const int n = static_cast<int>(x_true.size());
    std::vector<float> solved;
    ASSERT_TRUE(Solve(A, MultiplyMatVec(A, x_true, n), &solved));
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(solved[i], x_true[i], kSolutionTolerance) << "i = " << i;
    }
  }

  CudaStream stream_;
  SparseLinearSolverPtr solver_;
};

TEST_P(DenseSolverTest, SolveDenseSystemAcrossDifferentSizes) {
  for (const int n : {2, 3, 4, 8, 16, 24, 32}) {
    SCOPED_TRACE("Matrix size = " + std::to_string(n));
    std::vector<float> x_true(n);
    for (int i = 0; i < n; ++i) {
      x_true[i] = 0.25f + 0.3f * i;
    }
    ExpectSolves(RandomSPDMatrix(n, 7 + n), x_true);
  }
}

// DenseLDLT (pivoted) and DenseQR solve symmetric indefinite systems;
// DenseCholesky requires SPD and must report the failure.
TEST_P(DenseSolverTest, SymmetricIndefiniteSystem) {
  const std::vector<float> witness = {1.0f, -1.0f, 0.0f, 0.0f};
  const std::vector<float> Aw = MultiplyMatVec(kIndefiniteMatrix, witness, 4);
  ASSERT_LT(witness[0] * Aw[0] + witness[1] * Aw[1], 0.0f);

  if (GetParam() == SparseLinearSolverType::DenseCholesky) {
    EXPECT_FALSE(Solve(kIndefiniteMatrix, {1.0f, 2.0f, 3.0f, 4.0f}));
  } else {
    ExpectSolves(kIndefiniteMatrix, {1.0f, -2.0f, 0.5f, 3.0f});
  }
}

TEST_P(DenseSolverTest, SolveReturnsFalseForZeroMatrix) {
  EXPECT_FALSE(Solve(std::vector<float>(16, 0.0f), std::vector<float>(4, 1.0f)));
}

TEST_P(DenseSolverTest, SolveReturnsFalseForRankDeficientMatrix) {
  EXPECT_FALSE(Solve(kRankOneMatrix, {1.0f, 2.0f, 3.0f}));
}

TEST_P(DenseSolverTest, SolveReturnsTrueForValidSystemAfterSingularOne) {
  ASSERT_FALSE(Solve(std::vector<float>(9, 0.0f), std::vector<float>(3, 1.0f)));
  ExpectSolves(RandomSPDMatrix(3, 42), {1.0f, -0.5f, 2.0f});
}

/** Projects a world point through the world_from_camera pose state. */
Vector<2> ProjectPoint(const SE3Transform &world_from_cam, const Vector<3> &point_world) {
  const SE3Transform pose = test_utils::InverseSE3(world_from_cam);  // camera_from_world
  const float x =
      pose[0] * point_world[0] + pose[1] * point_world[1] + pose[2] * point_world[2] + pose[3];
  const float y =
      pose[4] * point_world[0] + pose[5] * point_world[1] + pose[6] * point_world[2] + pose[7];
  const float z =
      pose[8] * point_world[0] + pose[9] * point_world[1] + pose[10] * point_world[2] + pose[11];
  return Vector<2>{x / z, y / z};
}

SE3Transform TranslationPose(float x, float y, float z) {
  return {1.0f, 0.0f, 0.0f, x, 0.0f, 1.0f, 0.0f, y, 0.0f, 0.0f, 1.0f, z, 0.0f, 0.0f, 0.0f, 1.0f};
}

// One GaussNewtonMinimizer recovers a camera translation (PnP against fixed
// points) for growing point counts, reusing the dense solver across sizes.
TEST_P(DenseSolverTest, SolvePnPSequentiallyForDifferentPointCounts) {
  MinimizerOptions options;
  options.max_num_iterations = 20;
  options.state_tolerance = 1e-10f;
  options.cost_tolerance = 1e-12f;
  options.disable_safety_checks = false;
  options.sparse_linear_solver_type = GetParam();
  GaussNewtonMinimizer minimizer(options);

  const SE3Transform true_pose = TranslationPose(0.2f, -0.15f, 0.35f);
  for (const size_t num_points : {1000, 2000, 3000, 4000, 5000}) {
    SCOPED_TRACE("PnP point count = " + std::to_string(num_points));

    std::mt19937 rng(19 + static_cast<int>(num_points));
    std::uniform_real_distribution<float> xy_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> z_dist(4.0f, 8.0f);
    std::vector<Vector<3>> points_host(num_points);
    std::vector<Vector<2>> observations_host(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      points_host[i] = Vector<3>{xy_dist(rng), xy_dist(rng), z_dist(rng)};
      observations_host[i] = ProjectPoint(true_pose, points_host[i]);
    }

    dvector<SE3Transform> poses_device(
        std::vector<SE3Transform>{TranslationPose(0.45f, -0.45f, 0.8f)});
    dvector<Vector<3>> points_device(points_host);
    dvector<Vector<2>> observations_device(observations_host);
    dvector<int> point_const_ids_device(test_utils::MakeSequentialIds(num_points));

    SE3StateBatch pose_batch(reinterpret_cast<const float *>(poses_device.data()), 1);
    pose_batch.SetNumActiveStates(pose_batch.Capacity(), pose_batch.ConstCapacity());
    VectorStateBatch<3> point_batch(reinterpret_cast<const float *>(points_device.data()),
                                    num_points, point_const_ids_device.data(), num_points);
    point_batch.SetNumActiveStates(point_batch.Capacity(), point_batch.ConstCapacity());
    ReprojectionFactorBatch reprojection_factor(observations_device.data(), num_points);
    reprojection_factor.SetNumActiveFactors(reprojection_factor.Capacity());

    std::vector<float *> state_pointers;
    state_pointers.reserve(num_points * 2);
    for (size_t i = 0; i < num_points; ++i) {
      state_pointers.push_back(pose_batch.StateDevicePtr(0));
      state_pointers.push_back(point_batch.StateDevicePtr(i));
    }

    Problem problem;
    problem.AddStateBatch(&pose_batch);
    problem.AddStateBatch(&point_batch);
    problem.AddFactorBatch(&reprojection_factor, state_pointers);
    ASSERT_TRUE(problem.CheckConsistency());

    const auto summary = minimizer.Minimize(stream_.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));

    ASSERT_LT(summary.final_cost, summary.initial_cost);
    ASSERT_LT(summary.final_cost, 1e-5f);

    std::vector<SE3Transform> optimized_pose_host(1);
    poses_device.CopyToHost(optimized_pose_host.data(), 1);
    for (const int t : {3, 7, 11}) {
      ASSERT_NEAR(optimized_pose_host[0][t], true_pose[t], 5e-3f) << "translation entry " << t;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DenseBackends, DenseSolverTest,
                         ::testing::Values(SparseLinearSolverType::DenseLDLT,
                                           SparseLinearSolverType::DenseCholesky,
                                           SparseLinearSolverType::DenseQR),
                         [](const ::testing::TestParamInfo<SparseLinearSolverType> &info) {
                           switch (info.param) {
                             case SparseLinearSolverType::DenseLDLT:
                               return std::string("DenseLDLT");
                             case SparseLinearSolverType::DenseCholesky:
                               return std::string("DenseCholesky");
                             default:
                               return std::string("DenseQR");
                           }
                         });

}  // namespace
}  // namespace cunls
