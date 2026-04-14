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

#include "cunls/linear_solver/dense_qr_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

void GenerateRandomSPDMatrix(int n, int seed, float value_abs_bound,
                             std::vector<float>& matrix) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-value_abs_bound, value_abs_bound);
  std::vector<float> m(static_cast<size_t>(n) * n);
  for (float& v : m) {
    v = dist(rng);
  }

  matrix.assign(static_cast<size_t>(n) * n, 0.0f);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      float value = 0.0f;
      for (int k = 0; k < n; ++k) {
        value += m[row * n + k] * m[col * n + k];
      }
      if (row == col) {
        value += static_cast<float>(n);
      }
      matrix[row * n + col] = value;
    }
  }
}

std::vector<float> MultiplyMatVec(const std::vector<float>& A,
                                  const std::vector<float>& x, int n) {
  std::vector<float> out(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
      sum += A[i * n + j] * x[j];
    }
    out[i] = sum;
  }
  return out;
}

void DenseToCSR(const std::vector<float>& dense, int n,
                std::vector<int>& row_ptr, std::vector<int>& col_idx,
                std::vector<float>& values, float zero_threshold = 0.0f) {
  row_ptr.clear();
  col_idx.clear();
  values.clear();
  row_ptr.reserve(static_cast<size_t>(n) + 1);
  row_ptr.push_back(0);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      const float value = dense[row * n + col];
      if (fabsf(value) <= zero_threshold) {
        continue;
      }
      col_idx.push_back(col);
      values.push_back(value);
    }
    row_ptr.push_back(static_cast<int>(col_idx.size()));
  }
}

SE3Transform MakeIdentityPose() {
  SE3Transform pose{};
  pose.fill(0.0f);
  pose[0] = 1.0f;
  pose[5] = 1.0f;
  pose[10] = 1.0f;
  pose[15] = 1.0f;
  return pose;
}

Vector<2> ProjectPoint(const SE3Transform& pose,
                       const Vector<3>& point_world) {
  const float x = pose[0] * point_world[0] + pose[1] * point_world[1] +
                  pose[2] * point_world[2] + pose[3];
  const float y = pose[4] * point_world[0] + pose[5] * point_world[1] +
                  pose[6] * point_world[2] + pose[7];
  const float z = pose[8] * point_world[0] + pose[9] * point_world[1] +
                  pose[10] * point_world[2] + pose[11];
  return Vector<2>{x / z, y / z};
}

class DenseQRSolverTestFixture : public ::testing::Test {
 protected:
  int spd_generation_seed_ = 7;
  float spd_value_abs_bound_ = 0.5f;
  std::vector<int> dense_solver_validation_sizes_ = {2, 3, 4, 8, 16, 24, 32};

  size_t pnp_num_points_ = 1000;
  int pnp_generation_seed_ = 19;
  float pnp_xy_abs_bound_ = 1.0f;
  float pnp_depth_min_ = 4.0f;
  float pnp_depth_max_ = 8.0f;
  Vector<3> true_pose_translation_ = {0.2f, -0.15f, 0.35f};
  Vector<3> initial_pose_translation_ = {0.45f, -0.45f, 0.8f};

  float linear_solver_rhs_scale_ = 0.3f;
  float linear_solver_base_value_ = 0.25f;
  float linear_solver_solution_tolerance_ = 1e-3f;

  CudaStream stream_;
  cuBLASHandle cublas_handle_;
  profiler::Domain profiler_domain_{"DenseQRSolverTestFixture"};
};

TEST_F(DenseQRSolverTestFixture, SolveDenseSystemAcrossDifferentSizes) {
  DenseQRSolver solver;

  for (const int n : dense_solver_validation_sizes_) {
    SCOPED_TRACE("Matrix size = " + std::to_string(n));

    std::vector<float> host_A;
    GenerateRandomSPDMatrix(n, spd_generation_seed_ + n, spd_value_abs_bound_,
                            host_A);

    std::vector<float> x_true(n, 0.0f);
    for (int i = 0; i < n; ++i) {
      x_true[i] = linear_solver_base_value_ + linear_solver_rhs_scale_ * i;
    }
    const std::vector<float> rhs_host = MultiplyMatVec(host_A, x_true, n);

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    DenseToCSR(host_A, n, row_ptr, col_idx, values);

    CSRSparseMatrix matrix;
    test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
    dvector<float> rhs(rhs_host);
    dvector<float> result(n);

    ASSERT_TRUE(
        solver.Initialize(stream_.GetStream(), matrix, rhs, result));
    ASSERT_TRUE(
        solver.Solve(stream_.GetStream(), matrix, rhs, result));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));

    std::vector<float> solved_host(n, 0.0f);
    result.CopyToHost(solved_host.data(), solved_host.size());
    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solved_host[i], x_true[i],
                  linear_solver_solution_tolerance_);
    }
  }
}

TEST_F(DenseQRSolverTestFixture, SolveSymmetricIndefiniteSystem) {
  DenseQRSolver solver;
  constexpr int n = 4;
  const std::vector<float> host_A = {
      0.0f, 2.0f, 0.0f, 0.0f,
      2.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 1.0f, 3.0f, 1.0f,
      0.0f, 0.0f, 1.0f, 2.0f,
  };

  const std::vector<float> x_true = {1.0f, -2.0f, 0.5f, 3.0f};
  const std::vector<float> rhs_host = MultiplyMatVec(host_A, x_true, n);

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  DenseToCSR(host_A, n, row_ptr, col_idx, values);

  CSRSparseMatrix matrix;
  test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
  dvector<float> rhs(rhs_host);
  dvector<float> result(n);

  ASSERT_TRUE(
      solver.Initialize(stream_.GetStream(), matrix, rhs, result));
  ASSERT_TRUE(solver.Solve(stream_.GetStream(), matrix, rhs, result));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));

  std::vector<float> solved_host(n, 0.0f);
  result.CopyToHost(solved_host.data(), solved_host.size());
  for (int i = 0; i < n; ++i) {
    ASSERT_NEAR(solved_host[i], x_true[i],
                linear_solver_solution_tolerance_);
  }
}

TEST_F(DenseQRSolverTestFixture, SolveReturnsFalseForZeroMatrix) {
  DenseQRSolver solver;
  constexpr int n = 4;

  const std::vector<float> host_A(n * n, 0.0f);
  const std::vector<float> rhs_host(n, 1.0f);

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  DenseToCSR(host_A, n, row_ptr, col_idx, values);

  CSRSparseMatrix matrix;
  test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
  dvector<float> rhs(rhs_host);
  dvector<float> result(n);

  ASSERT_TRUE(solver.Initialize(stream_.GetStream(), matrix, rhs, result));
  ASSERT_FALSE(solver.Solve(stream_.GetStream(), matrix, rhs, result));
}

TEST_F(DenseQRSolverTestFixture, SolveReturnsFalseForRankDeficientMatrix) {
  DenseQRSolver solver;
  constexpr int n = 3;

  const std::vector<float> host_A = {
      1.0f, 2.0f, 3.0f,
      2.0f, 4.0f, 6.0f,
      3.0f, 6.0f, 9.0f,
  };
  const std::vector<float> rhs_host = {1.0f, 2.0f, 3.0f};

  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  DenseToCSR(host_A, n, row_ptr, col_idx, values);

  CSRSparseMatrix matrix;
  test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
  dvector<float> rhs(rhs_host);
  dvector<float> result(n);

  ASSERT_TRUE(solver.Initialize(stream_.GetStream(), matrix, rhs, result));
  ASSERT_FALSE(solver.Solve(stream_.GetStream(), matrix, rhs, result));
}

TEST_F(DenseQRSolverTestFixture,
       SolveReturnsTrueForValidSystemAfterSingularOne) {
  DenseQRSolver solver;

  {
    constexpr int n = 3;
    const std::vector<float> host_A(n * n, 0.0f);
    const std::vector<float> rhs_host(n, 1.0f);
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    DenseToCSR(host_A, n, row_ptr, col_idx, values);

    CSRSparseMatrix matrix;
    test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
    dvector<float> rhs(rhs_host);
    dvector<float> result(n);

    ASSERT_TRUE(solver.Initialize(stream_.GetStream(), matrix, rhs, result));
    ASSERT_FALSE(solver.Solve(stream_.GetStream(), matrix, rhs, result));
  }

  {
    constexpr int n = 3;
    std::vector<float> host_A;
    GenerateRandomSPDMatrix(n, 42, spd_value_abs_bound_, host_A);

    const std::vector<float> x_true = {1.0f, -0.5f, 2.0f};
    const std::vector<float> rhs_host = MultiplyMatVec(host_A, x_true, n);

    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    DenseToCSR(host_A, n, row_ptr, col_idx, values);

    CSRSparseMatrix matrix;
    test_utils::CreateCSRSparseMatrix(row_ptr, col_idx, values, matrix);
    dvector<float> rhs(rhs_host);
    dvector<float> result(n);

    ASSERT_TRUE(solver.Initialize(stream_.GetStream(), matrix, rhs, result));
    ASSERT_TRUE(solver.Solve(stream_.GetStream(), matrix, rhs, result));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));

    std::vector<float> solved_host(n, 0.0f);
    result.CopyToHost(solved_host.data(), solved_host.size());
    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solved_host[i], x_true[i],
                  linear_solver_solution_tolerance_);
    }
  }
}

TEST_F(DenseQRSolverTestFixture, SolvePnPWithQRSolver) {
  const size_t num_points = pnp_num_points_;
  std::mt19937 rng(pnp_generation_seed_);
  std::uniform_real_distribution<float> xy_dist(-pnp_xy_abs_bound_,
                                                pnp_xy_abs_bound_);
  std::uniform_real_distribution<float> z_dist(pnp_depth_min_, pnp_depth_max_);

  std::vector<Vector<3>> points_host(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    points_host[i] = Vector<3>{xy_dist(rng), xy_dist(rng), z_dist(rng)};
  }

  SE3Transform true_pose = MakeIdentityPose();
  true_pose[3] = true_pose_translation_[0];
  true_pose[7] = true_pose_translation_[1];
  true_pose[11] = true_pose_translation_[2];

  std::vector<Vector<2>> observations_host(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    observations_host[i] = ProjectPoint(true_pose, points_host[i]);
  }

  SE3Transform initial_pose = MakeIdentityPose();
  initial_pose[3] = initial_pose_translation_[0];
  initial_pose[7] = initial_pose_translation_[1];
  initial_pose[11] = initial_pose_translation_[2];

  dvector<SE3Transform> poses_device(std::vector<SE3Transform>{initial_pose});
  dvector<Vector<3>> points_device(points_host);
  dvector<Vector<2>> observations_device(observations_host);
  auto point_const_ids = test_utils::MakeSequentialIds(num_points);
  dvector<int> point_const_ids_device(point_const_ids);

  SE3StateBatch pose_batch(
      cublas_handle_, reinterpret_cast<const float*>(poses_device.data()), 1);
  VectorStateBatch<3> point_batch(
      reinterpret_cast<const float*>(points_device.data()), num_points,
      point_const_ids_device.data(), point_const_ids.size());
  ReprojectionFactorBatch reprojection_factor(observations_device.data(), num_points);

  std::vector<float*> state_pointers;
  state_pointers.reserve(num_points * 2);
  for (size_t i = 0; i < num_points; ++i) {
    state_pointers.push_back(pose_batch.StateBlockDevicePtr(0));
    state_pointers.push_back(point_batch.StateBlockDevicePtr(i));
  }

  Problem problem;
  problem.AddStateBatch(&pose_batch);
  problem.AddStateBatch(&point_batch);
  problem.AddFactorBatch(&reprojection_factor, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  MinimizerOptions options;
  options.max_num_iterations = 20;
  options.state_tolerance = 1e-10f;
  options.cost_tolerance = 1e-12f;
  options.disable_safety_checks = false;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseQR;
  GaussNewtonMinimizer minimizer(options);

  const auto summary = minimizer.Minimize(stream_.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));

  ASSERT_LT(summary.final_cost, summary.initial_cost);
  ASSERT_LT(summary.final_cost, 1e-5f);

  std::vector<SE3Transform> optimized_pose_host(1);
  poses_device.CopyToHost(optimized_pose_host.data(), 1);
  ASSERT_NEAR(optimized_pose_host[0][3], true_pose[3], 5e-3f);
  ASSERT_NEAR(optimized_pose_host[0][7], true_pose[7], 5e-3f);
  ASSERT_NEAR(optimized_pose_host[0][11], true_pose[11], 5e-3f);
}

}  // namespace
}  // namespace cunls
