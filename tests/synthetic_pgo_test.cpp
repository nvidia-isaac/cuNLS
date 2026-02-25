/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @file synthetic_pgo_test.cpp
 * @brief Integration tests for synthetic pose graph optimization (PGO).
 *
 * Constructs two sets of random SE3 poses with between-pose constraints,
 * optimizes using Levenberg-Marquardt, and verifies that the relative
 * transforms converge to the expected deltas (identity or given offset).
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/common/helper.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/math/lie_math.h"
#include "cunls/minimizer/problem.h"
#include "cunls/common/profiler.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/common/types.h"
#include "cunls/common/device_vector.h"

namespace cunls {

/**
 * @brief Test fixture for synthetic pose graph optimization test.
 *
 * Generates two sets of SE3 poses and creates between constraints between
 * consecutive pairs. After optimization, verifies that consecutive poses
 * are equal (relative transform is identity).
 */
class SyntheticPGOTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> rotation_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);

    // Generate poses for both sets
    GenerateRandomPoses(num_poses_, rng, rotation_dist, translation_dist,
                        poses_set1_);
    GenerateRandomPoses(num_poses_, rng, rotation_dist, translation_dist,
                        poses_set2_);
    GenerateRandomPoses(num_poses_, rng, rotation_dist, translation_dist,
                        pose_deltas_);
  }

 protected:
  /**
   * @brief Generates a random set of SE3 poses.
   *
   * Creates random SE3 transformation matrices by generating random twists
   * (rotation + translation) and converting them to SE3 transforms using the
   * exponential map.
   *
   * @param num_poses Number of poses to generate
   * @param rng Random number generator
   * @param rotation_dist Distribution for rotation components (twist[0:2])
   * @param translation_dist Distribution for translation components
   * (twist[3:5])
   * @return Device vector containing the generated SE3 transforms
   */
  void GenerateRandomPoses(
      size_t num_poses, std::mt19937& rng,
      std::uniform_real_distribution<float>& rotation_dist,
      std::uniform_real_distribution<float>& translation_dist,
      std::vector<SE3Transform>& poses) {
    // Generate random twists
    hvector<Vector<6>> twists(num_poses);
    for (size_t i = 0; i < num_poses; i++) {
      Vector<6>& twist = twists[i];
      twist[0] = rotation_dist(rng);
      twist[1] = rotation_dist(rng);
      twist[2] = rotation_dist(rng);
      twist[3] = translation_dist(rng);
      twist[4] = translation_dist(rng);
      twist[5] = translation_dist(rng);
    }

    // Convert twists to SE3 transforms
    CudaStream stream;
    constexpr size_t twist_stride = 6;
    constexpr size_t transform_pitch = 4;
    constexpr size_t transform_stride = 16;

    dvector<Vector<6>> twists_device(twists);
    dvector<SE3Transform> poses_device(num_poses);

    auto twists_ptr = reinterpret_cast<const float*>(twists_device.data());

    auto poses_ptr = reinterpret_cast<float*>(poses_device.data());
    ComputeExpSE3(stream.GetStream(), twists_ptr, twist_stride, transform_pitch,
           transform_stride, num_poses, poses_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    poses.resize(num_poses);
    hvector<SE3Transform> poses_host(num_poses);
    poses_device.CopyToHost(poses_host.data(), num_poses);
    for (size_t i = 0; i < num_poses; i++) {
      poses[i] = poses_host[i];
    }
  }

  const size_t num_poses_ = 10000;
  const uint32_t fixed_seed_ = 42;
  std::vector<SE3Transform> poses_set1_;
  std::vector<SE3Transform> poses_set2_;
  std::vector<SE3Transform> pose_deltas_;

  cuBLASHandle cublas_handle_;  ///< cuBLAS handle for factor constructors

  profiler::Domain profiler_domain_{"SyntheticPGOTest"};
};

/**
 * @brief Tests pose graph optimization with consecutive between constraints.
 *
 * Creates two sets of SE3 poses, adds between constraints between consecutive
 * pairs (pose[i] and pose[i+1]) for each set, optimizes with Gauss-Newton,
 * and verifies that consecutive poses are equal (relative transform is
 * identity).
 */
TEST_F(SyntheticPGOTest, OptimizeConsecutiveBetweenConstraints) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "OptimizeConsecutiveBetweenConstraints");

  // Create device vectors from host vectors
  dvector<SE3Transform> poses_set1_device(this->poses_set1_);
  dvector<SE3Transform> poses_set2_device(this->poses_set2_);

  // Create state blocks for both sets
  const float* poses_set1_ptr =
      reinterpret_cast<const float*>(poses_set1_device.data());
  const float* poses_set2_ptr =
      reinterpret_cast<const float*>(poses_set2_device.data());
  SE3StateBatch state_batch_set1(this->cublas_handle_, poses_set1_ptr, this->num_poses_);
  SE3StateBatch state_batch_set2(this->cublas_handle_, poses_set2_ptr, this->num_poses_);

  // Create between constraints for consecutive pairs in set 1
  // For N poses, we have N-1 consecutive constraints
  size_t num_constraints = this->num_poses_;
  dvector<SE3Transform> pose_deltas_device(this->pose_deltas_);
  SE3BetweenFactorBatch between_factor_batch(this->cublas_handle_, pose_deltas_device.data(),
                                                     num_constraints);

  // Create state pointers for set 1 constraints
  // Each constraint connects pose[i] from set1 (left) and pose[i] from set2 (right)
  std::vector<float*> state_pointers;
  for (size_t i = 0; i < num_constraints; i++) {
    state_pointers.push_back(
        state_batch_set1.StateBlockDevicePtr(i));  // left
    state_pointers.push_back(
        state_batch_set2.StateBlockDevicePtr(i));  // right
  }

  // Build problem
  Problem problem;
  problem.AddStateBatch(&state_batch_set1);
  problem.AddStateBatch(&state_batch_set2);

  problem.AddFactorBatch(&between_factor_batch, state_pointers);

  ASSERT_TRUE(problem.CheckConsistency());

  // Optimize with Levenberg-Marquardt
  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-6f;
  options.cost_tolerance = 1e-6f;
  // GaussNewtonMinimizer minimizer(options);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary;
  {
    auto minimize_range = this->profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // // Verify optimization converged
  ASSERT_LT(summary.final_cost, 1e-2f);
  ASSERT_GT(summary.num_iterations, 0);

  // Get optimized poses from state blocks
  const float* opt_poses_set1_ptr = state_batch_set1.StateBlockDevicePtr(0);
  const float* opt_poses_set2_ptr = state_batch_set2.StateBlockDevicePtr(0);

  // Compute inverse of poses_set1
  dvector<SE3Transform> poses_set1_inverse(this->num_poses_);
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  auto poses_set1_inv_ptr = reinterpret_cast<float*>(poses_set1_inverse.data());
  ComputeInverseSE3(stream.GetStream(), opt_poses_set1_ptr, transform_pitch,
             transform_stride, transform_pitch, transform_stride,
             this->num_poses_, poses_set1_inv_ptr);

  // Use cuBLAS for batch matrix multiplication
  cuBLASHandle cublas_handle;
  auto handle = cublas_handle.GetHandle(stream.GetStream());
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr size_t mat_size = 4;

  dvector<SE3Transform> results(this->num_poses_);
  float* results_ptr = reinterpret_cast<float*>(results.data());

  // Multiply: relative = poses_set1_inv * poses_set2
  // Note: cuBLAS uses column-major, but our matrices are row-major
  // CUBLAS_OP_N for both means we compute the equivalent of row-major
  // multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      opt_poses_set2_ptr, mat_size, transform_stride, poses_set1_inv_ptr, mat_size,
      transform_stride, &beta, results_ptr, mat_size, transform_stride,
      this->num_poses_));

  dvector<SE3Transform> deltas(pose_deltas_);
  float* deltas_ptr = reinterpret_cast<float*>(deltas.data());

  // Compute results = delta @ poses_set1_inv @ poses_set2
  // Note: cuBLAS uses column-major, but our matrices are row-major
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      results_ptr, mat_size, transform_stride, deltas_ptr, mat_size,
      transform_stride, &beta, results_ptr, mat_size, transform_stride,
      this->num_poses_));

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Copy results transforms to host for verification
  hvector<SE3Transform> results_host(this->num_poses_);
  results.CopyToHost(results_host.data(), this->num_poses_);

  // Verify that relative transforms are identity matrices
  constexpr float tolerance = 1e-2f;
  for (size_t i = 0; i < this->num_poses_; i++) {
    const SE3Transform& rel = results_host[i];

    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        ASSERT_NEAR(rel[row * 4 + col], (row == col) ? 1.0f : 0.0f, tolerance);
      }
    }
  }
}

/**
 * @brief Tests pose graph optimization with InformationFactorBatch
 * wrapper.
 *
 * Creates two sets of SE3 poses, adds between constraints wrapped with
 * InformationFactorBatch, optimizes with Levenberg-Marquardt,
 * and verifies that the optimized poses satisfy the constraints (relative
 * transform matches the expected delta).
 */
TEST_F(SyntheticPGOTest, InformationBetweenFactorBatch) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "OptimizeConsecutiveBetweenConstraints");

  // Create device vectors from host vectors
  dvector<SE3Transform> poses_set1_device(this->poses_set1_);
  dvector<SE3Transform> poses_set2_device(this->poses_set2_);

  // Create state blocks for both sets
  const float* poses_set1_ptr =
      reinterpret_cast<const float*>(poses_set1_device.data());
  const float* poses_set2_ptr =
      reinterpret_cast<const float*>(poses_set2_device.data());
  SE3StateBatch state_batch_set1(this->cublas_handle_, poses_set1_ptr, this->num_poses_);
  SE3StateBatch state_batch_set2(this->cublas_handle_, poses_set2_ptr, this->num_poses_);

  // Create between constraints for consecutive pairs in set 1
  // For N poses, we have N-1 consecutive constraints
  size_t num_constraints = this->num_poses_;

  std::vector<Matrix<6>> sqrt_information_matrices_host(num_constraints);
  for (size_t i = 0; i < num_constraints; i++) {
    sqrt_information_matrices_host[i].fill(0.0f);
    for (int j = 0; j < 6; j++) {
      sqrt_information_matrices_host[i][j * 6 + j] = 1.0f;
    }
  }
  dvector<Matrix<6>> sqrt_information_matrices_device(sqrt_information_matrices_host);
  dvector<SE3Transform> pose_deltas_device(this->pose_deltas_);
  InformationFactorBatch<SE3BetweenFactorBatch>
      between_factor_batch(this->cublas_handle_, sqrt_information_matrices_device.data(),
                            num_constraints,
                            pose_deltas_device.data(), num_constraints);

  // Create state pointers for set 1 constraints
  // Each constraint connects pose[i] from set1 (left) and pose[i] from set2
  // (right)
  std::vector<float*> state_pointers;
  for (size_t i = 0; i < num_constraints; i++) {
    state_pointers.push_back(
        state_batch_set1.StateBlockDevicePtr(i));  // left
    state_pointers.push_back(
        state_batch_set2.StateBlockDevicePtr(i));  // right
  }

  // Build problem
  Problem problem;
  problem.AddStateBatch(&state_batch_set1);
  problem.AddStateBatch(&state_batch_set2);

  problem.AddFactorBatch(&between_factor_batch, state_pointers);

  ASSERT_TRUE(problem.CheckConsistency());

  // Optimize with Levenberg-Marquardt
  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 50;
  options.state_tolerance = 1e-6f;
  options.cost_tolerance = 1e-6f;
  // GaussNewtonMinimizer minimizer(options);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary;
  {
    auto minimize_range = this->profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // // Verify optimization converged
  ASSERT_LT(summary.final_cost, 1e-2f);
  ASSERT_GT(summary.num_iterations, 0);

  // Get optimized poses from state blocks
  const float* opt_poses_set1_ptr = state_batch_set1.StateBlockDevicePtr(0);
  const float* opt_poses_set2_ptr = state_batch_set2.StateBlockDevicePtr(0);

  // Compute inverse of poses_set1
  dvector<SE3Transform> poses_set1_inverse(this->num_poses_);
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  auto poses_set1_inv_ptr = reinterpret_cast<float*>(poses_set1_inverse.data());
  ComputeInverseSE3(stream.GetStream(), opt_poses_set1_ptr, transform_pitch,
                    transform_stride, transform_pitch, transform_stride,
                    this->num_poses_, poses_set1_inv_ptr);

  // Use cuBLAS for batch matrix multiplication
  cuBLASHandle cublas_handle;
  auto handle = cublas_handle.GetHandle(stream.GetStream());
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr size_t mat_size = 4;

  dvector<SE3Transform> results(this->num_poses_);
  float* results_ptr = reinterpret_cast<float*>(results.data());

  // Multiply: relative = poses_set1_inv * poses_set2
  // Note: cuBLAS uses column-major, but our matrices are row-major
  // CUBLAS_OP_N for both means we compute the equivalent of row-major
  // multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      opt_poses_set2_ptr, mat_size, transform_stride, poses_set1_inv_ptr, mat_size,
      transform_stride, &beta, results_ptr, mat_size, transform_stride,
      this->num_poses_));

  dvector<SE3Transform> deltas(pose_deltas_);
  float* deltas_ptr = reinterpret_cast<float*>(deltas.data());

  // Compute results = delta @ poses_set1_inv @ poses_set2
  // Note: cuBLAS uses column-major, but our matrices are row-major
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      results_ptr, mat_size, transform_stride, deltas_ptr, mat_size,
      transform_stride, &beta, results_ptr, mat_size, transform_stride,
      this->num_poses_));

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Copy results transforms to host for verification
  hvector<SE3Transform> results_host(this->num_poses_);
  results.CopyToHost(results_host.data(), this->num_poses_);

  // Verify that relative transforms are identity matrices
  constexpr float tolerance = 1e-2f;
  for (size_t i = 0; i < this->num_poses_; i++) {
    const SE3Transform& rel = results_host[i];

    for (int row = 0; row < 4; row++) {
      for (int col = 0; col < 4; col++) {
        ASSERT_NEAR(rel[row * 4 + col], (row == col) ? 1.0f : 0.0f, tolerance);
      }
    }
  }
}

}  // namespace cunls
