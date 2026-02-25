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
 * @file symmetric_point_to_plane_factor_batch_test.cpp
 * @brief Unit tests for SymmetricPointToPlaneFactorBatch.
 *
 * Tests the symmetric point-to-plane factor by:
 * 1. Verifying residuals with identity and known transforms
 * 2. Verifying analytical Jacobians against numerical differentiation
 * 3. Verifying optimization can recover a disturbed SE(3) pose
 */

#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/math/lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/common/profiler.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Test fixture for SymmetricPointToPlaneFactorBatch unit tests.
 *
 * Creates synthetic 3D point correspondences with normals and SE(3) transforms
 * to verify residual evaluation, Jacobian correctness, and optimization
 * convergence.
 *
 * Ground truth setup: Given a transform T_gt, we generate p and q such that
 * T_gt @ p = T_gt^{-1} @ q, which is achieved by setting q = T_gt^2 @ p.
 * This ensures the residual (T@p - T^{-1}@q).(Np+Nq) = 0 at T = T_gt
 * regardless of the normals.
 */
class SymmetricPointToPlaneFactorBatchTest : public ::testing::Test {
 public:
  using Point3D = Vector<3>;

  void SetUp() override {
    std::mt19937 rng(fixed_seed_);
    std::uniform_real_distribution<float> point_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> rotation_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> normal_dist(-1.0f, 1.0f);

    // Generate random source points (p)
    p_points_.resize(num_correspondences_);
    for (size_t i = 0; i < num_correspondences_; i++) {
      p_points_[i][0] = point_dist(rng);
      p_points_[i][1] = point_dist(rng);
      p_points_[i][2] = point_dist(rng);
    }

    // Generate random unit normals (Np and Nq)
    np_normals_.resize(num_correspondences_);
    nq_normals_.resize(num_correspondences_);
    for (size_t i = 0; i < num_correspondences_; i++) {
      // Generate Np
      float nx = normal_dist(rng);
      float ny = normal_dist(rng);
      float nz = normal_dist(rng);
      float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len < 1e-6f) {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
        len = 1.0f;
      }
      np_normals_[i][0] = nx / len;
      np_normals_[i][1] = ny / len;
      np_normals_[i][2] = nz / len;

      // Generate Nq
      nx = normal_dist(rng);
      ny = normal_dist(rng);
      nz = normal_dist(rng);
      len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len < 1e-6f) {
        nx = 1.0f;
        ny = 0.0f;
        nz = 0.0f;
        len = 1.0f;
      }
      nq_normals_[i][0] = nx / len;
      nq_normals_[i][1] = ny / len;
      nq_normals_[i][2] = nz / len;
    }

    // Generate a random SE3 transform via the exponential map
    ground_truth_twist_.resize(1);
    ground_truth_twist_[0][0] = rotation_dist(rng);
    ground_truth_twist_[0][1] = rotation_dist(rng);
    ground_truth_twist_[0][2] = rotation_dist(rng);
    ground_truth_twist_[0][3] = translation_dist(rng);
    ground_truth_twist_[0][4] = translation_dist(rng);
    ground_truth_twist_[0][5] = translation_dist(rng);

    // Compute the SE3 transform from the twist
    CudaStream stream;
    dvector<Vector<6>> twist_device(ground_truth_twist_);
    ground_truth_pose_device_.resize(1);

    ComputeExpSE3(stream.GetStream(),
                  reinterpret_cast<const float*>(twist_device.data()),
                  /*twist_stride=*/6, /*transform_pitch=*/4,
                  /*transform_stride=*/16, /*size=*/1,
                  reinterpret_cast<float*>(ground_truth_pose_device_.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    // Copy pose to host
    ground_truth_pose_.resize(1);
    ground_truth_pose_device_.CopyToHost(ground_truth_pose_.data(), 1);

    // Compute T^2 = T * T for generating q points
    SE3Transform T_squared;
    MatMul4x4(ground_truth_pose_[0], ground_truth_pose_[0], T_squared);

    // Generate q = T^2 @ p so that T@p = T^{-1}@q at ground truth
    q_points_.resize(num_correspondences_);
    for (size_t i = 0; i < num_correspondences_; i++) {
      TransformPoint(T_squared, p_points_[i], q_points_[i]);
    }
  }

 protected:
  /**
   * @brief Transforms a 3D point by an SE(3) matrix on CPU: result = R*point + t.
   */
  void TransformPoint(const SE3Transform& pose, const Point3D& point,
                      Point3D& result) {
    for (int i = 0; i < 3; i++) {
      result[i] = pose[i * 4 + 3];  // translation
      for (int j = 0; j < 3; j++) {
        result[i] += pose[i * 4 + j] * point[j];
      }
    }
  }

  /**
   * @brief Computes the inverse of an SE(3) matrix on CPU.
   *
   * T^{-1} = [R^T, -R^T*t; 0, 1]
   */
  void InvertPose(const SE3Transform& pose, SE3Transform& inv) {
    inv.fill(0.0f);
    // R^T
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        inv[i * 4 + j] = pose[j * 4 + i];
      }
    }
    // -R^T * t
    for (int i = 0; i < 3; i++) {
      inv[i * 4 + 3] = 0.0f;
      for (int j = 0; j < 3; j++) {
        inv[i * 4 + 3] -= inv[i * 4 + j] * pose[j * 4 + 3];
      }
    }
    inv[15] = 1.0f;
  }

  /**
   * @brief Multiplies two 4x4 matrices on CPU: C = A * B.
   */
  void MatMul4x4(const SE3Transform& A, const SE3Transform& B,
                 SE3Transform& C) {
    C.fill(0.0f);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 4; k++) {
          C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
        }
      }
    }
  }

  /**
   * @brief Creates an identity SE3 transform.
   */
  SE3Transform MakeIdentity() {
    SE3Transform identity;
    identity.fill(0.0f);
    identity[0] = 1.0f;
    identity[5] = 1.0f;
    identity[10] = 1.0f;
    identity[15] = 1.0f;
    return identity;
  }

  /**
   * @brief Computes the dot product of two 3D vectors on CPU.
   */
  float Dot3(const Point3D& a, const Point3D& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  /**
   * @brief Computes a perturbed SE3 transform: T_perturbed = T * Exp(delta).
   *
   * Uses the GPU Exp map and CPU matrix multiplication.
   *
   * @param pose The base SE3 transform
   * @param delta The 6D tangent vector perturbation
   * @return The perturbed SE3 transform
   */
  SE3Transform PerturbPose(const SE3Transform& pose, const Vector<6>& delta) {
    CudaStream stream;

    dvector<Vector<6>> delta_device({delta});
    dvector<SE3Transform> exp_delta_device(1);

    // Compute Exp(delta)
    ComputeExpSE3(stream.GetStream(),
                  reinterpret_cast<const float*>(delta_device.data()),
                  /*twist_stride=*/6, /*transform_pitch=*/4,
                  /*transform_stride=*/16, /*size=*/1,
                  reinterpret_cast<float*>(exp_delta_device.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    SE3Transform exp_delta;
    exp_delta_device.CopyToHost(&exp_delta, 1);

    // Compute T * Exp(delta) on CPU
    SE3Transform result;
    MatMul4x4(pose, exp_delta, result);
    return result;
  }

  // Test configuration
  const size_t num_correspondences_ = 1000;
  const uint32_t fixed_seed_ = 42;

  // Data
  std::vector<Point3D> p_points_;                ///< Target points
  std::vector<Point3D> q_points_;                ///< Source points (q = T^2 @ p)
  std::vector<Point3D> np_normals_;              ///< Normal vectors at target points
  std::vector<Point3D> nq_normals_;              ///< Normal vectors at source points
  std::vector<Vector<6>> ground_truth_twist_;    ///< Ground truth twist
  std::vector<SE3Transform> ground_truth_pose_;  ///< Ground truth pose (host)
  dvector<SE3Transform> ground_truth_pose_device_;  ///< Ground truth pose (device)

  profiler::Domain profiler_domain_{
      "SymmetricPointToPlaneFactorBatchTest"};
};

/**
 * @brief Tests that StateBlockSizes() reports the correct sizes.
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, StateBlockSizes) {
  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  auto state_block_sizes = factor_batch.StateBlockSizes();
  ASSERT_EQ(state_block_sizes.size(), 1);
  EXPECT_EQ(state_block_sizes[0], 6);
  EXPECT_EQ(factor_batch.ResidualsSize(), 1);
  EXPECT_EQ(factor_batch.NumFactors(), num_correspondences_);
}

/**
 * @brief Tests residual evaluation with identity transform.
 *
 * When T = I, residual = (p - q) . (Np + Nq) for each correspondence.
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, ResidualIdentity) {
  auto test_range = profiler_domain_.CreateDomainRange("ResidualIdentity");
  CudaStream stream;

  auto identity = MakeIdentity();
  dvector<SE3Transform> pose_device({identity});
  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  std::vector<const float*> param_ptrs(num_correspondences_,
      reinterpret_cast<const float*>(pose_device.data()));
  dvector<const float*> param_ptrs_device(param_ptrs);

  dvector<float> residuals(num_correspondences_);

  factor_batch.Evaluate(residuals.data(), nullptr,
                         param_ptrs_device.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_residuals(num_correspondences_);
  residuals.CopyToHost(host_residuals.data(), host_residuals.size());

  for (size_t i = 0; i < num_correspondences_; i++) {
    // Expected: (p - q) . (Np + Nq)  when T = I
    Point3D diff;
    diff[0] = p_points_[i][0] - q_points_[i][0];
    diff[1] = p_points_[i][1] - q_points_[i][1];
    diff[2] = p_points_[i][2] - q_points_[i][2];

    Point3D N;
    N[0] = np_normals_[i][0] + nq_normals_[i][0];
    N[1] = np_normals_[i][1] + nq_normals_[i][1];
    N[2] = np_normals_[i][2] + nq_normals_[i][2];

    float expected = Dot3(diff, N);
    EXPECT_NEAR(host_residuals[i], expected, 1e-4f)
        << "Mismatch at correspondence " << i;
  }
}

/**
 * @brief Tests residual evaluation with the ground truth transform.
 *
 * Since q = T^2 @ p by construction, T@p = T^{-1}@q and therefore the
 * difference (T@p - T^{-1}@q) = 0. The residual is thus zero regardless
 * of normals.
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, ResidualGroundTruth) {
  auto test_range = profiler_domain_.CreateDomainRange("ResidualGroundTruth");
  CudaStream stream;

  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  std::vector<const float*> param_ptrs(num_correspondences_,
      reinterpret_cast<const float*>(ground_truth_pose_device_.data()));
  dvector<const float*> param_ptrs_device(param_ptrs);

  dvector<float> residuals(num_correspondences_);

  factor_batch.Evaluate(residuals.data(), nullptr,
                         param_ptrs_device.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_residuals(num_correspondences_);
  residuals.CopyToHost(host_residuals.data(), host_residuals.size());

  float max_residual = 0.0f;
  for (float r : host_residuals) {
    max_residual = std::max(max_residual, std::abs(r));
  }
  EXPECT_LT(max_residual, 1e-4f)
      << "Residuals should be near zero for ground truth transform";
}

/**
 * @brief Tests Jacobian evaluation with identity transform.
 *
 * When T = I (R = I, t = 0):
 *   N = Np + Nq, n_R = N, v = q
 *   - d(r)/d(omega) = -N^T * [p]_x - N^T * [q]_x = -N^T * [p+q]_x
 *   - d(r)/d(rho)   = N + N = 2*N
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, JacobianIdentity) {
  auto test_range = profiler_domain_.CreateDomainRange("JacobianIdentity");
  CudaStream stream;

  auto identity = MakeIdentity();
  dvector<SE3Transform> pose_device({identity});
  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  std::vector<const float*> param_ptrs(num_correspondences_,
      reinterpret_cast<const float*>(pose_device.data()));
  dvector<const float*> param_ptrs_device(param_ptrs);

  dvector<float> residuals(num_correspondences_);
  dvector<float> jacobians(num_correspondences_ * 6);

  factor_batch.Evaluate(residuals.data(), jacobians.data(),
                         param_ptrs_device.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_jacobians(num_correspondences_ * 6);
  jacobians.CopyToHost(host_jacobians.data(), host_jacobians.size());

  for (size_t i = 0; i < num_correspondences_; i++) {
    const float* jac = host_jacobians.data() + i * 6;
    const auto& pp = p_points_[i];
    const auto& qq = q_points_[i];

    // Combined normal N = Np + Nq
    float N[3];
    N[0] = np_normals_[i][0] + nq_normals_[i][0];
    N[1] = np_normals_[i][1] + nq_normals_[i][1];
    N[2] = np_normals_[i][2] + nq_normals_[i][2];

    // Sum s = p + q for the skew-symmetric shortcut
    float s[3];
    s[0] = pp[0] + qq[0];
    s[1] = pp[1] + qq[1];
    s[2] = pp[2] + qq[2];

    // With R = I, n_R = N, v = q
    // Rotation part: -N^T * [p+q]_x
    //   col 0: -(N[1]*s[2] - N[2]*s[1])
    //   col 1: -(N[2]*s[0] - N[0]*s[2])
    //   col 2: -(N[0]*s[1] - N[1]*s[0])
    EXPECT_NEAR(jac[0], -(N[1] * s[2] - N[2] * s[1]), 1e-4f)
        << "Rotation Jacobian col 0 at " << i;
    EXPECT_NEAR(jac[1], -(N[2] * s[0] - N[0] * s[2]), 1e-4f)
        << "Rotation Jacobian col 1 at " << i;
    EXPECT_NEAR(jac[2], -(N[0] * s[1] - N[1] * s[0]), 1e-4f)
        << "Rotation Jacobian col 2 at " << i;

    // Translation part: 2*N
    EXPECT_NEAR(jac[3], 2.0f * N[0], 1e-4f)
        << "Translation Jacobian col 0 at " << i;
    EXPECT_NEAR(jac[4], 2.0f * N[1], 1e-4f)
        << "Translation Jacobian col 1 at " << i;
    EXPECT_NEAR(jac[5], 2.0f * N[2], 1e-4f)
        << "Translation Jacobian col 2 at " << i;
  }
}

/**
 * @brief Tests analytical Jacobian against numerical differentiation.
 *
 * Uses central differences to verify the 1x6 Jacobian at a non-trivial
 * SE(3) pose. For each tangent dimension k:
 *   J_numerical[k] = (r(T*Exp(eps*e_k)) - r(T*Exp(-eps*e_k))) / (2*eps)
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, NumericalJacobian) {
  auto test_range = profiler_domain_.CreateDomainRange("NumericalJacobian");
  CudaStream stream;

  // Use a smaller batch for this test
  const size_t num_test_points = 50;

  std::vector<Point3D> test_p(p_points_.begin(),
                              p_points_.begin() + num_test_points);
  std::vector<Point3D> test_q(q_points_.begin(),
                              q_points_.begin() + num_test_points);
  std::vector<Point3D> test_np(np_normals_.begin(),
                               np_normals_.begin() + num_test_points);
  std::vector<Point3D> test_nq(nq_normals_.begin(),
                               nq_normals_.begin() + num_test_points);

  dvector<Point3D> p_device(test_p);
  dvector<Point3D> q_device(test_q);
  dvector<Point3D> np_device(test_np);
  dvector<Point3D> nq_device(test_nq);

  // Get analytical Jacobian at the ground truth pose
  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_test_points);

  std::vector<const float*> param_ptrs(num_test_points,
      reinterpret_cast<const float*>(ground_truth_pose_device_.data()));
  dvector<const float*> param_ptrs_device(param_ptrs);

  dvector<float> residuals(num_test_points);
  dvector<float> jacobians(num_test_points * 6);

  factor_batch.Evaluate(residuals.data(), jacobians.data(),
                         param_ptrs_device.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_jacobians(num_test_points * 6);
  jacobians.CopyToHost(host_jacobians.data(), host_jacobians.size());

  // Compute numerical Jacobian using central differences
  constexpr float eps = 1e-4f;
  constexpr int kTangentDim = 6;

  for (int k = 0; k < kTangentDim; k++) {
    Vector<6> delta_plus, delta_minus;
    delta_plus.fill(0.0f);
    delta_minus.fill(0.0f);
    delta_plus[k] = eps;
    delta_minus[k] = -eps;

    SE3Transform pose_plus = PerturbPose(ground_truth_pose_[0], delta_plus);
    SE3Transform pose_minus = PerturbPose(ground_truth_pose_[0], delta_minus);

    dvector<SE3Transform> pose_plus_device({pose_plus});
    dvector<SE3Transform> pose_minus_device({pose_minus});

    std::vector<const float*> ptrs_plus(num_test_points,
        reinterpret_cast<const float*>(pose_plus_device.data()));
    std::vector<const float*> ptrs_minus(num_test_points,
        reinterpret_cast<const float*>(pose_minus_device.data()));

    dvector<const float*> ptrs_plus_device(ptrs_plus);
    dvector<const float*> ptrs_minus_device(ptrs_minus);

    dvector<float> residuals_plus(num_test_points);
    dvector<float> residuals_minus(num_test_points);

    factor_batch.Evaluate(residuals_plus.data(), nullptr,
                           ptrs_plus_device.data(), stream.GetStream());
    factor_batch.Evaluate(residuals_minus.data(), nullptr,
                           ptrs_minus_device.data(), stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    std::vector<float> host_res_plus(num_test_points);
    std::vector<float> host_res_minus(num_test_points);
    residuals_plus.CopyToHost(host_res_plus.data(), host_res_plus.size());
    residuals_minus.CopyToHost(host_res_minus.data(), host_res_minus.size());

    // Compare numerical Jacobian column k with analytical
    for (size_t i = 0; i < num_test_points; i++) {
      float numerical = (host_res_plus[i] - host_res_minus[i]) / (2.0f * eps);
      float analytical = host_jacobians[i * kTangentDim + k];

      EXPECT_NEAR(analytical, numerical, 1e-2f)
          << "Jacobian mismatch at correspondence " << i
          << ", col " << k
          << " (analytical=" << analytical
          << ", numerical=" << numerical << ")";
    }
  }
}

/**
 * @brief Tests that Evaluate executes without errors when jacobians is nullptr.
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, EvaluateWithoutJacobians) {
  auto test_range =
      profiler_domain_.CreateDomainRange("EvaluateWithoutJacobians");
  CudaStream stream;

  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  std::vector<const float*> param_ptrs(num_correspondences_,
      reinterpret_cast<const float*>(ground_truth_pose_device_.data()));
  dvector<const float*> param_ptrs_device(param_ptrs);

  dvector<float> residuals(num_correspondences_);

  bool result = factor_batch.Evaluate(residuals.data(), nullptr,
                                       param_ptrs_device.data(),
                                       stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_TRUE(result);
}

/**
 * @brief Tests optimization to recover a disturbed SE(3) pose.
 *
 * Given ground truth correspondences where q = T_gt^2 @ p (so that
 * T_gt@p = T_gt^{-1}@q), starts from a disturbed pose T_init and
 * optimizes to minimize sum of ((T@p - T^{-1}@q).(Np+Nq))^2.
 *
 * With enough correspondences and varied normals, the full SE(3) pose
 * should be recoverable.
 */
TEST_F(SymmetricPointToPlaneFactorBatchTest, OptimizeDisturbedPose) {
  auto test_range = profiler_domain_.CreateDomainRange("OptimizeDisturbedPose");
  CudaStream stream;

  // Create a disturbed pose: T_init = T_gt * Exp(small_delta)
  Vector<6> disturbance;
  disturbance[0] = 0.05f;   // rotation x
  disturbance[1] = -0.03f;  // rotation y
  disturbance[2] = 0.04f;   // rotation z
  disturbance[3] = 0.3f;    // translation x
  disturbance[4] = -0.2f;   // translation y
  disturbance[5] = 0.15f;   // translation z

  SE3Transform disturbed_pose = PerturbPose(ground_truth_pose_[0], disturbance);

  // Verify the pose is actually disturbed
  float frobenius_sq = 0.0f;
  for (int i = 0; i < 16; i++) {
    float diff = ground_truth_pose_[0][i] - disturbed_pose[i];
    frobenius_sq += diff * diff;
  }
  ASSERT_GT(frobenius_sq, 0.01f) << "Pose should be significantly disturbed";

  // Copy data to device
  dvector<SE3Transform> pose_device({disturbed_pose});
  dvector<Point3D> p_device(p_points_);
  dvector<Point3D> q_device(q_points_);
  dvector<Point3D> np_device(np_normals_);
  dvector<Point3D> nq_device(nq_normals_);

  // Create state batch
  const float* pose_ptr = reinterpret_cast<const float*>(pose_device.data());
  cuBLASHandle cublas_handle;
  SE3StateBatch state_batch(cublas_handle, pose_ptr, 1);

  // Create factor batch
  SymmetricPointToPlaneFactorBatch factor_batch(
      p_device.data(), q_device.data(), np_device.data(), nq_device.data(),
      num_correspondences_);

  // All correspondences share the same pose (state block 0)
  std::vector<float*> state_pointers(num_correspondences_,
      state_batch.StateBlockDevicePtr(0));

  // Build problem
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&factor_batch, state_pointers);
  ASSERT_TRUE(problem.CheckConsistency());

  // Optimize using Levenberg-Marquardt
  MinimizerOptions options;
  options.max_num_iterations = 100;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e-10f;

  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  MinimizerSummary summary;
  {
    auto minimize_range = profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify optimization converged
  EXPECT_LT(summary.final_cost, 1e-6f) << "Optimization should converge";
  EXPECT_GT(summary.num_iterations, 0) << "Should take at least one iteration";

  // Copy optimized pose back to host
  SE3Transform optimized_pose;
  pose_device.CopyToHost(&optimized_pose, 1);

  // Verify pose recovered to ground truth
  float final_frobenius_sq = 0.0f;
  for (int i = 0; i < 16; i++) {
    float diff = ground_truth_pose_[0][i] - optimized_pose[i];
    final_frobenius_sq += diff * diff;
  }
  EXPECT_LT(final_frobenius_sq, 1e-6f)
      << "Pose should converge to ground truth";
  EXPECT_LT(final_frobenius_sq, frobenius_sq * 0.001f)
      << "Final error should be much smaller than initial";
}

}  // namespace cunls
