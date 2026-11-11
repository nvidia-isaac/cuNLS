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
 * @file registration_factor_batch_test.cpp
 * @brief Unit tests for the point registration factors (PointToPointFactorBatch,
 * PointToPlaneFactorBatch, SymmetricPointToPlaneFactorBatch), each with one
 * SE(3) pose state shared by all correspondences:
 * 1. sizes, and residuals / analytic Jacobians at the identity in closed form;
 * 2. zero residuals at the ground-truth pose;
 * 3. analytic Jacobians against central differences at the ground truth;
 * 4. Levenberg-Marquardt recovering a disturbed pose.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"

namespace cunls {
namespace {

using Point3D = Vector<3>;

SE3Transform ExpSE3(const Vector<6> &twist) {
  CudaStream stream;
  dvector<Vector<6>> twist_device({twist});
  dvector<SE3Transform> pose_device(1);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(twist_device.data()),
                /*twist_stride=*/6, /*transform_pitch=*/4, /*transform_stride=*/16, /*size=*/1,
                reinterpret_cast<float *>(pose_device.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  SE3Transform pose;
  pose_device.CopyToHost(&pose, 1);
  return pose;
}

SE3Transform MatMul4x4(const SE3Transform &a, const SE3Transform &b) {
  SE3Transform c{};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 4; k++) c[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return c;
}

/** pose * Exp(delta). */
SE3Transform PerturbPose(const SE3Transform &pose, const Vector<6> &delta) {
  return MatMul4x4(pose, ExpSE3(delta));
}

/** R * point + t. */
Point3D TransformPoint(const SE3Transform &pose, const Point3D &point) {
  Point3D result;
  for (int i = 0; i < 3; i++) {
    result[i] = pose[i * 4 + 3];
    for (int j = 0; j < 3; j++) result[i] += pose[i * 4 + j] * point[j];
  }
  return result;
}

float Dot3(const Point3D &a, const Point3D &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

float FrobeniusSq(const SE3Transform &a, const SE3Transform &b) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
  return sum;
}

/**
 * Random source points, unit normals and a ground-truth SE(3) pose (the
 * exponential of a random twist), shared by the three factors' tests.
 */
class RegistrationFactorTest : public ::testing::Test {
 protected:
  static constexpr size_t kNumCorrespondences = 1000;
  static constexpr size_t kNumJacobianChecks = 50;

  void SetUp() override {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> point_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> rotation_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> translation_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> normal_dist(-1.0f, 1.0f);
    auto unit_normal = [&] {
      Point3D n{normal_dist(rng), normal_dist(rng), normal_dist(rng)};
      const float len = std::sqrt(Dot3(n, n));
      if (len < 1e-6f) return Point3D{0.0f, 0.0f, 1.0f};
      return Point3D{n[0] / len, n[1] / len, n[2] / len};
    };

    for (size_t i = 0; i < kNumCorrespondences; i++) {
      source_.push_back({point_dist(rng), point_dist(rng), point_dist(rng)});
      normals_a_.push_back(unit_normal());
      normals_b_.push_back(unit_normal());
    }
    gt_pose_ = ExpSE3({rotation_dist(rng), rotation_dist(rng), rotation_dist(rng),
                       translation_dist(rng), translation_dist(rng), translation_dist(rng)});
    for (const Point3D &q : source_) transformed_.push_back(TransformPoint(gt_pose_, q));
  }

  struct Evaluation {
    std::vector<float> residuals;
    std::vector<float> jacobians;  ///< empty unless requested
  };

  /** Evaluates all active factors with every one reading `pose`; Evaluate must succeed. */
  static Evaluation EvaluateAt(const FactorBatch &factor, const SE3Transform &pose,
                               bool with_jacobians) {
    const size_t n = factor.NumActiveFactors();
    const size_t m = factor.ResidualsSize();
    CudaStream stream;
    dvector<SE3Transform> pose_device({pose});
    dvector<const float *> ptrs(
        std::vector<const float *>(n, reinterpret_cast<const float *>(pose_device.data())));
    dvector<float> residuals(n * m), jacobians(with_jacobians ? n * m * 6 : 0);
    EXPECT_TRUE(factor.Evaluate(residuals.data(), with_jacobians ? jacobians.data() : nullptr,
                                ptrs.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    Evaluation e{std::vector<float>(n * m), std::vector<float>(jacobians.size())};
    residuals.CopyToHost(e.residuals.data(), e.residuals.size());
    if (with_jacobians) jacobians.CopyToHost(e.jacobians.data(), e.jacobians.size());
    return e;
  }

  static void ExpectSizes(const FactorBatch &factor, size_t residual_size) {
    ASSERT_EQ(factor.StateSizes(), std::vector<size_t>{6});
    EXPECT_EQ(factor.ResidualsSize(), residual_size);
    EXPECT_EQ(factor.NumActiveFactors(), kNumCorrespondences);
  }

  void ExpectZeroResidualsAtGroundTruth(const FactorBatch &factor, float tolerance) const {
    const auto e = EvaluateAt(factor, gt_pose_, false);
    float max_residual = 0.0f;
    for (float r : e.residuals) max_residual = std::max(max_residual, std::abs(r));
    EXPECT_LT(max_residual, tolerance)
        << "Residuals should be near zero for ground truth transform";
  }

  /**
   * Compares the analytic Jacobians of the first kNumJacobianChecks factors at
   * the ground-truth pose with central differences r(T Exp(±eps e_k)).
   */
  void ExpectJacobiansMatchCentralDifferences(FactorBatch &factor) const {
    factor.SetNumActiveFactors(kNumJacobianChecks);
    const size_t m = factor.ResidualsSize();
    const auto base = EvaluateAt(factor, gt_pose_, true);
    constexpr float eps = 1e-4f;
    for (int k = 0; k < 6; k++) {
      Vector<6> delta{};
      delta[k] = eps;
      const auto plus = EvaluateAt(factor, PerturbPose(gt_pose_, delta), false);
      delta[k] = -eps;
      const auto minus = EvaluateAt(factor, PerturbPose(gt_pose_, delta), false);
      for (size_t i = 0; i < kNumJacobianChecks * m; i++) {
        const float numerical = (plus.residuals[i] - minus.residuals[i]) / (2.0f * eps);
        EXPECT_NEAR(base.jacobians[i * 6 + k], numerical, 1e-2f)
            << "correspondence " << i / m << ", row " << i % m << ", col " << k;
      }
    }
  }

  /** LM from gt_pose_ * Exp(disturbance) must return to gt_pose_. */
  void ExpectRecoversDisturbedPose(FactorBatch &factor) const {
    const SE3Transform disturbed_pose =
        PerturbPose(gt_pose_, {0.05f, -0.03f, 0.04f, 0.3f, -0.2f, 0.15f});
    const float initial_error = FrobeniusSq(gt_pose_, disturbed_pose);
    ASSERT_GT(initial_error, 0.01f) << "Pose should be significantly disturbed";

    dvector<SE3Transform> pose_device({disturbed_pose});
    SE3StateBatch state_batch(reinterpret_cast<const float *>(pose_device.data()), 1);
    state_batch.SetNumActiveStates(state_batch.Capacity(), state_batch.ConstCapacity());
    Problem problem;
    problem.AddStateBatch(&state_batch);
    problem.AddFactorBatch(
        &factor, std::vector<float *>(factor.NumActiveFactors(), state_batch.StateDevicePtr(0)));
    ASSERT_TRUE(problem.CheckConsistency());

    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options.max_num_iterations = 100;
    lm_options.base_options.state_tolerance = 1e-8f;
    lm_options.base_options.cost_tolerance = 1e-10f;
    lm_options.base_options.disable_safety_checks = false;
    lm_options.initial_lambda = 1e-3f;
    LevenbergMarquardtMinimizer minimizer(lm_options);
    CudaStream stream;
    const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    EXPECT_LT(summary.final_cost, 1e-6f) << "Optimization should converge";
    EXPECT_GT(summary.num_iterations, 0) << "Should take at least one iteration";
    SE3Transform optimized_pose;
    pose_device.CopyToHost(&optimized_pose, 1);
    const float final_error = FrobeniusSq(gt_pose_, optimized_pose);
    EXPECT_LT(final_error, 1e-6f) << "Pose should converge to ground truth";
    EXPECT_LT(final_error, initial_error * 0.001f) << "Final error should be much smaller";
  }

  std::vector<Point3D> source_;       ///< random points
  std::vector<Point3D> transformed_;  ///< gt_pose_ * source_
  std::vector<Point3D> normals_a_, normals_b_;
  SE3Transform gt_pose_;
  const SE3Transform identity_ = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

// ============================================================================
// PointToPointFactorBatch: r = p - T q, with p = T_gt q.
// ============================================================================

class PointToPointFactorBatchTest : public RegistrationFactorTest {
 protected:
  void SetUp() override {
    RegistrationFactorTest::SetUp();
    p_ = dvector<Point3D>(transformed_);
    q_ = dvector<Point3D>(source_);
    factor_ = std::make_unique<PointToPointFactorBatch>(p_.data(), q_.data(), kNumCorrespondences);
    factor_->SetNumActiveFactors(kNumCorrespondences);
  }
  dvector<Point3D> p_, q_;
  std::unique_ptr<PointToPointFactorBatch> factor_;
};

// At T = I: r = p - q and J = [[q]_x | -I].
TEST_F(PointToPointFactorBatchTest, SizesAndIdentityResidualsAndJacobians) {
  ExpectSizes(*factor_, 3);
  const auto e = EvaluateAt(*factor_, identity_, true);
  for (size_t i = 0; i < kNumCorrespondences; i++) {
    const Point3D &p = transformed_[i], &q = source_[i];
    const float skew_q[9] = {0, -q[2], q[1], q[2], 0, -q[0], -q[1], q[0], 0};
    for (int row = 0; row < 3; row++) {
      EXPECT_NEAR(e.residuals[3 * i + row], p[row] - q[row], 1e-5f) << i << " row " << row;
      for (int col = 0; col < 6; col++) {
        const float expected = col < 3 ? skew_q[row * 3 + col] : (col - 3 == row ? -1.0f : 0.0f);
        EXPECT_NEAR(e.jacobians[(3 * i + row) * 6 + col], expected, 1e-5f)
            << i << " row " << row << " col " << col;
      }
    }
  }
}

TEST_F(PointToPointFactorBatchTest, ResidualGroundTruth) {
  ExpectZeroResidualsAtGroundTruth(*factor_, 1e-5f);
}

TEST_F(PointToPointFactorBatchTest, NumericalJacobian) {
  ExpectJacobiansMatchCentralDifferences(*factor_);
}

TEST_F(PointToPointFactorBatchTest, OptimizeDisturbedPose) {
  ExpectRecoversDisturbedPose(*factor_);
}

// ============================================================================
// PointToPlaneFactorBatch: r = Nq . (p - T q), with p = T_gt q.
// ============================================================================

class PointToPlaneFactorBatchTest : public RegistrationFactorTest {
 protected:
  void SetUp() override {
    RegistrationFactorTest::SetUp();
    p_ = dvector<Point3D>(transformed_);
    q_ = dvector<Point3D>(source_);
    nq_ = dvector<Point3D>(normals_a_);
    factor_ = std::make_unique<PointToPlaneFactorBatch>(p_.data(), q_.data(), nq_.data(),
                                                        kNumCorrespondences);
    factor_->SetNumActiveFactors(kNumCorrespondences);
  }
  dvector<Point3D> p_, q_, nq_;
  std::unique_ptr<PointToPlaneFactorBatch> factor_;
};

// At T = I: r = Nq . (p - q) and J = [Nq^T [q]_x | -Nq^T].
TEST_F(PointToPlaneFactorBatchTest, SizesAndIdentityResidualsAndJacobians) {
  ExpectSizes(*factor_, 1);
  const auto e = EvaluateAt(*factor_, identity_, true);
  for (size_t i = 0; i < kNumCorrespondences; i++) {
    const Point3D &p = transformed_[i], &q = source_[i], &n = normals_a_[i];
    const Point3D diff{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    EXPECT_NEAR(e.residuals[i], Dot3(n, diff), 1e-5f) << "correspondence " << i;
    const float expected[6] = {n[1] * q[2] - n[2] * q[1],
                               n[2] * q[0] - n[0] * q[2],
                               n[0] * q[1] - n[1] * q[0],
                               -n[0],
                               -n[1],
                               -n[2]};
    for (int col = 0; col < 6; col++) {
      EXPECT_NEAR(e.jacobians[i * 6 + col], expected[col], 1e-5f) << i << " col " << col;
    }
  }
}

TEST_F(PointToPlaneFactorBatchTest, ResidualGroundTruth) {
  ExpectZeroResidualsAtGroundTruth(*factor_, 1e-5f);
}

TEST_F(PointToPlaneFactorBatchTest, NumericalJacobian) {
  ExpectJacobiansMatchCentralDifferences(*factor_);
}

// Random normals give 6 DOF of constraints, so the full pose is recoverable.
TEST_F(PointToPlaneFactorBatchTest, OptimizeDisturbedPose) {
  ExpectRecoversDisturbedPose(*factor_);
}

// ============================================================================
// SymmetricPointToPlaneFactorBatch: r = (T p - T^{-1} q) . (Np + Nq), with
// q = T_gt^2 p so that T_gt p = T_gt^{-1} q and r = 0 at T_gt for any normals.
// ============================================================================

class SymmetricPointToPlaneFactorBatchTest : public RegistrationFactorTest {
 protected:
  void SetUp() override {
    RegistrationFactorTest::SetUp();
    const SE3Transform gt_squared = MatMul4x4(gt_pose_, gt_pose_);
    for (const Point3D &p : source_) q_host_.push_back(TransformPoint(gt_squared, p));
    p_ = dvector<Point3D>(source_);
    q_ = dvector<Point3D>(q_host_);
    np_ = dvector<Point3D>(normals_a_);
    nq_ = dvector<Point3D>(normals_b_);
    factor_ = std::make_unique<SymmetricPointToPlaneFactorBatch>(p_.data(), q_.data(), np_.data(),
                                                                 nq_.data(), kNumCorrespondences);
    factor_->SetNumActiveFactors(kNumCorrespondences);
  }
  std::vector<Point3D> q_host_;
  dvector<Point3D> p_, q_, np_, nq_;
  std::unique_ptr<SymmetricPointToPlaneFactorBatch> factor_;
};

// At T = I, with N = Np + Nq and s = p + q: r = (p - q) . N and
// J = [-N^T [s]_x | 2 N^T].
TEST_F(SymmetricPointToPlaneFactorBatchTest, SizesAndIdentityResidualsAndJacobians) {
  ExpectSizes(*factor_, 1);
  const auto e = EvaluateAt(*factor_, identity_, true);
  for (size_t i = 0; i < kNumCorrespondences; i++) {
    const Point3D &p = source_[i], &q = q_host_[i];
    const Point3D N{normals_a_[i][0] + normals_b_[i][0], normals_a_[i][1] + normals_b_[i][1],
                    normals_a_[i][2] + normals_b_[i][2]};
    const Point3D s{p[0] + q[0], p[1] + q[1], p[2] + q[2]};
    const Point3D diff{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    EXPECT_NEAR(e.residuals[i], Dot3(diff, N), 1e-4f) << "correspondence " << i;
    const float expected[6] = {-(N[1] * s[2] - N[2] * s[1]),
                               -(N[2] * s[0] - N[0] * s[2]),
                               -(N[0] * s[1] - N[1] * s[0]),
                               2.0f * N[0],
                               2.0f * N[1],
                               2.0f * N[2]};
    for (int col = 0; col < 6; col++) {
      EXPECT_NEAR(e.jacobians[i * 6 + col], expected[col], 1e-4f) << i << " col " << col;
    }
  }
}

TEST_F(SymmetricPointToPlaneFactorBatchTest, ResidualGroundTruth) {
  ExpectZeroResidualsAtGroundTruth(*factor_, 1e-4f);
}

TEST_F(SymmetricPointToPlaneFactorBatchTest, NumericalJacobian) {
  ExpectJacobiansMatchCentralDifferences(*factor_);
}

TEST_F(SymmetricPointToPlaneFactorBatchTest, OptimizeDisturbedPose) {
  ExpectRecoversDisturbedPose(*factor_);
}

}  // namespace
}  // namespace cunls
