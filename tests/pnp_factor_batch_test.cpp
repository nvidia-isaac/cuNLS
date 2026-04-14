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
 * @file pnp_factor_batch_test.cpp
 * @brief Tests for PnPFactorBatch (fixed 3D structure, pose-only Jacobian).
 */

#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"

namespace cunls {
namespace {

using Point3D = Vector<3>;
using Observation2D = Vector<2>;

constexpr uint32_t kSeed = 4242u;

void TransformWorldToCamera(const SE3Transform& pose, const Point3D& x,
                            Point3D& y) {
  for (int i = 0; i < 3; i++) {
    y[i] = pose[i * 4 + 3];
    for (int j = 0; j < 3; j++) {
      y[i] += pose[i * 4 + j] * x[j];
    }
  }
}

void ProjectNormalized(const Point3D& p_cam, Observation2D& obs) {
  float inv_z = 1.0f / p_cam[2];
  obs[0] = p_cam[0] * inv_z;
  obs[1] = p_cam[1] * inv_z;
}

/**
 * @brief Builds one ground-truth world-to-camera pose and N visible world points
 *        with matching normalized observations.
 */
void MakeSinglePosePnPDataset(size_t num_points, SE3Transform& world_to_cam,
                              std::vector<Point3D>& points_world,
                              std::vector<Observation2D>& observations) {
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<float> rot_small(-0.25f, 0.25f);
  std::uniform_real_distribution<float> trans_small(-0.4f, 0.4f);

  hvector<Vector<6>> twist(1);
  twist[0][0] = rot_small(rng);
  twist[0][1] = rot_small(rng);
  twist[0][2] = rot_small(rng);
  twist[0][3] = trans_small(rng);
  twist[0][4] = trans_small(rng);
  twist[0][5] = 8.0f + trans_small(rng);

  CudaStream stream;
  dvector<Vector<6>> twist_d(twist);
  dvector<SE3Transform> pose_d(1);
  ComputeExpSE3(stream.GetStream(),
                reinterpret_cast<const float*>(twist_d.data()), 6, 4, 16, 1,
                reinterpret_cast<float*>(pose_d.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<SE3Transform> pose_h(1);
  pose_d.CopyToHost(pose_h.data(), 1);
  world_to_cam = pose_h[0];

  std::uniform_real_distribution<float> xy(-2.0f, 2.0f);
  points_world.resize(num_points);
  observations.resize(num_points);

  for (size_t i = 0; i < num_points; i++) {
    Point3D p_cam{};
    int tries = 0;
    do {
      Point3D p{};
      p[0] = xy(rng);
      p[1] = xy(rng);
      p[2] = xy(rng);
      TransformWorldToCamera(world_to_cam, p, p_cam);
      points_world[i] = p;
      if (++tries > 500) {
        FAIL() << "Could not sample visible point for PnP synthetic data";
      }
    } while (p_cam[2] < 1.0f);
    ProjectNormalized(p_cam, observations[i]);
  }
}

/** T_dist = exp(delta) * T on the GPU (same pattern as BA reprojection tests). */
void DisturbPoseOnDevice(cuBLASHandle& cublas, SE3Transform* pose_device,
                         float rot_noise, float trans_noise) {
  std::mt19937 rng(kSeed + 7);
  std::uniform_real_distribution<float> r(-rot_noise, rot_noise);
  std::uniform_real_distribution<float> t(-trans_noise, trans_noise);

  hvector<Vector<6>> delta(1);
  delta[0][0] = r(rng);
  delta[0][1] = r(rng);
  delta[0][2] = r(rng);
  delta[0][3] = t(rng);
  delta[0][4] = t(rng);
  delta[0][5] = t(rng);

  CudaStream stream;
  dvector<Vector<6>> delta_d(delta);
  dvector<SE3Transform> exp_d(1);
  ComputeExpSE3(stream.GetStream(),
                reinterpret_cast<const float*>(delta_d.data()), 6, 4, 16, 1,
                reinterpret_cast<float*>(exp_d.data()));

  auto handle =
      static_cast<cublasHandle_t>(cublas.GetHandle(stream.GetStream()));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr size_t mat_size = 4;
  constexpr size_t stride = 16;

  dvector<SE3Transform> out(1);
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(pose_device), mat_size, stride,
      reinterpret_cast<float*>(exp_d.data()), mat_size, stride, &beta,
      reinterpret_cast<float*>(out.data()), mat_size, stride, 1));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(pose_device, out.data(), sizeof(SE3Transform),
                                      cudaMemcpyDeviceToDevice,
                                      stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

float PoseFrobeniusSq(const SE3Transform& a, const SE3Transform& b) {
  float s = 0.0f;
  for (int i = 0; i < 16; i++) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

}  // namespace

/**
 * @brief Registers a single optimized pose with N PnP factors (same pose pointer
 *        for every correspondence). Caller constructs `pnp_batch` and
 *        `pose_state_batch` first; this only wires `Problem`.
 */
void RegisterPnPMinimizationProblem(Problem& problem, PnPFactorBatch& pnp_batch,
                                    SE3StateBatch& pose_state_batch) {
  const size_t n = pnp_batch.NumFactors();
  std::vector<float*> state_ptrs;
  state_ptrs.reserve(n);
  float* pose_block = pose_state_batch.StateBlockDevicePtr(0);
  for (size_t i = 0; i < n; ++i) {
    state_ptrs.push_back(pose_block);
  }
  problem.AddStateBatch(&pose_state_batch);
  problem.AddFactorBatch(&pnp_batch, state_ptrs);
}

class PnPFactorBatchTest : public ::testing::Test {
 protected:
  using Point3D = Vector<3>;
  using Observation2D = Vector<2>;

  /** Full dataset size (LM test); other tests use the first n entries only. */
  static constexpr size_t kMaxCorrespondences = 10000;
  static constexpr float kZThr = 1e-3f;

  void SetUp() override {
    MakeSinglePosePnPDataset(kMaxCorrespondences, gt_pose_, points_host_,
                             obs_host_);
    std::vector<SE3Transform> pose_host = {gt_pose_};
    pose_device_ = dvector<SE3Transform>(pose_host);
    points_device_ = dvector<Point3D>(points_host_);
    obs_device_ = dvector<Observation2D>(obs_host_);
  }

  SE3Transform PoseOnHostFromDevice() const {
    std::vector<SE3Transform> tmp(1);
    pose_device_.CopyToHost(tmp.data(), 1);
    return tmp[0];
  }

  cuBLASHandle cublas_handle_;
  SE3Transform gt_pose_{};
  std::vector<Point3D> points_host_;
  std::vector<Observation2D> obs_host_;
  dvector<SE3Transform> pose_device_;
  dvector<Point3D> points_device_;
  dvector<Observation2D> obs_device_;
};

TEST_F(PnPFactorBatchTest, JacobianMatchesReprojectionPoseBlock) {
  const size_t n = 6;
  PnPFactorBatch pnp(obs_device_.data(), points_device_.data(), n, kZThr);
  ReprojectionFactorBatch reproj(obs_device_.data(), n, kZThr);
  VectorStateBatch<3> point_batch(reinterpret_cast<float*>(points_device_.data()),
                                  n);
  SE3StateBatch pose_state(cublas_handle_,
                           reinterpret_cast<const float*>(pose_device_.data()), 1);

  std::vector<const float*> sp_reproj;
  sp_reproj.reserve(2 * n);
  std::vector<const float*> sp_pnp;
  sp_pnp.reserve(n);
  for (size_t i = 0; i < n; i++) {
    sp_reproj.push_back(
        reinterpret_cast<const float*>(pose_state.StateBlockDevicePtr(0)));
    sp_reproj.push_back(
        reinterpret_cast<const float*>(point_batch.StateBlockDevicePtr(i)));
    sp_pnp.push_back(
        reinterpret_cast<const float*>(pose_state.StateBlockDevicePtr(0)));
  }

  dvector<float> r_pnp(n * 2);
  dvector<float> j_pnp(n * 12);
  dvector<float> j_r(n * 18);
  dvector<const float*> dev_reproj(sp_reproj);
  dvector<const float*> dev_pnp(sp_pnp);

  CudaStream stream;
  ASSERT_TRUE(pnp.Evaluate(r_pnp.data(), j_pnp.data(), dev_pnp.data(),
                           stream.GetStream()));
  ASSERT_TRUE(
      reproj.Evaluate(r_pnp.data(), j_r.data(), dev_reproj.data(),
                      stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> hj_pnp(n * 12);
  std::vector<float> hj_r(n * 18);
  j_pnp.CopyToHost(hj_pnp.data(), hj_pnp.size());
  j_r.CopyToHost(hj_r.data(), hj_r.size());

  for (size_t i = 0; i < n; i++) {
    for (int row = 0; row < 2; row++) {
      for (int c = 0; c < 6; c++) {
        EXPECT_NEAR(hj_pnp[i * 12 + row * 6 + c], hj_r[i * 18 + row * 9 + c],
                    1e-4f);
      }
    }
  }
}

TEST_F(PnPFactorBatchTest, EvaluateNearZeroAtGroundTruth) {
  const size_t n = 20;
  PnPFactorBatch pnp(obs_device_.data(), points_device_.data(), n, kZThr);
  SE3StateBatch pose_state(cublas_handle_,
                           reinterpret_cast<const float*>(pose_device_.data()), 1);

  std::vector<const float*> sp;
  sp.reserve(n);
  for (size_t i = 0; i < n; i++) {
    sp.push_back(
        reinterpret_cast<const float*>(pose_state.StateBlockDevicePtr(0)));
  }
  dvector<const float*> dev_sp(sp);

  dvector<float> residuals(n * 2);
CudaStream stream;
  ASSERT_TRUE(pnp.Evaluate(residuals.data(), nullptr, dev_sp.data(),
                           stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> hr(n * 2);
  residuals.CopyToHost(hr.data(), hr.size());
  float m = 0.0f;
  for (float x : hr) {
    m = std::max(m, std::fabs(x));
  }
  EXPECT_LT(m, 1e-5f);
}

TEST_F(PnPFactorBatchTest, LevenbergMarquardtConverges) {
  const size_t n = kMaxCorrespondences;
  DisturbPoseOnDevice(cublas_handle_, pose_device_.data(), 0.08f, 0.25f);
  EXPECT_GT(PoseFrobeniusSq(PoseOnHostFromDevice(), gt_pose_), 1e-4f);

  PnPFactorBatch pnp(obs_device_.data(), points_device_.data(), n, kZThr);
  SE3StateBatch pose_state(cublas_handle_,
                           reinterpret_cast<const float*>(pose_device_.data()), 1);

  Problem problem;
  RegisterPnPMinimizationProblem(problem, pnp, pose_state);
  ASSERT_TRUE(problem.CheckConsistency());

  CudaStream stream;
  MinimizerOptions base;
  base.max_num_iterations = 100;
  base.state_tolerance = 1e-9f;
  base.cost_tolerance = 1e-9f;
  base.disable_safety_checks = false;
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = base;
  lm.initial_lambda = 1e-2f;
  LevenbergMarquardtMinimizer minimizer(lm);
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-4f);
  EXPECT_GT(summary.num_iterations, 0);
  EXPECT_LT(PoseFrobeniusSq(PoseOnHostFromDevice(), gt_pose_), 5e-3f);
}

// ---------------------------------------------------------------------------
// Parametrized LM convergence test over all solver backends
// ---------------------------------------------------------------------------

class PnPSolverTest
    : public ::testing::TestWithParam<SparseLinearSolverType> {
 protected:
  static constexpr size_t kNumCorrespondences = 10000;
  static constexpr float kZThr = 1e-3f;

  void SetUp() override {
    MakeSinglePosePnPDataset(kNumCorrespondences, gt_pose_, points_host_,
                             obs_host_);
    std::vector<SE3Transform> pose_host = {gt_pose_};
    pose_device_ = dvector<SE3Transform>(pose_host);
    points_device_ = dvector<Point3D>(points_host_);
    obs_device_ = dvector<Observation2D>(obs_host_);
  }

  SE3Transform PoseOnHostFromDevice() const {
    std::vector<SE3Transform> tmp(1);
    pose_device_.CopyToHost(tmp.data(), 1);
    return tmp[0];
  }

  cuBLASHandle cublas_handle_;
  SE3Transform gt_pose_{};
  std::vector<Point3D> points_host_;
  std::vector<Observation2D> obs_host_;
  dvector<SE3Transform> pose_device_;
  dvector<Point3D> points_device_;
  dvector<Observation2D> obs_device_;
};

TEST_P(PnPSolverTest, LevenbergMarquardtConverges) {
  const size_t n = kNumCorrespondences;
  DisturbPoseOnDevice(cublas_handle_, pose_device_.data(), 0.08f, 0.25f);
  EXPECT_GT(PoseFrobeniusSq(PoseOnHostFromDevice(), gt_pose_), 1e-4f);

  PnPFactorBatch pnp(obs_device_.data(), points_device_.data(), n, kZThr);
  SE3StateBatch pose_state(cublas_handle_,
                           reinterpret_cast<const float*>(pose_device_.data()), 1);

  Problem problem_1;
  RegisterPnPMinimizationProblem(problem_1, pnp, pose_state);
  ASSERT_TRUE(problem_1.CheckConsistency());

  Problem problem_2;
  RegisterPnPMinimizationProblem(problem_2, pnp, pose_state);
  ASSERT_TRUE(problem_2.CheckConsistency());

  CudaStream stream;
  MinimizerOptions base;
  base.max_num_iterations = 100;
  base.state_tolerance = 1e-9f;
  base.cost_tolerance = 1e-9f;
  base.sparse_linear_solver_type = GetParam();
  base.disable_safety_checks = false;
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = base;
  lm.initial_lambda = 1e-2f;
  LevenbergMarquardtMinimizer minimizer(lm);
  MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem_1);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  MinimizerSummary summary_2 =
      minimizer.Minimize(stream.GetStream(), problem_2);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_LT(summary.final_cost, 1e-4f);
  EXPECT_GT(summary.num_iterations, 0);
  EXPECT_LT(PoseFrobeniusSq(PoseOnHostFromDevice(), gt_pose_), 5e-3f);
}

INSTANTIATE_TEST_SUITE_P(
    AllSolvers, PnPSolverTest,
    ::testing::Values(SparseLinearSolverType::cuDSS,
                      SparseLinearSolverType::DenseLDLT,
                      SparseLinearSolverType::DenseCholesky,
                      SparseLinearSolverType::DenseQR),
    [](const ::testing::TestParamInfo<SparseLinearSolverType>& info) {
      switch (info.param) {
        case SparseLinearSolverType::cuDSS: return std::string("cuDSS");
        case SparseLinearSolverType::DenseLDLT: return std::string("DenseLDLT");
        case SparseLinearSolverType::DenseCholesky: return std::string("DenseCholesky");
        case SparseLinearSolverType::DenseQR: return std::string("DenseQR");
        default: return std::string("Unknown");
      }
    });

}  // namespace cunls
