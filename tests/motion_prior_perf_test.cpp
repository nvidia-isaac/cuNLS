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
 * @file motion_prior_perf_test.cpp
 * @brief Large-scale (PGO-comparable) performance test for the
 * constant-velocity SE(3) motion-prior factor and its covariance-weighted
 * (MotionPriorInformationFactorBatch) variant.
 *
 * Builds a single long pose+velocity chain (up to tens of thousands of
 * poses, i.e. as many CV factors as SyntheticPGOTest's largest loop-closure
 * case has between factors) and times/profiles Problem construction and
 * LevenbergMarquardtMinimizer::Minimize with NVTX ranges, so the run can be
 * inspected under `nsys profile` (build with -DENABLE_PROFILING=ON).
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/motion_prior_information.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

struct MotionPriorPerfParams {
  int n_poses;
  const char *label;
};

inline std::ostream &operator<<(std::ostream &os, const MotionPriorPerfParams &p) {
  return os << p.label;
}

/**
 * @brief PGO/SBA-scale performance test fixture for the constant-velocity
 * SE(3) motion prior, with and without the fused Q(dt)^-1 covariance
 * weighting.
 *
 * The chain is generated exactly as in examples/motion_prior/main.cpp
 * (integrate forward from a random anchor pose/velocity so the ground truth
 * has zero residual), scaled up to tens of thousands of poses.
 */
class MotionPriorPerfTest : public ::testing::TestWithParam<MotionPriorPerfParams> {
 protected:
  SE3Transform ComposeSE3(const SE3Transform &a, const SE3Transform &b) {
    SE3Transform c{};
    for (int r = 0; r < 4; ++r) {
      for (int cix = 0; cix < 4; ++cix) {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += a[r * 4 + k] * b[k * 4 + cix];
        c[r * 4 + cix] = s;
      }
    }
    return c;
  }

  // Builds a chain that approximately satisfies the constant-velocity
  // residual (exact ground truth would require transporting the velocity
  // through the per-step left Jacobian sequentially, which is too slow to
  // set up at 10k-50k poses); a single constant body velocity with small
  // per-step twists is close enough to give the minimizer a well-posed,
  // non-trivial problem without O(num_poses) small kernel launches.
  void GenerateChain(int num_poses, float dt, std::vector<SE3Transform> &gt_poses,
                     std::vector<Vector<6>> &gt_vels, std::vector<SE3Transform> &init_poses,
                     std::vector<Vector<6>> &init_vels) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> rot(-0.15f, 0.15f);
    std::uniform_real_distribution<float> trans(-0.5f, 0.5f);
    std::uniform_real_distribution<float> pose_noise(-0.05f, 0.05f);
    std::uniform_real_distribution<float> vel_noise(-0.1f, 0.1f);

    const Vector<6> anchor_vel{rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
    Vector<6> step_twist{};
    for (int d = 0; d < 6; ++d) step_twist[d] = dt * anchor_vel[d];

    dvector<Vector<6>> step_twist_dev({step_twist});
    dvector<SE3Transform> step_pose_dev(1);
    ComputeExpSE3(stream_.GetStream(), reinterpret_cast<const float *>(step_twist_dev.data()), 6, 4,
                  16, 1, reinterpret_cast<float *>(step_pose_dev.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));
    SE3Transform step_pose;
    step_pose_dev.CopyToHost(&step_pose, 1);

    gt_poses.assign(num_poses, SE3Transform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    gt_vels.assign(num_poses, anchor_vel);
    for (int i = 0; i + 1 < num_poses; ++i) {
      gt_poses[i + 1] = ComposeSE3(gt_poses[i], step_pose);
    }

    std::vector<Vector<6>> disturbance_twists(num_poses - 1);
    for (auto &t : disturbance_twists) {
      t = {rot(rng) * 0.5f, rot(rng) * 0.5f, rot(rng) * 0.5f,
           pose_noise(rng), pose_noise(rng), pose_noise(rng)};
    }
    dvector<Vector<6>> disturbance_dev(disturbance_twists);
    dvector<SE3Transform> disturbance_pose_dev(num_poses - 1);
    ComputeExpSE3(stream_.GetStream(), reinterpret_cast<const float *>(disturbance_dev.data()), 6,
                  4, 16, num_poses - 1, reinterpret_cast<float *>(disturbance_pose_dev.data()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));
    std::vector<SE3Transform> disturbance_poses(num_poses - 1);
    disturbance_pose_dev.CopyToHost(disturbance_poses.data(), num_poses - 1);

    init_poses = gt_poses;
    init_vels = gt_vels;
    for (int i = 1; i < num_poses; ++i) {
      init_poses[i] = ComposeSE3(disturbance_poses[i - 1], gt_poses[i]);
      for (int d = 0; d < 6; ++d) init_vels[i][d] = gt_vels[i][d] + vel_noise(rng);
    }
  }

  CudaStream stream_;
  cuBLASHandle cublas_handle_;
  profiler::Domain profiler_domain_ = profiler::Domain("MotionPriorPerfTest");
};

// Unweighted ConstantVelocitySE3FactorBatch at PGO/SBA scale.
TEST_P(MotionPriorPerfTest, UnweightedConverges) {
  auto p = GetParam();
  SCOPED_TRACE(std::string("MotionPriorPerfTest.UnweightedConverges: ") + p.label);

  const float dt = 0.1f;
  const int num_factors = p.n_poses - 1;

  std::vector<SE3Transform> gt_poses, init_poses;
  std::vector<Vector<6>> gt_vels, init_vels;
  {
    auto range = profiler_domain_.CreateDomainRange("GenerateChain");
    GenerateChain(p.n_poses, dt, gt_poses, gt_vels, init_poses, init_vels);
  }

  MinimizerSummary summary;
  {
    auto range = profiler_domain_.CreateDomainRange("BuildProblem+Minimize");

    dvector<SE3Transform> poses_device(init_poses);
    dvector<Vector<6>> vels_device(init_vels);
    std::vector<float> dt_values(num_factors, dt);
    dvector<float> dt_device(dt_values);
    std::vector<int> const_ids = {0};
    dvector<int> const_ids_device(const_ids);

    SE3StateBatch pose_states(cublas_handle_, reinterpret_cast<const float *>(poses_device.data()),
                              p.n_poses, const_ids_device.data(), 1);
    VectorStateBatch<6> vel_states(reinterpret_cast<const float *>(vels_device.data()), p.n_poses,
                                   const_ids_device.data(), 1);
    ConstantVelocitySE3FactorBatch motion_prior(dt_device.data(), num_factors);

    std::vector<float *> state_pointers;
    state_pointers.reserve(4 * num_factors);
    for (int i = 0; i < num_factors; ++i) {
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i + 1));
    }

    Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&vel_states);
    problem.AddFactorBatch(&motion_prior, state_pointers);
    ASSERT_TRUE(problem.CheckConsistency());

    MinimizerOptions options;
    options.max_num_iterations = 30;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;
    options.disable_safety_checks = true;
    options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
    options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
        test_utils::PCGBlockSizeFromEnv(6);
    options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
        test_utils::PCGMaxIterFromEnv(400);
    options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance =
        test_utils::PCGTolFromEnv(1e-4f);

    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    LevenbergMarquardtMinimizer minimizer(lm_options);

    auto minimize_range = profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream_.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));
  }

  EXPECT_TRUE(std::isfinite(summary.initial_cost));
  EXPECT_TRUE(std::isfinite(summary.final_cost));
  EXPECT_LE(summary.final_cost, summary.initial_cost);
}

// Q(dt)^-1-weighted ConstantVelocityInformationSE3FactorBatch at the same
// scale, to measure the overhead of the covariance fusion.
TEST_P(MotionPriorPerfTest, InformationWeightedConverges) {
  auto p = GetParam();
  SCOPED_TRACE(std::string("MotionPriorPerfTest.InformationWeightedConverges: ") + p.label);

  const float dt = 0.1f;
  const int num_factors = p.n_poses - 1;

  std::vector<SE3Transform> gt_poses, init_poses;
  std::vector<Vector<6>> gt_vels, init_vels;
  {
    auto range = profiler_domain_.CreateDomainRange("GenerateChain");
    GenerateChain(p.n_poses, dt, gt_poses, gt_vels, init_poses, init_vels);
  }

  MinimizerSummary summary;
  {
    auto range = profiler_domain_.CreateDomainRange("BuildProblem+Minimize");

    dvector<SE3Transform> poses_device(init_poses);
    dvector<Vector<6>> vels_device(init_vels);
    std::vector<float> dt_values(num_factors, dt);
    dvector<float> dt_device(dt_values);
    std::vector<float> qc_diag = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    dvector<float> qc_device(qc_diag);
    std::vector<int> const_ids = {0};
    dvector<int> const_ids_device(const_ids);

    SE3StateBatch pose_states(cublas_handle_, reinterpret_cast<const float *>(poses_device.data()),
                              p.n_poses, const_ids_device.data(), 1);
    VectorStateBatch<6> vel_states(reinterpret_cast<const float *>(vels_device.data()), p.n_poses,
                                   const_ids_device.data(), 1);

    ConstantVelocityInformationSE3FactorBatch motion_prior = [&] {
      auto build_info_range = profiler_domain_.CreateDomainRange("BuildInformationFactor");
      return ConstantVelocityInformationSE3FactorBatch(
          cublas_handle_, stream_.GetStream(), dt_device.data(), qc_device.data(), num_factors);
    }();

    std::vector<float *> state_pointers;
    state_pointers.reserve(4 * num_factors);
    for (int i = 0; i < num_factors; ++i) {
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
      state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i));
      state_pointers.push_back(vel_states.StateBlockDevicePtr(i + 1));
    }

    Problem problem;
    problem.AddStateBatch(&pose_states);
    problem.AddStateBatch(&vel_states);
    problem.AddFactorBatch(&motion_prior, state_pointers);
    ASSERT_TRUE(problem.CheckConsistency());

    MinimizerOptions options;
    options.max_num_iterations = 100;
    options.state_tolerance = 1e-8f;
    options.cost_tolerance = 1e-8f;
    options.disable_safety_checks = true;
    options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
    options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
        test_utils::PCGBlockSizeFromEnv(6);
    options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
        test_utils::PCGMaxIterFromEnv(400);
    options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance =
        test_utils::PCGTolFromEnv(1e-4f);

    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.initial_lambda = 1e-3f;
    LevenbergMarquardtMinimizer minimizer(lm_options);

    auto minimize_range = profiler_domain_.CreateDomainRange("Minimize");
    summary = minimizer.Minimize(stream_.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream_.GetStream()));
  }

  EXPECT_TRUE(std::isfinite(summary.initial_cost));
  EXPECT_TRUE(std::isfinite(summary.final_cost));
  EXPECT_LE(summary.final_cost, summary.initial_cost);
}

INSTANTIATE_TEST_SUITE_P(Sizes, MotionPriorPerfTest,
                         ::testing::Values(MotionPriorPerfParams{1000, "P1k"},
                                           MotionPriorPerfParams{5000, "P5k"},
                                           MotionPriorPerfParams{20000, "P20k"},
                                           MotionPriorPerfParams{50000, "P50k"}));

}  // namespace cunls
