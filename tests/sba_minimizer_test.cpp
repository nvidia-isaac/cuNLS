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
 * @file sba_binary_minimizer_test.cpp
 * @brief Tests ReprojectionFactorBatch + InformationFactorBatch with
 *        Levenberg-Marquardt on binary SBA (Sparse Bundle Adjustment) problems.
 *
 * Loads one or more problems from a binary file (filtered_ba_problems.bin),
 * builds a Problem with pose/point state batches and information-weighted
 * reprojection factors (with Huber loss), then runs LM and checks convergence
 * (finite cost, cost decrease).
 *
 * Binary format (one problem):
 * - int32 Nposes, Npoints, N_obs, N_fixed_points, N_fixed_poses, Ncameras
 * - N_obs blobs: 4 floats (2x2 sqrt information), 2 floats (observation),
 *   int32 camera_id, int32 point_id, int32 pose_id
 * - Nposes×16 floats, Npoints×3 floats, Ncameras×16 floats (camera_from_rig)
 *
 * To run a specific problem only, set CUNLS_SBA_PROBLEM_INDEX to 0-based index.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

#ifndef CUNLS_TEST_DATA_DIR
#error                                                                         \
    "CUNLS_TEST_DATA_DIR must be defined by CMake (target_compile_definitions)"
#endif
/// Directory containing test data (e.g. filtered_ba_problems.bin); set by
/// CMake.
constexpr const char *kTestDataDir = CUNLS_TEST_DATA_DIR;
/// Minimum depth in camera frame for valid reprojection; used by factors.
constexpr float kZThreshold = 1e-3f;

// -----------------------------------------------------------------------------
// Binary format: host struct and reader
// -----------------------------------------------------------------------------

/**
 * @brief Host-side representation of one SBA problem read from binary file.
 *
 * Matches the on-disk layout: dimensions first, then per-observation data
 * (sqrt information, observation, ids), then poses, points, and camera rig
 * transforms.
 */
struct SbaProblemHost {
  int32_t Nposes = 0;         ///< Number of camera poses (rig poses).
  int32_t Npoints = 0;        ///< Number of 3D points.
  int32_t N_obs = 0;          ///< Number of observations (measurements).
  int32_t N_fixed_points = 0; ///< Number of point indices held constant.
  int32_t N_fixed_poses = 0;  ///< Number of pose indices held constant.
  int32_t Ncameras =
      0; ///< Number of cameras (rig size); each obs has camera_id.
  /// Per-obs: 4 floats (2x2 sqrt info), then 2 floats (observation xy); size
  /// n_obs*6.
  std::vector<float> obs_sqrt_info_and_xy;
  std::vector<int32_t> camera_ids; ///< Camera index for each observation.
  std::vector<int32_t> point_ids;  ///< Point index for each observation.
  std::vector<int32_t> pose_ids;   ///< Pose index for each observation.
  std::vector<SE3Transform>
      poses;                     ///< Rig poses (world-to-rig), 16 floats each.
  std::vector<Vector<3>> points; ///< 3D points in world frame.
  std::vector<SE3Transform>
      camera_from_rig; ///< Camera-in-rig transforms per camera.
};

/**
 * @brief Reads one int32 from binary stream.
 * @param in Input binary stream.
 * @param out Output value.
 * @return true if read succeeded, false on EOF or error.
 */
static bool ReadInt32(std::istream &in, int32_t *out) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char *>(out), sizeof(int32_t)));
}

/**
 * @brief Reads a block of floats from binary stream.
 * @param in Input binary stream.
 * @param ptr Destination buffer.
 * @param count Number of floats to read.
 * @return true if all bytes were read, false otherwise.
 */
static bool ReadFloats(std::istream &in, float *ptr, size_t count) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char *>(ptr), count * sizeof(float)));
}

/**
 * @brief Reads one full SBA problem from binary stream into host struct.
 *
 * Order: header (6 int32s), then for each observation 6 floats + 3 int32s,
 * then poses (Nposes*16 floats), points (Npoints*3), camera_from_rig
 * (Ncameras*16). Throws on negative dimensions; returns false on read failure
 * or EOF.
 *
 * @param in Input binary stream.
 * @param out Filled problem; only valid if function returns true.
 * @return true if the full problem was read, false on EOF or read error.
 */
static bool ReadOneSbaProblem(std::istream &in, SbaProblemHost &out) {
  if (!ReadInt32(in, &out.Nposes) || !ReadInt32(in, &out.Npoints) ||
      !ReadInt32(in, &out.N_obs) || !ReadInt32(in, &out.N_fixed_points) ||
      !ReadInt32(in, &out.N_fixed_poses) || !ReadInt32(in, &out.Ncameras)) {
    return false;
  }
  if (out.Nposes < 0 || out.Npoints < 0 || out.N_obs < 0 ||
      out.N_fixed_points < 0 || out.N_fixed_poses < 0 || out.Ncameras < 0) {
    throw std::runtime_error("SbaProblem: negative dimension");
  }
  const size_t n_obs = static_cast<size_t>(out.N_obs);
  const size_t n_poses = static_cast<size_t>(out.Nposes);
  const size_t n_points = static_cast<size_t>(out.Npoints);
  const size_t n_cameras = static_cast<size_t>(out.Ncameras);

  out.obs_sqrt_info_and_xy.resize(n_obs * 6);
  out.camera_ids.resize(n_obs);
  out.point_ids.resize(n_obs);
  out.pose_ids.resize(n_obs);
  for (size_t i = 0; i < n_obs; i++) {
    if (!ReadFloats(in, out.obs_sqrt_info_and_xy.data() + i * 6, 6)) {
      return false;
    }
    if (!ReadInt32(in, &out.camera_ids[i]) ||
        !ReadInt32(in, &out.point_ids[i]) || !ReadInt32(in, &out.pose_ids[i])) {
      return false;
    }
  }
  out.poses.resize(n_poses);
  if (!ReadFloats(in, reinterpret_cast<float *>(out.poses.data()),
                  n_poses * 16)) {
    return false;
  }
  out.points.resize(n_points);
  if (!ReadFloats(in, reinterpret_cast<float *>(out.points.data()),
                  n_points * 3)) {
    return false;
  }
  out.camera_from_rig.resize(n_cameras);
  if (!ReadFloats(in, reinterpret_cast<float *>(out.camera_from_rig.data()),
                  n_cameras * 16)) {
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
// Test fixture: owns device data and batches, builds Problem from host data
// -----------------------------------------------------------------------------

/**
 * @brief Test fixture for SBA minimizer tests using binary problem files.
 *
 * Does not own the binary file; tests open it and call ReadOneSbaProblem.
 * BuildProblemFromHost() allocates device buffers, builds SE3StateBatch and
 * VectorStateBatch for poses/points (with fixed blocks from N_fixed_*),
 * InformationFactorBatch wrapping ReprojectionFactorBatch, and wires factor
 * to state blocks via state_pointers_ (pose then point per observation).
 * Huber loss is applied to the information-weighted residuals.
 */
class SbaMinimizerTestFixture : public ::testing::Test {
protected:
  /**
   * @brief Builds a minimizer Problem from a host SBA problem and stores device
   * state in fixture.
   *
   * Copies poses, points, observations, sqrt information, and camera_from_rig
   * to device; builds pose and point state batches with fixed blocks for the
   * first N_fixed_poses and N_fixed_points; builds InformationFactorBatch over
   * ReprojectionFactorBatch and assigns each factor to (pose_id, point_id).
   * Adds state batches and one factor batch (with Huber loss) to @p problem.
   * Throws on invalid dimensions or indices.
   *
   * @param host Host-side problem (from ReadOneSbaProblem).
   * @param problem Output problem to add state and factor batches to.
   */
  void BuildProblemFromHost(const SbaProblemHost &host, Problem *problem) {
    const size_t n_obs = static_cast<size_t>(host.N_obs);
    const size_t n_poses = static_cast<size_t>(host.Nposes);
    const size_t n_points = static_cast<size_t>(host.Npoints);
    const size_t n_cameras = static_cast<size_t>(host.Ncameras);
    const int n_fixed_poses = host.N_fixed_poses;
    const int n_fixed_points = host.N_fixed_points;

    if (n_obs == 0 || n_poses == 0 || n_points == 0) {
      throw std::runtime_error("SbaProblem: empty problem");
    }
    if (n_cameras == 0) {
      throw std::runtime_error("SbaProblem: Ncameras must be > 0");
    }

    // Unpack per-observation: 2x2 sqrt info (row-major), then observation
    // (x,y).
    std::vector<Vector<2>> observations(n_obs);
    std::vector<Matrix<2>> sqrt_info(n_obs);
    std::vector<SE3Transform> camera_from_rig_per_obs(n_obs);
    for (size_t i = 0; i < n_obs; i++) {
      const float *row = host.obs_sqrt_info_and_xy.data() + i * 6;
      sqrt_info[i][0] = row[0];
      sqrt_info[i][1] = row[1];
      sqrt_info[i][2] = row[2];
      sqrt_info[i][3] = row[3];
      observations[i][0] = row[4];
      observations[i][1] = row[5];
      const int32_t cam_id = host.camera_ids[i];
      if (cam_id < 0 || static_cast<size_t>(cam_id) >= n_cameras) {
        throw std::runtime_error("SbaProblem: invalid camera_id");
      }
      camera_from_rig_per_obs[i] =
          host.camera_from_rig[static_cast<size_t>(cam_id)];
    }

    poses_device_ = dvector<SE3Transform>(host.poses);
    points_device_ = dvector<Vector<3>>(host.points);
    observations_device_ = dvector<Vector<2>>(observations);
    sqrt_information_device_ = dvector<Matrix<2>>(sqrt_info);
    camera_from_rig_per_obs_device_ =
        dvector<SE3Transform>(camera_from_rig_per_obs);

    // Fixed blocks: first N_fixed_poses poses and N_fixed_points points are
    // constant.
    std::vector<int> const_pose_ids;
    for (int i = 0; i < n_fixed_poses && i < static_cast<int>(n_poses); i++) {
      const_pose_ids.push_back(i);
    }
    std::vector<int> const_point_ids;
    for (int i = 0; i < n_fixed_points && i < static_cast<int>(n_points); i++) {
      const_point_ids.push_back(i);
    }
    const_pose_ids_device_ = dvector<int>(const_pose_ids);
    const_point_ids_device_ = dvector<int>(const_point_ids);

    const float *poses_ptr =
        reinterpret_cast<const float *>(poses_device_.data());
    const float *points_ptr =
        reinterpret_cast<const float *>(points_device_.data());

    pose_batch_ = std::make_unique<SE3StateBatch>(
        cublas_handle_, poses_ptr, n_poses, const_pose_ids_device_.data(),
        const_pose_ids.size());
    point_batch_ = std::make_unique<VectorStateBatch<3>>(
        points_ptr, n_points, const_point_ids_device_.data(),
        const_point_ids.size());

    reproj_batch_ = std::make_unique<ReprojectionFactorBatch>(
        observations_device_.data(), camera_from_rig_per_obs_device_.data(),
        n_obs, kZThreshold);

    info_factor_batch_ =
        std::make_unique<InformationFactorBatch<ReprojectionFactorBatch>>(
            cublas_handle_, sqrt_information_device_.data(), n_obs,
            observations_device_.data(), camera_from_rig_per_obs_device_.data(),
            n_obs, kZThreshold);

    // Factor i connects pose host.pose_ids[i] and point host.point_ids[i].
    state_pointers_.clear();
    state_pointers_.reserve(n_obs * 2);
    for (size_t i = 0; i < n_obs; i++) {
      const int32_t pose_id = host.pose_ids[i];
      const int32_t point_id = host.point_ids[i];
      if (pose_id < 0 || static_cast<size_t>(pose_id) >= n_poses ||
          point_id < 0 || static_cast<size_t>(point_id) >= n_points) {
        throw std::runtime_error("SbaProblem: invalid pose_id or point_id");
      }
      state_pointers_.push_back(
          pose_batch_->StateBlockDevicePtr(static_cast<size_t>(pose_id)));
      state_pointers_.push_back(
          point_batch_->StateBlockDevicePtr(static_cast<size_t>(point_id)));
    }

    problem->AddStateBatch(pose_batch_.get());
    problem->AddStateBatch(point_batch_.get());

    huber_loss_batch_ = std::make_unique<HuberLossFunctionBatch>(1.0f);
    problem->AddFactorBatch(info_factor_batch_.get(), huber_loss_batch_.get(),
                            state_pointers_);
  }

  cuBLASHandle cublas_handle_;
  dvector<SE3Transform> poses_device_;
  dvector<Vector<3>> points_device_;
  dvector<Vector<2>> observations_device_;
  dvector<Matrix<2>> sqrt_information_device_;
  dvector<SE3Transform> camera_from_rig_per_obs_device_;
  dvector<int> const_pose_ids_device_;
  dvector<int> const_point_ids_device_;
  /// Device pointers for each factor: [pose_ptr, point_ptr] per observation, in
  /// order.
  std::vector<float *> state_pointers_;

  std::unique_ptr<SE3StateBatch> pose_batch_;
  std::unique_ptr<VectorStateBatch<3>> point_batch_;
  std::unique_ptr<ReprojectionFactorBatch> reproj_batch_;
  std::unique_ptr<InformationFactorBatch<ReprojectionFactorBatch>>
      info_factor_batch_;
  std::unique_ptr<HuberLossFunctionBatch> huber_loss_batch_;

  profiler::Domain profiler_domain_ =
      profiler::Domain("SbaMinimizerTestFixture");
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
/**
 * @brief Runs LM on each (or the selected) binary SBA problem and checks
 * convergence.
 *
 * Opens filtered_ba_problems.bin; if CUNLS_SBA_PROBLEM_INDEX is set, runs only
 * that 0-based problem index, otherwise runs all problems in the file. For each
 * problem: builds Problem via BuildProblemFromHost, runs
 * LevenbergMarquardtMinimizer, and asserts initial/final cost are finite and
 * final cost <= initial cost (within tolerance). Skips empty problems; skips if
 * data file or selected index is missing.
 */
TEST_F(SbaMinimizerTestFixture, OptimizeAndCheckConvergence) {
  std::string path = std::string(kTestDataDir) + "/filtered_ba_problems.bin";
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    GTEST_SKIP() << "Binary file not found: " << path;
  }

  // Optional: run only one problem (0-based index).
  const char *index_env = std::getenv("CUNLS_SBA_PROBLEM_INDEX");
  int problem_index = -1;
  if (index_env) {
    problem_index = std::atoi(index_env);
    if (problem_index < 0)
      problem_index = -1;
  }

  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 15;
  options.state_tolerance = 1e-8f;
  options.cost_tolerance = 1e4;
  options.disable_safety_checks = false;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  // SBA has both 6x6 (poses) and 3x3 (landmarks) diagonal blocks.  Using
  // block_size=3 keeps the preconditioner cheap and well-conditioned for the
  // landmark blocks while still capturing useful structure inside the 6x6
  // pose tiles (every 6x6 splits into a 2x2 grid of 3x3 sub-blocks).
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size =
      test_utils::PCGBlockSizeFromEnv(3);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations =
      test_utils::PCGMaxIterFromEnv(400);
  options.sparse_linear_solver_config.block_sparse_pcg_options
      .relative_tolerance = test_utils::PCGTolFromEnv(1e-3f);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-3f;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  SbaProblemHost host;
  int current_index = 0;
  bool processed_requested = false;

  while (ReadOneSbaProblem(in, host)) {
    if (problem_index >= 0 && current_index != problem_index) {
      current_index++;
      continue;
    }
    if (host.N_obs == 0 || host.Nposes == 0 || host.Npoints == 0) {
      if (problem_index >= 0)
        processed_requested = true;
      current_index++;
      continue;
    }

    SCOPED_TRACE("Problem index " + std::to_string(current_index));

    Problem problem;
    BuildProblemFromHost(host, &problem);
    ASSERT_TRUE(problem.CheckConsistency());

    MinimizerSummary summary;
    {
      auto minimize_range = profiler_domain_.CreateDomainRange("Minimize");
      summary = minimizer.Minimize(stream.GetStream(), problem);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    }

    EXPECT_TRUE(std::isfinite(summary.initial_cost))
        << "Initial cost must be finite (no NaNs)";
    EXPECT_TRUE(std::isfinite(summary.final_cost))
        << "Final cost must be finite (no NaNs)";
    EXPECT_LE(summary.final_cost, summary.initial_cost + 1e-6f)
        << "Final cost must be <= initial cost";

    if (problem_index >= 0) {
      processed_requested = true;
      break;
    }
    current_index++;
  }

  if (problem_index >= 0 && !processed_requested) {
    GTEST_SKIP() << "Problem index " << problem_index << " not found in file";
  }
}

} // namespace cunls
