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
 * @file pgo_minimizer_test.cpp
 * @brief Tests SE3BetweenFactorBatch + InformationFactorBatch with
 *        Levenberg-Marquardt on binary pose-graph optimization (PGO) problems.
 *
 * Loads one or more problems from a binary file (pgo_problems.bin), builds a
 * Problem with SE3 pose state batches and information-weighted between factors,
 * then runs LM and checks convergence (finite cost, cost decrease).
 *
 * A single LevenbergMarquardtMinimizer instance is reused across all problems
 * to verify that internal caches (cuSPARSE GEMM, cuDSS solver) are correctly
 * re-initialized when the sparsity pattern changes between problems.
 *
 * Binary format (one problem, repeated until EOF):
 *   int32  Nposes          – number of SE3 poses
 *   int32  Ndeltas         – number of relative-pose constraints
 *   int32  NFixed          – number of fixed (anchored) poses
 *   float[Nposes × 16]    – SE3 poses (4×4 column-major)
 *   For each delta (Ndeltas times):
 *     float[16]            – relative SE3 pose delta (4×4 column-major)
 *     float[36]            – 6×6 sqrt-information matrix (row-major)
 *     int32 pose1_id       – source pose index
 *     int32 pose2_id       – target pose index
 *   int32[NFixed]          – indices of fixed poses
 *
 */

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/information_factor_batch.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "tests/utils.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"

namespace cunls {

#ifndef CUNLS_TEST_DATA_DIR
#error                                                                         \
    "CUNLS_TEST_DATA_DIR must be defined by CMake (target_compile_definitions)"
#endif
/// Directory containing test data files; set by CMake.
constexpr const char *kTestDataDir = CUNLS_TEST_DATA_DIR;

// =============================================================================
// Binary I/O helpers
// =============================================================================

bool ReadInt32(std::istream &in, int32_t *out) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char *>(out), sizeof(int32_t)));
}

bool ReadFloats(std::istream &in, float *ptr, size_t count) {
  return static_cast<bool>(
      in.read(reinterpret_cast<char *>(ptr), count * sizeof(float)));
}

// =============================================================================
// Host-side PGO problem representation
// =============================================================================

/**
 * @brief Host-side representation of one PGO problem read from the binary file.
 *
 * Stores SE3 poses, relative-pose deltas with sqrt-information matrices, edge
 * connectivity (pose1/pose2 ids), and the set of fixed (anchored) pose indices.
 */
struct PgoProblemHost {
  int32_t Nposes = 0;  ///< Number of SE3 poses.
  int32_t Ndeltas = 0; ///< Number of relative-pose constraints.
  int32_t NFixed = 0;  ///< Number of fixed (anchored) poses.

  std::vector<SE3Transform> poses;           ///< Initial SE3 poses.
  std::vector<SE3Transform> pose_deltas;     ///< Relative-pose measurements.
  std::vector<Matrix<6>> sqrt_info_matrices; ///< Per-edge 6×6 sqrt-info.
  std::vector<int32_t> pose1_ids;            ///< Source pose per edge.
  std::vector<int32_t> pose2_ids;            ///< Target pose per edge.
  std::vector<int32_t> fixed_pose_ids;       ///< Indices of anchored poses.
};

/**
 * @brief Reads one full PGO problem from a binary stream.
 *
 * See the file-level comment for the binary layout.
 *
 * @param[in]  in   Binary input stream positioned at the start of a problem.
 * @param[out] out  Populated on success; partially written on failure.
 * @return true if a complete problem was read, false on EOF or read error.
 */
bool ReadOnePgoProblem(std::istream &in, PgoProblemHost &out) {
  // -- Header: three int32 dimensions --
  if (!ReadInt32(in, &out.Nposes) || !ReadInt32(in, &out.Ndeltas) ||
      !ReadInt32(in, &out.NFixed)) {
    return false;
  }
  if (out.Nposes < 0 || out.Ndeltas < 0 || out.NFixed < 0) {
    throw std::runtime_error("PgoProblem: negative dimension in header");
  }

  const auto n_poses = static_cast<size_t>(out.Nposes);
  const auto n_deltas = static_cast<size_t>(out.Ndeltas);
  const auto n_fixed = static_cast<size_t>(out.NFixed);

  // -- Poses: Nposes × 16 floats (4×4 SE3 matrices) --
  out.poses.resize(n_poses);
  if (!ReadFloats(in, reinterpret_cast<float *>(out.poses.data()),
                  n_poses * 16)) {
    return false;
  }

  // -- Per-edge data: delta pose + sqrt-information + connectivity --
  out.pose_deltas.resize(n_deltas);
  out.sqrt_info_matrices.resize(n_deltas);
  out.pose1_ids.resize(n_deltas);
  out.pose2_ids.resize(n_deltas);

  for (size_t i = 0; i < n_deltas; i++) {
    if (!ReadFloats(in, out.pose_deltas[i].data(), 16))
      return false;
    if (!ReadFloats(in, out.sqrt_info_matrices[i].data(), 36))
      return false;
    if (!ReadInt32(in, &out.pose1_ids[i]))
      return false;
    if (!ReadInt32(in, &out.pose2_ids[i]))
      return false;
  }

  // -- Fixed pose indices --
  out.fixed_pose_ids.resize(n_fixed);
  for (size_t i = 0; i < n_fixed; i++) {
    if (!ReadInt32(in, &out.fixed_pose_ids[i]))
      return false;
  }

  // Validate that all indices are within bounds
  for (size_t i = 0; i < n_deltas; i++) {
    if (out.pose1_ids[i] < 0 ||
        static_cast<size_t>(out.pose1_ids[i]) >= n_poses) {
      throw std::runtime_error("PgoProblem: pose1_id out of range at edge " +
                               std::to_string(i));
    }
    if (out.pose2_ids[i] < 0 ||
        static_cast<size_t>(out.pose2_ids[i]) >= n_poses) {
      throw std::runtime_error("PgoProblem: pose2_id out of range at edge " +
                               std::to_string(i));
    }
  }
  for (size_t i = 0; i < n_fixed; i++) {
    if (out.fixed_pose_ids[i] < 0 ||
        static_cast<size_t>(out.fixed_pose_ids[i]) >= n_poses) {
      throw std::runtime_error(
          "PgoProblem: fixed_pose_id out of range at index " +
          std::to_string(i));
    }
  }

  return true;
}

// =============================================================================
// Test fixture
// =============================================================================

/**
 * @brief Test fixture for PGO minimizer tests.
 *
 * Owns all device-side storage (poses, deltas, sqrt-information, factor
 * batches) and provides BuildProblemFromHost() to wire a PgoProblemHost into
 * a cunls::Problem suitable for LevenbergMarquardtMinimizer.
 */
class PgoMinimizerTestFixture : public ::testing::Test {
protected:
  /**
   * @brief Builds a cunls::Problem from a host PGO problem.
   *
   * Uploads poses, deltas, and sqrt-information matrices to device memory,
   * constructs an SE3StateBatch with fixed-pose anchoring, wraps
   * SE3BetweenFactorBatch inside InformationFactorBatch, and wires factor
   * to state blocks via state_pointers_ (target-pose then source-pose per
   * edge, matching SE3BetweenFactorBatch convention).
   *
   * @param host    Host-side problem data read from the binary file.
   * @param[out] problem  Empty Problem to populate.
   */
  void BuildProblemFromHost(const PgoProblemHost &host, Problem *problem) {
    const auto n_poses = static_cast<size_t>(host.Nposes);
    const auto n_deltas = static_cast<size_t>(host.Ndeltas);

    if (n_poses == 0 || n_deltas == 0) {
      throw std::runtime_error("PgoProblem: empty problem");
    }

    // Upload pose and delta data to device
    poses_device_ = dvector<SE3Transform>(host.poses);
    pose_deltas_device_ = dvector<SE3Transform>(host.pose_deltas);
    fixed_pose_ids_device_ = dvector<int32_t>(host.fixed_pose_ids);

    const float *poses_ptr =
        reinterpret_cast<const float *>(poses_device_.data());

    // SE3StateBatch: one 4×4 block per pose; fixed poses are anchored
    pose_batch_ = std::make_unique<SE3StateBatch>(
        cublas_handle_, poses_ptr, n_poses, fixed_pose_ids_device_.data(),
        host.fixed_pose_ids.size());

    // Upload sqrt-information matrices and create the factor batch
    sqrt_info_device_ = dvector<Matrix<6>>(host.sqrt_info_matrices);
    info_factor_batch_ =
        std::make_unique<InformationFactorBatch<SE3BetweenFactorBatch>>(
            cublas_handle_, sqrt_info_device_.data(), n_deltas,
            pose_deltas_device_.data(), n_deltas);

    // Wire each edge to its two pose state blocks
    state_pointers_.clear();
    state_pointers_.reserve(n_deltas * 2);
    for (size_t i = 0; i < n_deltas; i++) {
      state_pointers_.push_back(pose_batch_->StateBlockDevicePtr(
          static_cast<size_t>(host.pose2_ids[i])));
      state_pointers_.push_back(pose_batch_->StateBlockDevicePtr(
          static_cast<size_t>(host.pose1_ids[i])));
    }

    problem->AddStateBatch(pose_batch_.get());
    problem->AddFactorBatch(info_factor_batch_.get(), state_pointers_);
  }

  // -- Device-side storage (lifetime must outlive the Problem) ---------------
  cuBLASHandle cublas_handle_;
  dvector<SE3Transform> poses_device_;
  dvector<SE3Transform> pose_deltas_device_;
  dvector<Matrix<6>> sqrt_info_device_;
  dvector<int32_t> fixed_pose_ids_device_;
  std::vector<float *> state_pointers_;

  // -- Factor and state batches ----------------------------------------------
  std::unique_ptr<SE3StateBatch> pose_batch_;
  std::unique_ptr<InformationFactorBatch<SE3BetweenFactorBatch>>
      info_factor_batch_;

  profiler::Domain profiler_domain_ =
      profiler::Domain("PgoMinimizerTestFixture");
};

// =============================================================================
// Test
// =============================================================================

/**
 * @brief Runs LM on each (or a selected) PGO problem and checks convergence.
 *
 * Opens pgo_problems.bin and runs every problem in the file.  A single
 * LevenbergMarquardtMinimizer is reused across all problems.  For each
 * problem: builds Problem via BuildProblemFromHost, runs LM, and asserts that
 * both initial and final costs are finite and that the optimizer did not
 * increase cost.  Skips gracefully if the data file or selected index is
 * missing.
 */
TEST_F(PgoMinimizerTestFixture, Optimize) {
  const std::string path = std::string(kTestDataDir) + "/pgo_problems.bin";
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    GTEST_SKIP() << "Binary file not found: " << path;
  }

  // Configure and create the LM minimizer (reused across all problems)
  CudaStream stream;
  MinimizerOptions options;
  options.max_num_iterations = 10;
  options.state_tolerance = 1e-10f;
  options.cost_tolerance = 1e-2f;
  options.disable_safety_checks = false;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  options.sparse_linear_solver_config.block_sparse_pcg_options.block_size = test_utils::PCGBlockSizeFromEnv(6);
  options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations = test_utils::PCGMaxIterFromEnv(200);
  options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance = test_utils::PCGTolFromEnv(1e-3f);
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1000.0;
  LevenbergMarquardtMinimizer minimizer(lm_options);

  // Iterate over all problems in the binary file
  PgoProblemHost host;
  int current_index = 0;

  while (ReadOnePgoProblem(in, host)) {
    if (host.Nposes == 0 || host.Ndeltas == 0) {
      current_index++;
      continue;
    }

    SCOPED_TRACE("Problem index " + std::to_string(current_index));

    // Build the optimization problem on the GPU
    Problem problem;
    BuildProblemFromHost(host, &problem);
    ASSERT_TRUE(problem.CheckConsistency());

    // Run Levenberg-Marquardt optimization
    MinimizerSummary summary;
    {
      auto range = profiler_domain_.CreateDomainRange("Minimize");
      summary = minimizer.Minimize(stream.GetStream(), problem);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    }

    // Verify convergence: costs must be finite and non-increasing
    ASSERT_TRUE(std::isfinite(summary.initial_cost))
        << "Initial cost must be finite (no NaNs)";
    ASSERT_TRUE(std::isfinite(summary.final_cost))
        << "Final cost must be finite (no NaNs)";
    ASSERT_LE(summary.final_cost, summary.initial_cost)
        << "Final cost must not exceed initial cost";

    current_index++;
  }
}

} // namespace cunls
