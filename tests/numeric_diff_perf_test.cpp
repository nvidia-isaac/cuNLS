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
 * @file numeric_diff_perf_test.cpp
 * @brief Wall-clock comparison of JacobianMode::kAnalytic vs kNumeric, across
 * problem types (PGO / SBA / PnP) and scales (small / medium / large).
 *
 * Times repeated `GaussNewtonMinimizer::BuildSystem` calls (not full
 * `Minimize()` runs) via CUDA events: `BuildSystem` is exactly the call that
 * computes residuals + Jacobians and assembles the normal equations, so it
 * isolates the cost the two Jacobian modes actually differ on. A full
 * `Minimize()` would also fold in the sparse linear solve, whose cost is
 * independent of `JacobianMode` and (per the profiling note in commit
 * e5c4612) dominates total GPU time -- mixing it in would wash out the very
 * difference this benchmark exists to measure.
 *
 * Modeled directly on tests/motion_prior_perf_test.cpp's structure (gtest
 * TestWithParam, NVTX instrumentation gated by ENABLE_PROFILING) and on the
 * `SystemBuilder` BuildSystem-exposing pattern from
 * tests/block_hessian_assembler_test.cpp. Meant to be run under `nsys
 * profile`; also appends CUDA-event timings to a CSV for offline plotting.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

enum class ProblemType { kPGO, kSBA, kPnP };
enum class Scale { kSmall, kMedium, kLarge };

const char *ToString(ProblemType t) {
  switch (t) {
    case ProblemType::kPGO:
      return "PGO";
    case ProblemType::kSBA:
      return "SBA";
    case ProblemType::kPnP:
      return "PnP";
  }
  return "?";
}

const char *ToString(Scale s) {
  switch (s) {
    case Scale::kSmall:
      return "small";
    case Scale::kMedium:
      return "medium";
    case Scale::kLarge:
      return "large";
  }
  return "?";
}

const char *ToString(JacobianMode m) {
  return m == JacobianMode::kAnalytic ? "analytic" : "numeric";
}

/**
 * @brief Exposes GaussNewtonMinimizer::Initialize/BuildSystem (both
 * protected). Same rationale/pattern as SystemBuilder in
 * tests/block_hessian_assembler_test.cpp: least invasive way to time
 * assembly in isolation from the rest of Minimize().
 */
class SystemBuilder : public GaussNewtonMinimizer {
 public:
  explicit SystemBuilder(const MinimizerOptions &options) : GaussNewtonMinimizer(options) {}

  void Prepare(cudaStream_t stream, Problem &problem) {
    Initialize(stream, problem);
    current_state_.Recreate(stream, problem);
  }

  void Build(cudaStream_t stream, const Problem &problem) {
    BuildSystem(stream, problem, current_state_);
  }
};

// ---------------------------------------------------------------------------
// Synthetic dataset generators (kept minimal; correctness of these factor
// types is already covered by synthetic_pgo_test.cpp / synthetic_sba_test.cpp
// / pnp_factor_batch_test.cpp -- this file only needs *some* well-posed
// problem of the right scale to time BuildSystem on).
// ---------------------------------------------------------------------------

SE3Transform ComposeSE3Host(const SE3Transform &a, const SE3Transform &b) {
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

std::vector<SE3Transform> RandomPoses(size_t n, std::mt19937 &rng) {
  std::uniform_real_distribution<float> rot(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans(-1.0f, 1.0f);
  std::vector<Vector<6>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    twists[i] = {rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), 8.0f + trans(rng)};
  }
  CudaStream stream;
  dvector<Vector<6>> d_twists(twists);
  dvector<SE3Transform> d_poses(n);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(d_twists.data()), 6, 4, 16, n,
               reinterpret_cast<float *>(d_poses.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<SE3Transform> poses(n);
  d_poses.CopyToHost(poses.data(), n);
  return poses;
}

size_t PGONumPoses(Scale s) {
  switch (s) {
    case Scale::kSmall:
      return 200;
    case Scale::kMedium:
      return 2000;
    case Scale::kLarge:
      return 10000;
  }
  return 0;
}

void SBAScaleParams(Scale s, int &n_poses, int &n_points) {
  switch (s) {
    case Scale::kSmall:
      n_poses = 4;
      n_points = 100;
      return;
    case Scale::kMedium:
      n_poses = 6;
      n_points = 800;
      return;
    case Scale::kLarge:
      n_poses = 8;
      n_points = 4000;
      return;
  }
}

size_t PnPNumCorrespondences(Scale s) {
  switch (s) {
    case Scale::kSmall:
      return 2000;
    case Scale::kMedium:
      return 20000;
    case Scale::kLarge:
      return 100000;
  }
  return 0;
}

struct PerfParams {
  ProblemType problem_type;
  Scale scale;
  JacobianMode mode;
};

std::string ParamLabel(const PerfParams &p) {
  return std::string(ToString(p.problem_type)) + "_" + ToString(p.scale) + "_" +
        ToString(p.mode);
}

std::ostream &operator<<(std::ostream &os, const PerfParams &p) {
  return os << ParamLabel(p);
}

class NumericDiffPerfTest : public ::testing::TestWithParam<PerfParams> {
 protected:
  static constexpr int kWarmupIters = 3;
  static constexpr int kTimedIters = 10;

  // Appends one CSV row; writes the header once (file created fresh at
  // SetUpTestSuite time).
  static void AppendCsvRow(const std::string &problem_type, const std::string &scale,
                           const std::string &mode, double mean_ms) {
    std::ofstream f(CsvPath(), std::ios::app);
    f << problem_type << "," << scale << "," << mode << "," << mean_ms << "\n";
  }

  static std::string CsvPath() { return "/tmp/cunls_numeric_diff_perf/results.csv"; }

  static void SetUpTestSuite() {
    ::mkdir("/tmp/cunls_numeric_diff_perf", 0755);
    std::ofstream f(CsvPath(), std::ios::trunc);
    f << "problem_type,scale,jacobian_mode,mean_ms_per_build_system\n";
  }

  /**
   * @brief Runs kWarmupIters untimed + kTimedIters CUDA-event-timed
   * BuildSystem calls, wrapped in a per-iteration NVTX range named after
   * (problem_type, scale, mode) so it is identifiable in an nsys timeline.
   * Records the mean elapsed ms to the CSV and returns it.
   */
  double TimeBuildSystem(SystemBuilder &builder, Problem &problem, const PerfParams &p) {
    CudaStream stream;
    builder.Prepare(stream.GetStream(), problem);

    for (int i = 0; i < kWarmupIters; ++i) {
      builder.Build(stream.GetStream(), problem);
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    cudaEvent_t start, stop;
    THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
    THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));

    profiler::Domain domain("NumericDiffPerfTest");
    double total_ms = 0.0;
    for (int i = 0; i < kTimedIters; ++i) {
      auto range = domain.CreateDomainRange(ParamLabel(p) + "/BuildSystem");
      THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream.GetStream()));
      builder.Build(stream.GetStream(), problem);
      THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream.GetStream()));
      THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));
      float ms = 0.f;
      THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
      total_ms += ms;
    }

    THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
    THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));

    double mean_ms = total_ms / kTimedIters;
    AppendCsvRow(ToString(p.problem_type), ToString(p.scale), ToString(p.mode), mean_ms);
    return mean_ms;
  }

  MinimizerOptions MakeOptions(JacobianMode mode) {
    MinimizerOptions options;
    options.jacobian_mode = mode;
    options.disable_safety_checks = true;
    options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
    return options;
  }

  cuBLASHandle cublas_handle_;
};

TEST_P(NumericDiffPerfTest, BuildSystemTiming) {
  const PerfParams p = GetParam();
  SCOPED_TRACE(ParamLabel(p));

  double mean_ms = 0.0;

  switch (p.problem_type) {
    case ProblemType::kPGO: {
      const size_t num_poses = PGONumPoses(p.scale);
      const size_t num_factors = num_poses - 1;
      std::mt19937 rng(1000);
      std::vector<SE3Transform> gt_poses = RandomPoses(num_poses, rng);

      // Deltas satisfying delta_i = T_i^{-1} * T_{i+1} exactly, so the
      // problem is well-posed (not required for a timing-only benchmark, but
      // cheap and keeps BuildSystem's cost path realistic).
      std::vector<SE3Transform> deltas(num_factors);
      {
        // delta = T_i^{-1} * T_{i+1}; computed via device inverse, matching
        // other tests' conventions.
        CudaStream stream;
        dvector<SE3Transform> d_poses(gt_poses);
        dvector<SE3Transform> d_inv(num_poses);
        ComputeInverseSE3(stream.GetStream(), reinterpret_cast<const float *>(d_poses.data()), 4,
                          16, 4, 16, num_poses, reinterpret_cast<float *>(d_inv.data()));
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        std::vector<SE3Transform> inv(num_poses);
        d_inv.CopyToHost(inv.data(), num_poses);
        for (size_t i = 0; i < num_factors; ++i) {
          deltas[i] = ComposeSE3Host(inv[i], gt_poses[i + 1]);
        }
      }

      dvector<SE3Transform> poses_device(gt_poses);
      dvector<SE3Transform> deltas_device(deltas);
      std::vector<int> const_ids = {0};
      dvector<int> const_ids_device(const_ids);

      SE3StateBatch pose_states(cublas_handle_,
                                reinterpret_cast<const float *>(poses_device.data()), num_poses,
                                const_ids_device.data(), 1);
      SE3BetweenFactorBatch between_factor(deltas_device.data(), num_factors);

      std::vector<float *> state_pointers;
      state_pointers.reserve(2 * num_factors);
      for (size_t i = 0; i < num_factors; ++i) {
        state_pointers.push_back(pose_states.StateBlockDevicePtr(i));
        state_pointers.push_back(pose_states.StateBlockDevicePtr(i + 1));
      }

      Problem problem;
      problem.AddStateBatch(&pose_states);
      problem.AddFactorBatch(&between_factor, state_pointers);
      ASSERT_TRUE(problem.CheckConsistency());

      SystemBuilder builder(MakeOptions(p.mode));
      mean_ms = TimeBuildSystem(builder, problem, p);
      break;
    }
    case ProblemType::kSBA: {
      int n_poses = 0, n_points = 0;
      SBAScaleParams(p.scale, n_poses, n_points);
      std::mt19937 rng(2000);
      std::vector<SE3Transform> gt_poses = RandomPoses(n_poses, rng);

      std::uniform_real_distribution<float> xy(-8.f, 8.f);
      std::uniform_real_distribution<float> zz(3.f, 15.f);
      std::vector<Vector<3>> points(n_points);
      for (int i = 0; i < n_points; ++i) points[i] = {xy(rng), xy(rng), zz(rng)};

      // Every point observed by every pose (dense visibility) -- simplest
      // well-posed construction at this synthetic scale.
      std::vector<Vector<2>> observations;
      std::vector<float *> state_pointers;
      observations.reserve(static_cast<size_t>(n_poses) * n_points);
      state_pointers.reserve(2 * static_cast<size_t>(n_poses) * n_points);

      dvector<SE3Transform> poses_device(gt_poses);
      dvector<Vector<3>> points_device(points);
      std::vector<int> const_pose_ids = {0};
      dvector<int> const_pose_ids_device(const_pose_ids);

      SE3StateBatch pose_states(cublas_handle_,
                                reinterpret_cast<const float *>(poses_device.data()), n_poses,
                                const_pose_ids_device.data(), 1);
      VectorStateBatch<3> point_states(reinterpret_cast<const float *>(points_device.data()),
                                       n_points);

      for (int pose_idx = 0; pose_idx < n_poses; ++pose_idx) {
        const SE3Transform &T = gt_poses[pose_idx];
        for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
          const Vector<3> &p = points[pt_idx];
          float pc[3];
          pc[0] = T[3] + T[0] * p[0] + T[1] * p[1] + T[2] * p[2];
          pc[1] = T[7] + T[4] * p[0] + T[5] * p[1] + T[6] * p[2];
          pc[2] = T[11] + T[8] * p[0] + T[9] * p[1] + T[10] * p[2];
          Vector<2> obs{pc[0] / pc[2], pc[1] / pc[2]};
          observations.push_back(obs);
          state_pointers.push_back(pose_states.StateBlockDevicePtr(pose_idx));
          state_pointers.push_back(point_states.StateBlockDevicePtr(pt_idx));
        }
      }

      dvector<Vector<2>> observations_device(observations);
      ReprojectionFactorBatch reproj(observations_device.data(), observations.size(), 1e-3f);

      Problem problem;
      problem.AddStateBatch(&pose_states);
      problem.AddStateBatch(&point_states);
      problem.AddFactorBatch(&reproj, state_pointers);
      ASSERT_TRUE(problem.CheckConsistency());

      SystemBuilder builder(MakeOptions(p.mode));
      mean_ms = TimeBuildSystem(builder, problem, p);
      break;
    }
    case ProblemType::kPnP: {
      const size_t n = PnPNumCorrespondences(p.scale);
      std::mt19937 rng(3000);
      std::vector<SE3Transform> gt_pose_vec = RandomPoses(1, rng);
      const SE3Transform &T = gt_pose_vec[0];

      std::uniform_real_distribution<float> xy(-4.f, 4.f);
      std::uniform_real_distribution<float> zz(3.f, 12.f);
      std::vector<Vector<3>> points(n);
      std::vector<Vector<2>> observations(n);
      for (size_t i = 0; i < n; ++i) {
        Vector<3> p{xy(rng), xy(rng), zz(rng)};
        float pc[3];
        pc[0] = T[3] + T[0] * p[0] + T[1] * p[1] + T[2] * p[2];
        pc[1] = T[7] + T[4] * p[0] + T[5] * p[1] + T[6] * p[2];
        pc[2] = T[11] + T[8] * p[0] + T[9] * p[1] + T[10] * p[2];
        points[i] = p;
        observations[i] = {pc[0] / pc[2], pc[1] / pc[2]};
      }

      dvector<Vector<3>> points_device(points);
      dvector<Vector<2>> observations_device(observations);
      dvector<SE3Transform> pose_device(gt_pose_vec);

      SE3StateBatch pose_states(cublas_handle_,
                                reinterpret_cast<const float *>(pose_device.data()), 1);
      PnPFactorBatch pnp(observations_device.data(), points_device.data(), n, 1e-3f);

      std::vector<float *> state_pointers(n, pose_states.StateBlockDevicePtr(0));

      Problem problem;
      problem.AddStateBatch(&pose_states);
      problem.AddFactorBatch(&pnp, state_pointers);
      ASSERT_TRUE(problem.CheckConsistency());

      SystemBuilder builder(MakeOptions(p.mode));
      mean_ms = TimeBuildSystem(builder, problem, p);
      break;
    }
  }

  EXPECT_GT(mean_ms, 0.0);
  std::cout << "[NumericDiffPerfTest] " << ParamLabel(p) << ": " << mean_ms
            << " ms/BuildSystem\n";
}

std::vector<PerfParams> AllParams() {
  std::vector<PerfParams> out;
  for (ProblemType pt : {ProblemType::kPGO, ProblemType::kSBA, ProblemType::kPnP}) {
    for (Scale s : {Scale::kSmall, Scale::kMedium, Scale::kLarge}) {
      for (JacobianMode m : {JacobianMode::kAnalytic, JacobianMode::kNumeric}) {
        out.push_back({pt, s, m});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(Sweep, NumericDiffPerfTest, ::testing::ValuesIn(AllParams()),
                         [](const ::testing::TestParamInfo<PerfParams> &info) {
                           return ParamLabel(info.param);
                         });

}  // namespace
}  // namespace cunls
