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
 * @file numeric_diff_e2e_perf_test.cpp
 * @brief Wall-clock comparison of JacobianMode::kAnalytic vs kNumeric across
 * problem types (PGO / SBA / PnP) and named problem sizes, this time timing
 * full `LevenbergMarquardtMinimizer::Minimize()` calls rather than just
 * `GaussNewtonMinimizer::BuildSystem()` (tests/numeric_diff_perf_test.cpp).
 *
 * The synthetic problem generators/sizes here are copy-identical to
 * numeric_diff_perf_test.cpp so the two benchmarks are directly comparable
 * in problem definition -- only what's timed differs. `Minimize()` also
 * folds in the sparse linear solve, which is independent of JacobianMode
 * and (per the profiling note in commit e5c4612) dominates total GPU time;
 * this file exists to show the realistic end-to-end overhead of picking
 * numeric diff for a real solve, complementing (not replacing) the
 * BuildSystem-only isolation benchmark.
 *
 * Modeled on tests/motion_prior_perf_test.cpp's structure (gtest
 * TestWithParam, NVTX instrumentation) and reuses LevenbergMarquardtMinimizer
 * since that's what examples/pose_graph_optimization, examples/pnp and
 * examples/sparse_bundle_adjustment all default to.
 *
 * See the size-selection comment in numeric_diff_perf_test.cpp for how the
 * PGO / SBA / PnP problem sizes (and, for SBA, the sparse visibility
 * pattern) were chosen; the same reasoning/config tables are duplicated
 * here verbatim.
 */

#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
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
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

enum class ProblemType { kPGO, kSBA, kPnP };

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

const char *ToString(JacobianMode m) {
  return m == JacobianMode::kAnalytic ? "analytic" : "numeric";
}

// ---------------------------------------------------------------------------
// Synthetic dataset generators -- copy-identical to numeric_diff_perf_test.cpp
// so the two benchmarks time the exact same problems.
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

// Perturbs each pose in `gt_poses` by a small random right-multiplied twist
// (poses[skip_first_n:] only, so an anchor pose used as a fixed/constant
// state block stays exactly at ground truth). Used to build an initial
// state that is *not* already at the ground-truth optimum: the
// BuildSystem-only benchmark (numeric_diff_perf_test.cpp) doesn't care
// about this since it never runs the solve loop, but a full end-to-end
// Minimize() benchmark does -- an exact-ground-truth start converges in 0
// iterations (Minimize's `initial_cost < cost_tolerance` early exit) and
// never exercises the linear solve this benchmark exists to include.
std::vector<SE3Transform> PerturbPoses(const std::vector<SE3Transform> &gt_poses, std::mt19937 &rng,
                                       size_t skip_first_n = 0) {
  const size_t n = gt_poses.size();
  std::uniform_real_distribution<float> rot(-0.05f, 0.05f);
  std::uniform_real_distribution<float> trans(-0.1f, 0.1f);
  std::vector<Vector<6>> twists(n, Vector<6>{0, 0, 0, 0, 0, 0});
  for (size_t i = skip_first_n; i < n; ++i) {
    twists[i] = {rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), trans(rng)};
  }
  CudaStream stream;
  dvector<Vector<6>> d_twists(twists);
  dvector<SE3Transform> d_disturb(n);
  ComputeExpSE3(stream.GetStream(), reinterpret_cast<const float *>(d_twists.data()), 6, 4, 16, n,
                reinterpret_cast<float *>(d_disturb.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<SE3Transform> disturb(n);
  d_disturb.CopyToHost(disturb.data(), n);

  std::vector<SE3Transform> init_poses(n);
  for (size_t i = 0; i < n; ++i) init_poses[i] = ComposeSE3Host(gt_poses[i], disturb[i]);
  return init_poses;
}

Vector<3> PerturbPoint(const Vector<3> &p, std::mt19937 &rng) {
  std::uniform_real_distribution<float> trans(-0.1f, 0.1f);
  return {p[0] + trans(rng), p[1] + trans(rng), p[2] + trans(rng)};
}

// ---------------------------------------------------------------------------
// Named problem sizes -- see numeric_diff_perf_test.cpp for full rationale.
// ---------------------------------------------------------------------------

struct PGOConfig {
  size_t num_poses;
  const char *size_label;
};
constexpr std::array<PGOConfig, 3> kPGOConfigs = {{
    {10000, "10k_poses"},
    {100000, "100k_poses"},
    {1000000, "1M_poses"},
}};

struct SBAConfig {
  int n_poses;
  int n_points;
  int obs_per_landmark;
  const char *size_label;
};
constexpr std::array<SBAConfig, 3> kSBAConfigs = {{
    {1000, 5000, 3, "1kposes_5klandmarks"},
    {5000, 25000, 3, "5kposes_25klandmarks"},
    {10000, 100000, 3, "10kposes_100klandmarks"},
}};

struct PnPConfig {
  size_t n;
  const char *size_label;
};
constexpr std::array<PnPConfig, 3> kPnPConfigs = {{
    {1000, "1k_correspondences"},
    {100000, "100k_correspondences"},
    {1000000, "1M_correspondences"},
}};

std::string SizeLabel(ProblemType pt, int size_index) {
  switch (pt) {
    case ProblemType::kPGO:
      return kPGOConfigs[size_index].size_label;
    case ProblemType::kSBA:
      return kSBAConfigs[size_index].size_label;
    case ProblemType::kPnP:
      return kPnPConfigs[size_index].size_label;
  }
  return "?";
}

struct PerfParams {
  ProblemType problem_type;
  int size_index;  // 0, 1, 2 -- indexes into the per-problem-type config table above.
  JacobianMode mode;
};

std::string ParamLabel(const PerfParams &p) {
  return std::string(ToString(p.problem_type)) + "_" + SizeLabel(p.problem_type, p.size_index) +
         "_" + ToString(p.mode);
}

std::ostream &operator<<(std::ostream &os, const PerfParams &p) { return os << ParamLabel(p); }

class NumericDiffE2EPerfTest : public ::testing::TestWithParam<PerfParams> {
 protected:
  // Same iteration counts as numeric_diff_perf_test.cpp, for consistency
  // between the two benchmarks.
  static constexpr int kWarmupIters = 3;
  static constexpr int kTimedIters = 10;

  static void AppendCsvRow(const std::string &problem_type, const std::string &size_label,
                           int size_index, size_t n_primary, size_t n_secondary,
                           const std::string &mode, double mean_ms, size_t num_iterations) {
    std::ofstream f(CsvPath(), std::ios::app);
    f << problem_type << "," << size_label << "," << size_index << "," << n_primary << ","
      << n_secondary << "," << mode << "," << mean_ms << "," << num_iterations << "\n";
  }

  static std::string CsvPath() { return "/tmp/cunls_numeric_diff_perf/e2e_results.csv"; }

  static void SetUpTestSuite() {
    ::mkdir("/tmp/cunls_numeric_diff_perf", 0755);
    std::ofstream f(CsvPath(), std::ios::trunc);
    f << "problem_type,size_label,size_index,n_primary,n_secondary,jacobian_mode,"
         "mean_ms_per_minimize,num_iterations_to_convergence\n";
  }

  /**
   * @brief Runs kWarmupIters untimed + kTimedIters CUDA-event-timed full
   * `Minimize()` calls (a fresh minimizer instance per call), wrapped in a
   * per-iteration NVTX range named after (problem_type, size, mode).
   * Records the mean elapsed ms and the last run's iteration count to the
   * CSV.
   *
   * `Minimize()` writes the converged state back into the state batches'
   * backing device buffers (GaussNewtonMinimizer::Minimize's final `Copy(
   * stream, current_state_, problem)`), so without `reset_state` every call
   * after the first would start from the previous call's converged (or
   * near-converged) state and trivially finish in ~1 iteration. `reset_state`
   * re-uploads the original (unconverged) initial values before every
   * warmup and timed call so each run solves the exact same problem.
   */
  double TimeMinimize(const LevenbergMarquardtMinimizerOptions &lm_options, Problem &problem,
                      const PerfParams &p, size_t n_primary, size_t n_secondary,
                      const std::function<void()> &reset_state, size_t &num_iterations) {
    CudaStream stream;

    for (int i = 0; i < kWarmupIters; ++i) {
      reset_state();
      LevenbergMarquardtMinimizer minimizer(lm_options);
      minimizer.Minimize(stream.GetStream(), problem);
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    cudaEvent_t start, stop;
    THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
    THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));

    profiler::Domain domain("NumericDiffE2EPerfTest");
    double total_ms = 0.0;
    MinimizerSummary summary;
    for (int i = 0; i < kTimedIters; ++i) {
      reset_state();
      LevenbergMarquardtMinimizer minimizer(lm_options);
      auto range = domain.CreateDomainRange(ParamLabel(p) + "/Minimize");
      THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream.GetStream()));
      summary = minimizer.Minimize(stream.GetStream(), problem);
      THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream.GetStream()));
      THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));
      float ms = 0.f;
      THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
      total_ms += ms;
    }

    THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
    THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));

    double mean_ms = total_ms / kTimedIters;
    num_iterations = summary.num_iterations;
    AppendCsvRow(ToString(p.problem_type), SizeLabel(p.problem_type, p.size_index), p.size_index,
                 n_primary, n_secondary, ToString(p.mode), mean_ms, num_iterations);
    return mean_ms;
  }

  LevenbergMarquardtMinimizerOptions MakeOptions(JacobianMode mode) {
    MinimizerOptions options;
    options.jacobian_mode = mode;
    options.disable_safety_checks = true;
    options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();

    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    return lm_options;
  }

  cuBLASHandle cublas_handle_;
};

TEST_P(NumericDiffE2EPerfTest, MinimizeTiming) {
  const PerfParams p = GetParam();
  SCOPED_TRACE(ParamLabel(p));

  double mean_ms = 0.0;
  size_t num_iterations = 0;

  switch (p.problem_type) {
    case ProblemType::kPGO: {
      const size_t num_poses = kPGOConfigs[p.size_index].num_poses;
      const size_t num_factors = num_poses - 1;
      std::mt19937 rng(1000);
      std::vector<SE3Transform> gt_poses = RandomPoses(num_poses, rng);

      std::vector<SE3Transform> deltas(num_factors);
      {
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

      // Initial state (what the state batch is constructed from) is
      // perturbed off ground truth -- pose 0 is the anchor/constant state,
      // so it's left exact -- so Minimize() actually has to iterate/solve
      // instead of hitting the initial_cost < cost_tolerance early exit.
      std::vector<SE3Transform> init_poses = PerturbPoses(gt_poses, rng, /*skip_first_n=*/1);

      dvector<SE3Transform> poses_device(init_poses);
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

      auto reset_state = [&]() { poses_device.CopyFromHost(init_poses.data(), num_poses); };
      mean_ms =
          TimeMinimize(MakeOptions(p.mode), problem, p, num_poses, 0, reset_state, num_iterations);
      break;
    }
    case ProblemType::kSBA: {
      const SBAConfig &cfg = kSBAConfigs[p.size_index];
      const int n_poses = cfg.n_poses;
      const int n_points = cfg.n_points;
      const int obs_per_landmark = cfg.obs_per_landmark;
      std::mt19937 rng(2000);
      std::vector<SE3Transform> gt_poses = RandomPoses(n_poses, rng);

      std::uniform_real_distribution<float> xy(-8.f, 8.f);
      std::uniform_real_distribution<float> zz(3.f, 15.f);
      std::vector<Vector<3>> points(n_points);
      for (int i = 0; i < n_points; ++i) points[i] = {xy(rng), xy(rng), zz(rng)};

      // Sparse visibility -- see numeric_diff_perf_test.cpp / the SBAConfig
      // comment there for the full rationale (dense pose x landmark grid is
      // infeasible at these sizes; this stays well under the ~1.3M
      // factors/FactorBatch GPU-safety limit while guaranteeing every pose
      // and landmark is observed).
      std::uniform_int_distribution<int> pose_pick(0, n_poses - 1);

      // State batches are initialized from perturbed poses/points (pose 0 is
      // the anchor/constant state, left exact); observations below are
      // still computed from the exact ground truth, so the problem is
      // well-posed and Minimize() has real work to do instead of hitting
      // the initial_cost < cost_tolerance early exit.
      std::vector<SE3Transform> init_poses = PerturbPoses(gt_poses, rng, /*skip_first_n=*/1);
      std::vector<Vector<3>> init_points(n_points);
      for (int i = 0; i < n_points; ++i) init_points[i] = PerturbPoint(points[i], rng);

      dvector<SE3Transform> poses_device(init_poses);
      dvector<Vector<3>> points_device(init_points);
      std::vector<int> const_pose_ids = {0};
      dvector<int> const_pose_ids_device(const_pose_ids);

      SE3StateBatch pose_states(cublas_handle_,
                                reinterpret_cast<const float *>(poses_device.data()), n_poses,
                                const_pose_ids_device.data(), 1);
      VectorStateBatch<3> point_states(reinterpret_cast<const float *>(points_device.data()),
                                       n_points);

      std::vector<Vector<2>> observations;
      std::vector<float *> state_pointers;
      const size_t expected_factors = static_cast<size_t>(n_points) * obs_per_landmark;
      observations.reserve(expected_factors);
      state_pointers.reserve(2 * expected_factors);

      std::vector<int> chosen_poses;
      chosen_poses.reserve(obs_per_landmark);
      for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
        chosen_poses.clear();
        chosen_poses.push_back(pt_idx % n_poses);
        while (static_cast<int>(chosen_poses.size()) < obs_per_landmark &&
               static_cast<int>(chosen_poses.size()) < n_poses) {
          int candidate = pose_pick(rng);
          if (std::find(chosen_poses.begin(), chosen_poses.end(), candidate) ==
              chosen_poses.end()) {
            chosen_poses.push_back(candidate);
          }
        }

        const Vector<3> &pt = points[pt_idx];
        for (int pose_idx : chosen_poses) {
          const SE3Transform &T = gt_poses[pose_idx];
          float pc[3];
          pc[0] = T[3] + T[0] * pt[0] + T[1] * pt[1] + T[2] * pt[2];
          pc[1] = T[7] + T[4] * pt[0] + T[5] * pt[1] + T[6] * pt[2];
          pc[2] = T[11] + T[8] * pt[0] + T[9] * pt[1] + T[10] * pt[2];
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

      auto reset_state = [&]() {
        poses_device.CopyFromHost(init_poses.data(), n_poses);
        points_device.CopyFromHost(init_points.data(), n_points);
      };
      mean_ms = TimeMinimize(MakeOptions(p.mode), problem, p, n_poses, n_points, reset_state,
                             num_iterations);
      break;
    }
    case ProblemType::kPnP: {
      const size_t n = kPnPConfigs[p.size_index].n;
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

      // Pose is fully optimizable (no constant ids here), so perturb it off
      // ground truth -- otherwise Minimize() starts exactly at the optimum
      // (zero residual) and hits the initial_cost < cost_tolerance early
      // exit without ever running the linear solve.
      std::vector<SE3Transform> init_pose_vec = PerturbPoses(gt_pose_vec, rng);

      dvector<Vector<3>> points_device(points);
      dvector<Vector<2>> observations_device(observations);
      dvector<SE3Transform> pose_device(init_pose_vec);

      SE3StateBatch pose_states(cublas_handle_, reinterpret_cast<const float *>(pose_device.data()),
                                1);
      PnPFactorBatch pnp(observations_device.data(), points_device.data(), n, 1e-3f);

      std::vector<float *> state_pointers(n, pose_states.StateBlockDevicePtr(0));

      Problem problem;
      problem.AddStateBatch(&pose_states);
      problem.AddFactorBatch(&pnp, state_pointers);
      ASSERT_TRUE(problem.CheckConsistency());

      auto reset_state = [&]() { pose_device.CopyFromHost(init_pose_vec.data(), 1); };
      mean_ms = TimeMinimize(MakeOptions(p.mode), problem, p, n, 0, reset_state, num_iterations);
      break;
    }
  }

  EXPECT_GT(mean_ms, 0.0);
  std::cout << "[NumericDiffE2EPerfTest] " << ParamLabel(p) << ": " << mean_ms << " ms/Minimize ("
            << num_iterations << " iterations)\n";
}

std::vector<PerfParams> AllParams() {
  std::vector<PerfParams> out;
  for (ProblemType pt : {ProblemType::kPGO, ProblemType::kSBA, ProblemType::kPnP}) {
    for (int size_index = 0; size_index < 3; ++size_index) {
      for (JacobianMode m : {JacobianMode::kAnalytic, JacobianMode::kNumeric}) {
        out.push_back({pt, size_index, m});
      }
    }
  }
  return out;
}

INSTANTIATE_TEST_SUITE_P(Sweep, NumericDiffE2EPerfTest, ::testing::ValuesIn(AllParams()),
                         [](const ::testing::TestParamInfo<PerfParams> &info) {
                           return ParamLabel(info.param);
                         });

}  // namespace
}  // namespace cunls
