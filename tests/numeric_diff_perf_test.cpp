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
 * problem types (PGO / SBA / PnP) and named problem sizes, timed two ways on
 * the same synthetic problems:
 *
 *  - NumericDiffPerfTest.BuildSystemTiming times repeated
 *    `internal::BuildSystem` calls via CUDA events: `BuildSystem` is exactly
 *    the call that computes residuals + Jacobians and assembles the normal
 *    equations, so it isolates the cost the two Jacobian modes differ on.
 *  - NumericDiffE2EPerfTest.MinimizeTiming (built only with the CMake option
 *    CUNLS_BUILD_NUMERIC_DIFF_E2E_PERF_TEST; slow) times full
 *    `LevenbergMarquardtMinimizer::Minimize()` calls from a perturbed start,
 *    which also fold in the sparse linear solve (independent of JacobianMode
 *    and, per the profiling note in commit e5c4612, dominant in total GPU
 *    time): the realistic end-to-end overhead of numeric differentiation.
 *    LevenbergMarquardtMinimizer is what examples/pose_graph_optimization,
 *    examples/pnp and examples/sparse_bundle_adjustment default to.
 *
 * Modeled on tests/motion_prior_perf_test.cpp (gtest TestWithParam, NVTX
 * instrumentation gated by ENABLE_PROFILING) and the `SystemBuilder` pattern
 * of tests/block_hessian_assembler_test.cpp. Meant to be run under `nsys
 * profile`; also appends CUDA-event timings to CSVs in
 * /tmp/cunls_numeric_diff_perf for offline plotting.
 *
 * Problem sizes (each problem type sweeps 3 named sizes, small->large):
 *  - PGO: 10k / 100k / 1M poses, an SE3 chain (SE3BetweenFactorBatch).
 *  - SBA: (1k poses, 5k landmarks) / (5k poses, 25k landmarks) /
 *    (10k poses, 100k landmarks), via ReprojectionFactorBatch, with a
 *    *sparse* visibility pattern (each landmark observed by a small fixed
 *    number of poses -- see SBAConfig::obs_per_landmark below) rather than a
 *    dense pose x landmark grid. A dense grid at these sizes would produce
 *    an infeasible number of factors (e.g. 10k poses x 100k landmarks would
 *    be 1e9 factors); this codebase's GPU-safety guidance caps any single
 *    FactorBatch at roughly 1.3M factors (much beyond ~6M states in one
 *    batch has been observed to crash the GPU process), so sparse
 *    visibility is required to stay in a safe, realistic regime.
 *  - PnP: 1k / 100k / 1M correspondences (landmarks observed by the single
 *    pose being estimated), via PnPFactorBatch.
 */

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/minimizer.h"
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

/**
 * @brief One system assembly of Minimizer (internal::SetUpStructure +
 * internal::BuildSystem), to time assembly in isolation from the rest of
 * Minimize(). Same pattern as SystemBuilder in
 * tests/block_hessian_assembler_test.cpp.
 */
class SystemBuilder {
 public:
  explicit SystemBuilder(const MinimizerOptions &options) : options_(options) {
    system_.linear_system = internal::CreateLinearSystem(options);
  }

  void Prepare(cudaStream_t stream, Problem &problem) {
    internal::SetUpStructure(stream, problem, options_, state_ops_, system_);
    state_.Recreate(stream, problem);
  }

  void Build(cudaStream_t stream, const Problem &problem) {
    internal::BuildSystem(stream, problem, state_, options_, system_, scratch_);
  }

 private:
  MinimizerOptions options_;
  StateBatchOps state_ops_;
  internal::MinimizerSystem system_;
  internal::MinimizerScratch scratch_;
  MinimizerState state_;
};

// ---------------------------------------------------------------------------
// Named problem sizes.
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

// Sparse SBA visibility: each landmark is observed by `obs_per_landmark`
// poses (one deterministic "primary" pose -- landmark_idx % n_poses, which
// guarantees every pose is observed at least once since n_points >=
// n_poses in every config below -- plus obs_per_landmark - 1 further
// distinct poses chosen pseudo-randomly). This keeps total factor counts
// (n_points * obs_per_landmark) far below the ~1.3M/FactorBatch safety
// limit for all three sizes while giving every landmark >1 observation
// (needed to constrain its 3D position) and every pose several
// observations, unlike a dense pose x landmark grid which would be
// infeasible at these sizes.
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

// ---------------------------------------------------------------------------
// Synthetic problems (kept minimal; correctness of these factor types is
// covered by synthetic_pgo_test.cpp / synthetic_sba_test.cpp /
// pnp_factor_batch_test.cpp -- this file only needs *some* well-posed problem
// of the right scale).
// ---------------------------------------------------------------------------

SE3Transform ComposeSE3Host(const SE3Transform &a, const SE3Transform &b) {
  SE3Transform c{};
  for (int r = 0; r < 4; ++r)
    for (int cix = 0; cix < 4; ++cix)
      for (int k = 0; k < 4; ++k) c[r * 4 + cix] += a[r * 4 + k] * b[k * 4 + cix];
  return c;
}

/** Exp of n twists with rotation in ±rot, translation in ±trans (z offset by z_offset). */
std::vector<SE3Transform> ExpOfRandomTwists(size_t n, std::mt19937 &rng, float rot_mag,
                                            float trans_mag, float z_offset, size_t skip_first_n) {
  std::uniform_real_distribution<float> rot(-rot_mag, rot_mag);
  std::uniform_real_distribution<float> trans(-trans_mag, trans_mag);
  std::vector<Vector<6>> twists(n, Vector<6>{0, 0, 0, 0, 0, 0});
  for (size_t i = skip_first_n; i < n; ++i) {
    twists[i] = {rot(rng), rot(rng), rot(rng), trans(rng), trans(rng), z_offset + trans(rng)};
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

std::vector<SE3Transform> RandomPoses(size_t n, std::mt19937 &rng) {
  return ExpOfRandomTwists(n, rng, 0.3f, 1.0f, 8.0f, 0);
}

/**
 * Perturbs each pose in `gt_poses` by a small random right-multiplied twist
 * (poses[skip_first_n:] only, so an anchor pose used as a constant state
 * stays exactly at ground truth).
 */
std::vector<SE3Transform> PerturbPoses(const std::vector<SE3Transform> &gt_poses, std::mt19937 &rng,
                                       size_t skip_first_n = 0) {
  const auto disturb = ExpOfRandomTwists(gt_poses.size(), rng, 0.05f, 0.1f, 0.0f, skip_first_n);
  std::vector<SE3Transform> init_poses(gt_poses.size());
  for (size_t i = 0; i < gt_poses.size(); ++i)
    init_poses[i] = ComposeSE3Host(gt_poses[i], disturb[i]);
  return init_poses;
}

Vector<2> Project(const SE3Transform &T, const Vector<3> &p) {
  const float x = T[3] + T[0] * p[0] + T[1] * p[1] + T[2] * p[2];
  const float y = T[7] + T[4] * p[0] + T[5] * p[1] + T[6] * p[2];
  const float z = T[11] + T[8] * p[0] + T[9] * p[1] + T[10] * p[2];
  return {x / z, y / z};
}

/**
 * @brief The synthetic problem of one PerfParams. Measurements come from the
 * ground truth; with `perturbed`, the free states start off it (pose 0 of
 * PGO / SBA is the constant anchor and stays exact), so a full Minimize()
 * iterates and solves instead of hitting the initial_cost < cost_tolerance
 * early exit. ResetStates() re-uploads the initial values.
 */
class SyntheticProblem {
 public:
  SyntheticProblem(const PerfParams &p, bool perturbed) {
    switch (p.problem_type) {
      case ProblemType::kPGO:
        BuildPGO(kPGOConfigs[p.size_index].num_poses, perturbed);
        break;
      case ProblemType::kSBA:
        BuildSBA(kSBAConfigs[p.size_index], perturbed);
        break;
      case ProblemType::kPnP:
        BuildPnP(kPnPConfigs[p.size_index].n, perturbed);
        break;
    }
  }

  void ResetStates() {
    poses_.CopyFromHost(init_poses_.data(), init_poses_.size());
    if (!init_points_.empty()) points_.CopyFromHost(init_points_.data(), init_points_.size());
  }

  Problem problem;
  size_t n_primary = 0, n_secondary = 0;

 private:
  void BuildPGO(size_t num_poses, bool perturbed) {
    const size_t num_factors = num_poses - 1;
    std::mt19937 rng(1000);
    const std::vector<SE3Transform> gt_poses = RandomPoses(num_poses, rng);
    // Deltas satisfying delta_i = T_i^{-1} * T_{i+1} exactly (device inverse).
    std::vector<SE3Transform> deltas(num_factors);
    {
      CudaStream stream;
      dvector<SE3Transform> d_poses(gt_poses);
      dvector<SE3Transform> d_inv(num_poses);
      ComputeInverseSE3(stream.GetStream(), reinterpret_cast<const float *>(d_poses.data()), 4, 16,
                        4, 16, num_poses, reinterpret_cast<float *>(d_inv.data()));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
      std::vector<SE3Transform> inv(num_poses);
      d_inv.CopyToHost(inv.data(), num_poses);
      for (size_t i = 0; i < num_factors; ++i) deltas[i] = ComposeSE3Host(inv[i], gt_poses[i + 1]);
    }
    init_poses_ = perturbed ? PerturbPoses(gt_poses, rng, /*skip_first_n=*/1) : gt_poses;
    poses_ = dvector<SE3Transform>(init_poses_);
    deltas_ = dvector<SE3Transform>(deltas);
    const_ids_ = dvector<int>(std::vector<int>{0});
    pose_states_ = std::make_unique<SE3StateBatch>(reinterpret_cast<const float *>(poses_.data()),
                                                   num_poses, const_ids_.data(), 1);
    pose_states_->SetNumActiveStates(num_poses, 1);
    factor_ = std::make_unique<SE3BetweenFactorBatch>(deltas_.data(), num_factors);
    factor_->SetNumActiveFactors(num_factors);

    std::vector<float *> state_pointers;
    state_pointers.reserve(2 * num_factors);
    for (size_t i = 0; i < num_factors; ++i) {
      state_pointers.push_back(pose_states_->StateDevicePtr(i));
      state_pointers.push_back(pose_states_->StateDevicePtr(i + 1));
    }
    problem.AddStateBatch(pose_states_.get());
    problem.AddFactorBatch(factor_.get(), state_pointers);
    n_primary = num_poses;
  }

  void BuildSBA(const SBAConfig &cfg, bool perturbed) {
    const int n_poses = cfg.n_poses, n_points = cfg.n_points;
    std::mt19937 rng(2000);
    const std::vector<SE3Transform> gt_poses = RandomPoses(n_poses, rng);
    std::uniform_real_distribution<float> xy(-8.f, 8.f);
    std::uniform_real_distribution<float> zz(3.f, 15.f);
    std::vector<Vector<3>> points(n_points);
    for (int i = 0; i < n_points; ++i) points[i] = {xy(rng), xy(rng), zz(rng)};

    // Sparse visibility (see SBAConfig): one deterministic primary pose plus
    // (obs_per_landmark - 1) further distinct random poses per landmark.
    std::uniform_int_distribution<int> pose_pick(0, n_poses - 1);
    std::vector<std::pair<int, int>> pose_point;  // per observation
    std::vector<Vector<2>> observations;
    std::vector<int> chosen_poses;
    for (int pt_idx = 0; pt_idx < n_points; ++pt_idx) {
      chosen_poses.assign(1, pt_idx % n_poses);
      while (static_cast<int>(chosen_poses.size()) < std::min(cfg.obs_per_landmark, n_poses)) {
        const int candidate = pose_pick(rng);
        if (std::find(chosen_poses.begin(), chosen_poses.end(), candidate) == chosen_poses.end()) {
          chosen_poses.push_back(candidate);
        }
      }
      for (int pose_idx : chosen_poses) {
        observations.push_back(Project(gt_poses[pose_idx], points[pt_idx]));
        pose_point.emplace_back(pose_idx, pt_idx);
      }
    }

    init_poses_ = perturbed ? PerturbPoses(gt_poses, rng, /*skip_first_n=*/1) : gt_poses;
    init_points_ = points;
    if (perturbed) {
      std::uniform_real_distribution<float> trans(-0.1f, 0.1f);
      for (auto &q : init_points_) q = {q[0] + trans(rng), q[1] + trans(rng), q[2] + trans(rng)};
    }
    poses_ = dvector<SE3Transform>(init_poses_);
    points_ = dvector<Vector<3>>(init_points_);
    const_ids_ = dvector<int>(std::vector<int>{0});
    pose_states_ = std::make_unique<SE3StateBatch>(reinterpret_cast<const float *>(poses_.data()),
                                                   n_poses, const_ids_.data(), 1);
    pose_states_->SetNumActiveStates(n_poses, 1);
    point_states_ = std::make_unique<VectorStateBatch<3>>(
        reinterpret_cast<const float *>(points_.data()), n_points);
    point_states_->SetNumActiveStates(n_points);
    observations_ = dvector<Vector<2>>(observations);
    factor_ =
        std::make_unique<ReprojectionFactorBatch>(observations_.data(), observations.size(), 1e-3f);
    factor_->SetNumActiveFactors(observations.size());

    std::vector<float *> state_pointers;
    state_pointers.reserve(2 * pose_point.size());
    for (const auto &[pose_idx, pt_idx] : pose_point) {
      state_pointers.push_back(pose_states_->StateDevicePtr(pose_idx));
      state_pointers.push_back(point_states_->StateDevicePtr(pt_idx));
    }
    problem.AddStateBatch(pose_states_.get());
    problem.AddStateBatch(point_states_.get());
    problem.AddFactorBatch(factor_.get(), state_pointers);
    n_primary = n_poses;
    n_secondary = n_points;
  }

  void BuildPnP(size_t n, bool perturbed) {
    std::mt19937 rng(3000);
    const std::vector<SE3Transform> gt_pose = RandomPoses(1, rng);
    std::uniform_real_distribution<float> xy(-4.f, 4.f);
    std::uniform_real_distribution<float> zz(3.f, 12.f);
    std::vector<Vector<3>> landmarks(n);
    std::vector<Vector<2>> observations(n);
    for (size_t i = 0; i < n; ++i) {
      landmarks[i] = {xy(rng), xy(rng), zz(rng)};
      observations[i] = Project(gt_pose[0], landmarks[i]);
    }
    // The pose is fully optimizable (no constant ids here).
    init_poses_ = perturbed ? PerturbPoses(gt_pose, rng) : gt_pose;
    poses_ = dvector<SE3Transform>(init_poses_);
    landmarks_ = dvector<Vector<3>>(landmarks);
    observations_ = dvector<Vector<2>>(observations);
    pose_states_ =
        std::make_unique<SE3StateBatch>(reinterpret_cast<const float *>(poses_.data()), 1);
    pose_states_->SetNumActiveStates(1);
    factor_ = std::make_unique<PnPFactorBatch>(observations_.data(), landmarks_.data(), n, 1e-3f);
    factor_->SetNumActiveFactors(n);
    problem.AddStateBatch(pose_states_.get());
    problem.AddFactorBatch(factor_.get(), std::vector<float *>(n, pose_states_->StateDevicePtr(0)));
    n_primary = n;
  }

  std::vector<SE3Transform> init_poses_;
  std::vector<Vector<3>> init_points_;
  dvector<SE3Transform> poses_, deltas_;
  dvector<Vector<3>> points_, landmarks_;
  dvector<Vector<2>> observations_;
  dvector<int> const_ids_;
  std::unique_ptr<SE3StateBatch> pose_states_;
  std::unique_ptr<VectorStateBatch<3>> point_states_;
  std::unique_ptr<FactorBatch> factor_;
};

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

constexpr int kWarmupIters = 3;
constexpr int kTimedIters = 10;
const char *const kCsvDir = "/tmp/cunls_numeric_diff_perf";

/** Starts `path` fresh with `header`. */
void StartCsv(const std::string &path, const std::string &header) {
  ::mkdir(kCsvDir, 0755);
  std::ofstream(path, std::ios::trunc) << header << "\n";
}

/** Appends `p`'s row (sizes, mode, mean ms, then `extra` columns) to `path`. */
void AppendCsvRow(const std::string &path, const PerfParams &p, const SyntheticProblem &s,
                  double mean_ms, const std::string &extra = "") {
  std::ofstream(path, std::ios::app)
      << ToString(p.problem_type) << "," << SizeLabel(p.problem_type, p.size_index) << ","
      << p.size_index << "," << s.n_primary << "," << s.n_secondary << "," << ToString(p.mode)
      << "," << mean_ms << extra << "\n";
}

/**
 * Mean CUDA-event time (ms) of `run`: kWarmupIters untimed calls, then
 * kTimedIters timed ones, each in an NVTX range `range` of `domain`.
 * `before` runs untimed ahead of every call.
 */
double MeanMs(cudaStream_t stream, const char *domain_name, const std::string &range,
              const std::function<void()> &before, const std::function<void()> &run) {
  for (int i = 0; i < kWarmupIters; ++i) {
    before();
    run();
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
  THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));
  profiler::Domain domain(domain_name);
  double total_ms = 0.0;
  for (int i = 0; i < kTimedIters; ++i) {
    before();
    auto nvtx_range = domain.CreateDomainRange(range);
    THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream));
    run();
    THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream));
    THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));
    float ms = 0.f;
    THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }
  THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
  THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));
  return total_ms / kTimedIters;
}

MinimizerOptions MakeOptions(JacobianMode mode) {
  MinimizerOptions options;
  options.jacobian_mode = mode;
  options.disable_safety_checks = true;
  options.sparse_linear_solver_type = test_utils::SolverTypeFromEnv();
  return options;
}

class NumericDiffPerfTest : public ::testing::TestWithParam<PerfParams> {
 protected:
  static std::string CsvPath() { return std::string(kCsvDir) + "/results.csv"; }
  static void SetUpTestSuite() {
    StartCsv(CsvPath(),
             "problem_type,size_label,size_index,n_primary,n_secondary,jacobian_mode,"
             "mean_ms_per_build_system");
  }
};

TEST_P(NumericDiffPerfTest, BuildSystemTiming) {
  const PerfParams p = GetParam();
  SCOPED_TRACE(ParamLabel(p));
  SyntheticProblem s(p, /*perturbed=*/false);
  ASSERT_TRUE(s.problem.CheckConsistency());

  CudaStream stream;
  SystemBuilder builder(MakeOptions(p.mode));
  builder.Prepare(stream.GetStream(), s.problem);
  const double mean_ms = MeanMs(
      stream.GetStream(), "NumericDiffPerfTest", ParamLabel(p) + "/BuildSystem", [] {},
      [&] { builder.Build(stream.GetStream(), s.problem); });
  AppendCsvRow(CsvPath(), p, s, mean_ms);

  EXPECT_GT(mean_ms, 0.0);
  std::cout << "[NumericDiffPerfTest] " << ParamLabel(p) << ": " << mean_ms << " ms/BuildSystem\n";
}

#ifdef CUNLS_NUMERIC_DIFF_E2E_PERF_TEST  // CMake option CUNLS_BUILD_NUMERIC_DIFF_E2E_PERF_TEST
                                         // (slow)
class NumericDiffE2EPerfTest : public ::testing::TestWithParam<PerfParams> {
 protected:
  static std::string CsvPath() { return std::string(kCsvDir) + "/e2e_results.csv"; }
  static void SetUpTestSuite() {
    StartCsv(CsvPath(),
             "problem_type,size_label,size_index,n_primary,n_secondary,jacobian_mode,"
             "mean_ms_per_minimize,num_iterations_to_convergence");
  }
};

// Each call uses a fresh minimizer and first re-uploads the initial states:
// Minimize() writes the converged state back into the state batches' buffers,
// so otherwise every call after the first would start converged and finish in
// ~1 iteration.
TEST_P(NumericDiffE2EPerfTest, MinimizeTiming) {
  const PerfParams p = GetParam();
  SCOPED_TRACE(ParamLabel(p));
  SyntheticProblem s(p, /*perturbed=*/true);
  ASSERT_TRUE(s.problem.CheckConsistency());

  CudaStream stream;
  LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = MakeOptions(p.mode);
  std::unique_ptr<LevenbergMarquardtMinimizer> minimizer;
  MinimizerSummary summary;
  const double mean_ms = MeanMs(
      stream.GetStream(), "NumericDiffE2EPerfTest", ParamLabel(p) + "/Minimize",
      [&] {
        s.ResetStates();
        minimizer = std::make_unique<LevenbergMarquardtMinimizer>(lm_options);
      },
      [&] { summary = minimizer->Minimize(stream.GetStream(), s.problem); });
  AppendCsvRow(CsvPath(), p, s, mean_ms, "," + std::to_string(summary.num_iterations));

  EXPECT_GT(mean_ms, 0.0);
  std::cout << "[NumericDiffE2EPerfTest] " << ParamLabel(p) << ": " << mean_ms << " ms/Minimize ("
            << summary.num_iterations << " iterations)\n";
}
#endif

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

std::string ParamName(const ::testing::TestParamInfo<PerfParams> &info) {
  return ParamLabel(info.param);
}

INSTANTIATE_TEST_SUITE_P(Sweep, NumericDiffPerfTest, ::testing::ValuesIn(AllParams()), ParamName);
#ifdef CUNLS_NUMERIC_DIFF_E2E_PERF_TEST
INSTANTIATE_TEST_SUITE_P(Sweep, NumericDiffE2EPerfTest, ::testing::ValuesIn(AllParams()),
                         ParamName);
#endif

}  // namespace
}  // namespace cunls
