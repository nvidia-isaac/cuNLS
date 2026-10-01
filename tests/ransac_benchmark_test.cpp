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
 * @file ransac_benchmark_test.cpp
 * @brief Benchmarks of the RANSAC minimizers against the regular minimizers:
 *   1. speed at scale (PnP size sweep, hypothesis-count sweep, multi-camera rig),
 *   2. robustness to outliers (success rate vs outlier ratio and initial error),
 *   3. outlier detection quality (precision / recall of the inlier mask).
 *
 * Skipped unless CUNLS_RANSAC_BENCHMARK=1. Results are printed as markdown
 * tables and written as CSV to /tmp/cunls_ransac_bench/.
 * CUNLS_RANSAC_BENCH_TRIALS overrides the number of trials per robustness cell.
 */

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace {

using ransac_test::Compose;
using ransac_test::ExpSE3;
using ransac_test::PnPScene;
using ransac_test::RandomTwist;
using ransac_test::RotationErrorDeg;
using ransac_test::TranslationError;

constexpr double kNoise = 1e-3;  // ~0.5 px at f = 500
constexpr double kTau = 5e-3;    // inlier threshold (5 sigma)
constexpr double kMinOutlier = 2e-2;

bool Enabled() {
  const char *e = std::getenv("CUNLS_RANSAC_BENCHMARK");
  return e != nullptr && std::string(e) == "1";
}

int Trials(int fallback) {
  const char *e = std::getenv("CUNLS_RANSAC_BENCH_TRIALS");
  return e != nullptr ? std::max(1, std::atoi(e)) : fallback;
}

/** Optional filters for profiling one configuration: CUNLS_RANSAC_BENCH_POINTS / _METHOD. */
bool Selected(size_t points, const char *label) {
  const char *p = std::getenv("CUNLS_RANSAC_BENCH_POINTS");
  const char *m = std::getenv("CUNLS_RANSAC_BENCH_METHOD");
  return (p == nullptr || std::stoull(p) == points) && (m == nullptr || std::string(m) == label);
}

std::string OutDir() {
  ::mkdir("/tmp/cunls_ransac_bench", 0755);
  return "/tmp/cunls_ransac_bench/";
}

enum class Method { kGN, kLM, kLMHuber, kLMCauchy, kRansacGN, kRansacLM };

const char *Name(Method m) {
  switch (m) {
    case Method::kGN:
      return "GN";
    case Method::kLM:
      return "LM";
    case Method::kLMHuber:
      return "LM+Huber";
    case Method::kLMCauchy:
      return "LM+Cauchy";
    case Method::kRansacGN:
      return "RANSAC-GN";
    case Method::kRansacLM:
      return "RANSAC-LM";
  }
  return "?";
}

bool IsRansac(Method m) { return m == Method::kRansacGN || m == Method::kRansacLM; }

/**
 * R rig cameras (R free SE3 poses), one PnP batch per camera, an always-on
 * SE3 between factor between consecutive cameras. R = 1 is plain PnP.
 */
struct RigProblem {
  std::vector<PnPScene> scenes;
  std::vector<SE3Transform> init;
  cuBLASHandle cublas;
  dvector<SE3Transform> poses;
  std::unique_ptr<SE3StateBatch> state;
  std::vector<dvector<Vector<2>>> obs;
  std::vector<dvector<Vector<3>>> pts;
  std::vector<std::unique_ptr<PnPFactorBatch>> pnp;
  dvector<SE3Transform> deltas;
  std::unique_ptr<SE3BetweenFactorBatch> between;
  std::unique_ptr<LossFunctionBatch> loss;
  Problem problem;

  RigProblem(int cameras, size_t points_per_camera, double outlier_ratio, double rot_err,
             double trans_err, uint32_t seed, Method method, double min_outlier = kMinOutlier,
             double noise = kNoise, const std::array<double, 6> *coherent_twist = nullptr) {
    std::mt19937 rng(seed);
    std::vector<SE3Transform> gt(cameras);
    gt[0] = ExpSE3(RandomTwist(rng, 0.5, 1.0));
    for (int c = 1; c < cameras; ++c) {
      gt[c] = Compose(ExpSE3(RandomTwist(rng, 0.15, 0.4)), gt[c - 1]);
    }
    for (int c = 0; c < cameras; ++c) {
      if (coherent_twist != nullptr) {  // single camera: the scene draws its own pose
        scenes.push_back(ransac_test::MakeCoherentPnPScene(
            points_per_camera, outlier_ratio, noise, min_outlier, seed * 131 + c, *coherent_twist));
        gt[c] = scenes.back().world_to_cam;
      } else {
        scenes.push_back(ransac_test::MakePnPScene(points_per_camera, outlier_ratio, noise,
                                                   min_outlier, seed * 131 + c, &gt[c]));
      }
      init.push_back(Compose(gt[c], ExpSE3(RandomTwist(rng, rot_err, trans_err))));
    }
    poses.resize(cameras);
    poses.CopyFromHost(init.data(), cameras);
    state =
        std::make_unique<SE3StateBatch>(cublas, reinterpret_cast<float *>(poses.data()), cameras);
    state->SetNumStateBlocks(state->Capacity(), state->ConstCapacity());
    problem.AddStateBatch(state.get());
    if (method == Method::kLMHuber) {
      loss = std::make_unique<HuberLossFunctionBatch>(static_cast<float>(kTau));
    } else if (method == Method::kLMCauchy) {
      const float a2 = static_cast<float>(kTau * kTau);
      loss = std::make_unique<CauchyLossFunctionBatch>(a2, 1.f / a2);
    }
    obs.resize(cameras);
    pts.resize(cameras);
    for (int c = 0; c < cameras; ++c) {
      const PnPScene &s = scenes[c];
      obs[c].resize(s.observations.size());
      obs[c].CopyFromHost(s.observations.data(), s.observations.size());
      pts[c].resize(s.points_world.size());
      pts[c].CopyFromHost(s.points_world.data(), s.points_world.size());
      pnp.push_back(
          std::make_unique<PnPFactorBatch>(obs[c].data(), pts[c].data(), s.observations.size()));
      std::vector<float *> ptrs(s.observations.size(), state->StateBlockDevicePtr(c));
      if (loss) {
        problem.AddFactorBatch(pnp.back().get(), loss.get(), ptrs);
      } else {
        problem.AddFactorBatch(pnp.back().get(), ptrs);
      }
    }
    if (cameras > 1) {
      std::vector<SE3Transform> d;
      std::vector<float *> ptrs;
      for (int c = 0; c + 1 < cameras; ++c) {
        d.push_back(Compose(ransac_test::Inverse(gt[c + 1]), gt[c]));
        ptrs.push_back(state->StateBlockDevicePtr(c));
        ptrs.push_back(state->StateBlockDevicePtr(c + 1));
      }
      deltas.resize(d.size());
      deltas.CopyFromHost(d.data(), d.size());
      between = std::make_unique<SE3BetweenFactorBatch>(deltas.data(), d.size());
      between->SetNumFactors(between->Capacity());
      problem.AddFactorBatch(between.get(), ptrs);
    }
  }

  void Reset() { poses.CopyFromHost(init.data(), init.size()); }

  std::vector<SE3Transform> Poses() const {
    std::vector<SE3Transform> p(init.size());
    poses.CopyToHost(p.data(), p.size());
    return p;
  }

  size_t TotalPoints() const {
    size_t n = 0;
    for (const auto &s : scenes) n += s.observations.size();
    return n;
  }
};

RansacMinimizerOptions RansacOptions(int cameras, size_t hypotheses, size_t max_rounds) {
  RansacMinimizerOptions o;
  o.hypotheses_per_round = hypotheses;
  o.max_rounds = max_rounds;
  o.seed = 1;
  for (int c = 0; c < cameras; ++c) {
    o.factor_batches.push_back({RansacRole::kSampled, static_cast<float>(kTau)});
  }
  if (cameras > 1) {
    o.factor_batches.push_back({RansacRole::kAlwaysOn, 0.f});
  }
  return o;
}

/** A configured minimizer that can run a RigProblem repeatedly. */
struct Runner {
  Method method;
  std::unique_ptr<GaussNewtonMinimizer> regular;
  std::unique_ptr<RansacGaussNewtonMinimizer> ransac;
  RansacSummary last;

  Runner(Method m, int cameras, size_t hypotheses = 256, size_t max_rounds = 8,
         SparseLinearSolverType solver = SparseLinearSolverType::DenseLDLT)
      : method(m) {
    if (IsRansac(m)) {
      RansacMinimizerOptions o = RansacOptions(cameras, hypotheses, max_rounds);
      if (m == Method::kRansacGN) {
        ransac = std::make_unique<RansacGaussNewtonMinimizer>(o);
      } else {
        RansacLevenbergMarquardtMinimizerOptions lm;
        lm.base_options = o;
        ransac = std::make_unique<RansacLevenbergMarquardtMinimizer>(lm);
      }
    } else if (m == Method::kGN) {
      MinimizerOptions o;
      o.sparse_linear_solver_type = solver;
      regular = std::make_unique<GaussNewtonMinimizer>(o);
    } else {
      LevenbergMarquardtMinimizerOptions o;
      o.base_options.sparse_linear_solver_type = solver;
      o.base_options.max_num_iterations = 100;
      regular = std::make_unique<LevenbergMarquardtMinimizer>(o);
    }
  }

  void Run(cudaStream_t stream, Problem &problem) {
    if (ransac) {
      last = ransac->Minimize(stream, problem);
    } else {
      regular->Minimize(stream, problem);
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
};

double MedianMs(const std::function<void()> &reset, const std::function<void()> &run, int reps) {
  reset();
  run();  // warm-up
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    reset();
    THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    run();
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

struct PoseErrors {
  double max_rot_deg = 0;
  double max_trans = 0;
  bool success(double rot = 0.5, double trans = 0.05) const {
    return max_rot_deg < rot && max_trans < trans;
  }
};

PoseErrors Errors(const RigProblem &rp) {
  PoseErrors e;
  const auto est = rp.Poses();
  for (size_t c = 0; c < est.size(); ++c) {
    e.max_rot_deg = std::max(e.max_rot_deg, RotationErrorDeg(est[c], rp.scenes[c].world_to_cam));
    e.max_trans = std::max(e.max_trans, TranslationError(est[c], rp.scenes[c].world_to_cam));
    if (!std::isfinite(e.max_rot_deg) || !std::isfinite(e.max_trans)) {
      e.max_rot_deg = e.max_trans = INFINITY;
    }
  }
  return e;
}

struct Detection {
  double tp = 0, fp = 0, fn = 0, tn = 0;
  void Add(const Detection &o) {
    tp += o.tp;
    fp += o.fp;
    fn += o.fn;
    tn += o.tn;
  }
  double precision() const { return tp + fp > 0 ? tp / (tp + fp) : 1.0; }
  double recall() const { return tp + fn > 0 ? tp / (tp + fn) : 1.0; }
  double outlier_recall() const { return tn + fp > 0 ? tn / (tn + fp) : 1.0; }
  double f1() const {
    const double p = precision(), r = recall();
    return p + r > 0 ? 2 * p * r / (p + r) : 0.0;
  }
};

/** Classification by the RANSAC mask (RANSAC) or by |r| <= tau at the final pose (others). */
Detection Detect(const RigProblem &rp, const Runner &runner) {
  Detection d;
  const auto est = rp.Poses();
  for (size_t c = 0; c < rp.scenes.size(); ++c) {
    const PnPScene &s = rp.scenes[c];
    std::vector<uint8_t> mask(s.observations.size());
    if (runner.ransac) {
      THROW_ON_CUDA_ERROR(cudaMemcpy(mask.data(), runner.ransac->InlierMask(c), mask.size(),
                                     cudaMemcpyDeviceToHost));
    } else {
      for (size_t i = 0; i < mask.size(); ++i) {
        const auto p = ransac_test::Transform(
            est[c], {s.points_world[i][0], s.points_world[i][1], s.points_world[i][2]});
        const double du = p[0] / p[2] - s.observations[i][0];
        const double dv = p[1] / p[2] - s.observations[i][1];
        mask[i] = p[2] > 1e-3 && std::hypot(du, dv) <= kTau;
      }
    }
    for (size_t i = 0; i < mask.size(); ++i) {
      if (s.is_outlier[i]) {
        (mask[i] ? d.fp : d.tn) += 1;
      } else {
        (mask[i] ? d.tp : d.fn) += 1;
      }
    }
  }
  return d;
}

class RansacBenchmark : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!Enabled()) {
      GTEST_SKIP() << "set CUNLS_RANSAC_BENCHMARK=1 to run";
    }
  }
};

// ----------------------------------------------------------------------------
// 1. Performance at scale
// ----------------------------------------------------------------------------

TEST_F(RansacBenchmark, PnPSizeSweep) {
  CudaStream stream;
  std::ofstream csv(OutDir() + "pnp_size_sweep.csv");
  csv << "points,method,median_ms,rounds,hypotheses,rot_err_deg,trans_err\n";
  std::printf("\n### PnP, 30%% outliers, initial error 0.1 rad / 0.3: median wall time (ms)\n\n");
  std::printf(
      "| points | GN (DenseLDLT) | GN (PCG) | LM+Cauchy | RANSAC-GN | RANSAC-LM | "
      "RANSAC rounds |\n|---|---|---|---|---|---|---|\n");
  for (size_t n : {100u, 1000u, 10000u, 100000u, 1000000u}) {
    const char *only_points = std::getenv("CUNLS_RANSAC_BENCH_POINTS");
    if (only_points != nullptr && std::stoull(only_points) != n) {
      continue;
    }
    struct Config {
      const char *label;
      Method method;
      SparseLinearSolverType solver;
    };
    const std::vector<Config> configs = {
        {"GN (DenseLDLT)", Method::kGN, SparseLinearSolverType::DenseLDLT},
        {"GN (PCG)", Method::kGN, SparseLinearSolverType::BlockSparsePCG},
        {"LM+Cauchy", Method::kLMCauchy, SparseLinearSolverType::DenseLDLT},
        {"RANSAC-GN", Method::kRansacGN, SparseLinearSolverType::DenseLDLT},
        {"RANSAC-LM", Method::kRansacLM, SparseLinearSolverType::DenseLDLT},
    };
    std::printf("| %zu |", n);
    size_t rounds = 0;
    for (const Config &cfg : configs) {
      if (!Selected(n, cfg.label)) {
        std::printf(" - |");
        continue;
      }
      RigProblem rp(1, n, 0.3, 0.1, 0.3, 7, cfg.method);
      Runner runner(cfg.method, 1, 256, 8, cfg.solver);
      const int reps = n >= 1000000 ? 3 : 7;
      const double ms =
          MedianMs([&] { rp.Reset(); }, [&] { runner.Run(stream.GetStream(), rp.problem); }, reps);
      const PoseErrors e = Errors(rp);
      rounds = IsRansac(cfg.method) ? runner.last.num_rounds : rounds;
      std::printf(" %.2f%s |", ms, e.success() ? "" : " (fail)");
      csv << n << "," << cfg.label << "," << ms << "," << runner.last.num_rounds << ","
          << runner.last.num_hypotheses << "," << e.max_rot_deg << "," << e.max_trans << "\n";
    }
    std::printf(" %zu |\n", rounds);
  }
}

TEST_F(RansacBenchmark, HypothesisCountSweep) {
  CudaStream stream;
  std::ofstream csv(OutDir() + "hypothesis_sweep.csv");
  csv << "points,hypotheses_per_round,median_ms,ms_per_1k_hypotheses\n";
  std::printf("\n### RANSAC-GN, one round (max_rounds = 1), time vs hypotheses per round (ms)\n\n");
  std::printf("| points | K | ms | ms per 1k hypotheses |\n|---|---|---|---|\n");
  for (size_t n : {1000u, 10000u}) {
    for (size_t k : {64u, 256u, 1024u, 4096u}) {
      RigProblem rp(1, n, 0.3, 0.1, 0.3, 8, Method::kRansacGN);
      RansacGaussNewtonMinimizer r(RansacOptions(1, k, 1));
      const double ms = MedianMs([&] { rp.Reset(); },
                                 [&] {
                                   r.Minimize(stream.GetStream(), rp.problem);
                                   THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
                                 },
                                 5);
      csv << n << "," << k << "," << ms << "," << ms * 1000.0 / k << "\n";
      std::printf("| %zu | %zu | %.2f | %.2f |\n", n, k, ms, ms * 1000.0 / k);
    }
  }
}

TEST_F(RansacBenchmark, MultiCameraRig) {
  CudaStream stream;
  std::ofstream csv(OutDir() + "rig.csv");
  csv << "cameras,dim,method,median_ms,rounds,success,rot_err_deg\n";
  std::printf("\n### Rig of R cameras (D = 6R), 1000 points/camera, 30%% outliers (ms)\n\n");
  std::printf(
      "| R | D | LM (DenseLDLT) | LM+Cauchy | RANSAC-GN | RANSAC-LM | RANSAC rounds |\n"
      "|---|---|---|---|---|---|---|\n");
  for (int cams : {1, 2, 5, 10}) {
    std::printf("| %d | %d |", cams, 6 * cams);
    size_t rounds = 0;
    for (Method m : {Method::kLM, Method::kLMCauchy, Method::kRansacGN, Method::kRansacLM}) {
      RigProblem rp(cams, 1000, 0.3, 0.05, 0.15, 9, m);
      Runner runner(m, cams, 1024, 8);
      const double ms =
          MedianMs([&] { rp.Reset(); }, [&] { runner.Run(stream.GetStream(), rp.problem); }, 5);
      const PoseErrors e = Errors(rp);
      if (IsRansac(m)) rounds = runner.last.num_rounds;
      std::printf(" %.2f%s |", ms, e.success() ? "" : " (fail)");
      csv << cams << "," << 6 * cams << "," << Name(m) << "," << ms << "," << runner.last.num_rounds
          << "," << e.success() << "," << e.max_rot_deg << "\n";
    }
    std::printf(" %zu |\n", rounds);
  }
}

// ----------------------------------------------------------------------------
// 2. Robustness and 3. detection
// ----------------------------------------------------------------------------

TEST_F(RansacBenchmark, RobustnessAndDetectionVsOutlierRatio) {
  CudaStream stream;
  const int trials = Trials(20);
  const std::vector<Method> methods = {Method::kGN,       Method::kLM,       Method::kLMHuber,
                                       Method::kLMCauchy, Method::kRansacGN, Method::kRansacLM};
  std::ofstream csv(OutDir() + "robustness.csv");
  csv << "init,outlier_ratio,method,success_rate,median_rot_deg,median_trans,precision,recall,"
         "outlier_recall,f1,mean_ms\n";
  struct Init {
    const char *label;
    double rot, trans;
  };
  for (const Init init :
       {Init{"small (0.05 rad, 0.15)", 0.05, 0.15}, Init{"large (0.3 rad, 0.8)", 0.3, 0.8}}) {
    std::printf(
        "\n### Success rate (rot < 0.5 deg, trans < 0.05), PnP 1000 points, "
        "%d trials, initial error %s\n\n| outliers |",
        trials, init.label);
    for (Method m : methods) std::printf(" %s |", Name(m));
    std::printf("\n|---|");
    for (size_t i = 0; i < methods.size(); ++i) std::printf("---|");
    std::printf("\n");
    std::vector<std::string> detection_rows;
    for (double ratio : {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}) {
      std::printf("| %.0f%% |", ratio * 100);
      char row[512];
      std::string drow;
      std::snprintf(row, sizeof(row), "| %.0f%% |", ratio * 100);
      drow = row;
      for (Method m : methods) {
        Runner runner(m, 1, 1024, 16);
        int success = 0;
        std::vector<double> rot, trans;
        Detection det;
        double total_ms = 0;
        for (int t = 0; t < trials; ++t) {
          RigProblem rp(1, 1000, ratio, init.rot, init.trans, 1000 + t, m);
          const auto t0 = std::chrono::steady_clock::now();
          runner.Run(stream.GetStream(), rp.problem);
          total_ms +=
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                  .count();
          const PoseErrors e = Errors(rp);
          success += e.success();
          rot.push_back(e.max_rot_deg);
          trans.push_back(e.max_trans);
          det.Add(Detect(rp, runner));
        }
        std::sort(rot.begin(), rot.end());
        std::sort(trans.begin(), trans.end());
        const double rate = static_cast<double>(success) / trials;
        std::printf(" %.0f%% |", rate * 100);
        if (m == Method::kLMCauchy || IsRansac(m)) {
          std::snprintf(row, sizeof(row), " %.4f / %.4f / %.4f |", det.precision(), det.recall(),
                        det.outlier_recall());
          drow += row;
        }
        csv << init.label << "," << ratio << "," << Name(m) << "," << rate << ","
            << rot[rot.size() / 2] << "," << trans[trans.size() / 2] << "," << det.precision()
            << "," << det.recall() << "," << det.outlier_recall() << "," << det.f1() << ","
            << total_ms / trials << "\n";
      }
      std::printf("\n");
      detection_rows.push_back(drow);
    }
    std::printf(
        "\n### Inlier-mask precision / inlier recall / outlier recall, initial error %s "
        "(LM+Cauchy: |r| <= tau at its final pose)\n\n"
        "| outliers | LM+Cauchy | RANSAC-GN | RANSAC-LM |\n|---|---|---|---|\n",
        init.label);
    for (const auto &r : detection_rows) std::printf("%s\n", r.c_str());
  }
}

TEST_F(RansacBenchmark, RobustnessCoherentOutliers) {
  // Outliers agree with a second, wrong pose (rotated 0.15 rad, shifted 0.4):
  // a competing mode that robust losses can lock onto.
  CudaStream stream;
  const int trials = Trials(20);
  const std::array<double, 6> twist = {0.1, -0.08, 0.06, 0.25, -0.2, 0.2};
  const std::vector<Method> methods = {Method::kLMHuber, Method::kLMCauchy, Method::kRansacGN,
                                       Method::kRansacLM};
  std::ofstream csv(OutDir() + "coherent.csv");
  csv << "init,outlier_ratio,method,success_rate,precision,recall,outlier_recall\n";
  struct Init {
    const char *label;
    double rot, trans;
  };
  for (const Init init :
       {Init{"small (0.05 rad, 0.15)", 0.05, 0.15}, Init{"large (0.3 rad, 0.8)", 0.3, 0.8}}) {
    std::printf(
        "\n### Coherent outliers (a competing pose), PnP 1000 points, %d trials, initial "
        "error %s: success rate | precision / inlier recall / outlier recall\n\n"
        "| outliers | LM+Huber | P / R / OR | LM+Cauchy | P / R / OR | RANSAC-GN | P / R / OR "
        "| RANSAC-LM | P / R / OR |\n|---|---|---|---|---|---|---|---|---|\n",
        trials, init.label);
    for (double ratio : {0.1, 0.2, 0.3, 0.4, 0.45}) {
      std::printf("| %.0f%% |", ratio * 100);
      for (Method m : methods) {
        Runner runner(m, 1, 1024, 16);
        int success = 0;
        Detection det;
        for (int t = 0; t < trials; ++t) {
          RigProblem rp(1, 1000, ratio, init.rot, init.trans, 3000 + t, m, kMinOutlier, kNoise,
                        &twist);
          runner.Run(stream.GetStream(), rp.problem);
          success += Errors(rp).success();
          det.Add(Detect(rp, runner));
        }
        const double rate = static_cast<double>(success) / trials;
        std::printf(" %.0f%% | %.3f / %.3f / %.3f |", rate * 100, det.precision(), det.recall(),
                    det.outlier_recall());
        csv << init.label << "," << ratio << "," << Name(m) << "," << rate << "," << det.precision()
            << "," << det.recall() << "," << det.outlier_recall() << "\n";
      }
      std::printf("\n");
    }
  }
}

TEST_F(RansacBenchmark, DetectionVsNoiseAndOutlierMagnitude) {
  // Harder detection: outliers barely above the threshold, and noise close to it.
  CudaStream stream;
  const int trials = Trials(20);
  std::ofstream csv(OutDir() + "detection.csv");
  csv << "tau_over_sigma,min_outlier_over_tau,outlier_ratio,precision,recall,outlier_recall,"
         "success_rate\n";
  std::printf(
      "\n### RANSAC-GN detection, 1000 points, 50%% outliers, %d trials: "
      "precision / inlier recall / outlier recall (success rate)\n\n"
      "| tau / sigma | outliers >= 1.2 tau | outliers >= 2 tau | outliers >= 4 tau |\n"
      "|---|---|---|---|\n",
      trials);
  for (double tau_over_sigma : {2.0, 3.0, 5.0}) {
    std::printf("| %.0f |", tau_over_sigma);
    for (double outlier_over_tau : {1.2, 2.0, 4.0}) {
      Runner runner(Method::kRansacGN, 1, 1024, 16);
      Detection det;
      int success = 0;
      for (int t = 0; t < trials; ++t) {
        RigProblem rp(1, 1000, 0.5, 0.05, 0.15, 2000 + t, Method::kRansacGN,
                      outlier_over_tau * kTau, kTau / tau_over_sigma);
        runner.Run(stream.GetStream(), rp.problem);
        det.Add(Detect(rp, runner));
        success += Errors(rp).success();
      }
      std::printf(" %.4f / %.4f / %.4f (%.0f%%) |", det.precision(), det.recall(),
                  det.outlier_recall(), 100.0 * success / trials);
      csv << tau_over_sigma << "," << outlier_over_tau << ",0.5," << det.precision() << ","
          << det.recall() << "," << det.outlier_recall() << ","
          << static_cast<double>(success) / trials << "\n";
    }
    std::printf("\n");
  }
}

}  // namespace
}  // namespace cunls
