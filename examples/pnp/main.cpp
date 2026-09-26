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

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/camera_utils.h"
#include "utils/se3_utils.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

namespace {

// Visibility threshold used when generating synthetic points.
constexpr float kMinDepth = 1.0f;
// Reprojection factor guard threshold to avoid unstable divisions near z=0.
constexpr float kZThreshold = 1e-3f;

// A single known camera pose (world -> camera), N known 3D world points, and
// their noisy normalized 2D observations.
struct PnPDataset {
  SE3Transform gt_pose;
  SE3Transform initial_pose;
  std::vector<Vector<3>> points_world;
  std::vector<Vector<2>> observations;
};

PnPDataset GenerateDataset(size_t num_points, std::mt19937 &rng) {
  PnPDataset data;

  // Ground-truth pose: a small twist away from identity, translated forward
  // so the world origin (and nearby points) stay in front of the camera.
  std::uniform_real_distribution<float> rot_dist(-0.3f, 0.3f);
  std::uniform_real_distribution<float> trans_dist(-1.0f, 1.0f);
  Vector<6> gt_twist;
  gt_twist[0] = rot_dist(rng);
  gt_twist[1] = rot_dist(rng);
  gt_twist[2] = rot_dist(rng);
  gt_twist[3] = trans_dist(rng);
  gt_twist[4] = trans_dist(rng);
  gt_twist[5] = 8.0f + trans_dist(rng);
  std::vector<SE3Transform> gt_pose_vec;
  examples::TwistsToSE3({gt_twist}, gt_pose_vec);
  data.gt_pose = gt_pose_vec[0];

  // Sample world points visible from the ground-truth pose.
  std::uniform_real_distribution<float> point_dist(-3.0f, 3.0f);
  std::normal_distribution<float> pixel_noise(0.0f, 3e-3f);
  data.points_world.resize(num_points);
  data.observations.resize(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    Vector<3> p;
    do {
      p[0] = point_dist(rng);
      p[1] = point_dist(rng);
      p[2] = point_dist(rng);
    } while (examples::ComputeDepth(data.gt_pose, p) < kMinDepth);
    data.points_world[i] = p;

    Vector<2> obs = examples::ProjectNormalized(data.gt_pose, p);
    obs[0] += pixel_noise(rng);
    obs[1] += pixel_noise(rng);
    data.observations[i] = obs;
  }

  // Perturb the pose to create a non-trivial initial estimate. Kept small
  // enough that every ground-truth-visible point stays visible.
  std::vector<SE3Transform> perturbation;
  examples::GenerateRandomSE3(1, rng, perturbation, 0.05f, 0.2f);
  data.initial_pose = examples::ComposeSE3(perturbation[0], data.gt_pose);

  return data;
}

// Runs LM on a fresh device copy of `data.initial_pose`, using the requested
// Jacobian mode, and returns the optimization summary plus final pose error.
struct RunResult {
  cunls::MinimizerSummary summary;
  float initial_pose_mse;
  float final_pose_mse;
};

RunResult RunPnP(const PnPDataset &data, cunls::JacobianMode jacobian_mode) {
  const size_t num_points = data.points_world.size();

  dvector<Vector<3>> points_device(data.points_world);
  dvector<Vector<2>> observations_device(data.observations);
  std::vector<SE3Transform> pose_host = {data.initial_pose};
  dvector<SE3Transform> pose_device(pose_host);

  cunls::cuBLASHandle cublas_handle;
  cunls::SE3StateBatch pose_states(cublas_handle,
                                   reinterpret_cast<const float *>(pose_device.data()), 1);

  cunls::PnPFactorBatch pnp_factor(observations_device.data(), points_device.data(), num_points,
                                   kZThreshold);

  std::vector<float *> state_pointers(num_points, pose_states.StateBlockDevicePtr(0));

  cunls::Problem problem;
  problem.AddStateBatch(&pose_states);
  problem.AddFactorBatch(&pnp_factor, state_pointers, jacobian_mode);
  if (!problem.CheckConsistency()) {
    throw std::runtime_error("PnP problem consistency check failed");
  }

  cunls::MinimizerOptions options;
  options.max_num_iterations = 60;
  options.state_tolerance = 1e-9f;
  options.cost_tolerance = 1e-9f;
  options.jacobian_mode = jacobian_mode;

  cunls::LevenbergMarquardtMinimizerOptions lm_options;
  lm_options.base_options = options;
  lm_options.initial_lambda = 1e-2f;
  cunls::LevenbergMarquardtMinimizer minimizer(lm_options);

  cunls::CudaStream stream;
  RunResult result;
  result.summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<SE3Transform> optimized_pose(1);
  pose_device.CopyToHost(optimized_pose.data(), 1);

  result.initial_pose_mse = examples::ComputePoseMSE({data.initial_pose}, {data.gt_pose});
  result.final_pose_mse = examples::ComputePoseMSE(optimized_pose, {data.gt_pose});
  return result;
}

const char *JacobianModeName(cunls::JacobianMode mode) {
  return mode == cunls::JacobianMode::kAnalytic ? "analytic" : "numeric";
}

void PrintResult(const char *label, const RunResult &r) {
  std::cout << "  [" << label << "]\n";
  std::cout << "    Initial cost: " << r.summary.initial_cost << "\n";
  std::cout << "    Final cost:   " << r.summary.final_cost << "\n";
  std::cout << "    Iterations:   " << r.summary.num_iterations << "\n";
  std::cout << "    Pose MSE:     " << r.initial_pose_mse << " -> " << r.final_pose_mse << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    // CLI: --num-points N (default 2000), --jacobian-mode {analytic,numeric,both}
    const std::string usage = std::string("Usage: ") + argv[0] +
                              " [--num-points N] [--jacobian-mode analytic|numeric|both]\n";
    size_t num_points = 2000;
    std::string mode_arg = "both";
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--num-points") == 0) {
        if (i + 1 >= argc) {
          std::cerr << "Missing value for --num-points\n" << usage;
          return 1;
        }
        const char *value = argv[++i];
        char *end = nullptr;
        const long long parsed = std::strtoll(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0) {
          std::cerr << "Invalid --num-points value '" << value
                    << "' (expected a positive integer)\n"
                    << usage;
          return 1;
        }
        num_points = static_cast<size_t>(parsed);
      } else if (std::strcmp(argv[i], "--jacobian-mode") == 0) {
        if (i + 1 >= argc) {
          std::cerr << "Missing value for --jacobian-mode\n" << usage;
          return 1;
        }
        mode_arg = argv[++i];
      } else if (std::strcmp(argv[i], "--help") == 0) {
        std::cout << usage;
        return 0;
      } else {
        std::cerr << "Unknown argument '" << argv[i] << "'\n" << usage;
        return 1;
      }
    }

    std::mt19937 rng(13579);
    PnPDataset data = GenerateDataset(num_points, rng);

    std::cout << "PnP Example\n";
    std::cout << "  Num correspondences: " << num_points << "\n";

    bool run_analytic = (mode_arg == "analytic" || mode_arg == "both");
    bool run_numeric = (mode_arg == "numeric" || mode_arg == "both");
    if (!run_analytic && !run_numeric) {
      std::cerr << "Unknown --jacobian-mode '" << mode_arg
                << "' (expected analytic|numeric|both)\n";
      return 1;
    }

    // Observations carry pixel noise, so the residual cost has a noise floor
    // and will not reach ~0; judge convergence by pose accuracy instead
    // (final pose MSE should collapse relative to the initial perturbation).
    bool ok = true;
    if (run_analytic) {
      RunResult analytic = RunPnP(data, cunls::JacobianMode::kAnalytic);
      PrintResult("JacobianMode::kAnalytic", analytic);
      ok = ok && analytic.summary.final_cost <= analytic.summary.initial_cost &&
           analytic.final_pose_mse < analytic.initial_pose_mse * 0.05f;
    }
    if (run_numeric) {
      RunResult numeric = RunPnP(data, cunls::JacobianMode::kNumeric);
      PrintResult("JacobianMode::kNumeric", numeric);
      ok = ok && numeric.summary.final_cost <= numeric.summary.initial_cost &&
           numeric.final_pose_mse < numeric.initial_pose_mse * 0.05f;
    }

    if (!ok) {
      std::cerr << "Optimization quality check failed.\n";
      return 2;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
