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

// PnP: recover one camera pose from known 3D points and their 2D normalized
// observations with PnPFactorBatch. The same problem is solved twice, with the
// factor's analytic Jacobian and with numeric (finite-difference) Jacobians.

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/se3_state_batch.h"
#include "utils/cli.h"
#include "utils/datasets.h"
#include "utils/report.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::SE3Transform;
using cunls::Vector;

namespace {

// Solves the PnP problem from scene.initial_pose with the given Jacobian mode;
// returns true if the pose got much closer to the ground truth.
bool SolvePnP(const examples::PnPScene &scene, cunls::JacobianMode jacobian_mode) {
  const size_t num_points = scene.points_world.size();

  // Upload the data; the pose is the only state (one SE(3) block).
  dvector<Vector<3>> points(scene.points_world);
  dvector<Vector<2>> observations(scene.observations);
  dvector<SE3Transform> pose(std::vector<SE3Transform>{scene.initial_pose});
  cunls::cuBLASHandle cublas;
  cunls::SE3StateBatch pose_state(cublas, reinterpret_cast<const float *>(pose.data()), 1);

  // One PnP factor per correspondence; every factor reads the same pose.
  cunls::PnPFactorBatch pnp(observations.data(), points.data(), num_points,
                            /*z_threshold=*/1e-3f);
  cunls::Problem problem;
  problem.AddStateBatch(&pose_state);
  problem.AddFactorBatch(&pnp, std::vector<float *>(num_points, pose_state.StateBlockDevicePtr(0)),
                         jacobian_mode);
  if (!problem.CheckConsistency()) throw std::runtime_error("PnP problem is inconsistent");

  // Levenberg-Marquardt; jacobian_mode selects analytic or numeric Jacobians.
  cunls::LevenbergMarquardtMinimizerOptions options;
  options.base_options.max_num_iterations = 60;
  options.base_options.state_tolerance = 1e-9f;
  options.base_options.cost_tolerance = 1e-9f;
  options.base_options.jacobian_mode = jacobian_mode;
  options.initial_lambda = 1e-2f;
  cunls::LevenbergMarquardtMinimizer minimizer(options);
  cunls::CudaStream stream;
  const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Read back the pose and compare with the ground truth.
  SE3Transform final_pose;
  pose.CopyToHost(&final_pose, 1);
  const float mse_before = examples::ComputePoseMSE({scene.initial_pose}, {scene.gt_pose});
  const float mse_after = examples::ComputePoseMSE({final_pose}, {scene.gt_pose});
  examples::PrintTitle(jacobian_mode == cunls::JacobianMode::kAnalytic
                           ? "  [JacobianMode::kAnalytic]"
                           : "  [JacobianMode::kNumeric]");
  examples::PrintSummary(summary);
  examples::PrintChange("Pose MSE", mse_before, mse_after);
  // The observations are noisy, so the cost has a floor; judge by the pose.
  return summary.final_cost <= summary.initial_cost && mse_after < mse_before * 0.05f;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const examples::CommandLine cli(argc, argv, {"--num-points", "--jacobian-mode"});
    examples::PnPSceneOptions data;
    data.num_points = cli.GetSize("--num-points", 2000);
    const std::string mode = cli.GetString("--jacobian-mode", "both");  // analytic|numeric|both
    if (mode != "analytic" && mode != "numeric" && mode != "both") {
      cli.Fail("Unknown --jacobian-mode '" + mode + "' (expected analytic|numeric|both)");
    }

    const examples::PnPScene scene = examples::MakePnPScene(data);
    examples::PrintTitle("PnP Example");
    examples::PrintValue("Num correspondences", data.num_points);

    bool ok = true;
    if (mode != "numeric") ok = SolvePnP(scene, cunls::JacobianMode::kAnalytic) && ok;
    if (mode != "analytic") ok = SolvePnP(scene, cunls::JacobianMode::kNumeric) && ok;
    return examples::QualityExitCode(ok);
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
