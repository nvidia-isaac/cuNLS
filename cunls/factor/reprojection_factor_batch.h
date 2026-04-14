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

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batched factor for camera reprojection error computation.
 *
 * Computes the reprojection error for a batch of 3D-to-2D correspondences.
 * Observations must be in **normalized image coordinates** (intrinsics removed).
 *
 * Residual per observation:
 *     P_cam = T_cam_from_world * P_world
 *     r = [P_cam.x / P_cam.z - x_n, P_cam.y / P_cam.z - y_n]
 *
 * Inherits from SizedFactorBatch<2, 6, 3>:
 *   - 2: Residual dimension (2D reprojection error)
 *   - 6: First state block (SE3 pose tangent)
 *   - 3: Second state block (3D point)
 */
class ReprojectionFactorBatch : public SizedFactorBatch<2, 6, 3> {
 public:
  /**
   * @brief Constructs with identity camera-from-rig transforms.
   *
   * @param observations  Device pointer to normalized 2D observations.
   * @param num_observations  Number of observations in the batch.
   * @param z_threshold  Minimum depth for valid projection.
   */
  ReprojectionFactorBatch(const Vector<2>* observations,
                          size_t num_observations,
                          float z_threshold = 1e-3f);

  /**
   * @brief Constructs with custom camera-from-rig transforms.
   *
   * The final camera pose is: T_cam_from_world = T_cam_from_rig * T_rig_from_world.
   *
   * @param observations  Device pointer to normalized 2D observations.
   * @param poses_camera_from_rig  Per-observation camera-from-rig SE3 transforms.
   * @param num_observations  Number of observations in the batch.
   * @param z_threshold  Minimum depth for valid projection.
   */
  ReprojectionFactorBatch(const Vector<2>* observations,
                          const SE3Transform* poses_camera_from_rig,
                          size_t num_observations,
                          float z_threshold = 1e-3f);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_observations_; }

 private:
  ReprojectionFactorBatch() = delete;

  const Vector<2>* observations_;
  const SE3Transform* poses_camera_from_rig_ = nullptr;
  size_t num_observations_;
  float z_threshold_ = 1e-3f;
};

}  // namespace cunls
