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
 * @brief Batched PnP reprojection factor (pose only, fixed 3D points).
 *
 * Residual per correspondence:
 *     P_cam = T_cam_from_world * P_world[i]
 *     r_i = [P_cam.x/P_cam.z - obs_i.x, P_cam.y/P_cam.z - obs_i.y]
 *
 * Jacobian: 2x6 (pose tangent only, no point derivatives).
 *
 * Inherits from SizedFactorBatch<2, 6>:
 *   - 2: Residual dimension
 *   - 6: SE(3) pose tangent
 */
class PnPFactorBatch : public SizedFactorBatch<2, 6> {
 public:
  /**
   * @brief Constructs with identity camera-from-rig extrinsics.
   */
  PnPFactorBatch(const Vector<2>* observations,
                 const Vector<3>* points_world, size_t num_observations,
                 float z_threshold = 1e-3f);

  /**
   * @brief Constructs with per-correspondence camera-from-rig transforms.
   */
  PnPFactorBatch(const Vector<2>* observations,
                 const SE3Transform* poses_camera_from_rig,
                 const Vector<3>* points_world, size_t num_observations,
                 float z_threshold = 1e-3f);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_observations_; }

 private:
  PnPFactorBatch() = delete;

  const Vector<2>* observations_;
  const Vector<3>* points_world_;
  const SE3Transform* poses_camera_from_rig_ = nullptr;
  size_t num_observations_;
  float z_threshold_ = 1e-3f;
};

}  // namespace cunls
