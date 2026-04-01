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

#include "cunls/common/cublas_helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batched Perspective-n-Point (PnP) reprojection factor (pose only).
 *
 * This class is a specialization of the pinhole reprojection model for the
 * common case where 3D landmarks are **fixed** and only the camera pose is
 * optimized. Mathematically the residual per correspondence is identical to
 * `ReprojectionFactorBatch`, but:
 *   - World points are **not** state variables: they are supplied once via a
 *     device pointer in the constructor (one `Vector<3>` per factor).
 *   - The analytic Jacobian is computed **only** with respect to the SE(3)
 *     pose, so each factor has tangent dimension 6 instead of 9.
 *
 * Typical use cases include:
 *   - Absolute pose estimation from known 3D–2D matches.
 *   - Tracking / localization with a fixed map.
 *   - Reducing problem size when structure is not bundled.
 *
 * @section pnp_vs_ba Relationship to ReprojectionFactorBatch
 *
 * `ReprojectionFactorBatch` couples one pose and one 3D point per residual and
 * exports a `2 × (6 + 3)` Jacobian. `PnPFactorBatch` exports `2 × 6` by
 * treating the point as a constant measurement. The first six columns match
 * the pose block of `ReprojectionFactorBatch` for the same pose, point, and
 * observation, under the same SE(3) right-multiplication and tangent ordering
 * used throughout cuNLS (rotation first, translation second in the
 * 6-vector).
 *
 * @section pnp_normalized_coords Normalized Coordinates (IMPORTANT)
 *
 * As with `ReprojectionFactorBatch`, **observations must be in normalized image
 * coordinates** (intrinsics already removed):
 *
 *     [x_n] = K^-1 [u v 1]^T   ⇒   (x_n, y_n) passed per observation.
 *
 * @section pnp_math Mathematical Model
 *
 * For correspondence i:
 *
 *     P_cam = T_world_to_cam * P_world[i]
 *     r_i   = [ P_cam.x / P_cam.z - obs_i.x ]
 *             [ P_cam.y / P_cam.z - obs_i.y ]
 *
 * Optional multi-camera rigs: if `poses_camera_from_rig` is provided,
 * `T_cam_from_world = T_cam_from_rig[i] * T_rig_from_world` (same convention
 * as `ReprojectionFactorBatch`).
 *
 * @section pnp_template_params Template Parameters
 *
 * Inherits from `SizedFactorBatch<2, 6>`:
 *   - 2: Residual dimension (x/y reprojection error).
 *   - 6: Single state block — SE(3) in tangent space (3 rotation + 3
 *        translation).
 *
 * @see ReprojectionFactorBatch for the joint pose–structure model.
 * @see pnp_factor_batch.cu for memory layout and derivative notes.
 */
class PnPFactorBatch : public SizedFactorBatch<2, 6> {
 public:
  /**
   * @brief Constructs a batched PnP factor with identity camera-from-rig
   *        extrinsics.
   *
   * Use this overload when the optimized `SE3Transform` is already expressed
   * in the camera frame (single camera, or extrinsics pre-composed into the
   * pose).
   *
   * @param cublas_handle       Externally owned cuBLAS handle (used when an
   *                            optional rig transform batch is supplied).
   * @param observations        Device pointer to normalized 2D observations:
   *                            `num_observations` vectors `[x_n, y_n]`.
   * @param points_world        Device pointer to fixed world points:
   *                            `num_observations` vectors `[x, y, z]`, each
   *                            paired with `observations[i]`.
   * @param num_observations    Number of correspondences (factors) in the
   *                            batch.
   * @param z_threshold         Minimum depth in the **camera** frame for a
   *                            residual to be active (avoids divide-by-zero).
   */
  PnPFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations,
                 const Vector<3>* points_world, size_t num_observations,
                 float z_threshold = 1e-3f);

  /**
   * @brief Constructs a batched PnP factor with per-correspondence
   *        camera-from-rig transforms.
   *
   * The effective camera pose is `T_cam_from_world = T_cam_from_rig[i] *
   * T_rig_from_world`, where `T_rig_from_world` is the `SE3Transform` passed
   * through `state_pointers[i]` at evaluation time.
   *
   * @param cublas_handle           Shared cuBLAS handle.
   * @param observations            Normalized 2D observations (device).
   * @param poses_camera_from_rig    Device array of `num_observations` rig
   *                                 extrinsics (camera-from-rig), same layout
   *                                 as `SE3Transform` elsewhere in cuNLS.
   * @param points_world            Fixed world points (device).
   * @param num_observations        Number of factors.
   * @param z_threshold             Minimum valid camera-frame depth.
   */
  PnPFactorBatch(cuBLASHandle& cublas_handle, const Vector<2>* observations,
                 const SE3Transform* poses_camera_from_rig,
                 const Vector<3>* points_world, size_t num_observations,
                 float z_threshold = 1e-3f);

  /**
   * @brief Evaluates PnP residuals and optionally pose Jacobians.
   *
   * @param residuals       Device buffer of `num_observations * 2` floats, or
   *                        `nullptr` (see implementation: Jacobians assume
   *                        residuals were requested together for a valid
   *                        projection pipeline).
   * @param jacobians       Device buffer of `num_observations * 12` floats
   *                        (`2 × 6` row-major per factor), or `nullptr`.
   *
   *                        Column layout per factor (matches the pose block of
   *                        `ReprojectionFactorBatch`):
   *                          - Cols 0–2: \(\partial r / \partial \omega\)
   *                            (SO(3) part of the SE(3) tangent).
   *                          - Cols 3–5: \(\partial r / \partial \rho\)
   *                            (translation part).
   *
   * @param state_pointers  Device array of `num_observations` pointers; entry
   *                        `i` points to the `SE3Transform` (rig-from-world)
   *                        for correspondence `i`.
   * @param stream          CUDA stream.
   *
   * @return `true` on success.
   */
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Number of correspondences in this batch.
   */
  size_t NumFactors() const final { return num_observations_; }

 private:
  PnPFactorBatch() = delete;

  const Vector<2>* observations_;
  const Vector<3>* points_world_;
  const SE3Transform* poses_camera_from_rig_ = nullptr;

  size_t num_observations_;

  mutable dvector<SE3Transform> poses_rig_from_world_;
  mutable dvector<SE3Transform> poses_cam_from_world_;

  cuBLASHandle& cublas_handle_;
  float z_threshold_ = 1e-3f;
};

}  // namespace cunls
