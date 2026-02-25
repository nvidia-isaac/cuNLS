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
 * @brief Batched factor for camera reprojection error computation.
 *
 * This class computes the reprojection error for a batch of 3D point
 * observations. It measures the difference between the observed 2D image
 * coordinates and the projected 3D world points through a camera pose
 * transformation.
 *
 * @section normalized_coords Normalized Coordinates (IMPORTANT)
 *
 * **Observations must be provided in NORMALIZED image coordinates**, meaning
 * the camera intrinsic matrix K has already been applied (inverted) to the
 * pixel coordinates. Normalized coordinates are computed as:
 *
 *     [x_n]       [u]       [x_n]   [(u - cx) / fx]
 *     [y_n] = K^-1 * [v]  =>  [y_n] = [(v - cy) / fy]
 *     [ 1 ]       [1]
 *
 * Where:
 *   - (u, v) are the raw pixel coordinates
 *   - (cx, cy) is the principal point
 *   - (fx, fy) are the focal lengths
 *   - (x_n, y_n) are the normalized coordinates passed to this class
 *
 * This design choice decouples the reprojection residual from camera intrinsics,
 * allowing:
 *   - Support for heterogeneous cameras in bundle adjustment problems
 *   - Simpler Jacobian computation (no intrinsic derivatives needed)
 *   - Flexibility in handling different camera models externally
 *
 * @section math_model Mathematical Model
 *
 * The residual for each observation is computed as:
 *
 *     P_cam = T_world_cam * P_world    (transform world point to camera frame)
 *     residual = [P_cam.x / P_cam.z - x_n]
 *                [P_cam.y / P_cam.z - y_n]
 *
 * Where T_world_cam is the SE(3) camera pose (world-to-camera transformation).
 *
 * @section template_params Template Parameters
 *
 * Inherits from SizedFactorBatch<2, 6, 3>:
 *   - 2: Residual dimension (2D reprojection error: x and y components)
 *   - 6: First state block dimension (SE3 pose: 6 DOF in tangent space)
 *   - 3: Second state block dimension (3D point in world coordinates)
 *
 * @see SizedFactorBatch for the base class interface
 * @see reprojection_factor_batch.cu for implementation details and
 * memory layout
 */
class ReprojectionFactorBatch : public SizedFactorBatch<2, 6, 3> {
 public:
  /**
   * @brief Constructs a batched reprojection factor with identity
   *        camera-from-rig transforms.
   *
   * This constructor assumes that camera_from_rig poses are all identity
   * transforms, meaning the camera frame coincides with the rig frame. Use this
   * overload for single-camera systems or when camera extrinsics relative to
   * the rig are already baked into the poses.
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   * @param observations  Device pointer to an array of 2D observations in
   *                      **normalized coordinates** (camera intrinsics
   * removed). The array must have at least @p num_observations elements and
   * remain valid for the lifetime of this object. Each Vector<2> contains
   * [x_normalized, y_normalized].
   *
   * @param num_observations  Number of point observations in the batch. This
   * defines the number of factors (residuals) to compute.
   *
   * @param z_threshold  Minimum allowed depth (Z coordinate in camera frame)
   * for valid projection. Points with depth below this threshold produce zero
   * residuals and zero Jacobians to avoid numerical instability from division
   * by very small Z values. Default: 1e-3f (1mm in metric units).
   *
   * @note The observations pointer must point to device memory (GPU
   * accessible).
   * @note For points behind the camera (z < z_threshold), the residual and
   * Jacobian are set to zero to prevent numerical issues and gradient
   * corruption.
   */
  ReprojectionFactorBatch(cuBLASHandle& cublas_handle,
                                const Vector<2>* observations,
                                size_t num_observations,
                                float z_threshold = 1e-3f);

  /**
   * @brief Constructs a batched reprojection factor with custom
   *        camera-from-rig transforms.
   *
   * This constructor allows specifying custom camera_from_rig poses for
   * multi-camera systems where cameras have different extrinsic calibrations
   * relative to the rig body frame. The final camera pose is computed as:
   *     pose_cam_from_world = pose_camera_from_rig * pose_rig_from_world
   *
   * @param cublas_handle Reference to an externally-owned cuBLAS handle.
   *
   * @param observations  Device pointer to an array of 2D observations in
   *                      **normalized coordinates** (camera intrinsics
   * removed). The array must have at least @p num_observations elements and
   * remain valid for the lifetime of this object. Each Vector<2> contains
   * [x_normalized, y_normalized].
   *
   * @param poses_camera_from_rig  Device pointer to an array of SE3 transforms
   *                               representing the camera pose relative to the
   * rig body frame for each observation. Must have at least @p num_observations
   * elements. Use this for multi-camera rigs where each camera has its own
   * extrinsic calibration.
   *
   * @param num_observations  Number of point observations in the batch. This
   * defines the number of factors (residuals) to compute.
   *
   * @param z_threshold  Minimum allowed depth (Z coordinate in camera frame)
   * for valid projection. Points with depth below this threshold produce zero
   * residuals and zero Jacobians to avoid numerical instability from division
   * by very small Z values. Default: 1e-3f (1mm in metric units).
   *
   * @note The observations and poses_camera_from_rig pointers must point to
   * device memory (GPU accessible).
   * @note For points behind the camera (z < z_threshold), the residual and
   * Jacobian are set to zero to prevent numerical issues and gradient
   * corruption.
   */
  ReprojectionFactorBatch(cuBLASHandle& cublas_handle,
                                const Vector<2>* observations,
                                const SE3Transform* poses_camera_from_rig,
                                size_t num_observations,
                                float z_threshold = 1e-3f);

  /**
   * @brief Evaluates residuals and optionally Jacobians for all observations.
   *
   * @param residuals   Output device pointer for residuals. If non-null, must
   *                    have space for num_observations * 2 floats.
   *                    Layout: [r0_x, r0_y, r1_x, r1_y, ...]
   *
   * @param jacobians   Output device pointer for Jacobians. If non-null, must
   *                    have space for num_observations * 18 floats (2 rows × 9
   *                    cols per observation). Pass nullptr to skip Jacobian
   *                    computation.
   *
   *                    Jacobian layout per observation (row-major, 2×9):
   *                      - Columns 0-2: d(r)/d(translation) = d(r)/d(P_cam)
   *                      - Columns 3-5: d(r)/d(rotation) in SO(3) tangent space
   *                      - Columns 6-8: d(r)/d(point) = d(r)/d(P_cam) * R
   *
   * @param state_pointers  Device pointer to array of state block pointers.
   *                    For observation i:
   *                      - state_pointers[i*2]   -> SE3Transform (camera pose, 16
   *                        floats)
   *                      - state_pointers[i*2+1] -> Vector<3> (3D point, 3 floats)
   *
   * @param stream      CUDA stream for asynchronous execution.
   *
   * @return true on success, false on failure.
   *
   * @see reprojection_factor_batch.cu for detailed memory layout
   *      documentation
   */
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Returns the number of factors (observations) in this batch.
   * @return Number of observations passed to the constructor.
   */
  size_t NumFactors() const final { return num_observations_; }

 private:
  ReprojectionFactorBatch() = delete;

  /// Device pointer to normalized 2D observations (intrinsics-free coordinates)
  const Vector<2>* observations_;
  const SE3Transform* poses_camera_from_rig_ = nullptr;

  /// Number of observations in the batch
  size_t num_observations_;

  mutable dvector<SE3Transform> poses_rig_from_world_;
  mutable dvector<Vector<3>> points_world_;

  mutable dvector<SE3Transform> poses_cam_from_world_;

  cuBLASHandle& cublas_handle_;  ///< cuBLAS handle for matrix operations

  /// Minimum depth threshold for valid projection (avoids division by ~zero)
  float z_threshold_ = 1e-3f;
};

}  // namespace cunls
