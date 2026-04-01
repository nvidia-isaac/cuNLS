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

/**
 * @file reprojection_factor_batch.cu
 * @brief CUDA implementation of batched camera reprojection error computation.
 *
 * This file contains the GPU kernel and host-side evaluation function for
 * computing reprojection residuals and Jacobians in parallel across a batch of
 * 3D-to-2D point correspondences.
 *
 * =============================================================================
 * COORDINATE SYSTEM AND NORMALIZATION
 * =============================================================================
 *
 * All 2D observations are expected in NORMALIZED image coordinates. This means
 * the camera intrinsic state_pointers (focal length, principal point) have
 * already been factored out:
 *
 *     Normalized coords: (x_n, y_n) = ((u - cx)/fx, (v - cy)/fy)
 *
 * This allows the reprojection model to be camera-agnostic, supporting:
 *   - Multi-camera systems with different intrinsics
 *   - Simplified Jacobian computation (no K matrix derivatives)
 *   - Clean separation between geometric and photometric calibration
 *
 * =============================================================================
 * MEMORY LAYOUT DOCUMENTATION
 * =============================================================================
 *
 * This section describes the memory layout of the input state_pointers, output
 * residuals, and output Jacobians. Understanding this layout is crucial for
 * correctly interfacing with the optimizer and for efficient GPU memory access.
 *
 * -----------------------------------------------------------------------------
 * INPUT: States Array
 * -----------------------------------------------------------------------------
 *
 * The `state_pointers` array is a device pointer to an array of state block
 * pointers. For N observations, the layout is:
 *
 *     state_pointers[0]   -> SE3Transform for observation 0 (pose, 16 floats)
 *     state_pointers[1]   -> Vector<3> for observation 0 (3D point, 3 floats)
 *     state_pointers[2]   -> SE3Transform for observation 1
 *     state_pointers[3]   -> Vector<3> for observation 1
 *     ...
 *     state_pointers[2*i]     -> SE3Transform for observation i
 *     state_pointers[2*i + 1] -> Vector<3> for observation i
 *
 * SE3Transform storage (16 floats, row-major 4x4 matrix, last row omitted in
 * use):
 *
 *     [ R00  R01  R02  tx  ]     indices: [0]  [1]  [2]  [3]
 *     [ R10  R11  R12  ty  ]              [4]  [5]  [6]  [7]
 *     [ R20  R21  R22  tz  ]              [8]  [9]  [10] [11]
 *     [  0    0    0    1  ]              [12] [13] [14] [15] (implicit)
 *
 * Vector<3> storage (3 floats): [x, y, z]
 *
 * -----------------------------------------------------------------------------
 * OUTPUT: Residuals Memory Layout
 * -----------------------------------------------------------------------------
 *
 * The residuals are stored contiguously for all observations. For N
 * observations with residual dimension 2, the total size is N * 2 floats.
 *
 * Memory layout (contiguous, flat array):
 *
 *     residuals[0]  = r0_x   (observation 0, x-component of reprojection error)
 *     residuals[1]  = r0_y   (observation 0, y-component of reprojection error)
 *     residuals[2]  = r1_x   (observation 1, x-component)
 *     residuals[3]  = r1_y   (observation 1, y-component)
 *     ...
 *     residuals[2*i]     = ri_x
 *     residuals[2*i + 1] = ri_y
 *
 * Stride per observation: sizeof(Vector<2>) = 2 floats = 8 bytes
 *
 * Visual representation for 3 observations:
 *
 *     +------+------+------+------+------+------+
 *     | r0_x | r0_y | r1_x | r1_y | r2_x | r2_y |
 *     +------+------+------+------+------+------+
 *       obs 0         obs 1         obs 2
 *
 * -----------------------------------------------------------------------------
 * OUTPUT: Jacobians Memory Layout
 * -----------------------------------------------------------------------------
 *
 * The Jacobian matrix for each observation has dimensions:
 *   - Rows: 2 (residual dimension: x and y reprojection error)
 *   - Cols: 9 (6 for SE3 pose tangent + 3 for 3D point = 6 + 3)
 *
 * Total Jacobian block size per observation: 2 * 9 = 18 floats
 *
 * The Jacobian is stored in ROW-MAJOR order within each observation's block.
 * The columns are ordered as: [pose (6 DOF)] [point (3 DOF)]
 *
 * For a single observation, the Jacobian block layout is:
 *
 *                  |<--- SE3 Pose (6 cols) --->|<- Point (3) ->|
 *                  | t_x  t_y  t_z  r_x  r_y  r_z  p_x  p_y  p_z |
 *     +-----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 *     | row 0 (x) | J00 | J01 | J02 | J03 | J04 | J05 | J06 | J07 | J08 |
 *     +-----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 *     | row 1 (y) | J10 | J11 | J12 | J13 | J14 | J15 | J16 | J17 | J18 |
 *     +-----------+-----+-----+-----+-----+-----+-----+-----+-----+-----+
 *
 * Where:
 *   - J[0,0:6] = d(r_x) / d(pose)   (6 values, SE(3) tangent space)
 *   - J[0,6:9] = d(r_x) / d(point)  (3 values)
 *   - J[1,0:6] = d(r_y) / d(pose)
 *   - J[1,6:9] = d(r_y) / d(point)
 *
 * Jacobian layout per observation (row-major, 2×9):
 *   - Columns 0-5: d(r)/d(pose) in SE(3) tangent space
 *   - Columns 6-8: d(r)/d(point) = d(r)/d(P_cam) * R
 *
 * Memory layout for multiple observations (N observations):
 *
 *     jacobians[0..17]   = Jacobian block for observation 0 (18 floats)
 *     jacobians[18..35]  = Jacobian block for observation 1
 *     ...
 *     jacobians[18*i .. 18*i+17] = Jacobian block for observation i
 *
 * Flattened memory for observation i:
 *
 *     base = jacobians + i * 18
 *     base[0]  = J00 (d(r_x)/d(t_x))
 *     base[1]  = J01 (d(r_x)/d(t_y))
 *     base[2]  = J02 (d(r_x)/d(t_z))
 *     base[3]  = J03 (d(r_x)/d(r_x))
 *     base[4]  = J04 (d(r_x)/d(r_y))
 *     base[5]  = J05 (d(r_x)/d(r_z))
 *     base[6]  = J06 (d(r_x)/d(p_x))
 *     base[7]  = J07 (d(r_x)/d(p_y))
 *     base[8]  = J08 (d(r_x)/d(p_z))
 *     base[9]  = J10 (d(r_y)/d(t_x))
 *     base[10] = J11 (d(r_y)/d(t_y))
 *     ...
 *     base[17] = J18 (d(r_y)/d(p_z))
 *
 * Note: The Jacobian has 9 columns (6 for pose tangent + 3 for point).
 *       Each observation's Jacobian block is 2 rows x 9 cols = 18 floats.
 *       The stride between observations is 18 floats.
 *
 * =============================================================================
 * MATHEMATICAL DERIVATION
 * =============================================================================
 *
 * Projection model (pinhole, normalized coordinates):
 *
 *     P_cam = R * P_world + t       (transform to camera frame)
 *     proj  = [P_cam.x / P_cam.z]   (perspective projection)
 *             [P_cam.y / P_cam.z]
 *     r     = proj - observation    (residual)
 *
 * Let P_cam = [x, y, z]^T, then:
 *
 *     r = [x/z - obs_x]
 *         [y/z - obs_y]
 *
 * Jacobian layout per observation (row-major, 2×9):
 *   - Columns 0-5: d(r)/d(pose) in SE(3) tangent space
 *   - Columns 6-8: d(r)/d(point) = d(r)/d(P_cam) * R
 *
 * -----------------------------------------------------------------------------
 * Projection Jacobian (Jc):
 * -----------------------------------------------------------------------------
 *
 *     Jc = d(r)/d(P_cam) = [ 1/z    0   -x/z^2 ]
 *                          [  0    1/z  -y/z^2 ]
 *
 * -----------------------------------------------------------------------------
 * Pose Jacobian (columns 0-5): d(r)/d(pose) in SE(3) tangent space
 * -----------------------------------------------------------------------------
 *
 * The pose Jacobian combines rotation and translation derivatives.
 *
 * Translation part:
 *     d(P_cam)/d(t) = I (identity)
 *     d(r)/d(t) = d(r)/d(P_cam) * I = Jc
 *
 * Rotation part (SO(3) right perturbation R' = R * exp([w]_x)):
 *     P_cam' = R * exp([w]_x) * P_world + t
 *            ≈ R * (I + [w]_x) * P_world + t
 *            = P_cam + R * (w × P_world)
 *
 *     d(P_cam)/d(w) = -R * [P_world]_x
 *
 *     d(r)/d(w) = d(r)/d(P_cam) * d(P_cam)/d(w)
 *               = Jc * (-R * [P_world]_x)
 *               = -(Jc * R) * [P_world]_x
 *               = -Jp * [P_world]_x
 *
 * -----------------------------------------------------------------------------
 * Point Jacobian (columns 6-8): d(r)/d(P_world)
 * -----------------------------------------------------------------------------
 *
 *     d(P_cam)/d(P_world) = R
 *     Jp = d(r)/d(P_world) = d(r)/d(P_cam) * R = Jc * R
 *
 * =============================================================================
 */

#include <cublas_v2.h>

#include "cunls/factor/reprojection_factor_batch.h"

namespace cunls {

/// CUDA block size for kernel launch configuration
constexpr size_t kBlockSize = 256;

/**
 * @brief CUDA kernel to collect poses and 3D points from state pointers.
 *
 * Extracts interleaved SE(3) poses and 3D points from the flat state
 * pointer array into separate contiguous arrays for efficient batch processing.
 *
 * @param state_pointers         Device pointer to state block pointers.
 *                           Layout: [pose_0, point_0, pose_1, point_1, ...].
 * @param poses_rig_from_world  Output array for SE(3) rig-from-world poses
 *                              (num_observations elements).
 * @param points_world       Output array for 3D world points
 *                           (num_observations elements).
 * @param num_observations   Number of observations (one thread per
 * observation).
 *
 * @note Launch configuration: <<<ceil(num_observations / kBlockSize),
 * kBlockSize>>>
 */
__global__ void collect_states_kernel(float const* const* state_pointers,
                                      SE3Transform* poses_rig_from_world,
                                      Vector<3>* points_world,
                                      int num_observations) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_observations) {
    return;
  }

  poses_rig_from_world[tid] =
      *reinterpret_cast<const SE3Transform*>(state_pointers[tid * 2]);
  points_world[tid] =
      *reinterpret_cast<const Vector<3>*>(state_pointers[tid * 2 + 1]);
}

/**
 * @brief CUDA kernel for computing reprojection residuals and Jacobians.
 *
 * Each thread processes one observation independently. This provides
 * excellent parallelism as observations are independent of each other.
 *
 * @param observations   Device pointer to normalized 2D observations (N
 * elements)
 * @param state_pointers     Device pointer to state block pointers (2*N
 * elements)
 * @param residuals      Output: device pointer for residuals (N * 2 floats)
 * @param jacobians      Output: device pointer for Jacobians (N * 18
 * floats), or nullptr
 * @param z_threshold    Minimum depth for valid projection
 * @param num_observations  Number of observations to process
 *
 * @note Thread assignment: thread i processes observation i
 * @note Memory coalescing: residuals and Jacobians are written with stride,
 *       which may not be optimal. Consider structure-of-arrays for better
 * coalescing.
 */
__global__ void reprojection_cost_kernel(
    const Vector<2>* observations, const Vector<3>* points_world,
    const SE3Transform* poses_cam_from_world, float* residuals,
    float* jacobians, float z_threshold, int num_observations) {
  // Compute global thread ID - each thread handles one observation
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_observations) {
    return;
  }

  // -------------------------------------------------------------------------
  // Extract state_pointers for this observation
  // -------------------------------------------------------------------------
  const Vector<3>& P = points_world[tid];
  const SE3Transform& pose = poses_cam_from_world[tid];

  // Stride constants for output arrays (in number of floats, not bytes)
  constexpr int kResidualDim = 2;   // 2D reprojection error
  constexpr int kJacobianCols = 9;  // 6 (pose tangent) + 3 (point)

  // Point in camera frame after transformation
  float point_cam[3];
  float inv_z;

  // -------------------------------------------------------------------------
  // Compute residuals: r = project(T * P) - observation
  // -------------------------------------------------------------------------
  if (residuals != nullptr) {
    // Initialize with translation component: point_cam = t
    // SE3Transform is row-major 4x4, translation is in column 3
    point_cam[0] = pose[0 * 4 + 3];  // tx at index 3
    point_cam[1] = pose[1 * 4 + 3];  // ty at index 7
    point_cam[2] = pose[2 * 4 + 3];  // tz at index 11

    // Add rotation contribution: point_cam = R * P + t
    // R is the upper-left 3x3 of the 4x4 matrix
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        point_cam[i] += pose[i * 4 + j] * P[j];
      }
    }

    // Compute residual output pointer for this observation
    float* res_ptr = residuals + tid * kResidualDim;

    // Check if point is in front of camera (positive depth)
    if (point_cam[2] < z_threshold) {
      // Point behind camera or too close - zero out residual to avoid
      // numerical instability and prevent gradient from pushing solution
      // through the singularity
      res_ptr[0] = 0.0f;
      res_ptr[1] = 0.0f;
    } else {
      // Compute inverse depth using fast reciprocal (CUDA intrinsic)
      inv_z = __frcp_rn(point_cam[2]);
      // Residual = projected_point - observed_point (in normalized coords)
      // projected_point = [x/z, y/z] in normalized image coordinates
      const auto& obs = observations[tid];
      res_ptr[0] = point_cam[0] * inv_z - obs[0];  // x-component error
      res_ptr[1] = point_cam[1] * inv_z - obs[1];  // y-component error
    }
  }

  // -------------------------------------------------------------------------
  // Compute Jacobians (only if both residuals and jacobians are requested)
  // -------------------------------------------------------------------------
  if (residuals != nullptr && jacobians != nullptr) {
    // Jacobian block size per observation: 2 rows x 9 cols = 18 floats
    constexpr int kJacobianBlockSize = kResidualDim * kJacobianCols;

    // Compute Jacobian output pointer for this observation
    float* jac_ptr = jacobians + tid * kJacobianBlockSize;

    // Zero out Jacobians for invalid projections
    if (point_cam[2] < z_threshold) {
#pragma unroll
      for (int i = 0; i < kJacobianBlockSize; i++) {
        jac_ptr[i] = 0.0f;
      }
      return;
    }

    // -----------------------------------------------------------------------
    // Compute Jacobian w.r.t. point in world frame: Jp = d(r)/d(P_world)
    // -----------------------------------------------------------------------
    // This is the chain rule: d(r)/d(P_world) = d(r)/d(P_cam) *
    // d(P_cam)/d(P_world)
    //                                         = d(r)/d(P_cam) * R
    //
    // d(r)/d(P_cam) = [ 1/z    0   -x/z^2 ]  (2x3 matrix)
    //                 [  0    1/z  -y/z^2 ]
    //
    // Jp = d(r)/d(P_cam) * R combines projection and rotation Jacobians
    float Jp[2][3];
    {
      float inv_z_sq = inv_z * inv_z;
      const float& x = point_cam[0];
      const float& y = point_cam[1];

      // Precompute terms for the third row of R scaled by 1/z^2
      // These appear in -x/z^2 * R[2,:] and -y/z^2 * R[2,:]
      float a = pose[2 * 4 + 0] * inv_z_sq;  // R[2,0] / z^2
      float b = pose[2 * 4 + 1] * inv_z_sq;  // R[2,1] / z^2
      float c = pose[2 * 4 + 2] * inv_z_sq;  // R[2,2] / z^2

      // Jp[0,:] = d(r_x)/d(P) = (1/z) * R[0,:] - (x/z^2) * R[2,:]
      Jp[0][0] = pose[0 * 4 + 0] * inv_z - a * x;
      Jp[0][1] = pose[0 * 4 + 1] * inv_z - b * x;
      Jp[0][2] = pose[0 * 4 + 2] * inv_z - c * x;

      // Jp[1,:] = d(r_y)/d(P) = (1/z) * R[1,:] - (y/z^2) * R[2,:]
      Jp[1][0] = pose[1 * 4 + 0] * inv_z - a * y;
      Jp[1][1] = pose[1 * 4 + 1] * inv_z - b * y;
      Jp[1][2] = pose[1 * 4 + 2] * inv_z - c * y;
    }

    // -----------------------------------------------------------------------
    // Write Jacobians to output buffer
    // -----------------------------------------------------------------------
    // Jacobian layout per observation (row-major, 2×9):
    //   - Columns 0-5: d(r)/d(pose) in SE(3) tangent space
    //   - Columns 6-8: d(r)/d(point) = d(r)/d(P_cam) * R
    constexpr int kPoseTangentDim = 6;  // Offset to point Jacobian columns

    // -----------------------------------------------------------------------
    // Pose Jacobian (columns 0-5) and Point Jacobian (columns 6-8)
    // -----------------------------------------------------------------------
#pragma unroll
    for (int i = 0; i < kResidualDim; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        // Pose Jacobian columns 3-5 (translation part)
        jac_ptr[i * kJacobianCols + 3 + j] = Jp[i][j];
        // Point Jacobian columns 6-8
        jac_ptr[i * kJacobianCols + kPoseTangentDim + j] = Jp[i][j];
      }
    }

    // Pose Jacobian columns 0-2 (rotation part)
    jac_ptr[0 * kJacobianCols + 0] = P[1] * Jp[0][2] - Jp[0][1] * P[2];
    jac_ptr[0 * kJacobianCols + 1] = P[2] * Jp[0][0] - Jp[0][2] * P[0];
    jac_ptr[0 * kJacobianCols + 2] = P[0] * Jp[0][1] - Jp[0][0] * P[1];

    jac_ptr[1 * kJacobianCols + 0] = P[1] * Jp[1][2] - Jp[1][1] * P[2];
    jac_ptr[1 * kJacobianCols + 1] = P[2] * Jp[1][0] - Jp[1][2] * P[0];
    jac_ptr[1 * kJacobianCols + 2] = P[0] * Jp[1][1] - Jp[1][0] * P[1];
  }
}

/**
 * @brief Constructs the reprojection factor batch with identity
 *        camera-from-rig transforms.
 *
 * This constructor assumes camera_from_rig poses are all identity transforms,
 * meaning the camera frame coincides with the rig frame. Suitable for
 * single-camera systems or when camera extrinsics are pre-baked into poses.
 *
 * @param cublas_handle  Reference to an externally-owned cuBLAS handle.
 * @param observations   Device pointer to normalized 2D observations
 * @param num_observations  Number of observations in the batch
 * @param z_threshold    Minimum depth threshold for valid projections
 */
ReprojectionFactorBatch::ReprojectionFactorBatch(
    cuBLASHandle& cublas_handle, const Vector<2>* observations,
    size_t num_observations, float z_threshold)
    : observations_(observations),
      num_observations_(num_observations),
      poses_rig_from_world_(num_observations),
      points_world_(num_observations),
      poses_cam_from_world_(num_observations),
      cublas_handle_(cublas_handle),
      z_threshold_(z_threshold) {}

/**
 * @brief Constructs the reprojection factor batch with custom
 *        camera-from-rig transforms.
 *
 * This constructor allows specifying custom camera_from_rig poses for
 * multi-camera systems. The final camera pose is computed as:
 *     pose_cam_from_world = pose_camera_from_rig * pose_rig_from_world
 *
 * @param cublas_handle  Reference to an externally-owned cuBLAS handle.
 * @param observations   Device pointer to normalized 2D observations
 * @param poses_camera_from_rig  Device pointer to SE3 transforms for each
 *                               observation, representing camera pose relative
 *                               to the rig body frame
 * @param num_observations  Number of observations in the batch
 * @param z_threshold    Minimum depth threshold for valid projections
 */
ReprojectionFactorBatch::ReprojectionFactorBatch(
    cuBLASHandle& cublas_handle, const Vector<2>* observations,
    const SE3Transform* poses_camera_from_rig, size_t num_observations,
    float z_threshold)
    : observations_(observations),
      poses_camera_from_rig_(poses_camera_from_rig),
      num_observations_(num_observations),
      poses_rig_from_world_(num_observations),
      points_world_(num_observations),
      poses_cam_from_world_(num_observations),
      cublas_handle_(cublas_handle),
      z_threshold_(z_threshold) {}

/**
 * @brief Evaluates the reprojection factor for all observations.
 *
 * Launches the CUDA kernel to compute residuals and optionally Jacobians
 * for all observations in parallel.
 *
 * @param residuals   Output buffer for residuals (device memory)
 * @param jacobians   Output buffer for Jacobians (device memory), or nullptr
 * @param state_pointers  Array of state block pointers (device memory)
 * @param stream      CUDA stream for asynchronous execution
 *
 * @return true on success
 * @throws CUDA error if kernel launch fails
 */
bool ReprojectionFactorBatch::Evaluate(float* residuals, float* jacobians,
                                       float const* const* state_pointers,
                                       cudaStream_t stream) const {
  if (num_observations_ == 0) {
    return true;
  }
  // Compute grid dimensions: one thread per observation
  size_t num_blocks = (num_observations_ + kBlockSize - 1) / kBlockSize;

  collect_states_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, poses_rig_from_world_.data(), points_world_.data(),
      num_observations_);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  if (poses_camera_from_rig_ != nullptr) {
    auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
    const size_t mat_size = 4;
    const size_t stride = 16;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    // cuBLAS uses column-major format, but SE3Transform is row-major.
    // To compute C = A * B in row-major using cuBLAS, we swap the order: pass B
    // first, then A. This exploits: (A_row * B_row)^T = B_row^T * A_row^T, and
    // cuBLAS interprets our row-major data as transposed column-major.
    // Result: poses_cam_from_world = poses_camera_from_rig * poses_rig_from_world
    THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
        reinterpret_cast<const float*>(poses_rig_from_world_.data()), mat_size,
        stride, reinterpret_cast<const float*>(poses_camera_from_rig_),
        mat_size, stride, &beta,
        reinterpret_cast<float*>(poses_cam_from_world_.data()), mat_size,
        stride, num_observations_));
  } else {
    // No custom camera_from_rig transforms provided - assume identity
    // (camera frame coincides with rig frame)
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
        poses_cam_from_world_.data(), poses_rig_from_world_.data(),
        num_observations_ * sizeof(SE3Transform), cudaMemcpyDeviceToDevice,
        stream));
  }

  // Launch kernel asynchronously on the specified stream
  reprojection_cost_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      observations_, points_world_.data(), poses_cam_from_world_.data(),
      residuals, jacobians, z_threshold_, num_observations_);

  // Check for kernel launch errors (asynchronous - doesn't wait for completion)
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  return true;
}

}  // namespace cunls