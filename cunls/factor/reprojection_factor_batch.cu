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
 * @file reprojection_factor_batch.cu
 * @brief CUDA implementation of batched camera reprojection error computation.
 *
 * =============================================================================
 * COORDINATE SYSTEM AND NORMALIZATION
 * =============================================================================
 *
 * All 2D observations are expected in NORMALIZED image coordinates:
 *     Normalized coords: (x_n, y_n) = ((u - cx)/fx, (v - cy)/fy)
 *
 * =============================================================================
 * MEMORY LAYOUT DOCUMENTATION
 * =============================================================================
 *
 * state_pointers[2*i]     -> SE3Transform for observation i (pose, 16 floats)
 * state_pointers[2*i + 1] -> Vector<3> for observation i (3D point, 3 floats)
 *
 * residuals: [r0_x, r0_y, r1_x, r1_y, ...] (N * 2 floats)
 *
 * jacobians: N blocks of 2x9 row-major matrices (N * 18 floats)
 *   Columns 0-2: d(r)/d(rotation), 3-5: d(r)/d(translation), 6-8: d(r)/d(point)
 *
 * =============================================================================
 * MATHEMATICAL DERIVATION
 * =============================================================================
 *
 * Projection model (pinhole, normalized coordinates):
 *     P_cam = R * P_world + t
 *     proj  = [P_cam.x / P_cam.z, P_cam.y / P_cam.z]
 *     r     = proj - observation
 *
 * Jacobians:
 *   Jc = d(r)/d(P_cam) = [1/z, 0, -x/z^2; 0, 1/z, -y/z^2]
 *   d(r)/d(point) = Jc * R
 *   d(r)/d(translation) = Jc
 *   d(r)/d(rotation) = -Jc * R * [P_world]_x
 */

#include "cunls/factor/reprojection_factor_batch.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

/**
 * @brief Fused kernel: read pose+point from state_pointers, optionally apply
 *        camera-from-rig transform, compute reprojection residual+Jacobian.
 *
 * Replaces: collect_states_kernel + cuBLAS SGEMM (or memcpy) +
 * reprojection_cost_kernel. The SE(3) camera-from-rig multiply is fully
 * unrolled inline, exploiting the [0 0 0 1] last row.
 */
__global__ void reprojection_fused_kernel(
    const Vector<2> *observations, float const *const *state_pointers,
    const SE3Transform *poses_camera_from_rig, float *residuals,
    float *jacobians, float z_threshold, int num_observations) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_observations)
    return;

  constexpr int kResidualDim = 2;
  constexpr int kJacobianCols = 9;

  const float *__restrict__ rig = state_pointers[tid * 2];
  const float *__restrict__ pt = state_pointers[tid * 2 + 1];
  const float P0 = pt[0], P1 = pt[1], P2 = pt[2];

  float pose[12];

  if (poses_camera_from_rig != nullptr) {
    const float *__restrict__ E = poses_camera_from_rig[tid].data();

    const float e00 = E[0], e01 = E[1], e02 = E[2], e03 = E[3];
    const float e10 = E[4], e11 = E[5], e12 = E[6], e13 = E[7];
    const float e20 = E[8], e21 = E[9], e22 = E[10], e23 = E[11];

    const float r00 = rig[0], r01 = rig[1], r02 = rig[2], r03 = rig[3];
    const float r10 = rig[4], r11 = rig[5], r12 = rig[6], r13 = rig[7];
    const float r20 = rig[8], r21 = rig[9], r22 = rig[10], r23 = rig[11];

    pose[0] = e00 * r00 + e01 * r10 + e02 * r20;
    pose[1] = e00 * r01 + e01 * r11 + e02 * r21;
    pose[2] = e00 * r02 + e01 * r12 + e02 * r22;
    pose[3] = e00 * r03 + e01 * r13 + e02 * r23 + e03;
    pose[4] = e10 * r00 + e11 * r10 + e12 * r20;
    pose[5] = e10 * r01 + e11 * r11 + e12 * r21;
    pose[6] = e10 * r02 + e11 * r12 + e12 * r22;
    pose[7] = e10 * r03 + e11 * r13 + e12 * r23 + e13;
    pose[8] = e20 * r00 + e21 * r10 + e22 * r20;
    pose[9] = e20 * r01 + e21 * r11 + e22 * r21;
    pose[10] = e20 * r02 + e21 * r12 + e22 * r22;
    pose[11] = e20 * r03 + e21 * r13 + e22 * r23 + e23;
  } else {
#pragma unroll
    for (int i = 0; i < 12; i++)
      pose[i] = rig[i];
  }

  float point_cam[3];
  float inv_z;

  if (residuals != nullptr) {
    point_cam[0] = pose[3] + pose[0] * P0 + pose[1] * P1 + pose[2] * P2;
    point_cam[1] = pose[7] + pose[4] * P0 + pose[5] * P1 + pose[6] * P2;
    point_cam[2] = pose[11] + pose[8] * P0 + pose[9] * P1 + pose[10] * P2;

    float *res_ptr = residuals + tid * kResidualDim;

    if (point_cam[2] < z_threshold) {
      res_ptr[0] = 0.0f;
      res_ptr[1] = 0.0f;
    } else {
      inv_z = __frcp_rn(point_cam[2]);
      const auto &obs = observations[tid];
      res_ptr[0] = point_cam[0] * inv_z - obs[0];
      res_ptr[1] = point_cam[1] * inv_z - obs[1];
    }
  }

  if (residuals != nullptr && jacobians != nullptr) {
    constexpr int kJacobianBlockSize = kResidualDim * kJacobianCols;
    float *jac_ptr = jacobians + tid * kJacobianBlockSize;

    if (point_cam[2] < z_threshold) {
#pragma unroll
      for (int i = 0; i < kJacobianBlockSize; i++)
        jac_ptr[i] = 0.0f;
      return;
    }

    float Jp[2][3];
    {
      float inv_z_sq = inv_z * inv_z;
      const float &x = point_cam[0];
      const float &y = point_cam[1];

      float a = pose[8] * inv_z_sq;
      float b = pose[9] * inv_z_sq;
      float c = pose[10] * inv_z_sq;

      Jp[0][0] = pose[0] * inv_z - a * x;
      Jp[0][1] = pose[1] * inv_z - b * x;
      Jp[0][2] = pose[2] * inv_z - c * x;

      Jp[1][0] = pose[4] * inv_z - a * y;
      Jp[1][1] = pose[5] * inv_z - b * y;
      Jp[1][2] = pose[6] * inv_z - c * y;
    }

    constexpr int kPoseTangentDim = 6;

#pragma unroll
    for (int i = 0; i < kResidualDim; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        jac_ptr[i * kJacobianCols + 3 + j] = Jp[i][j];
        jac_ptr[i * kJacobianCols + kPoseTangentDim + j] = Jp[i][j];
      }
    }

    jac_ptr[0 * kJacobianCols + 0] = P1 * Jp[0][2] - Jp[0][1] * P2;
    jac_ptr[0 * kJacobianCols + 1] = P2 * Jp[0][0] - Jp[0][2] * P0;
    jac_ptr[0 * kJacobianCols + 2] = P0 * Jp[0][1] - Jp[0][0] * P1;

    jac_ptr[1 * kJacobianCols + 0] = P1 * Jp[1][2] - Jp[1][1] * P2;
    jac_ptr[1 * kJacobianCols + 1] = P2 * Jp[1][0] - Jp[1][2] * P0;
    jac_ptr[1 * kJacobianCols + 2] = P0 * Jp[1][1] - Jp[1][0] * P1;
  }
}

ReprojectionFactorBatch::ReprojectionFactorBatch(const Vector<2> *observations,
                                                 size_t num_observations,
                                                 float z_threshold)
    : observations_(observations), num_observations_(num_observations),
      z_threshold_(z_threshold) {}

ReprojectionFactorBatch::ReprojectionFactorBatch(
    const Vector<2> *observations, const SE3Transform *poses_camera_from_rig,
    size_t num_observations, float z_threshold)
    : observations_(observations),
      poses_camera_from_rig_(poses_camera_from_rig),
      num_observations_(num_observations), z_threshold_(z_threshold) {}

bool ReprojectionFactorBatch::Evaluate(float *residuals, float *jacobians,
                                       float const *const *state_pointers,
                                       cudaStream_t stream) const {
  if (num_observations_ == 0) {
    return true;
  }
  size_t num_blocks = (num_observations_ + kBlockSize - 1) / kBlockSize;

  reprojection_fused_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      observations_, state_pointers, poses_camera_from_rig_, residuals,
      jacobians, z_threshold_, num_observations_);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

} // namespace cunls
