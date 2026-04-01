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
 * @file pnp_factor_batch.cu
 * @brief CUDA implementation of batched PnP (fixed-structure) reprojection.
 *
 * The residual computation and pose derivatives mirror `reprojection_cost_kernel`
 * in `reprojection_factor_batch.cu`, except world points are read from the
 * constructor-owned pointer `points_world_` (constant per factor instance)
 * and the Jacobian omits d r / d P_world.
 *
 * =============================================================================
 * MEMORY LAYOUT
 * =============================================================================
 *
 * **state_pointers** (length `N`):
 *     state_pointers[i] -> SE3Transform for correspondence i (16 floats).
 *
 * **residuals** (length `2 * N`):
 *     `[r0_x, r0_y, r1_x, r1_y, ...]`.
 *
 * **jacobians** (length `12 * N`, row-major `2 × 6` blocks):
 *     Block i starts at `jacobians + i * 12`.
 *     Columns 0–2: rotation tangent; columns 3–5: translation tangent, in the
 *     same convention as `ReprojectionFactorBatch` pose columns.
 *
 * =============================================================================
 * DERIVATIVES (sketch)
 * =============================================================================
 *
 * With \(P_{\mathrm{cam}} = R P_{\mathrm{world}} + t\) and normalized
 * projection \(\pi(x,y,z) = (x/z,\, y/z)\):
 *
 *     J_c = \partial r / \partial P_{\mathrm{cam}}
 *         = [ 1/z,   0, -x/z^2 ; 0, 1/z, -y/z^2 ].
 *
 * Pose derivatives follow the same SE(3) right-perturbation derivation as
 * bundle-adjustment reprojection; see `reprojection_factor_batch.cu`.
 */

#include <cublas_v2.h>

#include "cunls/factor/pnp_factor_batch.h"

namespace cunls {

constexpr size_t kPnPBlockSize = 256;

__global__ void pnp_collect_poses_kernel(float const* const* state_pointers,
                                         SE3Transform* poses_rig_from_world,
                                         int num_observations) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_observations) {
    return;
  }
  poses_rig_from_world[tid] =
      *reinterpret_cast<const SE3Transform*>(state_pointers[tid]);
}

__global__ void pnp_cost_kernel(const Vector<2>* observations,
                                const Vector<3>* points_world,
                                const SE3Transform* poses_cam_from_world,
                                float* residuals, float* jacobians,
                                float z_threshold, int num_observations) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_observations) {
    return;
  }

  constexpr int kResidualDim = 2;
  constexpr int kJacobianCols = 6;

  const Vector<3>& P = points_world[tid];
  const SE3Transform& pose = poses_cam_from_world[tid];

  float point_cam[3];
  float inv_z = 0.0f;

  if (residuals != nullptr) {
    point_cam[0] = pose[0 * 4 + 3];
    point_cam[1] = pose[1 * 4 + 3];
    point_cam[2] = pose[2 * 4 + 3];

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        point_cam[i] += pose[i * 4 + j] * P[j];
      }
    }

    float* res_ptr = residuals + tid * kResidualDim;

    if (point_cam[2] < z_threshold) {
      res_ptr[0] = 0.0f;
      res_ptr[1] = 0.0f;
    } else {
      inv_z = __frcp_rn(point_cam[2]);
      const auto& obs = observations[tid];
      res_ptr[0] = point_cam[0] * inv_z - obs[0];
      res_ptr[1] = point_cam[1] * inv_z - obs[1];
    }
  }

  if (residuals != nullptr && jacobians != nullptr) {
    constexpr int kJacobianBlockSize = kResidualDim * kJacobianCols;
    float* jac_ptr = jacobians + tid * kJacobianBlockSize;

    if (point_cam[2] < z_threshold) {
#pragma unroll
      for (int i = 0; i < kJacobianBlockSize; i++) {
        jac_ptr[i] = 0.0f;
      }
      return;
    }

    float Jp[2][3];
    {
      float inv_z_sq = inv_z * inv_z;
      const float& x = point_cam[0];
      const float& y = point_cam[1];

      float a = pose[2 * 4 + 0] * inv_z_sq;
      float b = pose[2 * 4 + 1] * inv_z_sq;
      float c = pose[2 * 4 + 2] * inv_z_sq;

      Jp[0][0] = pose[0 * 4 + 0] * inv_z - a * x;
      Jp[0][1] = pose[0 * 4 + 1] * inv_z - b * x;
      Jp[0][2] = pose[0 * 4 + 2] * inv_z - c * x;

      Jp[1][0] = pose[1 * 4 + 0] * inv_z - a * y;
      Jp[1][1] = pose[1 * 4 + 1] * inv_z - b * y;
      Jp[1][2] = pose[1 * 4 + 2] * inv_z - c * y;
    }

#pragma unroll
    for (int i = 0; i < kResidualDim; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        jac_ptr[i * kJacobianCols + 3 + j] = Jp[i][j];
      }
    }

    jac_ptr[0 * kJacobianCols + 0] = P[1] * Jp[0][2] - Jp[0][1] * P[2];
    jac_ptr[0 * kJacobianCols + 1] = P[2] * Jp[0][0] - Jp[0][2] * P[0];
    jac_ptr[0 * kJacobianCols + 2] = P[0] * Jp[0][1] - Jp[0][0] * P[1];

    jac_ptr[1 * kJacobianCols + 0] = P[1] * Jp[1][2] - Jp[1][1] * P[2];
    jac_ptr[1 * kJacobianCols + 1] = P[2] * Jp[1][0] - Jp[1][2] * P[0];
    jac_ptr[1 * kJacobianCols + 2] = P[0] * Jp[1][1] - Jp[1][0] * P[1];
  }
}

PnPFactorBatch::PnPFactorBatch(cuBLASHandle& cublas_handle,
                               const Vector<2>* observations,
                               const Vector<3>* points_world,
                               size_t num_observations, float z_threshold)
    : observations_(observations),
      points_world_(points_world),
      num_observations_(num_observations),
      poses_rig_from_world_(num_observations),
      poses_cam_from_world_(num_observations),
      cublas_handle_(cublas_handle),
      z_threshold_(z_threshold) {}

PnPFactorBatch::PnPFactorBatch(cuBLASHandle& cublas_handle,
                               const Vector<2>* observations,
                               const SE3Transform* poses_camera_from_rig,
                               const Vector<3>* points_world,
                               size_t num_observations, float z_threshold)
    : observations_(observations),
      points_world_(points_world),
      poses_camera_from_rig_(poses_camera_from_rig),
      num_observations_(num_observations),
      poses_rig_from_world_(num_observations),
      poses_cam_from_world_(num_observations),
      cublas_handle_(cublas_handle),
      z_threshold_(z_threshold) {}

bool PnPFactorBatch::Evaluate(float* residuals, float* jacobians,
                              float const* const* state_pointers,
                              cudaStream_t stream) const {
  if (num_observations_ == 0) {
    return true;
  }

  size_t num_blocks =
      (num_observations_ + kPnPBlockSize - 1) / kPnPBlockSize;

  pnp_collect_poses_kernel<<<num_blocks, kPnPBlockSize, 0, stream>>>(
      state_pointers, poses_rig_from_world_.data(),
      static_cast<int>(num_observations_));
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  if (poses_camera_from_rig_ != nullptr) {
    auto handle =
        static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
    const size_t mat_size = 4;
    const size_t stride = 16;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
        reinterpret_cast<const float*>(poses_rig_from_world_.data()), mat_size,
        stride, reinterpret_cast<const float*>(poses_camera_from_rig_),
        mat_size, stride, &beta,
        reinterpret_cast<float*>(poses_cam_from_world_.data()), mat_size,
        stride, num_observations_));
  } else {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
        poses_cam_from_world_.data(), poses_rig_from_world_.data(),
        num_observations_ * sizeof(SE3Transform), cudaMemcpyDeviceToDevice,
        stream));
  }

  pnp_cost_kernel<<<num_blocks, kPnPBlockSize, 0, stream>>>(
      observations_, points_world_, poses_cam_from_world_.data(), residuals,
      jacobians, z_threshold_, static_cast<int>(num_observations_));

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
