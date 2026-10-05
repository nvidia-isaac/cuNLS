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
 * @file pnp_factor_batch.cu
 * @brief CUDA implementation of batched PnP (fixed-structure) reprojection.
 *
 * Same residual as ReprojectionFactorBatch but world points are fixed
 * (not state variables). Jacobian is 2x6 (pose only).
 */

#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/projection.cuh"

namespace cunls {

constexpr size_t kPnPBlockSize = 256;

/**
 * @brief Fused kernel: read the world_from_rig pose from state_pointers,
 *        optionally apply camera-from-rig, compute the PnP residual and
 *        Jacobian in one pass (projection.cuh).
 */
__global__ void pnp_fused_kernel(const Vector<2> *observations, const Vector<3> *points_world,
                                 float const *const *state_pointers,
                                 const SE3Transform *poses_camera_from_rig, float *residuals,
                                 float *jacobians, float z_threshold, int num_items,
                                 const int *factor_ids, int num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_items) return;
  const int m = FactorMeasurementIndex(tid, factor_ids, num_factors);

  constexpr int kResidualDim = 2;
  constexpr int kJacobianCols = 6;

  if (residuals == nullptr) return;
  const float *__restrict__ rig = state_pointers[tid];
  const Vector<3> &Pw = points_world[m];
  const float P[3] = {Pw[0], Pw[1], Pw[2]};
  float T[12];  // world_from_rig, rows 0..2
#pragma unroll
  for (int i = 0; i < 12; i++) T[i] = rig[i];
  const float *E = poses_camera_from_rig != nullptr ? poses_camera_from_rig[m].data() : nullptr;

  float p_r[3], p_c[3];
  projection::RigAndCameraPoint(T, E, P, p_r, p_c);
  float *res_ptr = residuals + tid * kResidualDim;
  const bool valid = p_c[2] >= z_threshold;
  float inv_z = 0.0f;
  if (!valid) {
    res_ptr[0] = 0.0f;
    res_ptr[1] = 0.0f;
  } else {
    inv_z = __frcp_rn(p_c[2]);
    const auto &obs = observations[m];
    res_ptr[0] = p_c[0] * inv_z - obs[0];
    res_ptr[1] = p_c[1] * inv_z - obs[1];
  }

  if (jacobians == nullptr) return;
  constexpr int kJacobianBlockSize = kResidualDim * kJacobianCols;
  float *jac_ptr = jacobians + tid * kJacobianBlockSize;
  if (!valid) {
#pragma unroll
    for (int i = 0; i < kJacobianBlockSize; i++) jac_ptr[i] = 0.0f;
    return;
  }
  float J_pose[2][6];
  projection::Jacobian(T, E, p_r, p_c, inv_z, J_pose, nullptr);
#pragma unroll
  for (int i = 0; i < kResidualDim; i++) {
#pragma unroll
    for (int j = 0; j < 6; j++) jac_ptr[i * kJacobianCols + j] = J_pose[i][j];
  }
}

PnPFactorBatch::PnPFactorBatch(const Vector<2> *observations, const Vector<3> *points_world,
                               size_t capacity, float z_threshold)
    : SizedFactorBatch(capacity),
      observations_(observations),
      points_world_(points_world),
      z_threshold_(z_threshold) {}

PnPFactorBatch::PnPFactorBatch(const Vector<2> *observations,
                               const SE3Transform *poses_camera_from_rig,
                               const Vector<3> *points_world, size_t capacity, float z_threshold)
    : SizedFactorBatch(capacity),
      observations_(observations),
      points_world_(points_world),
      poses_camera_from_rig_(poses_camera_from_rig),
      z_threshold_(z_threshold) {}

bool PnPFactorBatch::Evaluate(float *residuals, float *jacobians,
                              float const *const *state_pointers, cudaStream_t stream,
                              const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  const size_t num_blocks = (num_items + kPnPBlockSize - 1) / kPnPBlockSize;
  pnp_fused_kernel<<<num_blocks, kPnPBlockSize, 0, stream>>>(
      observations_, points_world_, state_pointers, poses_camera_from_rig_, residuals, jacobians,
      z_threshold_, static_cast<int>(num_items), factor_ids, static_cast<int>(NumActiveFactors()));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
