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
 * Projection model (pinhole, normalized coordinates), pose state T =
 * world_from_rig = (R, t), optional camera_from_rig E:
 *     P_rig = R^T (P_world - t),   P_cam = E P_rig
 *     r     = [P_cam.x / P_cam.z, P_cam.y / P_cam.z] - observation
 *
 * Jacobians (right perturbation T Exp([phi, rho]), rig frame; see
 * the helpers below), with A = d(proj)/d(P_cam) * E_R:
 *   d(r)/d(rotation)    = A [P_rig]_x   (row i: A_i x P_rig)
 *   d(r)/d(translation) = -A
 *   d(r)/d(point)       = A R^T
 */

#include "cunls/factor/reprojection_factor_batch.h"

namespace cunls {

namespace {

/**
 * Factor (measurement) index of evaluation item `item`: factor_ids[item], or
 * item modulo the batch size when factor_ids is null (see FactorBatch::Evaluate).
 */
__device__ __forceinline__ int FactorMeasurementIndex(int item, const int *factor_ids,
                                                      int num_factors) {
  if (factor_ids != nullptr) {
    return factor_ids[item];
  }
  return item < num_factors ? item : item % num_factors;
}

// Pinhole projection through a world_from_rig pose state (the same helpers
// as in the other projection factor's kernel file).
//
// T = (R, t) is world_from_rig (rows 0..2 of a row-major 4x4, pitch 4), E an
// optional camera_from_rig (same layout; nullptr = identity):
//
//   p_r = Rᵀ (P - t),   p_c = E p_r,   r = p_c.xy / p_c.z - obs
//
// The state is perturbed on the right, T Exp(δ), δ = [φ, ρ] in the rig frame:
// p_r ← Exp(δ)⁻¹ p_r ≈ p_r + [p_r]x φ - ρ. With A = J_c E_R (2x3, J_c =
// ∂(x/z, y/z)/∂p_c), row i of the Jacobian is
//
//   ∂r_i/∂φ = A_i × p_r,   ∂r_i/∂ρ = -A_i,   ∂r_i/∂P = A_i Rᵀ.
//
// The rotation columns scale with the depth p_r, not with the distance from
// the world origin; P - t is formed first, so far-away scenes keep their
// precision.

/** p_r = Rᵀ (P - t) and p_c = E p_r (p_c = p_r without E). */
__device__ __forceinline__ void RigAndCameraPoint(const float *__restrict__ T,
                                                  const float *__restrict__ E, const float *P,
                                                  float *p_r, float *p_c) {
  const float d0 = P[0] - T[3], d1 = P[1] - T[7], d2 = P[2] - T[11];
#pragma unroll
  for (int j = 0; j < 3; ++j) p_r[j] = T[j] * d0 + T[4 + j] * d1 + T[8 + j] * d2;
  if (E != nullptr) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      p_c[i] = E[i * 4 + 3] + E[i * 4] * p_r[0] + E[i * 4 + 1] * p_r[1] + E[i * 4 + 2] * p_r[2];
  } else {
#pragma unroll
    for (int i = 0; i < 3; ++i) p_c[i] = p_r[i];
  }
}

/**
 * Jacobian rows (see the file comment) for a valid projection (inv_z = 1 /
 * p_c.z): pose[i][0..5] = [A_i × p_r, -A_i]; point[i][0..2] = A_i Rᵀ when
 * `point` is not null.
 */
__device__ __forceinline__ void ProjectionJacobian(const float *__restrict__ T,
                                                   const float *__restrict__ E, const float *p_r,
                                                   const float *p_c, float inv_z, float (*pose)[6],
                                                   float (*point)[3]) {
  const float u = p_c[0] * inv_z, v = p_c[1] * inv_z;
  // J_c = inv_z [[1, 0, -u], [0, 1, -v]].
  float A[2][3];
  if (E != nullptr) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      A[0][j] = inv_z * (E[j] - u * E[8 + j]);
      A[1][j] = inv_z * (E[4 + j] - v * E[8 + j]);
    }
  } else {
    A[0][0] = inv_z, A[0][1] = 0.f, A[0][2] = -u * inv_z;
    A[1][0] = 0.f, A[1][1] = inv_z, A[1][2] = -v * inv_z;
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float *a = A[i];
    pose[i][0] = a[1] * p_r[2] - a[2] * p_r[1];
    pose[i][1] = a[2] * p_r[0] - a[0] * p_r[2];
    pose[i][2] = a[0] * p_r[1] - a[1] * p_r[0];
    pose[i][3] = -a[0];
    pose[i][4] = -a[1];
    pose[i][5] = -a[2];
    if (point != nullptr) {
#pragma unroll
      for (int j = 0; j < 3; ++j)
        point[i][j] = a[0] * T[j * 4] + a[1] * T[j * 4 + 1] + a[2] * T[j * 4 + 2];
    }
  }
}

}  // namespace

constexpr size_t kBlockSize = 256;

/**
 * @brief Fused kernel: read the world_from_rig pose and the point from
 *        state_pointers, optionally apply camera-from-rig, compute the
 *        reprojection residual and Jacobian in one pass.
 */
__global__ void reprojection_fused_kernel(const Vector<2> *observations,
                                          float const *const *state_pointers,
                                          const SE3Transform *poses_camera_from_rig,
                                          float *residuals, float *jacobians, float z_threshold,
                                          int num_items, const int *factor_ids, int num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_items) return;
  const int m = FactorMeasurementIndex(tid, factor_ids, num_factors);

  constexpr int kResidualDim = 2;
  constexpr int kJacobianCols = 9;

  if (residuals == nullptr) return;
  const float *__restrict__ rig = state_pointers[tid * 2];
  const float *__restrict__ pt = state_pointers[tid * 2 + 1];
  const float P[3] = {pt[0], pt[1], pt[2]};
  float T[12];  // world_from_rig, rows 0..2
#pragma unroll
  for (int i = 0; i < 12; i++) T[i] = rig[i];
  const float *E = poses_camera_from_rig != nullptr ? poses_camera_from_rig[m].data() : nullptr;

  float p_r[3], p_c[3];
  RigAndCameraPoint(T, E, P, p_r, p_c);
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
  float J_pose[2][6], J_point[2][3];
  ProjectionJacobian(T, E, p_r, p_c, inv_z, J_pose, J_point);
#pragma unroll
  for (int i = 0; i < kResidualDim; i++) {
#pragma unroll
    for (int j = 0; j < 6; j++) jac_ptr[i * kJacobianCols + j] = J_pose[i][j];
#pragma unroll
    for (int j = 0; j < 3; j++) jac_ptr[i * kJacobianCols + 6 + j] = J_point[i][j];
  }
}

ReprojectionFactorBatch::ReprojectionFactorBatch(const Vector<2> *observations, size_t capacity,
                                                 float z_threshold)
    : SizedFactorBatch(capacity), observations_(observations), z_threshold_(z_threshold) {}

ReprojectionFactorBatch::ReprojectionFactorBatch(const Vector<2> *observations,
                                                 const SE3Transform *poses_camera_from_rig,
                                                 size_t capacity, float z_threshold)
    : SizedFactorBatch(capacity),
      observations_(observations),
      poses_camera_from_rig_(poses_camera_from_rig),
      z_threshold_(z_threshold) {}

bool ReprojectionFactorBatch::Evaluate(float *residuals, float *jacobians,
                                       float const *const *state_pointers, cudaStream_t stream,
                                       const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  const size_t num_blocks = (num_items + kBlockSize - 1) / kBlockSize;
  reprojection_fused_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      observations_, state_pointers, poses_camera_from_rig_, residuals, jacobians, z_threshold_,
      static_cast<int>(num_items), factor_ids, static_cast<int>(NumActiveFactors()));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
