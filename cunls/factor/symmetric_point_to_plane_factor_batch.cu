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
 * @file symmetric_point_to_plane_factor_batch.cu
 * @brief CUDA implementation of batched symmetric point-to-plane factor.
 *
 * This file contains the GPU kernel and Evaluate method implementation for
 * computing symmetric point-to-plane residuals and Jacobians in parallel
 * across a batch of 3D point correspondences with surface normals.
 *
 * =============================================================================
 * MATHEMATICAL MODEL
 * =============================================================================
 *
 * Residual (scalar):
 *     r = (T @ p - T^{-1} @ q) . (Nq + Np)
 *       = ((R*p + t) - R^T*(q - t))^T * N
 *
 * where:
 *   - p is the target 3D point (observation)
 *   - q is the source 3D point (observation)
 *   - Np is the normal vector at p in the target frame (observation)
 *   - Nq is the normal vector at q in the source frame (observation)
 *   - N = Np + Nq is the combined normal
 *   - T = [R, t; 0, 1] is the SE(3) transformation (state)
 *   - R is the 3x3 rotation matrix
 *   - t = [tx, ty, tz] is the translation vector
 *
 * The two terms expand as:
 *   - T @ p   = R*p + t
 *   - T^{-1} @ q = R^T*(q - t) = R^T*q - R^T*t
 *
 * =============================================================================
 * JACOBIAN DERIVATION
 * =============================================================================
 *
 * Using right perturbation: T' = T * Exp(delta), delta = [omega; rho]
 * where omega (3D) is the rotation component and rho (3D) is the translation.
 *
 * Let f = T@p - T^{-1}@q, so r = N^T * f.
 *
 * Perturbation of T@p:
 *     T'@p = R(I + [omega]_x)*p + R*rho + t
 *          = (R*p + t) + R*([omega]_x * p + rho)
 *     d(T@p)/d(omega) = -R * [p]_x
 *     d(T@p)/d(rho)   = R
 *
 * Perturbation of T^{-1}@q (using T'^{-1} = Exp(-delta) * T^{-1}):
 *     T'^{-1}@q = v - omega x v - rho   where v = R^T*(q - t)
 *     d(T^{-1}@q)/d(omega) = [v]_x
 *     d(T^{-1}@q)/d(rho)   = -I
 *
 * Therefore:
 *     df/d(omega) = -R*[p]_x - [v]_x
 *     df/d(rho)   = R + I
 *
 * Jacobian (1x6):
 *     dr/d(omega) = N^T * (-R*[p]_x - [v]_x)
 *                 = -n_R^T * [p]_x - N^T * [v]_x
 *     dr/d(rho)   = N^T * (R + I) = n_R^T + N^T
 *
 * where n_R = R^T * N (combined normal rotated into body frame).
 *
 * Expanded rotation Jacobian:
 *     dr/d(omega)[0] = -(n_R[1]*p[2] - n_R[2]*p[1]) - (N[1]*v[2] - N[2]*v[1])
 *     dr/d(omega)[1] = -(n_R[2]*p[0] - n_R[0]*p[2]) - (N[2]*v[0] - N[0]*v[2])
 *     dr/d(omega)[2] = -(n_R[0]*p[1] - n_R[1]*p[0]) - (N[0]*v[1] - N[1]*v[0])
 *
 * Expanded translation Jacobian:
 *     dr/d(rho)[j] = n_R[j] + N[j]
 *
 * =============================================================================
 * MEMORY LAYOUT
 * =============================================================================
 *
 * Input: States Array
 *     state_pointers[i] -> SE3Transform for correspondence i (16 floats, row-major)
 *
 * SE3Transform storage (row-major 4x4):
 *     [ R00  R01  R02  tx  ]     indices: [0]  [1]  [2]  [3]
 *     [ R10  R11  R12  ty  ]              [4]  [5]  [6]  [7]
 *     [ R20  R21  R22  tz  ]              [8]  [9]  [10] [11]
 *     [  0    0    0    1  ]              [12] [13] [14] [15]
 *
 * Output: Residuals (1 float per correspondence)
 *     residuals[i] = (T@p[i] - T^{-1}@q[i]) . (Nq[i] + Np[i])
 *
 * Output: Jacobians (1x6 = 6 floats per correspondence, row-major)
 *     | omega_x  omega_y  omega_z  rho_x  rho_y  rho_z |
 *     +------------------------------------------------+
 *     | J0       J1       J2       J3     J4     J5    |
 *     +------------------------------------------------+
 *
 *     Columns 0-2: dr/d(omega) = -n_R^T * [p]_x - N^T * [v]_x
 *     Columns 3-5: dr/d(rho)   = n_R^T + N^T
 *     where n_R = R^T * N, v = R^T * (q - t), N = Np + Nq
 */

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block for symmetric point-to-plane kernel launches.
constexpr size_t kBlockSize = 256;

/**
 * @brief CUDA kernel that computes symmetric point-to-plane residuals and
 *        Jacobians.
 *
 * For each correspondence i, computes:
 *   - residual[i] = (T@p[i] - T^{-1}@q[i]) . (Nq[i] + Np[i])
 *   - jacobian[i] = [-n_R^T*[p]_x - N^T*[v]_x | n_R^T + N^T]
 *
 * Each thread processes one correspondence independently.
 *
 * @param p_observations  Flattened array of target points (num * 3 floats).
 * @param q_observations  Flattened array of source points (num * 3 floats).
 * @param np_observations Flattened array of target normals (num * 3 floats).
 * @param nq_observations Flattened array of source normals (num * 3 floats).
 * @param state_pointers      Array of device pointers, one per correspondence,
 *                        each pointing to an SE3Transform (16 floats).
 * @param residuals       Output array for residuals (num floats), or nullptr.
 * @param jacobians       Output array for Jacobians (num * 6 floats,
 *                        row-major 1x6 per correspondence), or nullptr.
 * @param num_correspondences Number of point correspondences.
 *
 * @note Launch configuration: <<<ceil(num / kBlockSize), kBlockSize>>>
 */
__global__ void symmetric_point_to_plane_cost_kernel(
    const float* p_observations, const float* q_observations,
    const float* np_observations, const float* nq_observations,
    float const* const* state_pointers, float* residuals, float* jacobians,
    int num_correspondences) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_correspondences) {
    return;
  }

  constexpr int kDim = 3;
  constexpr int kTangentDim = 6;

  // Read the SE3 transform from the state block
  auto param_ptr = state_pointers[tid];
  assert(param_ptr != nullptr);

  // Read observation points and normals
  const float* p = p_observations + tid * kDim;
  const float* q = q_observations + tid * kDim;
  const float* np_obs = np_observations + tid * kDim;
  const float* nq = nq_observations + tid * kDim;

  // Extract rotation matrix R and translation t from the SE3 transform
  // SE3Transform is row-major 4x4: [R(3x3) | t(3x1); 0 | 1]
  float R[3][3];
  float t[3];

#pragma unroll
  for (int i = 0; i < 3; i++) {
    t[i] = param_ptr[i * 4 + 3];
#pragma unroll
    for (int j = 0; j < 3; j++) {
      R[i][j] = param_ptr[i * 4 + j];
    }
  }

  // Combined normal N = Np + Nq
  float N[3];
#pragma unroll
  for (int i = 0; i < kDim; i++) {
    N[i] = np_obs[i] + nq[i];
  }

  // Compute T @ p = R*p + t
  float Tp[3];
#pragma unroll
  for (int i = 0; i < kDim; i++) {
    Tp[i] = t[i];
#pragma unroll
    for (int j = 0; j < kDim; j++) {
      Tp[i] += R[i][j] * p[j];
    }
  }

  // Compute v = R^T * (q - t) = T^{-1} @ q
  float v[3];
#pragma unroll
  for (int j = 0; j < kDim; j++) {
    v[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < kDim; i++) {
      v[j] += R[i][j] * (q[i] - t[i]);
    }
  }

  if (residuals != nullptr) {
    // Compute r = (T@p - T^{-1}@q) . N where T^{-1}@q = v = R^T*(q - t)
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < kDim; i++) {
      r += N[i] * (Tp[i] - v[i]);
    }
    residuals[tid] = r;
  }

  if (jacobians != nullptr) {
    // Compute n_R = R^T * N (combined normal rotated into body frame)
    float n_R[3];
#pragma unroll
    for (int j = 0; j < kDim; j++) {
      n_R[j] = 0.0f;
#pragma unroll
      for (int i = 0; i < kDim; i++) {
        n_R[j] += R[i][j] * N[i];
      }
    }

    float* jac_ptr = jacobians + tid * kTangentDim;

    // Rotation Jacobian: dr/d(omega) = -n_R^T * [p]_x - N^T * [v]_x
    //
    // n_R^T * [p]_x:
    //   col 0: n_R[1]*p[2] - n_R[2]*p[1]
    //   col 1: n_R[2]*p[0] - n_R[0]*p[2]
    //   col 2: n_R[0]*p[1] - n_R[1]*p[0]
    //
    // N^T * [v]_x:
    //   col 0: N[1]*v[2] - N[2]*v[1]
    //   col 1: N[2]*v[0] - N[0]*v[2]
    //   col 2: N[0]*v[1] - N[1]*v[0]
    jac_ptr[0] = -(n_R[1] * p[2] - n_R[2] * p[1]) -
                  (N[1] * v[2] - N[2] * v[1]);
    jac_ptr[1] = -(n_R[2] * p[0] - n_R[0] * p[2]) -
                  (N[2] * v[0] - N[0] * v[2]);
    jac_ptr[2] = -(n_R[0] * p[1] - n_R[1] * p[0]) -
                  (N[0] * v[1] - N[1] * v[0]);

    // Translation Jacobian: dr/d(rho) = n_R^T + N^T
    jac_ptr[3] = n_R[0] + N[0];
    jac_ptr[4] = n_R[1] + N[1];
    jac_ptr[5] = n_R[2] + N[2];
  }
}

bool SymmetricPointToPlaneFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  auto p_data_ptr = reinterpret_cast<const float*>(p_observations_ptr_);
  auto q_data_ptr = reinterpret_cast<const float*>(q_observations_ptr_);
  auto np_data_ptr = reinterpret_cast<const float*>(np_observations_ptr_);
  auto nq_data_ptr = reinterpret_cast<const float*>(nq_observations_ptr_);
  size_t num_factors = NumFactors();

  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;
  symmetric_point_to_plane_cost_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      p_data_ptr, q_data_ptr, np_data_ptr, nq_data_ptr, state_pointers, residuals,
      jacobians, num_factors);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
