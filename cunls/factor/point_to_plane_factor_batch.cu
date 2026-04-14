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
 * @file point_to_plane_factor_batch.cu
 * @brief CUDA implementation of batched point-to-plane factor.
 *
 * This file contains the GPU kernel and Evaluate method implementation for
 * computing point-to-plane residuals and Jacobians in parallel across
 * a batch of 3D point correspondences with surface normals.
 *
 * =============================================================================
 * MATHEMATICAL MODEL
 * =============================================================================
 *
 * Residual (scalar):
 *     r = Nq^T * (p - T @ q) = Nq . (p - (R * q + t))
 *
 * where:
 *   - p is the target 3D point (observation)
 *   - q is the source 3D point (observation)
 *   - Nq is the normal vector at q in the source frame (observation)
 *   - T = [R, t; 0, 1] is the SE(3) transformation (state)
 *   - R is the 3x3 rotation matrix
 *   - t = [tx, ty, tz] is the translation vector
 *
 * =============================================================================
 * JACOBIAN DERIVATION
 * =============================================================================
 *
 * Using right perturbation: T' = T * Exp(delta), delta = [omega; rho]
 * where omega (3D) is the rotation component and rho (3D) is the translation.
 *
 * From the point-to-point derivation:
 *     d(p - T@q)/d(delta) = [R * [q]_x | -R]   (3x6 matrix)
 *
 * Since r = Nq^T * (p - T@q), by the chain rule:
 *     d(r)/d(delta) = Nq^T * d(p - T@q)/d(delta)
 *                   = Nq^T * [R * [q]_x | -R]   (1x6 row vector)
 *
 * Let n' = R^T * Nq (the normal rotated into the body frame), then:
 *     Nq^T * R = n'^T
 *
 * Jacobian (1x6):
 *     d(r)/d(omega) = n'^T * [q]_x   (columns 0-2, rotation part)
 *     d(r)/d(rho)   = -n'^T          (columns 3-5, translation part)
 *
 * Expanded:
 *     d(r)/d(omega) = [n'[1]*q[2] - n'[2]*q[1],
 *                      n'[2]*q[0] - n'[0]*q[2],
 *                      n'[0]*q[1] - n'[1]*q[0]]
 *     d(r)/d(rho)   = [-n'[0], -n'[1], -n'[2]]
 *
 * where [q]_x is the skew-symmetric matrix of q:
 *     [q]_x = [  0   -q_z   q_y ]
 *             [ q_z    0   -q_x ]
 *             [-q_y   q_x    0  ]
 *
 * =============================================================================
 * MEMORY LAYOUT
 * =============================================================================
 *
 * Input: States Array
 *     state_pointers[i] -> SE3Transform for correspondence i (16 floats,
 * row-major)
 *
 * SE3Transform storage (row-major 4x4):
 *     [ R00  R01  R02  tx  ]     indices: [0]  [1]  [2]  [3]
 *     [ R10  R11  R12  ty  ]              [4]  [5]  [6]  [7]
 *     [ R20  R21  R22  tz  ]              [8]  [9]  [10] [11]
 *     [  0    0    0    1  ]              [12] [13] [14] [15]
 *
 * Output: Residuals (1 float per correspondence)
 *     residuals[i] = Nq[i] . (p[i] - T @ q[i])
 *
 * Output: Jacobians (1x6 = 6 floats per correspondence, row-major)
 *     | omega_x  omega_y  omega_z  rho_x  rho_y  rho_z |
 *     +------------------------------------------------+
 *     | J0       J1       J2       J3     J4     J5    |
 *     +------------------------------------------------+
 *
 *     Columns 0-2: d(r)/d(omega) = n'^T * [q]_x
 *     Columns 3-5: d(r)/d(rho)   = -n'^T
 *     where n' = R^T * Nq
 */

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/factor/point_to_plane_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block for point-to-plane cost kernel launches.
constexpr size_t kBlockSize = 256;

/**
 * @brief CUDA kernel that computes point-to-plane residuals and Jacobians.
 *
 * For each correspondence i, computes:
 *   - residual[i] = Nq[i] . (p[i] - (R * q[i] + t))
 *   - jacobian[i] = [n'^T * [q[i]]_x | -n'^T]  where n' = R^T * Nq[i]
 *
 * Each thread processes one correspondence independently.
 *
 * @param p_observations  Flattened array of target points (num_correspondences
 * * 3 floats).
 * @param q_observations  Flattened array of source points (num_correspondences
 * * 3 floats).
 * @param nq_observations Flattened array of normal vectors (num_correspondences
 * * 3 floats).
 * @param state_pointers      Array of device pointers, one per correspondence,
 * each pointing to an SE3Transform (16 floats).
 * @param residuals       Output array for residuals (num_correspondences
 * floats), or nullptr to skip.
 * @param jacobians       Output array for Jacobians (num_correspondences * 6
 * floats, row-major 1x6 per correspondence), or nullptr to skip.
 * @param num_correspondences Number of point correspondences (one thread per
 * correspondence).
 *
 * @note Launch configuration: <<<ceil(num_correspondences / kBlockSize),
 * kBlockSize>>>
 */
__global__ void point_to_plane_cost_kernel(const float *p_observations,
                                           const float *q_observations,
                                           const float *nq_observations,
                                           float const *const *state_pointers,
                                           float *residuals, float *jacobians,
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

  // Read observation points and normal
  const float *p = p_observations + tid * kDim;
  const float *q = q_observations + tid * kDim;
  const float *nq = nq_observations + tid * kDim;

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

  if (residuals != nullptr) {
    // Compute r = Nq . (p - (R * q + t))
    float r = 0.0f;
#pragma unroll
    for (int i = 0; i < kDim; i++) {
      float diff_i = p[i] - t[i];
#pragma unroll
      for (int j = 0; j < kDim; j++) {
        diff_i -= R[i][j] * q[j];
      }
      r += nq[i] * diff_i;
    }
    residuals[tid] = r;
  }

  if (jacobians != nullptr) {
    // Jacobian is 1x6 (residual dim x tangent dim), stored row-major
    // J = Nq^T * [R * [q]_x | -R] = [n'^T * [q]_x | -n'^T]
    // where n' = R^T * Nq (normal rotated into body frame)

    // Compute n' = R^T * Nq
    // n'[j] = sum_i R[i][j] * Nq[i]
    float np[3];
#pragma unroll
    for (int j = 0; j < kDim; j++) {
      np[j] = 0.0f;
#pragma unroll
      for (int i = 0; i < kDim; i++) {
        np[j] += R[i][j] * nq[i];
      }
    }

    float *jac_ptr = jacobians + tid * kTangentDim;

    // Rotation Jacobian: n'^T * [q]_x
    //   col 0: np[1]*q[2] - np[2]*q[1]
    //   col 1: np[2]*q[0] - np[0]*q[2]
    //   col 2: np[0]*q[1] - np[1]*q[0]
    jac_ptr[0] = np[1] * q[2] - np[2] * q[1];
    jac_ptr[1] = np[2] * q[0] - np[0] * q[2];
    jac_ptr[2] = np[0] * q[1] - np[1] * q[0];

    // Translation Jacobian: -n'^T
    jac_ptr[3] = -np[0];
    jac_ptr[4] = -np[1];
    jac_ptr[5] = -np[2];
  }
}

bool PointToPlaneFactorBatch::Evaluate(float *residuals, float *jacobians,
                                       float const *const *state_pointers,
                                       cudaStream_t stream) const {
  auto p_data_ptr = reinterpret_cast<const float *>(p_observations_ptr_);
  auto q_data_ptr = reinterpret_cast<const float *>(q_observations_ptr_);
  auto nq_data_ptr = reinterpret_cast<const float *>(nq_observations_ptr_);
  size_t num_factors = NumFactors();

  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;
  point_to_plane_cost_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      p_data_ptr, q_data_ptr, nq_data_ptr, state_pointers, residuals, jacobians,
      num_factors);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

} // namespace cunls
