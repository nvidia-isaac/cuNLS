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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cunls/common/helper.h"
#include "cunls/math/lie_device.cuh"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t block_size = 256;  ///< Default thread block size for CUDA kernels.

//-------------------------------- KERNELS --------------------------------

/**
 * @brief CUDA kernel to compute the SO(3) logarithm map for a batch of
 * rotations.
 *
 * Each thread maps one 3x3 rotation matrix to a 3D twist vector.
 *
 * @param rotation_matrix Input rotation matrices (3x3, device pointer).
 * @param rotation_pitch Pitch (stride between rows) of each rotation matrix.
 * @param rotation_stride Stride between consecutive rotation matrices.
 * @param twist_stride Stride between consecutive output twist vectors.
 * @param size Number of rotation matrices to process.
 * @param twist Output twist vectors (3D, device pointer).
 */
__global__ void log_so3_kernel(const float *rotation_matrix, const size_t rotation_pitch,
                               const size_t rotation_stride, const size_t twist_stride, size_t size,
                               float *twist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *twist_ptr = twist + tid * twist_stride;
  const float *rotation_matrix_ptr = rotation_matrix + tid * rotation_stride;

  lie_device::LogSO3(rotation_matrix_ptr, rotation_pitch, twist_ptr);
}

/**
 * @brief CUDA kernel to compute the SE(3) exponential map for a batch of
 * twists.
 *
 * Each thread maps one 6D twist vector xi = [phi, rho] to a 4x4 homogeneous
 * transformation matrix T using the SO(3) left Jacobian for the translation
 * part.
 *
 * @param twist Input twist vectors (6D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param transform Output transformation matrices (4x4, device pointer).
 * @param transform_pitch Pitch (stride between rows) of each transform matrix.
 * @param transform_stride Stride between consecutive transform matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void exp_se3_kernel(const float *twist, const size_t twist_stride, float *transform,
                               const size_t transform_pitch, const size_t transform_stride,
                               size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];  // copied first: twist and transform may alias
  memcpy(xi, twist + tid * twist_stride, 6 * sizeof(float));
  lie_device::ExpSE3(xi, transform + tid * transform_stride, transform_pitch);
}

/**
 * @brief CUDA kernel to compute the inverse right Jacobian of SO(3):
 * J_r^{-1}(phi) = J_l^{-1}(-phi).
 *
 * @param twist Input twist vectors (3D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param jacobian_inv Output inverse Jacobian matrices (3x3, device pointer).
 * @param jacobian_inv_pitch Pitch (stride between rows) of each inverse
 * Jacobian matrix.
 * @param jacobian_inv_stride Stride between consecutive inverse Jacobian
 * matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void __launch_bounds__(256, 4)
    jacobian_right_inverse_so3_kernel(const float *twist, const size_t twist_stride,
                                      float *jacobian_inv, const size_t jacobian_inv_pitch,
                                      const size_t jacobian_inv_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *jacobian_inv_ptr = jacobian_inv + tid * jacobian_inv_stride;
  const float *twist_ptr = twist + tid * twist_stride;
  const float temp[3] = {-twist_ptr[0], -twist_ptr[1], -twist_ptr[2]};
  lie_device::SO3JacobianLeftInverse(temp, jacobian_inv_ptr, jacobian_inv_pitch);
}

/**
 * @brief CUDA kernel to compute the SE(3) logarithm map for a batch of
 * transforms.
 *
 * Each thread maps one 4x4 homogeneous transformation matrix T to a 6D twist
 * vector xi = [phi, rho] by computing Log(R) for the rotation part and applying
 * the inverse left Jacobian to the translation part.
 *
 * @param transform Input transformation matrices (4x4, device pointer).
 * @param transform_pitch Pitch (stride between rows) of each transform matrix.
 * @param transform_stride Stride between consecutive transform matrices.
 * @param twist_stride Stride between consecutive output twist vectors.
 * @param size Number of transforms to process.
 * @param twist Output twist vectors (6D, device pointer).
 */
__global__ void log_se3_kernel(const float *transform, const size_t transform_pitch,
                               const size_t transform_stride, const size_t twist_stride,
                               size_t size, float *twist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];  // written last: twist and transform may alias
  lie_device::LogSE3(transform + tid * transform_stride, transform_pitch, xi);
  memcpy(twist + tid * twist_stride, xi, 6 * sizeof(float));
}

/**
 * @brief CUDA kernel to compute the left Jacobian J_l(xi) of SE(3): a 6x6
 * matrix with structure [J_l(phi), 0; Q(xi), J_l(phi)].
 *
 * @param twist Input twist vectors (6D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param jacobian Output Jacobian matrices (6x6, device pointer).
 * @param jacobian_pitch Pitch (stride between rows) of each Jacobian matrix.
 * @param jacobian_stride Stride between consecutive Jacobian matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void jacobian_left_se3_kernel(const float *twist, const size_t twist_stride,
                                         float *jacobian, const size_t jacobian_pitch,
                                         const size_t jacobian_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];
#pragma unroll
  for (int i = 0; i < 6; i++) xi[i] = twist[tid * twist_stride + i];
  lie_device::SE3JacobianLeft(xi, jacobian + tid * jacobian_stride, jacobian_pitch);
}

/**
 * @brief CUDA kernel to compute the inverse right Jacobian of SE(3),
 * J_r^{-1}(xi) = J_l^{-1}(-xi), for a batch of 6D twist vectors
 * (lie_device::SE3JacobianLeftInverse):
 *   [ J_l^{-1}(phi)                          |      0          ]
 *   [ -J_l^{-1}(phi) @ Q(xi) @ J_l^{-1}(phi)| J_l^{-1}(phi)  ]
 * evaluated at -xi.
 */
constexpr size_t se3_jac_inv_block_size = 128;  ///< Block size for SE(3) inverse Jacobian kernel
                                                ///< (tuned for register pressure).

__global__ void __launch_bounds__(128, 4)
    jacobian_right_inverse_se3_kernel(const float *__restrict__ twist, const size_t twist_stride,
                                      float *__restrict__ jacobian, const size_t jacobian_pitch,
                                      const size_t jacobian_stride, size_t size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];
#pragma unroll
  for (int i = 0; i < 6; i++) xi[i] = -twist[tid * twist_stride + i];
  lie_device::SE3JacobianLeftInverse(xi, jacobian + tid * jacobian_stride, jacobian_pitch);
}

//-------------------------------- FUNCTIONS --------------------------------

/** @copydoc ComputeLogSO3 */
void ComputeLogSO3(cudaStream_t stream, const float *rotation, const size_t rotation_pitch,
                   const size_t rotation_stride, const size_t twist_stride, size_t size,
                   float *twist) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  log_so3_kernel<<<num_blocks, block_size, 0, stream>>>(rotation, rotation_pitch, rotation_stride,
                                                        twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSO3 */
void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_inv_pitch,
                                    const size_t jacobian_inv_stride, size_t size,
                                    float *jacobian_inv) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  jacobian_right_inverse_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      twist, twist_stride, jacobian_inv, jacobian_inv_pitch, jacobian_inv_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeExpSE3 */
void ComputeExpSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                   const size_t transform_pitch, const size_t transform_stride, size_t size,
                   float *transform) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  exp_se3_kernel<<<num_blocks, block_size, 0, stream>>>(twist, twist_stride, transform,
                                                        transform_pitch, transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeLogSE3 */
void ComputeLogSE3(cudaStream_t stream, const float *transform, const size_t transform_pitch,
                   const size_t transform_stride, const size_t twist_stride, size_t size,
                   float *twist) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  log_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      transform, transform_pitch, transform_stride, twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftSE3 */
void ComputeJacobianLeftSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                            const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                            float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  jacobian_left_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSE3 */
void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size, float *jacobian) {
  size_t num_blocks = (size + se3_jac_inv_block_size - 1) / se3_jac_inv_block_size;
  jacobian_right_inverse_se3_kernel<<<num_blocks, se3_jac_inv_block_size, 0, stream>>>(
      twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// SO(2) Kernels
// ============================================================================

constexpr size_t kSO2MathBlockSize = 256;

/**
 * @brief SO(2) logarithm map: 2x2 rotation matrix -> angle.
 *
 * Extracts the angle from R via Log(R) = atan2(R[1,0], R[0,0]).
 * For R = [[c,-s],[s,c]], this returns atan2(s, c) = theta.
 */
__global__ void log_so2_kernel(const float *rotations, size_t rotation_stride, float *angles,
                               size_t angle_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *R = rotations + idx * rotation_stride;
  angles[idx * angle_stride] = atan2f(R[2], R[0]);
}

void ComputeLogSO2(cudaStream_t stream, const float *rotations, size_t rotation_stride,
                   size_t angle_stride, size_t size, float *angles) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  log_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(rotations, rotation_stride, angles,
                                                               angle_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// SE(2) Kernels
// ============================================================================

constexpr size_t kSE2MathBlockSize = 256;

/**
 * @brief SE(2) logarithm map: 3x3 transform -> tangent vector.
 *
 * For T = [R t; 0 1], computes xi = [v_x, v_y, theta] where:
 *   theta = atan2(sin, cos) from R
 *   [v_x, v_y] = V(theta)^{-1} * t
 *
 * with V^{-1} = [[h, theta/2], [-theta/2, h]], h = (theta/2) cot(theta/2)
 * (float32-safe for all theta, including near 0).
 */
__global__ void log_se2_kernel(const float *transforms, size_t transform_stride, float *tangent,
                               size_t tangent_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  lie_device::LogSE2(transforms + idx * transform_stride, tangent + idx * tangent_stride);
}

/**
 * @brief Inverse right Jacobian of SE(2): J_r^{-1}(xi).
 *
 * For xi = [v_x, v_y, theta], computes the 3x3 matrix J_r^{-1}(xi) that
 * relates perturbations in tangent space.
 *
 * For large |alpha|:
 *   J_r^{-1} = [[alpha*cot(alpha/2)/2,  -alpha/2,      v1/alpha - v1*cot/2 +
 * v2/2], [alpha/2,                alpha*cot/2,    v2/alpha - v1/2 - v2*cot/2],
 *               [0,                      0,              1 ]] where cot =
 * sin(alpha)/(1 - cos(alpha)).
 *
 * Evaluated by lie_device::SE2JrInv (float32-safe series near alpha = 0).
 */
__global__ void __launch_bounds__(256, 4)
    jacobian_right_inverse_se2_kernel(const float *tangent, size_t tangent_stride, float *jacobians,
                                      size_t jacobian_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *xi = tangent + idx * tangent_stride;
  float *J = jacobians + idx * jacobian_stride;

  lie_device::SE2JrInv(xi[0], xi[1], xi[2], J);
}

void ComputeLogSE2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                   size_t tangent_stride, size_t size, float *tangent) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  log_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(transforms, transform_stride,
                                                               tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSE2(cudaStream_t stream, const float *tangent,
                                    size_t tangent_stride, size_t jacobian_stride, size_t size,
                                    float *jacobians) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  jacobian_right_inverse_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
