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

/**
 * @brief Device helper to swap two values.
 *
 * @tparam T Type of the values to swap.
 * @param a First value (swapped with b).
 * @param b Second value (swapped with a).
 */
template <typename T>
__device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

//-------------------------------- KERNELS --------------------------------

/**
 * @brief CUDA kernel to compute skew-symmetric matrices from 3D twists.
 *
 * Processes a batch of twist vectors and computes their skew-symmetric
 * matrices.
 *
 * @param twist Input twist vectors (3D, device pointer)
 * @param twist_stride Stride between twist vectors
 * @param skew Output skew-symmetric matrices (device pointer)
 * @param skew_pitch Pitch (stride between rows) of skew matrices
 * @param skew_stride Stride between skew matrices
 * @param size Number of twists to process
 */
__global__ void skew_so3_kernel(const float *twist, const size_t twist_stride, float *skew,
                                const size_t skew_pitch, const size_t skew_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *skew_ptr = skew + tid * skew_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  skew_ptr[0 * skew_pitch + 0] = 0;
  skew_ptr[0 * skew_pitch + 1] = -twist_ptr[2];
  skew_ptr[0 * skew_pitch + 2] = twist_ptr[1];

  skew_ptr[1 * skew_pitch + 0] = twist_ptr[2];
  skew_ptr[1 * skew_pitch + 1] = 0;
  skew_ptr[1 * skew_pitch + 2] = -twist_ptr[0];

  skew_ptr[2 * skew_pitch + 0] = -twist_ptr[1];
  skew_ptr[2 * skew_pitch + 1] = twist_ptr[0];
  skew_ptr[2 * skew_pitch + 2] = 0;
}

/**
 * @brief CUDA kernel to compute the SO(3) exponential map for a batch of
 * twists.
 *
 * Each thread processes one twist vector and outputs a 3x3 rotation matrix
 * using Rodrigues' formula.
 *
 * @param twist Input twist vectors (3D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param exp Output rotation matrices (3x3, device pointer).
 * @param exp_pitch Pitch (stride between rows) of each rotation matrix.
 * @param exp_stride Stride between consecutive rotation matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void exp_so3_kernel(const float *twist, const size_t twist_stride, float *exp,
                               const size_t exp_pitch, const size_t exp_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *exp_ptr = exp + tid * exp_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  lie_device::ExpSO3(twist_ptr, exp_ptr, exp_pitch);
}

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
 * @brief CUDA kernel to compute the left or right Jacobian of SO(3).
 *
 * When @p left is true, computes J_l(phi). When false, computes J_r(phi) by
 * negating the twist before computing J_l(-phi).
 *
 * @param left If true, computes the left Jacobian; otherwise the right
 * Jacobian.
 * @param twist Input twist vectors (3D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param jacobian Output Jacobian matrices (3x3, device pointer).
 * @param jacobian_pitch Pitch (stride between rows) of each Jacobian matrix.
 * @param jacobian_stride Stride between consecutive Jacobian matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void jacobian_so3_kernel(bool left, const float *twist, const size_t twist_stride,
                                    float *jacobian, const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *jacobian_ptr = jacobian + tid * jacobian_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  float temp[3];
  memcpy(temp, twist_ptr, 3 * sizeof(float));
  if (!left) {
#pragma unroll
    for (uint8_t i = 0; i < 3; i++) {
      temp[i] = -temp[i];
    }
  }

  lie_device::SO3JacobianLeft(temp, jacobian_ptr, jacobian_pitch);
}

/**
 * @brief CUDA kernel to compute the inverse of the left or right Jacobian of
 * SO(3).
 *
 * When @p left is true, computes J_l^{-1}(phi). When false, computes
 * J_r^{-1}(phi) by negating the twist before computing J_l^{-1}(-phi).
 *
 * @param left If true, computes the inverse left Jacobian; otherwise the
 * inverse right.
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
    jacobian_inverse_so3_kernel(bool left, const float *twist, const size_t twist_stride,
                                float *jacobian_inv, const size_t jacobian_inv_pitch,
                                const size_t jacobian_inv_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *jacobian_inv_ptr = jacobian_inv + tid * jacobian_inv_stride;
  const float *twist_ptr = twist + tid * twist_stride;
  float temp[3];
  memcpy(temp, twist_ptr, 3 * sizeof(float));
  if (!left) {
#pragma unroll
    for (uint8_t i = 0; i < 3; i++) {
      temp[i] = -temp[i];
    }
  }

  lie_device::SO3JacobianLeftInverse(temp, jacobian_inv_ptr, jacobian_inv_pitch);
}

/**
 * @brief CUDA kernel to negate a batch of matrices element-wise.
 *
 * Each thread processes one matrix, computing dst = -src for every element.
 *
 * @param rows Number of rows in each matrix.
 * @param cols Number of columns in each matrix.
 * @param src_matrix Input matrices (device pointer).
 * @param pitch Pitch (stride between rows) of each matrix.
 * @param stride Stride between consecutive matrices in the batch.
 * @param num_matrices Number of matrices to process.
 * @param dst_matrix Output negated matrices (device pointer).
 */
__global__ void negate_matrices_kernel(const size_t rows, const size_t cols,
                                       const float *src_matrix, const size_t pitch,
                                       const size_t stride, size_t num_matrices,
                                       float *dst_matrix) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_matrices) {
    return;
  }

  float *dst_matrix_ptr = dst_matrix + tid * stride;
  const float *src_matrix_ptr = src_matrix + tid * stride;

#pragma unroll
  for (uint8_t i = 0; i < rows; i++) {
#pragma unroll
    for (uint8_t j = 0; j < cols; j++) {
      dst_matrix_ptr[i * pitch + j] = -src_matrix_ptr[i * pitch + j];
    }
  }
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
 * @brief CUDA kernel to compute the adjoint or inverse adjoint of SE(3).
 *
 * When @p inverse is false, computes Ad_T = [R, 0; [t]_x R, R].
 * When @p inverse is true, computes Ad_T^{-1} = [R^T, 0; -[t]_x R^T, R^T].
 *
 * @param inverse If false, computes the adjoint; if true, the inverse adjoint.
 * @param transform Input transformation matrices (4x4, device pointer).
 * @param transform_pitch Pitch (stride between rows) of each transform matrix.
 * @param transform_stride Stride between consecutive transform matrices.
 * @param adjoint_pitch Pitch (stride between rows) of each adjoint matrix.
 * @param adjoint_stride Stride between consecutive adjoint matrices.
 * @param size Number of transforms to process.
 * @param adjoint Output adjoint matrices (6x6, device pointer).
 */
__global__ void adjoint_se3_kernel(bool inverse, const float *transform,
                                   const size_t transform_pitch, const size_t transform_stride,
                                   const size_t adjoint_pitch, const size_t adjoint_stride,
                                   size_t size, float *adjoint) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *adjoint_ptr = adjoint + tid * adjoint_stride;
  const float *transform_ptr = transform + tid * transform_stride;

  float translation[3];
  translation[0] = transform_ptr[0 * transform_pitch + 3];
  translation[1] = transform_ptr[1 * transform_pitch + 3];
  translation[2] = transform_ptr[2 * transform_pitch + 3];

  float R[9];
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *dst = &R[i * 3];
    const float *src = &transform_ptr[i * transform_pitch];
    memcpy(dst, src, 3 * sizeof(float));
  }

  // For Ad(T^{-1}) = Ad(T)^{-1}, use R' = R^T and t' = -R^T * t (the
  // rotation/translation of T^{-1}); the same block formula then applies.
  if (inverse) {
    swap(R[0 * 3 + 1], R[1 * 3 + 0]);
    swap(R[0 * 3 + 2], R[2 * 3 + 0]);
    swap(R[1 * 3 + 2], R[2 * 3 + 1]);

    float t0 = translation[0], t1 = translation[1], t2 = translation[2];
    translation[0] = -(R[0] * t0 + R[1] * t1 + R[2] * t2);
    translation[1] = -(R[3] * t0 + R[4] * t1 + R[5] * t2);
    translation[2] = -(R[6] * t0 + R[7] * t1 + R[8] * t2);
  }

  // Ad(T) = [[R, 0], [skew(t) * R, R]], one row at a time (shared with the
  // factor kernels that derive the adjoint of a measurement inline).
#pragma unroll
  for (int row = 0; row < 6; ++row) {
    float out[6];
    lie_device::AdjointSE3Row(R, translation, row, out);
#pragma unroll
    for (int j = 0; j < 6; ++j) adjoint_ptr[row * adjoint_pitch + j] = out[j];
  }
}

/**
 * @brief CUDA kernel to compute the left or right Jacobian of SE(3).
 *
 * When @p left is true, computes J_l(xi). When false, computes J_r(xi) by
 * negating the twist before computing J_l(-xi). The output is a 6x6 matrix
 * with structure [J_l(phi), 0; Q(xi), J_l(phi)].
 *
 * @param left If true, computes the left Jacobian; otherwise the right
 * Jacobian.
 * @param twist Input twist vectors (6D, device pointer).
 * @param twist_stride Stride between consecutive twist vectors.
 * @param jacobian Output Jacobian matrices (6x6, device pointer).
 * @param jacobian_pitch Pitch (stride between rows) of each Jacobian matrix.
 * @param jacobian_stride Stride between consecutive Jacobian matrices.
 * @param size Number of twist vectors to process.
 */
__global__ void jacobian_se3_kernel(bool left, const float *twist, const size_t twist_stride,
                                    float *jacobian, const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];
  const float sign = left ? 1.f : -1.f;  // J_r(xi) = J_l(-xi)
#pragma unroll
  for (int i = 0; i < 6; i++) xi[i] = sign * twist[tid * twist_stride + i];
  lie_device::SE3JacobianLeft(xi, jacobian + tid * jacobian_stride, jacobian_pitch);
}

/**
 * @brief CUDA kernel to compute the inverse Jacobian of SE(3).
 *
 * Computes J_l^{-1}(xi) or J_r^{-1}(xi) = J_l^{-1}(-xi) for a batch of 6D twist
 * vectors (lie_device::SE3JacobianLeftInverse):
 *   [ J_l^{-1}(phi)                          |      0          ]
 *   [ -J_l^{-1}(phi) @ Q(xi) @ J_l^{-1}(phi)| J_l^{-1}(phi)  ]
 */
constexpr size_t se3_jac_inv_block_size = 128;  ///< Block size for SE(3) inverse Jacobian kernel
                                                ///< (tuned for register pressure).

__global__ void __launch_bounds__(128, 4)
    jacobian_inverse_se3_kernel(bool left, const float *__restrict__ twist,
                                const size_t twist_stride, float *__restrict__ jacobian,
                                const size_t jacobian_pitch, const size_t jacobian_stride,
                                size_t size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  float xi[6];
  const float sign = left ? 1.f : -1.f;  // J_r^{-1}(xi) = J_l^{-1}(-xi)
#pragma unroll
  for (int i = 0; i < 6; i++) xi[i] = sign * twist[tid * twist_stride + i];
  lie_device::SE3JacobianLeftInverse(xi, jacobian + tid * jacobian_stride, jacobian_pitch);
}

/**
 * @brief CUDA kernel to compute the inverse of SE(3) transformation matrices.
 *
 * For T = [R, t; 0, 1], computes T^{-1} = [R^T, -R^T t; 0, 1].
 * Each thread processes one transformation matrix.
 *
 * @param transform Input transformation matrices (4x4, device pointer).
 * @param transform_pitch Pitch (stride between rows) of each transform matrix.
 * @param transform_stride Stride between consecutive transform matrices.
 * @param inverse_pitch Pitch (stride between rows) of each inverse matrix.
 * @param inverse_stride Stride between consecutive inverse matrices.
 * @param size Number of transforms to process.
 * @param inverse_transform Output inverse transformation matrices (4x4, device
 * pointer).
 */
__global__ void inverse_se3_kernel(const float *transform, const size_t transform_pitch,
                                   const size_t transform_stride, const size_t inverse_pitch,
                                   const size_t inverse_stride, size_t size,
                                   float *inverse_transform) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *inverse_transform_ptr = inverse_transform + tid * inverse_stride;
  const float *transform_ptr = transform + tid * transform_stride;

  float pose[16];
  lie_device::InverseSE3(transform_ptr, transform_pitch, pose);

#pragma unroll
  for (uint8_t i = 0; i < 4; i++) {
    const float *src = &pose[i * 4];
    float *dst = &inverse_transform_ptr[i * inverse_pitch];
    memcpy(dst, src, 4 * sizeof(float));
  }
}
//-------------------------------- FUNCTIONS --------------------------------

/**
 * @brief Host function wrapper for ComputeSkewSO3 kernel.
 *
 * Launches the CUDA kernel to compute skew-symmetric matrices from twist
 * vectors.
 */
void ComputeSkewSO3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                    const size_t skew_pitch, const size_t skew_stride, size_t size, float *skew) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  skew_so3_kernel<<<num_blocks, block_size, 0, stream>>>(twist, twist_stride, skew, skew_pitch,
                                                         skew_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeNegateMatrix */
void ComputeNegateMatrix(cudaStream_t stream, const float *matrix, size_t rows, size_t cols,
                         const size_t pitch, const size_t stride, size_t size,
                         float *negated_matrix) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  negate_matrices_kernel<<<num_blocks, block_size, 0, stream>>>(rows, cols, matrix, pitch, stride,
                                                                size, negated_matrix);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeInverseSE3 */
void ComputeInverseSE3(cudaStream_t stream, const float *transform, const size_t transform_pitch,
                       const size_t transform_stride, const size_t inverse_pitch,
                       const size_t inverse_stride, size_t size, float *inverse_transform) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  inverse_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      transform, transform_pitch, transform_stride, inverse_pitch, inverse_stride, size,
      inverse_transform);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeExpSO3 */
void ComputeExpSO3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                   const size_t rotation_pitch, const size_t rotation_stride, size_t size,
                   float *rotation) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  exp_so3_kernel<<<num_blocks, block_size, 0, stream>>>(twist, twist_stride, rotation,
                                                        rotation_pitch, rotation_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}
/** @copydoc ComputeLogSO3 */
void ComputeLogSO3(cudaStream_t stream, const float *rotation, const size_t rotation_pitch,
                   const size_t rotation_stride, const size_t twist_stride, size_t size,
                   float *twist) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  log_so3_kernel<<<num_blocks, block_size, 0, stream>>>(rotation, rotation_pitch, rotation_stride,
                                                        twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftSO3 */
void ComputeJacobianLeftSO3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                            const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                            float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_so3_kernel<<<num_blocks, block_size, 0, stream>>>(left, twist, twist_stride, jacobian,
                                                             jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightSO3 */
void ComputeJacobianRightSO3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                             const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                             float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_so3_kernel<<<num_blocks, block_size, 0, stream>>>(left, twist, twist_stride, jacobian,
                                                             jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftInverseSO3 */
void ComputeJacobianLeftInverseSO3(cudaStream_t stream, const float *twist,
                                   const size_t twist_stride, const size_t jacobian_inv_pitch,
                                   const size_t jacobian_inv_stride, size_t size,
                                   float *jacobian_inv) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_inverse_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian_inv, jacobian_inv_pitch, jacobian_inv_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSO3 */
void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_inv_pitch,
                                    const size_t jacobian_inv_stride, size_t size,
                                    float *jacobian_inv) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_inverse_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian_inv, jacobian_inv_pitch, jacobian_inv_stride, size);
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

/** @copydoc ComputeAdjointSE3 */
void ComputeAdjointSE3(cudaStream_t stream, const float *transform, const size_t transform_pitch,
                       const size_t transform_stride, const size_t adjoint_pitch,
                       const size_t adjoint_stride, size_t size, float *adjoint) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  bool inverse = false;
  adjoint_se3_kernel<<<num_blocks, block_size, 0, stream>>>(inverse, transform, transform_pitch,
                                                            transform_stride, adjoint_pitch,
                                                            adjoint_stride, size, adjoint);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeInverseAdjointSE3 */
void ComputeInverseAdjointSE3(cudaStream_t stream, const float *transform,
                              const size_t transform_pitch, const size_t transform_stride,
                              const size_t inv_adjoint_pitch, const size_t inv_adjoint_stride,
                              size_t size, float *inv_adjoint) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  bool inverse = true;
  adjoint_se3_kernel<<<num_blocks, block_size, 0, stream>>>(inverse, transform, transform_pitch,
                                                            transform_stride, inv_adjoint_pitch,
                                                            inv_adjoint_stride, size, inv_adjoint);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftSE3 */
void ComputeJacobianLeftSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                            const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                            float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_se3_kernel<<<num_blocks, block_size, 0, stream>>>(left, twist, twist_stride, jacobian,
                                                             jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftInverseSE3 */
void ComputeJacobianLeftInverseSE3(cudaStream_t stream, const float *twist,
                                   const size_t twist_stride, const size_t jacobian_pitch,
                                   const size_t jacobian_stride, size_t size, float *jacobian) {
  size_t num_blocks = (size + se3_jac_inv_block_size - 1) / se3_jac_inv_block_size;
  constexpr bool left = true;
  jacobian_inverse_se3_kernel<<<num_blocks, se3_jac_inv_block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightSE3 */
void ComputeJacobianRightSE3(cudaStream_t stream, const float *twist, const size_t twist_stride,
                             const size_t jacobian_pitch, const size_t jacobian_stride, size_t size,
                             float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_se3_kernel<<<num_blocks, block_size, 0, stream>>>(left, twist, twist_stride, jacobian,
                                                             jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSE3 */
void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride, const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size, float *jacobian) {
  size_t num_blocks = (size + se3_jac_inv_block_size - 1) / se3_jac_inv_block_size;
  constexpr bool left = false;
  jacobian_inverse_se3_kernel<<<num_blocks, se3_jac_inv_block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}
/**
 * @brief Transposes a batch of 3x3 rotation matrices.
 *
 * For R = [[a,b,c],[d,e,f],[g,h,i]], computes R^T = [[a,d,g],[b,e,h],[c,f,i]].
 */
__global__ void transpose_so3_kernel(const float *rotations, size_t input_stride, float *transposed,
                                     size_t output_stride, size_t n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)n) {
    return;
  }
  const float *R = rotations + tid * input_stride;
  float *Rt = transposed + tid * output_stride;
  Rt[0] = R[0];
  Rt[1] = R[3];
  Rt[2] = R[6];
  Rt[3] = R[1];
  Rt[4] = R[4];
  Rt[5] = R[7];
  Rt[6] = R[2];
  Rt[7] = R[5];
  Rt[8] = R[8];
}

/** @copydoc ComputeTransposeSO3 */
void ComputeTransposeSO3(cudaStream_t stream, const float *rotation, size_t input_stride,
                         size_t output_stride, size_t size, float *transposed) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  transpose_so3_kernel<<<num_blocks, block_size, 0, stream>>>(rotation, input_stride, transposed,
                                                              output_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// SO(2) Kernels
// ============================================================================

constexpr size_t kSO2MathBlockSize = 256;

/**
 * @brief SO(2) exponential map: angle -> 2x2 rotation matrix.
 *
 * For angle theta, computes:
 *   R = [[cos(theta), -sin(theta)],
 *        [sin(theta),  cos(theta)]]
 * stored in row-major order (4 floats).
 */
__global__ void exp_so2_kernel(const float *angles, size_t angle_stride, float *rotations,
                               size_t rotation_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  float theta = angles[idx * angle_stride];
  float c = cosf(theta);
  float s = sinf(theta);

  float *R = rotations + idx * rotation_stride;
  R[0] = c;
  R[1] = -s;
  R[2] = s;
  R[3] = c;
}

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

/**
 * @brief SO(2) transpose (= inverse): R^T = R^{-1}.
 *
 * For R = [[a,b],[c,d]], computes R^T = [[a,c],[b,d]].
 * Since R is orthogonal, R^T = R^{-1}.
 */
__global__ void transpose_so2_kernel(const float *rotations, size_t input_stride, float *transposed,
                                     size_t output_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *R = rotations + idx * input_stride;
  float *Rt = transposed + idx * output_stride;

  Rt[0] = R[0];
  Rt[1] = R[2];
  Rt[2] = R[1];
  Rt[3] = R[3];
}

void ComputeExpSO2(cudaStream_t stream, const float *angles, size_t angle_stride,
                   size_t rotation_stride, size_t size, float *rotations) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  exp_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(angles, angle_stride, rotations,
                                                               rotation_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSO2(cudaStream_t stream, const float *rotations, size_t rotation_stride,
                   size_t angle_stride, size_t size, float *angles) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  log_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(rotations, rotation_stride, angles,
                                                               angle_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeTransposeSO2(cudaStream_t stream, const float *rotations, size_t input_stride,
                         size_t output_stride, size_t size, float *transposed) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  transpose_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(
      rotations, input_stride, transposed, output_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// SE(2) Kernels
// ============================================================================

constexpr size_t kSE2MathBlockSize = 256;

/**
 * @brief SE(2) exponential map: tangent vector -> 3x3 transform.
 *
 * For tangent xi = [v_x, v_y, theta], computes:
 *   T = [cos(theta), -sin(theta), tx;
 *        sin(theta),  cos(theta), ty;
 *        0,           0,          1 ]
 *
 * where [tx, ty] = V(theta) * [v_x, v_y] and V is the SE(2) V-matrix:
 *   V = [[sin(theta)/theta,    -(1-cos(theta))/theta],
 *        [(1-cos(theta))/theta,  sin(theta)/theta    ]]
 *
 * Both coefficients are evaluated without cancellation (series near 0).
 */
__global__ void exp_se2_kernel(const float *tangent, size_t tangent_stride, float *transforms,
                               size_t transform_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  lie_device::ExpSE2(tangent + idx * tangent_stride, transforms + idx * transform_stride);
}

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
 * @brief SE(2) inverse: T^{-1} = [R^T, -R^T*t; 0, 1].
 *
 * For a 2D rigid transform T = [R t; 0 1] with R orthogonal,
 * the inverse is [R^T, -R^T*t; 0 1].
 */
__global__ void inverse_se2_kernel(const float *transforms, size_t transform_stride,
                                   float *inverse_transforms, size_t inverse_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  lie_device::InverseSE2(transforms + idx * transform_stride,
                         inverse_transforms + idx * inverse_stride);
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

void ComputeExpSE2(cudaStream_t stream, const float *tangent, size_t tangent_stride,
                   size_t transform_stride, size_t size, float *transforms) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  exp_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(tangent, tangent_stride, transforms,
                                                               transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSE2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                   size_t tangent_stride, size_t size, float *tangent) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  log_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(transforms, transform_stride,
                                                               tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeInverseSE2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                       size_t inverse_stride, size_t size, float *inverse_transforms) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  inverse_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(
      transforms, transform_stride, inverse_transforms, inverse_stride, size);
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
