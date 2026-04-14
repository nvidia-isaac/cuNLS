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
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t block_size =
    256; ///< Default thread block size for CUDA kernels.

/**
 * @brief Device helper to swap two values.
 *
 * @tparam T Type of the values to swap.
 * @param a First value (swapped with b).
 * @param b Second value (swapped with a).
 */
template <typename T> __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

/**
 * @brief Device function to compute a 3x3 skew-symmetric matrix.
 *
 * Computes the skew-symmetric matrix [v]_× for a 3D vector v.
 *
 * @param translation Input 3D vector
 * @param ptr Output matrix pointer (3x3, row-major)
 * @param pitch Pitch (stride between rows) of the output matrix
 */
__device__ void compute_skew_matrix(const float *translation, float *ptr,
                                    const size_t pitch) {
  ptr[0 * pitch + 0] = 0;
  ptr[0 * pitch + 1] = -translation[2];
  ptr[0 * pitch + 2] = translation[1];

  ptr[1 * pitch + 0] = translation[2];
  ptr[1 * pitch + 1] = 0;
  ptr[1 * pitch + 2] = -translation[0];

  ptr[2 * pitch + 0] = -translation[1];
  ptr[2 * pitch + 1] = translation[0];
  ptr[2 * pitch + 2] = 0;
}

/**
 * @brief Device function to compute a Rodrigues-style matrix.
 *
 * Computes k1*I + k2*skew(phi) + k3*phi*phi^T + k4*skew(phi) for small angles.
 * This is a general form used for various SO(3) operations.
 *
 * @param phi Input 3D vector
 * @param k1 Coefficient for identity matrix
 * @param k2 Coefficient for skew-symmetric matrix
 * @param k3 Coefficient for outer product
 * @param k4 Additional coefficient for skew-symmetric matrix (used for small
 * angles)
 * @param ptr Output matrix pointer (3x3, row-major)
 * @param pitch Pitch (stride between rows) of the output matrix
 * @param tol Tolerance for small angle approximation
 */
__device__ void compute_rodrigues_matrix(const float *phi, float k1, float k2,
                                         float k3, float k4, float *ptr,
                                         const size_t pitch, float tol = 1e-5) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  assert(theta >= 0);
  if (theta < tol) {
    // I + k4 * skew(phi)
    ptr[0 * pitch + 0] = 1;
    ptr[0 * pitch + 1] = -k4 * phi[2];
    ptr[0 * pitch + 2] = k4 * phi[1];

    ptr[1 * pitch + 0] = k4 * phi[2];
    ptr[1 * pitch + 1] = 1;
    ptr[1 * pitch + 2] = -k4 * phi[0];

    ptr[2 * pitch + 0] = -k4 * phi[1];
    ptr[2 * pitch + 1] = k4 * phi[0];
    ptr[2 * pitch + 2] = 1;
    return;
  }

  float a = k2 * phi[2];
  float b = k2 * phi[1];
  float c = k2 * phi[0];

  float phi01 = k3 * phi[0] * phi[1];
  float phi02 = k3 * phi[0] * phi[2];
  float phi12 = k3 * phi[1] * phi[2];

  // k1 * I + k2 * skew(phi) + k3 * phi * phi.T
  ptr[0 * pitch + 0] = k1 + k3 * phi[0] * phi[0];
  ptr[0 * pitch + 1] = -a + phi01;
  ptr[0 * pitch + 2] = b + phi02;

  ptr[1 * pitch + 0] = a + phi01;
  ptr[1 * pitch + 1] = k1 + k3 * phi[1] * phi[1];
  ptr[1 * pitch + 2] = -c + phi12;

  ptr[2 * pitch + 0] = -b + phi02;
  ptr[2 * pitch + 1] = c + phi12;
  ptr[2 * pitch + 2] = k1 + k3 * phi[2] * phi[2];
}

/**
 * @brief Device function to compute the exponential map for SO(3).
 *
 * Computes R = Exp(phi) using Rodrigues' formula.
 *
 * @param phi Input 3D twist vector
 * @param ptr Output rotation matrix pointer (3x3, row-major)
 * @param pitch Pitch (stride between rows) of the output matrix
 */
__device__ void compute_exp_so3(const float *phi, float *ptr,
                                const size_t pitch) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  float theta_squared = powf(theta, 2);

  float k1 = cosf(theta);
  float k2 = sinf(theta) / theta;
  float k3 = (1 - k1) / theta_squared;

  compute_rodrigues_matrix(phi, k1, k2, k3, 1, ptr, pitch);
}

/**
 * @brief Device function to compute the left Jacobian of SO(3).
 *
 * Computes J_l(phi), the left Jacobian of SO(3) at twist phi.
 *
 * @param phi Input 3D twist vector
 * @param ptr Output Jacobian matrix pointer (3x3, row-major)
 * @param pitch Pitch (stride between rows) of the output matrix
 */
__device__ void compute_so3_jacobian_left(const float *phi, float *ptr,
                                          const size_t pitch) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  float theta_squared = powf(theta, 2);

  float k1 = sinf(theta) / theta;
  float k2 = (1 - cosf(theta)) / theta_squared;
  float k3 = (1 - k1) / theta_squared;

  compute_rodrigues_matrix(phi, k1, k2, k3, 0.5, ptr, pitch);
}

/**
 * @brief Device function to compute the inverse left Jacobian of SO(3).
 *
 * Computes J_l(phi)^{-1}, the inverse of the left Jacobian of SO(3).
 *
 * @param phi Input 3D twist vector
 * @param ptr Output inverse Jacobian matrix pointer (3x3, row-major)
 * @param pitch Pitch (stride between rows) of the output matrix
 */
__device__ void compute_so3_jacobian_left_inverse(const float *phi, float *ptr,
                                                  const size_t pitch) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  float theta_half = 0.5 * theta;
  float theta_squared = powf(theta, 2);

  float k1 = theta_half / tanf(theta_half);
  float k2 = -theta_half / theta;
  float k3 = (1 - k1) / theta_squared;

  compute_rodrigues_matrix(phi, k1, k2, k3, -0.5, ptr, pitch);
}

/**
 * @brief Device function to compute the logarithm map for SO(3).
 *
 * Computes phi = Log(R), mapping a rotation matrix to a 3D twist vector.
 *
 * @param rotation_matrix Input rotation matrix pointer (3x3, row-major)
 * @param rotation_pitch Pitch (stride between rows) of the input matrix
 * @param twist Output 3D twist vector
 * @param tol Tolerance for detecting identity rotation
 */
__device__ void compute_log_so3(const float *rotation_matrix,
                                const size_t rotation_pitch, float *twist,
                                float tol = 1e-5) {
  float trace = 0;
#pragma unroll
  for (int i = 0; i < 3; i++) {
    trace += rotation_matrix[i * rotation_pitch + i];
  }

  memset(twist, 0, 3 * sizeof(float));

  if (fabsf(3.f - trace) < tol) {
    return;
  }

  float cos_theta = fminf(1.0f, fmaxf(-1.0f, 0.5f * (trace - 1.0f)));
  float theta = acosf(cos_theta);
  float sin_theta = sinf(theta);

  if (sin_theta < tol) {
    // theta ≈ π (or ≈ 0, already handled above): the standard formula
    // has a 0/0 singularity.  Extract the axis from the column of (R + I)
    // with the largest diagonal.  For theta = π, R = 2nnᵀ − I, so the
    // largest R[j][j] = 2n_j² − 1 gives the best-conditioned axis component.
    int best = 0;
    float best_diag = rotation_matrix[0];
    float d1 = rotation_matrix[rotation_pitch + 1];
    float d2 = rotation_matrix[2 * rotation_pitch + 2];
    if (d1 > best_diag) {
      best = 1;
      best_diag = d1;
    }
    if (d2 > best_diag) {
      best = 2;
      best_diag = d2;
    }

    // Read only the chosen column of (R + I) and normalize
    float v0 = rotation_matrix[best] + ((best == 0) ? 1.0f : 0.0f);
    float v1 =
        rotation_matrix[rotation_pitch + best] + ((best == 1) ? 1.0f : 0.0f);
    float v2 = rotation_matrix[2 * rotation_pitch + best] +
               ((best == 2) ? 1.0f : 0.0f);
    float sq = v0 * v0 + v1 * v1 + v2 * v2;
    if (sq > 0.0f) {
      float scale = theta * __frsqrt_rn(sq);
      twist[0] = v0 * scale;
      twist[1] = v1 * scale;
      twist[2] = v2 * scale;
    }
    return;
  }

  float k = (0.5f * theta) / sin_theta;

  twist[0] = k * (rotation_matrix[2 * rotation_pitch + 1] -
                  rotation_matrix[1 * rotation_pitch + 2]);
  twist[1] = k * (rotation_matrix[0 * rotation_pitch + 2] -
                  rotation_matrix[2 * rotation_pitch + 0]);
  twist[2] = k * (rotation_matrix[1 * rotation_pitch + 0] -
                  rotation_matrix[0 * rotation_pitch + 1]);
}

/**
 * @brief Device function to multiply two 3x3 matrices: C = A * B.
 *
 * Both input and output matrices use a contiguous row-major layout with
 * pitch 3.
 *
 * @param A Left input matrix (3x3, row-major).
 * @param B Right input matrix (3x3, row-major).
 * @param C Output matrix (3x3, row-major).
 */
__device__ void matmul_3x3(const float *A, const float *B, float *C) {
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
#pragma unroll
    for (uint8_t j = 0; j < 3; j++) {
      C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] +
                     A[i * 3 + 2] * B[2 * 3 + j];
    }
  }
}

/**
 * @brief Device function to accumulate a scaled 3x3 matrix: B += scale * A.
 *
 * @param A Input matrix (3x3, contiguous row-major).
 * @param scale Scalar multiplier.
 * @param B Output matrix (3x3, contiguous row-major), accumulated in-place.
 */
__device__ void scale_add_3x3(const float *A, float scale, float *B) {
#pragma unroll
  for (uint8_t i = 0; i < 9; i++) {
    B[i] += scale * A[i];
  }
}

/**
 * @brief Device function to accumulate a scaled matrix product: C += scale * A
 * * B.
 *
 * @param A Left input matrix (3x3, contiguous row-major).
 * @param B Right input matrix (3x3, contiguous row-major).
 * @param scale Scalar multiplier applied to the product.
 * @param C Output matrix (3x3, contiguous row-major), accumulated in-place.
 */
__device__ void matmul_add_3x3(const float *A, const float *B, float scale,
                               float *C) {
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
#pragma unroll
    for (uint8_t j = 0; j < 3; j++) {
      C[i * 3 + j] +=
          scale * (A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] +
                   A[i * 3 + 2] * B[2 * 3 + j]);
    }
  }
}

/**
 * @brief Device function to compute the Q matrix for SE(3) Jacobian
 * computation.
 *
 * Computes Q_left(xi), a 3x3 matrix used in the computation of the left
 * Jacobian of SE(3). This matrix relates the translation part of the twist to
 * the rotation part.
 *
 * @param twist Input 6D twist vector [phi, rho]
 * @param Q_pitch Pitch (stride between rows) of the output matrix
 * @param Q Output Q matrix pointer (3x3, row-major)
 * @param tol Tolerance for small angle approximation
 */
__device__ void compute_Q_left(const float *twist, const size_t Q_pitch,
                               float *Q, float tol = 1e-5) {
  float phi = norm3df(twist[0], twist[1], twist[2]);

  float A = 1.f / 6.f;
  float B = 1.f / 24.f;
  float C = 1.f / 120.f;

  if (phi > tol) {
    float s = sinf(phi);
    float c = cosf(phi);

    float phi_squared = phi * phi;
    float phi_cubed = phi_squared * phi;
    float phi_fourth = phi_cubed * phi;
    float phi_fifth = phi_fourth * phi;

    A = (phi - s) / phi_cubed;
    B = (phi_squared * 0.5f + c - 1.f) / phi_fourth;
    C = 0.5f * ((2.f + c) / phi_fourth - 3.f * s / phi_fifth);
  }

  float result[9];
  memset(result, 0, 9 * sizeof(float));

  float temp[9];
  compute_skew_matrix(&twist[3], temp, 3);
  scale_add_3x3(temp, 0.5, result);

  float W[9];
  compute_skew_matrix(twist, W, 3);

  float VW[9];
  matmul_3x3(temp, W, VW);
  scale_add_3x3(VW, A, result);
  matmul_add_3x3(VW, W, B, result);

  float WV[9];
  matmul_3x3(W, temp, WV);
  scale_add_3x3(WV, A, result);

  matmul_add_3x3(W, WV, B, result);

  // compute WVW
  matmul_3x3(WV, W, temp);
  scale_add_3x3(temp, A - 3.f * B, result);

  matmul_add_3x3(temp, W, C, result);
  matmul_add_3x3(W, temp, C, result);

  for (uint8_t i = 0; i < 3; i++) {
    for (uint8_t j = 0; j < 3; j++) {
      Q[i * Q_pitch + j] = result[i * 3 + j];
    }
  }
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
__global__ void skew_so3_kernel(const float *twist, const size_t twist_stride,
                                float *skew, const size_t skew_pitch,
                                const size_t skew_stride, size_t size) {
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
__global__ void exp_so3_kernel(const float *twist, const size_t twist_stride,
                               float *exp, const size_t exp_pitch,
                               const size_t exp_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *exp_ptr = exp + tid * exp_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  compute_exp_so3(twist_ptr, exp_ptr, exp_pitch);
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
__global__ void log_so3_kernel(const float *rotation_matrix,
                               const size_t rotation_pitch,
                               const size_t rotation_stride,
                               const size_t twist_stride, size_t size,
                               float *twist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *twist_ptr = twist + tid * twist_stride;
  const float *rotation_matrix_ptr = rotation_matrix + tid * rotation_stride;

  compute_log_so3(rotation_matrix_ptr, rotation_pitch, twist_ptr);
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
__global__ void exp_se3_kernel(const float *twist, const size_t twist_stride,
                               float *transform, const size_t transform_pitch,
                               const size_t transform_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *transform_ptr = transform + tid * transform_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  float temp[6];
  memcpy(temp, twist_ptr, 6 * sizeof(float));

  float update[16];
#pragma unroll
  for (int i = 12; i < 15; i++) {
    update[i] = 0;
  }
  update[15] = 1; // set last row to [0, 0, 0, 1]

  const size_t update_pitch = 4;

  compute_so3_jacobian_left(temp, update, update_pitch);

  update[0 * update_pitch + 3] = update[0 * update_pitch + 0] * temp[3] +
                                 update[0 * update_pitch + 1] * temp[4] +
                                 update[0 * update_pitch + 2] * temp[5];
  update[1 * update_pitch + 3] = update[1 * update_pitch + 0] * temp[3] +
                                 update[1 * update_pitch + 1] * temp[4] +
                                 update[1 * update_pitch + 2] * temp[5];
  update[2 * update_pitch + 3] = update[2 * update_pitch + 0] * temp[3] +
                                 update[2 * update_pitch + 1] * temp[4] +
                                 update[2 * update_pitch + 2] * temp[5];

  compute_exp_so3(temp, update, update_pitch);

#pragma unroll
  for (uint8_t i = 0; i < 4; i++) {
#pragma unroll
    for (uint8_t j = 0; j < 4; j++) {
      transform_ptr[i * transform_pitch + j] = update[i * update_pitch + j];
    }
  }
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
__global__ void jacobian_so3_kernel(bool left, const float *twist,
                                    const size_t twist_stride, float *jacobian,
                                    const size_t jacobian_pitch,
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

  compute_so3_jacobian_left(temp, jacobian_ptr, jacobian_pitch);
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
    jacobian_inverse_so3_kernel(bool left, const float* twist,
                                            const size_t twist_stride,
                                            float* jacobian_inv,
                                            const size_t jacobian_inv_pitch,
                                            const size_t jacobian_inv_stride,
                                            size_t size) {
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

  compute_so3_jacobian_left_inverse(temp, jacobian_inv_ptr, jacobian_inv_pitch);
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
                                       const float *src_matrix,
                                       const size_t pitch, const size_t stride,
                                       size_t num_matrices, float *dst_matrix) {
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
__global__ void log_se3_kernel(const float *transform,
                               const size_t transform_pitch,
                               const size_t transform_stride,
                               const size_t twist_stride, size_t size,
                               float *twist) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *twist_ptr = twist + tid * twist_stride;
  const float *transform_ptr = transform + tid * transform_stride;

  float twist_se3[6];
  compute_log_so3(transform_ptr, transform_pitch, twist_se3);

  float translation[3];
  translation[0] = transform_ptr[0 * transform_pitch + 3];
  translation[1] = transform_ptr[1 * transform_pitch + 3];
  translation[2] = transform_ptr[2 * transform_pitch + 3];

  float J_inv[9];
  compute_so3_jacobian_left_inverse(twist_se3, J_inv, 3);

  twist_se3[3] = J_inv[0 * 3 + 0] * translation[0] +
                 J_inv[0 * 3 + 1] * translation[1] +
                 J_inv[0 * 3 + 2] * translation[2];
  twist_se3[4] = J_inv[1 * 3 + 0] * translation[0] +
                 J_inv[1 * 3 + 1] * translation[1] +
                 J_inv[1 * 3 + 2] * translation[2];
  twist_se3[5] = J_inv[2 * 3 + 0] * translation[0] +
                 J_inv[2 * 3 + 1] * translation[1] +
                 J_inv[2 * 3 + 2] * translation[2];

  memcpy(twist_ptr, twist_se3, 6 * sizeof(float));
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
                                   const size_t transform_pitch,
                                   const size_t transform_stride,
                                   const size_t adjoint_pitch,
                                   const size_t adjoint_stride, size_t size,
                                   float *adjoint) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *adjoint_ptr = adjoint + tid * adjoint_stride;
  const float *transform_ptr = transform + tid * transform_stride;

  float k = inverse ? -1.0f : 1.0f;

  float translation[3];
  translation[0] = k * transform_ptr[0 * transform_pitch + 3];
  translation[1] = k * transform_ptr[1 * transform_pitch + 3];
  translation[2] = k * transform_ptr[2 * transform_pitch + 3];

  float R[9];
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *dst = &R[i * 3];
    const float *src = &transform_ptr[i * transform_pitch];
    memcpy(dst, src, 3 * sizeof(float));
  }

  if (inverse) {
    // transpose R
    swap(R[0 * 3 + 1], R[1 * 3 + 0]);
    swap(R[0 * 3 + 2], R[2 * 3 + 0]);
    swap(R[1 * 3 + 2], R[2 * 3 + 1]);
  }

#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    // adjoint[0:3, 0:3] = R
    float *src = &R[i * 3];
    float *dst = &adjoint_ptr[i * adjoint_pitch];
    memcpy(dst, src, 3 * sizeof(float));

    // adjoint[3:6, 3:6] = R
    dst = &adjoint_ptr[(i + 3) * adjoint_pitch + 3];
    memcpy(dst, src, 3 * sizeof(float));

    // adjoint[0:3, 3:6] = 0
    dst = &adjoint_ptr[i * adjoint_pitch + 3];
    memset(dst, 0, 3 * sizeof(float));
  }

  float skew[9];
  compute_skew_matrix(translation, skew, 3);

  float temp[9];
  if (inverse) {
    matmul_3x3(skew, R, temp);
  } else {
    matmul_3x3(R, skew, temp);
  }

#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *dst = &adjoint_ptr[(3 + i) * adjoint_pitch];
    const float *src = &temp[i * 3];
    memcpy(dst, src, 3 * sizeof(float));
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
__global__ void jacobian_se3_kernel(bool left, const float *twist,
                                    const size_t twist_stride, float *jacobian,
                                    const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *jacobian_ptr = jacobian + tid * jacobian_stride;
  const float *twist_ptr = twist + tid * twist_stride;

  float temp[6];
  memcpy(temp, twist_ptr, 6 * sizeof(float));
  if (!left) {
#pragma unroll
    for (uint8_t i = 0; i < 6; i++) {
      temp[i] = -temp[i];
    }
  }

  float J[36];
  memset(J, 0, 36 * sizeof(float));

  //   J[0:3, 0:3] = so3_jacobian_left(twist[0:3])
  compute_so3_jacobian_left(temp, J, 6);

  // J[3:6, 3:6] = so3_jacobian_left(twist[0:3])
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *dst = &J[(3 + i) * 6 + 3];
    const float *src = &J[i * 6];
    memcpy(dst, src, 3 * sizeof(float));
  }

  //   J[3:6, 0:3] = Q_left(twist)
  compute_Q_left(temp, 6, &J[3 * 6]);

#pragma unroll
  for (uint8_t i = 0; i < 6; i++) {
    float *dst = &jacobian_ptr[i * jacobian_pitch];
    const float *src = &J[i * 6];
    memcpy(dst, src, 6 * sizeof(float));
  }
}

/**
 * @brief Optimized CUDA kernel to compute the inverse Jacobian of SE(3).
 *
 * Computes J_l^{-1}(xi) or J_r^{-1}(xi) for a batch of 6D twist vectors.
 * The output is a 6x6 matrix with the structure:
 *   [ J_l^{-1}(phi)                          |      0          ]
 *   [ -J_l^{-1}(phi) @ Q(xi) @ J_l^{-1}(phi)| J_l^{-1}(phi)  ]
 *
 * Performance optimizations:
 *   - __launch_bounds__ for controlled register allocation and improved
 * occupancy
 *   - __restrict__ pointers enable better load/store scheduling by the compiler
 *   - Eliminated redundant global memory zeroing (saves 36 wasted stores)
 *   - Single consolidated write pass instead of 3 scattered write phases
 *   - Q-block negation fused into the final store (eliminates separate negate
 * loop)
 */
constexpr size_t se3_jac_inv_block_size =
    128; ///< Block size for SE(3) inverse Jacobian kernel (tuned for register
         ///< pressure).

__global__ void __launch_bounds__(128, 4)
    jacobian_inverse_se3_kernel(bool left, const float *__restrict__ twist,
                                const size_t twist_stride,
                                float *__restrict__ jacobian,
                                const size_t jacobian_pitch,
                                const size_t jacobian_stride, size_t size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  // --- Load twist via explicit scalar loads (cleaner codegen than memcpy) ---
  const float *twist_ptr = twist + tid * twist_stride;
  float tw[6];
#pragma unroll
  for (uint8_t i = 0; i < 6; i++) {
    tw[i] = twist_ptr[i];
  }
  if (!left) {
#pragma unroll
    for (uint8_t i = 0; i < 6; i++) {
      tw[i] = -tw[i];
    }
  }

  // --- Compute J_left_inverse(phi) where phi = tw[0:3] ---
  float J[9];
  compute_so3_jacobian_left_inverse(tw, J, 3);

  // --- Compute Q_left(twist) ---
  float Q[9];
  compute_Q_left(tw, 3, Q);

  // --- Compute J_left_inv @ Q @ J_left_inv, stored back in Q ---
  float T[9];
  matmul_3x3(J, Q, T);
  matmul_3x3(T, J, Q);

  // --- Single consolidated write to global memory ---
  // Writes the complete 6x6 output in one pass, eliminating the
  // separate zeroing (36 wasted stores) and scattered partial writes.
  float *out = jacobian + tid * jacobian_stride;

  // Rows 0-2: [ J_left_inv | 0 ]
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *row = out + i * jacobian_pitch;
    row[0] = J[i * 3 + 0];
    row[1] = J[i * 3 + 1];
    row[2] = J[i * 3 + 2];
    row[3] = 0.0f;
    row[4] = 0.0f;
    row[5] = 0.0f;
  }

  // Rows 3-5: [ -(J @ Q @ J) | J_left_inv ]
  // Negation is fused into the write, avoiding a separate negate pass.
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *row = out + (3 + i) * jacobian_pitch;
    row[0] = -Q[i * 3 + 0];
    row[1] = -Q[i * 3 + 1];
    row[2] = -Q[i * 3 + 2];
    row[3] = J[i * 3 + 0];
    row[4] = J[i * 3 + 1];
    row[5] = J[i * 3 + 2];
  }
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
__global__ void inverse_se3_kernel(const float *transform,
                                   const size_t transform_pitch,
                                   const size_t transform_stride,
                                   const size_t inverse_pitch,
                                   const size_t inverse_stride, size_t size,
                                   float *inverse_transform) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }

  float *inverse_transform_ptr = inverse_transform + tid * inverse_stride;
  const float *transform_ptr = transform + tid * transform_stride;

  float pose[16];
  memset(pose, 0, 16 * sizeof(float));
  pose[15] = 1;

  float t1 = transform_ptr[0 * transform_pitch + 3];
  float t2 = transform_ptr[1 * transform_pitch + 3];
  float t3 = transform_ptr[2 * transform_pitch + 3];

#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
    float *dst = &pose[i * 4];
    const float *src = &transform_ptr[i * transform_pitch];
    memcpy(dst, src, 3 * sizeof(float));
  }

  // transpose R
  swap(pose[0 * 4 + 1], pose[1 * 4 + 0]);
  swap(pose[0 * 4 + 2], pose[2 * 4 + 0]);
  swap(pose[1 * 4 + 2], pose[2 * 4 + 1]);

  pose[0 * 4 + 3] =
      -(pose[0 * 4 + 0] * t1 + pose[0 * 4 + 1] * t2 + pose[0 * 4 + 2] * t3);
  pose[1 * 4 + 3] =
      -(pose[1 * 4 + 0] * t1 + pose[1 * 4 + 1] * t2 + pose[1 * 4 + 2] * t3);
  pose[2 * 4 + 3] =
      -(pose[2 * 4 + 0] * t1 + pose[2 * 4 + 1] * t2 + pose[2 * 4 + 2] * t3);

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
void ComputeSkewSO3(cudaStream_t stream, const float *twist,
                    const size_t twist_stride, const size_t skew_pitch,
                    const size_t skew_stride, size_t size, float *skew) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  skew_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      twist, twist_stride, skew, skew_pitch, skew_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeNegateMatrix */
void ComputeNegateMatrix(cudaStream_t stream, const float *matrix, size_t rows,
                         size_t cols, const size_t pitch, const size_t stride,
                         size_t size, float *negated_matrix) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  negate_matrices_kernel<<<num_blocks, block_size, 0, stream>>>(
      rows, cols, matrix, pitch, stride, size, negated_matrix);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeInverseSE3 */
void ComputeInverseSE3(cudaStream_t stream, const float *transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t inverse_pitch, const size_t inverse_stride,
                       size_t size, float *inverse_transform) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  inverse_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      transform, transform_pitch, transform_stride, inverse_pitch,
      inverse_stride, size, inverse_transform);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeExpSO3 */
void ComputeExpSO3(cudaStream_t stream, const float *twist,
                   const size_t twist_stride, const size_t rotation_pitch,
                   const size_t rotation_stride, size_t size, float *rotation) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  exp_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      twist, twist_stride, rotation, rotation_pitch, rotation_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}
/** @copydoc ComputeLogSO3 */
void ComputeLogSO3(cudaStream_t stream, const float *rotation,
                   const size_t rotation_pitch, const size_t rotation_stride,
                   const size_t twist_stride, size_t size, float *twist) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  log_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      rotation, rotation_pitch, rotation_stride, twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftSO3 */
void ComputeJacobianLeftSO3(cudaStream_t stream, const float *twist,
                            const size_t twist_stride,
                            const size_t jacobian_pitch,
                            const size_t jacobian_stride, size_t size,
                            float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride,
      size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightSO3 */
void ComputeJacobianRightSO3(cudaStream_t stream, const float *twist,
                             const size_t twist_stride,
                             const size_t jacobian_pitch,
                             const size_t jacobian_stride, size_t size,
                             float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride,
      size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftInverseSO3 */
void ComputeJacobianLeftInverseSO3(cudaStream_t stream, const float *twist,
                                   const size_t twist_stride,
                                   const size_t jacobian_inv_pitch,
                                   const size_t jacobian_inv_stride,
                                   size_t size, float *jacobian_inv) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_inverse_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian_inv, jacobian_inv_pitch,
      jacobian_inv_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSO3 */
void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride,
                                    const size_t jacobian_inv_pitch,
                                    const size_t jacobian_inv_stride,
                                    size_t size, float *jacobian_inv) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_inverse_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian_inv, jacobian_inv_pitch,
      jacobian_inv_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeExpSE3 */
void ComputeExpSE3(cudaStream_t stream, const float *twist,
                   const size_t twist_stride, const size_t transform_pitch,
                   const size_t transform_stride, size_t size,
                   float *transform) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  exp_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      twist, twist_stride, transform, transform_pitch, transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeLogSE3 */
void ComputeLogSE3(cudaStream_t stream, const float *transform,
                   const size_t transform_pitch, const size_t transform_stride,
                   const size_t twist_stride, size_t size, float *twist) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  log_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      transform, transform_pitch, transform_stride, twist_stride, size, twist);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeAdjointSE3 */
void ComputeAdjointSE3(cudaStream_t stream, const float *transform,
                       const size_t transform_pitch,
                       const size_t transform_stride,
                       const size_t adjoint_pitch, const size_t adjoint_stride,
                       size_t size, float *adjoint) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  bool inverse = false;
  adjoint_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      inverse, transform, transform_pitch, transform_stride, adjoint_pitch,
      adjoint_stride, size, adjoint);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeInverseAdjointSE3 */
void ComputeInverseAdjointSE3(cudaStream_t stream, const float *transform,
                              const size_t transform_pitch,
                              const size_t transform_stride,
                              const size_t inv_adjoint_pitch,
                              const size_t inv_adjoint_stride, size_t size,
                              float *inv_adjoint) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  bool inverse = true;
  adjoint_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      inverse, transform, transform_pitch, transform_stride, inv_adjoint_pitch,
      inv_adjoint_stride, size, inv_adjoint);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftSE3 */
void ComputeJacobianLeftSE3(cudaStream_t stream, const float *twist,
                            const size_t twist_stride,
                            const size_t jacobian_pitch,
                            const size_t jacobian_stride, size_t size,
                            float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = true;
  jacobian_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride,
      size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianLeftInverseSE3 */
void ComputeJacobianLeftInverseSE3(cudaStream_t stream, const float *twist,
                                   const size_t twist_stride,
                                   const size_t jacobian_pitch,
                                   const size_t jacobian_stride, size_t size,
                                   float *jacobian) {
  size_t num_blocks =
      (size + se3_jac_inv_block_size - 1) / se3_jac_inv_block_size;
  constexpr bool left = true;
  jacobian_inverse_se3_kernel<<<num_blocks, se3_jac_inv_block_size, 0,
                                stream>>>(left, twist, twist_stride, jacobian,
                                          jacobian_pitch, jacobian_stride,
                                          size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightSE3 */
void ComputeJacobianRightSE3(cudaStream_t stream, const float *twist,
                             const size_t twist_stride,
                             const size_t jacobian_pitch,
                             const size_t jacobian_stride, size_t size,
                             float *jacobian) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  constexpr bool left = false;
  jacobian_se3_kernel<<<num_blocks, block_size, 0, stream>>>(
      left, twist, twist_stride, jacobian, jacobian_pitch, jacobian_stride,
      size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** @copydoc ComputeJacobianRightInverseSE3 */
void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float *twist,
                                    const size_t twist_stride,
                                    const size_t jacobian_pitch,
                                    const size_t jacobian_stride, size_t size,
                                    float *jacobian) {
  size_t num_blocks =
      (size + se3_jac_inv_block_size - 1) / se3_jac_inv_block_size;
  constexpr bool left = false;
  jacobian_inverse_se3_kernel<<<num_blocks, se3_jac_inv_block_size, 0,
                                stream>>>(left, twist, twist_stride, jacobian,
                                          jacobian_pitch, jacobian_stride,
                                          size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}
/**
 * @brief Transposes a batch of 3x3 rotation matrices.
 *
 * For R = [[a,b,c],[d,e,f],[g,h,i]], computes R^T = [[a,d,g],[b,e,h],[c,f,i]].
 */
__global__ void transpose_so3_kernel(const float *rotations,
                                     size_t input_stride, float *transposed,
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
void ComputeTransposeSO3(cudaStream_t stream, const float *rotation,
                         size_t input_stride, size_t output_stride, size_t size,
                         float *transposed) {
  size_t num_blocks = (size + block_size - 1) / block_size;
  transpose_so3_kernel<<<num_blocks, block_size, 0, stream>>>(
      rotation, input_stride, transposed, output_stride, size);
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
__global__ void exp_so2_kernel(const float *angles, size_t angle_stride,
                               float *rotations, size_t rotation_stride,
                               size_t size) {
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
__global__ void log_so2_kernel(const float *rotations, size_t rotation_stride,
                               float *angles, size_t angle_stride,
                               size_t size) {
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
__global__ void transpose_so2_kernel(const float *rotations,
                                     size_t input_stride, float *transposed,
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

void ComputeExpSO2(cudaStream_t stream, const float *angles,
                   size_t angle_stride, size_t rotation_stride, size_t size,
                   float *rotations) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  exp_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(
      angles, angle_stride, rotations, rotation_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSO2(cudaStream_t stream, const float *rotations,
                   size_t rotation_stride, size_t angle_stride, size_t size,
                   float *angles) {
  size_t num_blocks = (size + kSO2MathBlockSize - 1) / kSO2MathBlockSize;
  log_so2_kernel<<<num_blocks, kSO2MathBlockSize, 0, stream>>>(
      rotations, rotation_stride, angles, angle_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeTransposeSO2(cudaStream_t stream, const float *rotations,
                         size_t input_stride, size_t output_stride, size_t size,
                         float *transposed) {
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
 * For |theta| < 1e-3, V approaches I and [tx, ty] ~ [v_x, v_y].
 */
__global__ void exp_se2_kernel(const float *tangent, size_t tangent_stride,
                               float *transforms, size_t transform_stride,
                               size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *xi = tangent + idx * tangent_stride;
  float vx = xi[0];
  float vy = xi[1];
  float w = xi[2];

  float c = cosf(w);
  float s = sinf(w);

  float tx, ty;
  if (fabsf(w) < 1e-3f) {
    tx = vx;
    ty = vy;
  } else {
    float sinw_over_w = s / w;
    float one_minus_cosw_over_w = (1.0f - c) / w;
    tx = vx * sinw_over_w - vy * one_minus_cosw_over_w;
    ty = vx * one_minus_cosw_over_w + vy * sinw_over_w;
  }

  float *T = transforms + idx * transform_stride;
  T[0] = c;
  T[1] = -s;
  T[2] = tx;
  T[3] = s;
  T[4] = c;
  T[5] = ty;
  T[6] = 0.0f;
  T[7] = 0.0f;
  T[8] = 1.0f;
}

/**
 * @brief SE(2) logarithm map: 3x3 transform -> tangent vector.
 *
 * For T = [R t; 0 1], computes xi = [v_x, v_y, theta] where:
 *   theta = atan2(sin, cos) from R
 *   [v_x, v_y] = V(theta)^{-1} * t
 *
 * Logmap:
 * For |theta| < 1e-3, V^{-1} ~ I and [v_x, v_y] ~ [tx, ty].
 *
 * The V^{-1} computation uses the identity:
 *   V^{-1} = (theta / (2*(1-cos))) * R_pi/2 * ((c-1)*I + s*J) * [tx,ty]
 * where R_pi/2 rotates by 90 degrees: (x,y) -> (-y, x).
 */
__global__ void log_se2_kernel(const float *transforms, size_t transform_stride,
                               float *tangent, size_t tangent_stride,
                               size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *T = transforms + idx * transform_stride;
  float *xi = tangent + idx * tangent_stride;

  float c = T[0];
  float s = T[3];
  float tx = T[2];
  float ty = T[5];

  float theta = atan2f(s, c);

  if (fabsf(theta) < 1e-3f) {
    xi[0] = tx;
    xi[1] = ty;
    xi[2] = theta;
  } else {
    float c_1 = c - 1.0f;
    float det = c_1 * c_1 + s * s;
    float dx = c_1 * tx + s * ty;
    float dy = -s * tx + c_1 * ty;
    float factor = theta / det;
    xi[0] = factor * (-dy);
    xi[1] = factor * dx;
    xi[2] = theta;
  }
}

/**
 * @brief SE(2) inverse: T^{-1} = [R^T, -R^T*t; 0, 1].
 *
 * For a 2D rigid transform T = [R t; 0 1] with R orthogonal,
 * the inverse is [R^T, -R^T*t; 0 1].
 */
__global__ void inverse_se2_kernel(const float *transforms,
                                   size_t transform_stride,
                                   float *inverse_transforms,
                                   size_t inverse_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *T = transforms + idx * transform_stride;
  float *Ti = inverse_transforms + idx * inverse_stride;

  float r00 = T[0], r01 = T[1], tx = T[2];
  float r10 = T[3], r11 = T[4], ty = T[5];

  Ti[0] = r00;
  Ti[1] = r10;
  Ti[2] = -(r00 * tx + r10 * ty);
  Ti[3] = r01;
  Ti[4] = r11;
  Ti[5] = -(r01 * tx + r11 * ty);
  Ti[6] = 0.0f;
  Ti[7] = 0.0f;
  Ti[8] = 1.0f;
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
 * For |alpha| < 1e-3 (near identity): J_r^{-1} ~ I + small corrections.
 */
__global__ void __launch_bounds__(256, 4)
    jacobian_right_inverse_se2_kernel(const float* tangent,
                                                  size_t tangent_stride,
                                                  float* jacobians,
                                                  size_t jacobian_stride,
                                                  size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *xi = tangent + idx * tangent_stride;
  float *J = jacobians + idx * jacobian_stride;

  float v1 = xi[0];
  float v2 = xi[1];
  float alpha = xi[2];

  if (fabsf(alpha) > 1e-3f) {
    float alpha_inv = 1.0f / alpha;
    float half_cot_half_alpha = 0.5f * sinf(alpha) / (1.0f - cosf(alpha));

    J[0] = alpha * half_cot_half_alpha;
    J[1] = -0.5f * alpha;
    J[2] = v1 * alpha_inv - v1 * half_cot_half_alpha + 0.5f * v2;
    J[3] = 0.5f * alpha;
    J[4] = alpha * half_cot_half_alpha;
    J[5] = v2 * alpha_inv - 0.5f * v1 - v2 * half_cot_half_alpha;
    J[6] = 0.0f;
    J[7] = 0.0f;
    J[8] = 1.0f;
  } else {
    J[0] = 1.0f;
    J[1] = 0.0f;
    J[2] = 0.5f * v2;
    J[3] = 0.0f;
    J[4] = 1.0f;
    J[5] = -0.5f * v1;
    J[6] = 0.0f;
    J[7] = 0.0f;
    J[8] = 1.0f;
  }
}

void ComputeExpSE2(cudaStream_t stream, const float *tangent,
                   size_t tangent_stride, size_t transform_stride, size_t size,
                   float *transforms) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  exp_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(
      tangent, tangent_stride, transforms, transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSE2(cudaStream_t stream, const float *transforms,
                   size_t transform_stride, size_t tangent_stride, size_t size,
                   float *tangent) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  log_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(
      transforms, transform_stride, tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeInverseSE2(cudaStream_t stream, const float *transforms,
                       size_t transform_stride, size_t inverse_stride,
                       size_t size, float *inverse_transforms) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  inverse_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0, stream>>>(
      transforms, transform_stride, inverse_transforms, inverse_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSE2(cudaStream_t stream, const float *tangent,
                                    size_t tangent_stride,
                                    size_t jacobian_stride, size_t size,
                                    float *jacobians) {
  size_t num_blocks = (size + kSE2MathBlockSize - 1) / kSE2MathBlockSize;
  jacobian_right_inverse_se2_kernel<<<num_blocks, kSE2MathBlockSize, 0,
                                      stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
