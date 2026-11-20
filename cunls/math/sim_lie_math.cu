/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/math/lie_device.cuh"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kSimMathBlockSize = 256;

// ============================================================================
// Sim(2) Kernels
// ============================================================================

/**
 * @brief Sim(2) logarithm map: 3x3 matrix -> tangent [u_x, u_y, theta, lambda].
 *
 * For T = [R t; 0 0 1/s], extracts theta = atan2(s, c), lambda = log(s),
 * and inverts [u_x, u_y] = V(theta, lambda)^{-1} * [tx, ty].
 *
 * V^{-1} = [[X, thetaY], [-thetaY, X]] / (X^2 + (thetaY)^2).
 */
__global__ void log_sim2_kernel(const float *transforms, size_t transform_stride, float *tangent,
                                size_t tangent_stride, size_t size) {
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
  float inv_s = T[8];

  float theta = atan2f(s, c);
  float scale = 1.0f / inv_s;
  float lambda = logf(scale);

  float X, thetaY;
  lie_device::Sim2VCoefficients(theta, lambda, &X, &thetaY);

  float det = X * X + thetaY * thetaY;
  float ux, uy;
  if (det > 1e-10f) {
    float inv_det = 1.0f / det;
    ux = (X * tx + thetaY * ty) * inv_det;
    uy = (-thetaY * tx + X * ty) * inv_det;
  } else {
    ux = tx;
    uy = ty;
  }

  xi[0] = ux;
  xi[1] = uy;
  xi[2] = theta;
  xi[3] = lambda;
}

__global__ void __launch_bounds__(256, 4)
    jacobian_right_inverse_sim2_kernel(const float *tangent, size_t tangent_stride,
                                       float *jacobians, size_t jacobian_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *xi = tangent + idx * tangent_stride;
  float *J = jacobians + idx * jacobian_stride;
  lie_device::Sim2JrInv(xi[0], xi[1], xi[2], xi[3], J);
}

// ============================================================================
// Sim(2) Host wrappers
// ============================================================================

void ComputeLogSim2(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  log_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(transforms, transform_stride,
                                                                tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSim2(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  jacobian_right_inverse_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// Sim(3) Device helpers for 3x3 matrix operations
// ============================================================================

/// out = M * v (3x3 * 3x1)
static __device__ void mat3_vec(const float *__restrict__ M, const float *__restrict__ v,
                                float *__restrict__ out) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      s += M[r * 3 + k] * v[k];
    }
    out[r] = s;
  }
}

/// Invert a 3x3 matrix via cofactors. Returns false if singular.
static __device__ bool mat3_inv(const float *__restrict__ M, float *__restrict__ Minv) {
  float c0 = M[4] * M[8] - M[5] * M[7];
  float c1 = -(M[3] * M[8] - M[5] * M[6]);
  float c2 = M[3] * M[7] - M[4] * M[6];
  float det = M[0] * c0 + M[1] * c1 + M[2] * c2;
  if (fabsf(det) < 1e-12f) {
    return false;
  }
  float inv_det = 1.0f / det;
  Minv[0] = c0 * inv_det;
  Minv[1] = (-(M[1] * M[8] - M[2] * M[7])) * inv_det;
  Minv[2] = (M[1] * M[5] - M[2] * M[4]) * inv_det;
  Minv[3] = c1 * inv_det;
  Minv[4] = (M[0] * M[8] - M[2] * M[6]) * inv_det;
  Minv[5] = (-(M[0] * M[5] - M[2] * M[3])) * inv_det;
  Minv[6] = c2 * inv_det;
  Minv[7] = (-(M[0] * M[7] - M[1] * M[6])) * inv_det;
  Minv[8] = (M[0] * M[4] - M[1] * M[3]) * inv_det;
  return true;
}

// ============================================================================
// Sim(3) Kernels
// ============================================================================

/**
 * @brief Sim(3) logarithm map: 4x4 matrix -> tangent [w, u, lambda].
 *
 * Extracts w = Log_SO3(R) (using the angle-axis formula on the rotation block),
 * lambda = log(s) from the (3,3) entry, and u = V^{-1} * t where V is the
 * Sim(3) V-matrix built from (theta, lambda).
 *
 * V is assembled as V = diag*I + Q*[w]_x + R_c*w*w^T, then inverted
 * via cofactor expansion of the resulting 3x3 matrix.
 */
__global__ void log_sim3_kernel(const float *transforms, size_t transform_stride, float *tangent,
                                size_t tangent_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float *T = transforms + idx * transform_stride;
  float *xi = tangent + idx * tangent_stride;

  float R[9] = {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]};
  float t[3] = {T[3], T[7], T[11]};
  float inv_s = T[15];
  float s_val = 1.0f / inv_s;
  float lambda = logf(s_val);

  float trace = R[0] + R[4] + R[8];
  float cos_a = fmaxf(-1.0f, fminf(1.0f, (trace - 1.0f) * 0.5f));
  float theta = acosf(cos_a);
  float theta2 = theta * theta;

  float vee[3] = {R[7] - R[5], R[2] - R[6], R[3] - R[1]};
  float w[3];
  if (fabsf(theta) < 1e-4f) {
    w[0] = 0.5f * vee[0];
    w[1] = 0.5f * vee[1];
    w[2] = 0.5f * vee[2];
  } else {
    float k = theta / (2.0f * sinf(theta));
    w[0] = k * vee[0];
    w[1] = k * vee[1];
    w[2] = k * vee[2];
  }

  float A1, A2, P_c, Q_c, R_c;
  lie_device::Sim3VCoefficients(theta2, lambda, &A1, &A2, &P_c, &Q_c, &R_c);

  float diag = P_c - R_c * theta2;
  float V[9], W_mat[9];
  lie_device::SkewMatrix(w, W_mat, 3);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      int ij = i * 3 + j;
      V[ij] = R_c * w[i] * w[j] + Q_c * W_mat[ij];
      if (i == j) {
        V[ij] += diag;
      }
    }
  }

  float Vinv[9];
  if (!mat3_inv(V, Vinv)) {
    xi[0] = w[0];
    xi[1] = w[1];
    xi[2] = w[2];
    xi[3] = t[0];
    xi[4] = t[1];
    xi[5] = t[2];
    xi[6] = lambda;
    return;
  }
  float u[3];
  mat3_vec(Vinv, t, u);

  xi[0] = w[0];
  xi[1] = w[1];
  xi[2] = w[2];
  xi[3] = u[0];
  xi[4] = u[1];
  xi[5] = u[2];
  xi[6] = lambda;
}

/** @brief J_r^{-1}(xi) of Sim(3), one element per thread (lie_device::Sim3JrInv). */
constexpr size_t kSim3JacBlockSize = 128;

__global__ void __launch_bounds__(128, 5)
    jacobian_right_inverse_sim3_kernel(const float *tangent, size_t tangent_stride,
                                       float *jacobians, size_t jacobian_stride, size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  lie_device::Sim3JrInv(tangent + idx * tangent_stride, jacobians + idx * jacobian_stride);
}

// ============================================================================
// Sim(3) Host wrappers
// ============================================================================

void ComputeLogSim3(cudaStream_t stream, const float *transforms, size_t transform_stride,
                    size_t tangent_stride, size_t size, float *tangent) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  log_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(transforms, transform_stride,
                                                                tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSim3(cudaStream_t stream, const float *tangent,
                                     size_t tangent_stride, size_t jacobian_stride, size_t size,
                                     float *jacobians) {
  size_t num_blocks = (size + kSim3JacBlockSize - 1) / kSim3JacBlockSize;
  jacobian_right_inverse_sim3_kernel<<<num_blocks, kSim3JacBlockSize, 0, stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
