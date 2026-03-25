/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kSimMathBlockSize = 256;

// ============================================================================
// Sim(2) Device helpers
// ============================================================================

/**
 * @brief Sim(2) V-matrix coefficients for the exponential/logarithm map.
 *
 * The Sim(2) V-matrix maps translation generators to actual translations:
 *   t = V(theta, lambda) * [u_x, u_y]^T
 * where V = [[X, -theta*Y], [theta*Y, X]].
 *
 * The coefficients X and Y interpolate between two regimes:
 *   - Pure rotation (|lambda| small): X = sin(theta)/theta, Y = (1-cos)/theta^2
 *   - Mixed rotation+scale: uses a blending parameter alpha = lambda^2/(lambda^2+theta^2)
 *
 * Reference: Eade, "Lie Groups for 2D and 3D Transformations", Sec. 3.
 *
 * @param theta  Rotation angle
 * @param lambda Log-scale (lambda = log(s))
 * @param[out] X Diagonal element of V
 * @param[out] thetaY Off-diagonal magnitude (theta * Y)
 */
static __device__ void ComputeVCoeffsSim2(float theta, float lambda,
                                          float& X, float& thetaY) {
  const float lambda2 = lambda * lambda;
  const float theta2 = theta * theta;

  if (fabsf(lambda) < 1e-4f) {
    float A, B;
    if (theta2 > 1e-6f) {
      float sv = sinf(theta);
      float cv = cosf(theta);
      A = sv / theta;
      B = (1.0f - cv) / theta2;
    } else {
      A = 1.0f - theta2 / 6.0f;
      B = 0.5f - theta2 / 24.0f;
    }
    X = A;
    thetaY = theta * B;
    return;
  }

  const float d2 = lambda2 + theta2;
  if (d2 < 1e-10f) {
    X = 1.0f;
    thetaY = 0.0f;
    return;
  }

  float A, B, C;
  if (theta2 > 1e-6f) {
    float sv = sinf(theta);
    float cv = cosf(theta);
    A = sv / theta;
    B = (1.0f - cv) / theta2;
    C = (1.0f - A) / theta2;
  } else {
    A = 1.0f - theta2 / 6.0f;
    B = 0.5f - theta2 / 24.0f;
    C = 1.0f / 6.0f - theta2 / 120.0f;
  }

  float alpha_coeff = lambda2 / d2;
  float s_inv = expf(-lambda);
  X = alpha_coeff * (1.0f - s_inv) / lambda +
      (1.0f - alpha_coeff) * (A - lambda * B);
  float Y = alpha_coeff * (s_inv - 1.0f + lambda) / lambda2 +
            (1.0f - alpha_coeff) * (B - lambda * C);
  thetaY = theta * Y;
}

// ============================================================================
// Sim(2) Kernels
// ============================================================================

/**
 * @brief Sim(2) exponential map: tangent [u_x, u_y, theta, lambda] -> 3x3 matrix.
 *
 * Computes:
 *   T = [[cos(theta), -sin(theta), tx],
 *        [sin(theta),  cos(theta), ty],
 *        [0,           0,          e^{-lambda}]]
 *
 * where [tx, ty] = V(theta, lambda) * [u_x, u_y] and
 * V = [[X, -theta*Y], [theta*Y, X]] is the Sim(2) V-matrix.
 */
__global__ void exp_sim2_kernel(const float* tangent, size_t tangent_stride,
                                float* transforms, size_t transform_stride,
                                size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* xi = tangent + idx * tangent_stride;
  float ux = xi[0];
  float uy = xi[1];
  float theta = xi[2];
  float lambda = xi[3];

  float c = cosf(theta);
  float s = sinf(theta);
  float inv_scale = expf(-lambda);

  float X, thetaY;
  ComputeVCoeffsSim2(theta, lambda, X, thetaY);

  float tx = X * ux - thetaY * uy;
  float ty = thetaY * ux + X * uy;

  float* T = transforms + idx * transform_stride;
  T[0] = c;     T[1] = -s;    T[2] = tx;
  T[3] = s;     T[4] = c;     T[5] = ty;
  T[6] = 0.0f;  T[7] = 0.0f;  T[8] = inv_scale;
}

/**
 * @brief Sim(2) logarithm map: 3x3 matrix -> tangent [u_x, u_y, theta, lambda].
 *
 * For T = [R t; 0 0 1/s], extracts theta = atan2(s, c), lambda = log(s),
 * and inverts [u_x, u_y] = V(theta, lambda)^{-1} * [tx, ty].
 *
 * V^{-1} = [[X, thetaY], [-thetaY, X]] / (X^2 + (thetaY)^2).
 */
__global__ void log_sim2_kernel(const float* transforms,
                                size_t transform_stride, float* tangent,
                                size_t tangent_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* T = transforms + idx * transform_stride;
  float* xi = tangent + idx * tangent_stride;

  float c = T[0];
  float s = T[3];
  float tx = T[2];
  float ty = T[5];
  float inv_s = T[8];

  float theta = atan2f(s, c);
  float scale = 1.0f / inv_s;
  float lambda = logf(scale);

  float X, thetaY;
  ComputeVCoeffsSim2(theta, lambda, X, thetaY);

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

/**
 * @brief Sim(2) inverse: T^{-1} = [R^T, -s*R^T*t; 0, s].
 *
 * For T = [R t; 0 0 1/s], the inverse is [R^T, -s*R^T*t; 0 0 s].
 * Note: unlike SE(2), the translation block includes the scale factor s = 1/(1/s).
 */
__global__ void inverse_sim2_kernel(const float* transforms,
                                    size_t transform_stride,
                                    float* inverse_transforms,
                                    size_t inverse_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* T = transforms + idx * transform_stride;
  float* Ti = inverse_transforms + idx * inverse_stride;

  float r00 = T[0], r01 = T[1], tx = T[2];
  float r10 = T[3], r11 = T[4], ty = T[5];
  float inv_s = T[8];
  float s = 1.0f / inv_s;

  Ti[0] = r00;   Ti[1] = r10;   Ti[2] = -s * (r00 * tx + r10 * ty);
  Ti[3] = r01;   Ti[4] = r11;   Ti[5] = -s * (r01 * tx + r11 * ty);
  Ti[6] = 0.0f;  Ti[7] = 0.0f;  Ti[8] = s;
}

__global__ void jacobian_right_inverse_sim2_kernel(
    const float* tangent, size_t tangent_stride, float* jacobians,
    size_t jacobian_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* xi = tangent + idx * tangent_stride;
  float* J = jacobians + idx * jacobian_stride;
  Sim2JrInv(xi[0], xi[1], xi[2], xi[3], J);
}

// ============================================================================
// Sim(2) Host wrappers
// ============================================================================

void ComputeExpSim2(cudaStream_t stream, const float* tangent,
                    size_t tangent_stride, size_t transform_stride,
                    size_t size, float* transforms) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  exp_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      tangent, tangent_stride, transforms, transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSim2(cudaStream_t stream, const float* transforms,
                    size_t transform_stride, size_t tangent_stride,
                    size_t size, float* tangent) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  log_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      transforms, transform_stride, tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeInverseSim2(cudaStream_t stream, const float* transforms,
                        size_t transform_stride, size_t inverse_stride,
                        size_t size, float* inverse_transforms) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  inverse_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      transforms, transform_stride, inverse_transforms, inverse_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSim2(cudaStream_t stream, const float* tangent,
                                     size_t tangent_stride,
                                     size_t jacobian_stride, size_t size,
                                     float* jacobians) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  jacobian_right_inverse_sim2_kernel<<<num_blocks, kSimMathBlockSize, 0,
                                       stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// Sim(3) Device helpers for 3x3 matrix operations
// ============================================================================

/// C = A * B (3x3, row-major)
static __device__ void mat3_mul(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        s += A[r * 3 + k] * B[k * 3 + c];
      }
      C[r * 3 + c] = s;
    }
  }
}

/// out = M * v (3x3 * 3x1)
static __device__ void mat3_vec(const float* __restrict__ M,
                                const float* __restrict__ v,
                                float* __restrict__ out) {
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
static __device__ bool mat3_inv(const float* __restrict__ M,
                                float* __restrict__ Minv) {
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

/// Build skew-symmetric matrix from 3-vector: S = [v]_x
static __device__ void skew3(const float* v, float* S) {
  S[0] = 0.0f;   S[1] = -v[2];  S[2] = v[1];
  S[3] = v[2];   S[4] = 0.0f;   S[5] = -v[0];
  S[6] = -v[1];  S[7] = v[0];   S[8] = 0.0f;
}

// ============================================================================
// Sim(3) Kernels
// ============================================================================

/**
 * @brief Sim(3) exponential map: tangent [w1,w2,w3, u1,u2,u3, lambda] -> 4x4 matrix.
 *
 * Computes:
 *   T = [R, t; 0, 1/s]
 * where R = Exp_SO3(w) via Rodrigues' formula,
 * t = V(w, lambda) * u with the Sim(3) V-matrix,
 * and s = exp(lambda).
 *
 * The V-matrix is: V = P*I + Q*[w]_x + R_c*w*w^T
 * where P, Q, R_c are computed from theta = |w| and lambda,
 * using a blending between the SE(3)-like and pure-scale regimes
 * (Eade, "Lie Groups for 2D and 3D Transformations").
 */
__global__ void exp_sim3_kernel(const float* tangent, size_t tangent_stride,
                                float* transforms, size_t transform_stride,
                                size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* xi = tangent + idx * tangent_stride;
  float w1 = xi[0], w2 = xi[1], w3 = xi[2];
  float u1 = xi[3], u2 = xi[4], u3 = xi[5];
  float lambda = xi[6];

  float theta2 = w1 * w1 + w2 * w2 + w3 * w3;
  float theta = sqrtf(theta2);

  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    A1 = st / theta;
    A2 = (1.0f - ct) / theta2;
    A3 = (1.0f - A1) / theta2;
    A4 = (0.5f - A2) / theta2;
  } else {
    A1 = 1.0f - theta2 / 6.0f;
    A2 = 0.5f - theta2 / 24.0f;
    A3 = 1.0f / 6.0f - theta2 / 120.0f;
    A4 = 1.0f / 24.0f - theta2 / 720.0f;
  }

  float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    float e = expf(-lambda);
    P_c = (1.0f - e) / lambda;
    float alpha = lambda2 / (lambda2 + theta2);
    float beta = (e - 1.0f + lambda) / lambda2;
    float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    float one_m_alpha = 1.0f - alpha;
    Q_c = alpha * beta + one_m_alpha * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_alpha * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  float ct_val = 1.0f - A2 * theta2;
  float* T = transforms + idx * transform_stride;
  T[0]  = ct_val + A2 * w1 * w1;
  T[1]  = A2 * w1 * w2 - A1 * w3;
  T[2]  = A2 * w1 * w3 + A1 * w2;
  T[4]  = A2 * w2 * w1 + A1 * w3;
  T[5]  = ct_val + A2 * w2 * w2;
  T[6]  = A2 * w2 * w3 - A1 * w1;
  T[8]  = A2 * w3 * w1 - A1 * w2;
  T[9]  = A2 * w3 * w2 + A1 * w1;
  T[10] = ct_val + A2 * w3 * w3;

  float diag = P_c - R_c * theta2;
  float dot_wu = w1 * u1 + w2 * u2 + w3 * u3;
  float cx = w2 * u3 - w3 * u2;
  float cy = w3 * u1 - w1 * u3;
  float cz = w1 * u2 - w2 * u1;
  T[3]  = diag * u1 + Q_c * cx + R_c * w1 * dot_wu;
  T[7]  = diag * u2 + Q_c * cy + R_c * w2 * dot_wu;
  T[11] = diag * u3 + Q_c * cz + R_c * w3 * dot_wu;

  float inv_s = expf(-lambda);
  T[12] = 0.0f;
  T[13] = 0.0f;
  T[14] = 0.0f;
  T[15] = inv_s;
}

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
__global__ void log_sim3_kernel(const float* transforms,
                                size_t transform_stride, float* tangent,
                                size_t tangent_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* T = transforms + idx * transform_stride;
  float* xi = tangent + idx * tangent_stride;

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

  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    A1 = st / theta;
    A2 = (1.0f - ct) / theta2;
    A3 = (1.0f - A1) / theta2;
    A4 = (0.5f - A2) / theta2;
  } else {
    A1 = 1.0f - theta2 / 6.0f;
    A2 = 0.5f - theta2 / 24.0f;
    A3 = 1.0f / 6.0f - theta2 / 120.0f;
    A4 = 1.0f / 24.0f - theta2 / 720.0f;
  }

  float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    float e = expf(-lambda);
    P_c = (1.0f - e) / lambda;
    float alpha = lambda2 / (lambda2 + theta2);
    float beta = (e - 1.0f + lambda) / lambda2;
    float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    float one_m_a = 1.0f - alpha;
    Q_c = alpha * beta + one_m_a * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_a * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  float diag = P_c - R_c * theta2;
  float V[9], W_mat[9];
  skew3(w, W_mat);
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
    xi[0] = w[0]; xi[1] = w[1]; xi[2] = w[2];
    xi[3] = t[0]; xi[4] = t[1]; xi[5] = t[2];
    xi[6] = lambda;
    return;
  }
  float u[3];
  mat3_vec(Vinv, t, u);

  xi[0] = w[0]; xi[1] = w[1]; xi[2] = w[2];
  xi[3] = u[0]; xi[4] = u[1]; xi[5] = u[2];
  xi[6] = lambda;
}

/**
 * @brief Sim(3) inverse: T^{-1} = [R^T, -s*R^T*t; 0, s].
 *
 * For T = [R t; 0 0 0 1/s], the inverse is [R^T, -s*R^T*t; 0 0 0 s].
 */
__global__ void inverse_sim3_kernel(const float* transforms,
                                    size_t transform_stride,
                                    float* inverse_transforms,
                                    size_t inverse_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* T = transforms + idx * transform_stride;
  float* Ti = inverse_transforms + idx * inverse_stride;

  float r00 = T[0], r01 = T[1], r02 = T[2], tx = T[3];
  float r10 = T[4], r11 = T[5], r12 = T[6], ty = T[7];
  float r20 = T[8], r21 = T[9], r22 = T[10], tz = T[11];
  float inv_s = T[15];
  float s = 1.0f / inv_s;

  Ti[0]  = r00;   Ti[1]  = r10;   Ti[2]  = r20;
  Ti[3]  = -s * (r00 * tx + r10 * ty + r20 * tz);
  Ti[4]  = r01;   Ti[5]  = r11;   Ti[6]  = r21;
  Ti[7]  = -s * (r01 * tx + r11 * ty + r21 * tz);
  Ti[8]  = r02;   Ti[9]  = r12;   Ti[10] = r22;
  Ti[11] = -s * (r02 * tx + r12 * ty + r22 * tz);
  Ti[12] = 0.0f;  Ti[13] = 0.0f;  Ti[14] = 0.0f;  Ti[15] = s;
}

/**
 * @brief Inverse right Jacobian of Sim(3): J_r^{-1}(xi).
 *
 * For xi = [w1,w2,w3, u1,u2,u3, lambda], computes the 7x7 J_r^{-1}(xi)
 * using a truncated Bernoulli series (up to order 6).
 *
 * The ad matrix of sim(3) decomposes as:
 *   ad(xi) = [[W,   0,      0  ],   (3x3, 3x3, 3x1)
 *             [U,   W+lI,  -u  ],
 *             [0,   0,      0  ]]
 *
 * where W = [w]_x, U = [u]_x, l = lambda.
 *
 * J_r^{-1} = I + Sum_{n=1}^{6} (B_n^+/n!) * ad^n is computed in blocks:
 *
 * Block(0,0) = J_r^{-1}_SO3(w): closed-form using Rodrigues-derived coefficients
 *   a0*I + a1*W + a2*W^2 where a1 = 1/2, a2 = 1/theta^2 - (1+cos)/(2*theta*sin).
 *
 * Block(1,1) = I + Sum c_n * (M^n decomposed as a_n*I + b_n*W + c_n*W^2)
 *   where M = W + lambda*I and M^n uses the recurrence
 *   (a', b', c') = (lam*a, lam*b + a - theta^2*c, lam*c + b)
 *   exploiting W^3 = -theta^2 * W (Cayley-Hamilton for skew-symmetric 3x3).
 *
 * Block(1,0) = Sum c_n * B_n where B_n = U*W^{n-1} + M*B_{n-1},
 *   using W^n scalar representation and full 3x3 B_n accumulation.
 *
 * Block(1,2) = -Sum d_n * M^n * u, computed via the same scalar M^n
 *   representation applied to u, w x u, and W^2*u.
 *
 * B_n^+/n! are the shifted Bernoulli coefficients:
 *   n=1: 1/2, n=2: 1/12, n=3: 0, n=4: -1/720, n=5: 0, n=6: 1/30240.
 */
__global__ void jacobian_right_inverse_sim3_kernel(
    const float* tangent, size_t tangent_stride, float* jacobians,
    size_t jacobian_stride, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  const float* xi = tangent + idx * tangent_stride;
  float* J = jacobians + idx * jacobian_stride;

  float w[3] = {xi[0], xi[1], xi[2]};
  float u[3] = {xi[3], xi[4], xi[5]};
  float lam = xi[6];

  float theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];

  if (theta2 + lam * lam < 1e-6f) {
    for (int i = 0; i < 49; ++i) {
      J[i] = 0.0f;
    }
    J[0] = 1.0f; J[8] = 1.0f; J[16] = 1.0f;
    J[24] = 1.0f; J[32] = 1.0f; J[40] = 1.0f;
    J[48] = 1.0f;
    J[1]  += -0.5f * w[2];  J[2]  +=  0.5f * w[1];
    J[7]  +=  0.5f * w[2];  J[9]  += -0.5f * w[0];
    J[14] += -0.5f * w[1];  J[15] +=  0.5f * w[0];
    J[24] += 0.5f * lam;    J[25] += -0.5f * w[2];  J[26] +=  0.5f * w[1];
    J[31] += 0.5f * w[2];   J[32] += 0.5f * lam;    J[33] += -0.5f * w[0];
    J[38] += -0.5f * w[1];  J[39] +=  0.5f * w[0];  J[40] += 0.5f * lam;
    J[21] += -0.5f * u[2];  J[22] +=  0.5f * u[1];
    J[28] +=  0.5f * u[2];  J[30] += -0.5f * u[0];
    J[35] += -0.5f * u[1];  J[36] +=  0.5f * u[0];
    J[27] = -0.5f * u[0];  J[34] = -0.5f * u[1];  J[41] = -0.5f * u[2];
    return;
  }

  float theta = sqrtf(theta2);

  float a0 = 1.0f, a1 = 0.5f, a2;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    a2 = 1.0f / theta2 - (1.0f + ct) / (2.0f * theta * st);
  } else {
    a2 = 1.0f / 12.0f + theta2 / 720.0f;
  }

  float W[9], W2[9], U_mat[9], UW[9], UW2[9];
  skew3(w, W);
  skew3(u, U_mat);
  mat3_mul(W, W, W2);
  mat3_mul(U_mat, W, UW);
  mat3_mul(U_mat, W2, UW2);

  float wu_cross[3] = {w[1]*u[2]-w[2]*u[1], w[2]*u[0]-w[0]*u[2], w[0]*u[1]-w[1]*u[0]};
  float dot_wu = w[0]*u[0] + w[1]*u[1] + w[2]*u[2];
  float W2u[3] = {w[0]*dot_wu - theta2*u[0],
                   w[1]*dot_wu - theta2*u[1],
                   w[2]*dot_wu - theta2*u[2]};

  constexpr float c1 = 0.5f, c2 = 1.0f/12.0f, c4 = -1.0f/720.0f, c6 = 1.0f/30240.0f;
  constexpr float d0 = 0.5f, d1 = 1.0f/12.0f, d3 = -1.0f/720.0f, d5 = 1.0f/30240.0f;

  float ma = lam, mb = 1.0f, mc = 0.0f;

  float wa_prev = 1.0f, wb_prev = 0.0f, wc_prev = 0.0f;

  float sa = 0.0f, sb = 0.0f, sc = 0.0f;

  float Bn[9], SB[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) { Bn[i] = U_mat[i]; SB[i] = 0.0f; }

  sa += c1 * ma; sb += c1 * mb; sc += c1 * mc;
#pragma unroll
  for (int i = 0; i < 9; ++i) { SB[i] += c1 * Bn[i]; }

  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  float wa_cur = 0.0f, wb_cur = 1.0f, wc_cur = 0.0f;
  { float UWn[9]; float WB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { UWn[i] = wa_cur * U_mat[i] + wb_cur * UW[i] + wc_cur * UW2[i]; }
    mat3_mul(W, Bn, WB);
#pragma unroll
    for (int i = 0; i < 9; ++i) { Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; }
    wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = 1.0f; }

  sa += c2 * ma; sb += c2 * mb; sc += c2 * mc;
#pragma unroll
  for (int i = 0; i < 9; ++i) { SB[i] += c2 * Bn[i]; }

  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = -theta2; wc_cur = 0.0f; }
  { float UWn[9]; float WB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i]; }
    mat3_mul(W, Bn, WB);
#pragma unroll
    for (int i = 0; i < 9; ++i) { Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; } }

  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = -theta2; }
  { float UWn[9]; float WB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i]; }
    mat3_mul(W, Bn, WB);
#pragma unroll
    for (int i = 0; i < 9; ++i) { Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; } }

  sa += c4 * ma; sb += c4 * mb; sc += c4 * mc;
#pragma unroll
  for (int i = 0; i < 9; ++i) { SB[i] += c4 * Bn[i]; }

  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    float t4 = theta2*theta2;
    wa_cur = 0.0f; wb_cur = t4; wc_cur = 0.0f; }
  { float UWn[9]; float WB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i]; }
    mat3_mul(W, Bn, WB);
#pragma unroll
    for (int i = 0; i < 9; ++i) { Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; } }

  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    float t4 = theta2*theta2;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = t4; }
  { float UWn[9]; float WB[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) { UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i]; }
    mat3_mul(W, Bn, WB);
#pragma unroll
    for (int i = 0; i < 9; ++i) { Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; } }

  sa += c6 * ma; sb += c6 * mb; sc += c6 * mc;
#pragma unroll
  for (int i = 0; i < 9; ++i) { SB[i] += c6 * Bn[i]; }

  for (int i = 0; i < 49; ++i) { J[i] = 0.0f; }

#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      int ij = r * 3 + c;
      float val = a1 * W[ij] + a2 * W2[ij];
      if (r == c) { val += a0; }
      J[r * 7 + c] = val;
    }
  }

#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      int ij = r * 3 + c;
      float val = sb * W[ij] + sc * W2[ij];
      if (r == c) { val += 1.0f + sa; }
      J[(r + 3) * 7 + (c + 3)] = val;
    }
  }

#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      J[(r + 3) * 7 + c] = SB[r * 3 + c];
    }
  }

  float ma_s[6], mb_s[6], mc_s[6];
  ma_s[0] = 1.0f; mb_s[0] = 0.0f; mc_s[0] = 0.0f;
  ma_s[1] = lam;  mb_s[1] = 1.0f; mc_s[1] = 0.0f;
#pragma unroll
  for (int n = 1; n < 5; ++n) {
    ma_s[n+1] = lam * ma_s[n];
    mb_s[n+1] = lam * mb_s[n] + ma_s[n] - theta2 * mc_s[n];
    mc_s[n+1] = lam * mc_s[n] + mb_s[n];
  }
  float sd_a = d0*ma_s[0] + d1*ma_s[1] + d3*ma_s[3] + d5*ma_s[5];
  float sd_b = d0*mb_s[0] + d1*mb_s[1] + d3*mb_s[3] + d5*mb_s[5];
  float sd_c = d0*mc_s[0] + d1*mc_s[1] + d3*mc_s[3] + d5*mc_s[5];

  J[3 * 7 + 6] = -(sd_a * u[0] + sd_b * wu_cross[0] + sd_c * W2u[0]);
  J[4 * 7 + 6] = -(sd_a * u[1] + sd_b * wu_cross[1] + sd_c * W2u[1]);
  J[5 * 7 + 6] = -(sd_a * u[2] + sd_b * wu_cross[2] + sd_c * W2u[2]);

  J[48] = 1.0f;
}

/**
 * @brief Compute the Sim(3) adjoint matrix Ad(T) from a Sim(3) transform.
 *
 * Given T = [[R, t], [0, 1/s]]  (row-major 4x4), computes the 7x7 adjoint:
 *
 *   Ad(T) = [[R,            0_{3x3},  0_{3x1} ],
 *            [s*[t]_x*R,    s*R,      -s*t     ],
 *            [0_{1x3},      0_{1x3},  1        ]]
 *
 * where [t]_x is the skew-symmetric matrix of t.
 */
__global__ void compute_adjoint_sim3_kernel(const float* transforms,
                                            size_t transform_stride,
                                            float* adjoints,
                                            size_t adjoint_stride,
                                            size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }

  const float* T = transforms + tid * transform_stride;
  float* Ad = adjoints + tid * adjoint_stride;

  float R00 = T[0], R01 = T[1], R02 = T[2];
  float R10 = T[4], R11 = T[5], R12 = T[6];
  float R20 = T[8], R21 = T[9], R22 = T[10];
  float tx = T[3], ty = T[7], tz = T[11];
  float s = 1.0f / T[15];

  // A = s * [t]_x * R, where [t]_x = [[0,-tz,ty],[tz,0,-tx],[-ty,tx,0]]
  float A00 = s * (-tz * R10 + ty * R20);
  float A01 = s * (-tz * R11 + ty * R21);
  float A02 = s * (-tz * R12 + ty * R22);
  float A10 = s * (tz * R00 - tx * R20);
  float A11 = s * (tz * R01 - tx * R21);
  float A12 = s * (tz * R02 - tx * R22);
  float A20 = s * (-ty * R00 + tx * R10);
  float A21 = s * (-ty * R01 + tx * R11);
  float A22 = s * (-ty * R02 + tx * R12);

  // Row 0-2: [R | 0 | 0]
  Ad[0] = R00; Ad[1] = R01; Ad[2] = R02;
  Ad[3] = 0;   Ad[4] = 0;   Ad[5] = 0;   Ad[6] = 0;
  Ad[7] = R10; Ad[8] = R11; Ad[9] = R12;
  Ad[10] = 0;  Ad[11] = 0;  Ad[12] = 0;  Ad[13] = 0;
  Ad[14] = R20; Ad[15] = R21; Ad[16] = R22;
  Ad[17] = 0;   Ad[18] = 0;   Ad[19] = 0;  Ad[20] = 0;

  // Row 3-5: [A | sR | -s*t]
  Ad[21] = A00; Ad[22] = A01; Ad[23] = A02;
  Ad[24] = s * R00; Ad[25] = s * R01; Ad[26] = s * R02; Ad[27] = -s * tx;
  Ad[28] = A10; Ad[29] = A11; Ad[30] = A12;
  Ad[31] = s * R10; Ad[32] = s * R11; Ad[33] = s * R12; Ad[34] = -s * ty;
  Ad[35] = A20; Ad[36] = A21; Ad[37] = A22;
  Ad[38] = s * R20; Ad[39] = s * R21; Ad[40] = s * R22; Ad[41] = -s * tz;

  // Row 6: [0 0 0 0 0 0 1]
  Ad[42] = 0; Ad[43] = 0; Ad[44] = 0;
  Ad[45] = 0; Ad[46] = 0; Ad[47] = 0; Ad[48] = 1;
}

// ============================================================================
// Sim(3) Host wrappers
// ============================================================================

void ComputeExpSim3(cudaStream_t stream, const float* tangent,
                    size_t tangent_stride, size_t transform_stride,
                    size_t size, float* transforms) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  exp_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      tangent, tangent_stride, transforms, transform_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeLogSim3(cudaStream_t stream, const float* transforms,
                    size_t transform_stride, size_t tangent_stride,
                    size_t size, float* tangent) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  log_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      transforms, transform_stride, tangent, tangent_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeInverseSim3(cudaStream_t stream, const float* transforms,
                        size_t transform_stride, size_t inverse_stride,
                        size_t size, float* inverse_transforms) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  inverse_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      transforms, transform_stride, inverse_transforms, inverse_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeJacobianRightInverseSim3(cudaStream_t stream, const float* tangent,
                                     size_t tangent_stride,
                                     size_t jacobian_stride, size_t size,
                                     float* jacobians) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  jacobian_right_inverse_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0,
                                       stream>>>(
      tangent, tangent_stride, jacobians, jacobian_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeAdjointSim3(cudaStream_t stream, const float* transforms,
                         size_t transform_stride, float* adjoints,
                         size_t adjoint_stride, size_t size) {
  size_t num_blocks = (size + kSimMathBlockSize - 1) / kSimMathBlockSize;
  compute_adjoint_sim3_kernel<<<num_blocks, kSimMathBlockSize, 0, stream>>>(
      transforms, transform_stride, adjoints, adjoint_stride, size);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
