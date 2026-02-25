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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity2_prior_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block.
constexpr size_t kSim2PriorBlockSize = 256;

/**
 * @brief Device helper: computes the Sim(2) V-matrix coefficients.
 *
 * The V-matrix maps tangent translation components to the translation
 * in the group element: t = V(theta, lambda) * [u_x, u_y].
 *
 * V = [[X, -theta*Y], [theta*Y, X]]
 *
 * Reference: Eade, "Lie Groups for 2D and 3D Transformations", page 6.
 *
 * @param theta Rotation angle
 * @param lambda Log-scale (lambda = log(s))
 * @param[out] X Diagonal element of V
 * @param[out] thetaY Off-diagonal magnitude (theta * Y)
 */
__device__ void ComputeVCoeffsSim2(float theta, float lambda,
                                   float& X, float& thetaY) {
  const float lambda2 = lambda * lambda;
  const float theta2 = theta * theta;

  // SE(2) or near-SE(2) case (|lambda| tiny)
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

  // General Sim(2) case
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

/**
 * @brief CUDA kernel to compute the inverse of Sim(2) transformation matrices.
 *
 * For T = [R t; 0 1/s], computes T^{-1} = [R^T  -s*R^T*t; 0  s].
 *
 * @param transforms Input transformation matrices (3x3, row-major, device pointer)
 * @param inverse_transforms Output inverse matrices (3x3, row-major, device pointer)
 * @param size Number of transforms to process
 */
__global__ void inverse_sim2_kernel(const float* transforms,
                                    float* inverse_transforms, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 9;
  float* T_inv = inverse_transforms + tid * 9;

  // T = [R t; 0 1/s] where R is 2x2, t is 2x1
  float r00 = T[0], r01 = T[1], tx = T[2];
  float r10 = T[3], r11 = T[4], ty = T[5];
  float inv_s = T[8];  // 1/s
  float s = 1.0f / inv_s;

  // T^{-1} = [R^T  -s*R^T*t; 0  s]
  // R^T = [[r00, r10], [r01, r11]]
  // -s*R^T*t = [-s*(r00*tx + r10*ty), -s*(r01*tx + r11*ty)]
  T_inv[0] = r00;   T_inv[1] = r10;   T_inv[2] = -s * (r00 * tx + r10 * ty);
  T_inv[3] = r01;   T_inv[4] = r11;   T_inv[5] = -s * (r01 * tx + r11 * ty);
  T_inv[6] = 0.0f;  T_inv[7] = 0.0f;  T_inv[8] = s;
}

/**
 * @brief CUDA kernel to collect Sim(2) transforms from state pointers.
 *
 * @param state_pointers Array of state block pointers
 * @param num_factors Number of factors
 * @param transforms Output array for collected transforms
 */
__global__ void collect_sim2_transforms_kernel(float const* const* state_pointers,
                                               size_t num_factors,
                                               Matrix<3>* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;

  auto transform_ptr = reinterpret_cast<const Matrix<3>*>(state_pointers[tid]);
  transforms[tid] = *transform_ptr;
}

/**
 * @brief CUDA kernel to compute the Sim(2) logarithm map.
 *
 * For a 3x3 Sim(2) transformation matrix T = [R t; 0 1/s], computes the tangent vector:
 *   xi = Log(T) = [u_x, u_y, theta, lambda]
 *
 * where theta = atan2(sin, cos) from R, lambda = log(s) = -log(1/s),
 * and [u_x, u_y] = V(theta, lambda)^{-1} * [tx, ty].
 *
 * @param transforms Input transformation matrices (3x3, row-major, device pointer)
 * @param tangent Output tangent vectors (4D, device pointer, stride 4)
 * @param size Number of transforms to process
 */
__global__ void log_sim2_kernel(const float* transforms, float* tangent,
                                size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 9;
  float* xi = tangent + tid * 4;

  float c = T[0];       // cos(theta)
  float s = T[3];       // sin(theta)
  float tx = T[2];
  float ty = T[5];
  float inv_s = T[8];   // 1/scale

  float theta = atan2f(s, c);
  float scale = 1.0f / inv_s;
  float lambda = logf(scale);

  // Compute V-matrix coefficients
  float X, thetaY;
  ComputeVCoeffsSim2(theta, lambda, X, thetaY);

  // u = V^{-1} * t
  // V = [[X, -thetaY], [thetaY, X]]
  // V^{-1} = [[X, thetaY], [-thetaY, X]] / (X^2 + thetaY^2)
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
 * @brief CUDA kernel to compute the inverse right Jacobian of Sim(2).
 *
 * For a tangent vector xi = [u_x, u_y, theta, lambda], computes J_r^{-1}(xi),
 * the inverse of the right Jacobian of Sim(2).
 *
 * The 4x4 Jacobian has block structure:
 *   J_r^{-1} = [[TL^{-1}, -TL^{-1}*TR], [0, I]]
 *
 * where TL and TR are 2x2 blocks derived from the adjoint representation:
 *   ad(xi) = [[lambda, -w, u2, -u1],
 *             [w, lambda, -u1, -u2],
 *             [0, 0, 0, 0],
 *             [0, 0, 0, 0]]
 *
 * TL = A^{-1}(I - e^{-A}),  TR = -A^{-2}(e^{-A} - I + A) * B,
 * where A = lambda*I + w*J is the top-left 2x2 of ad, and B is the top-right 2x2.
 *
 * @param tangent Input tangent vectors (4D, device pointer, stride 4)
 * @param jacobians Output 4x4 Jacobian matrices (device pointer, row-major, stride 16)
 * @param size Number of tangent vectors to process
 */
__global__ void jacobian_right_inverse_sim2_kernel(const float* tangent,
                                                   float* jacobians,
                                                   size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* xi = tangent + tid * 4;
  float* J = jacobians + tid * 16;

  float u1 = xi[0], u2 = xi[1], w = xi[2], lambda = xi[3];
  float d2 = lambda * lambda + w * w;

  if (d2 < 1e-5f) {
    // Small angle approximation: J_r^{-1} ≈ I + 0.5 * ad(xi)
    // ad(xi) = [[lambda, -w, u2, -u1],
    //           [w, lambda, -u1, -u2],
    //           [0, 0, 0, 0],
    //           [0, 0, 0, 0]]
    J[0]  = 1.0f + 0.5f * lambda;  J[1]  = -0.5f * w;
    J[2]  = 0.5f * u2;             J[3]  = -0.5f * u1;
    J[4]  = 0.5f * w;              J[5]  = 1.0f + 0.5f * lambda;
    J[6]  = -0.5f * u1;            J[7]  = -0.5f * u2;
    J[8]  = 0.0f;                  J[9]  = 0.0f;
    J[10] = 1.0f;                  J[11] = 0.0f;
    J[12] = 0.0f;                  J[13] = 0.0f;
    J[14] = 0.0f;                  J[15] = 1.0f;
    return;
  }

  float e_neg_lambda = expf(-lambda);
  float cos_w = cosf(w);
  float sin_w = sinf(w);

  // alpha = 1 - e^{-lambda} * cos(w)
  // beta  = e^{-lambda} * sin(w)
  float alpha = 1.0f - e_neg_lambda * cos_w;
  float beta = e_neg_lambda * sin_w;

  // TL = A^{-1}(I - e^{-A}) where A = lambda*I + w*J
  // I - e^{-A} = alpha*I + beta*J (after derivation)
  // A^{-1} = (lambda*I - w*J) / d2
  // TL = ((lambda*alpha + w*beta)/d2)*I + ((lambda*beta - w*alpha)/d2)*J
  //    = [[P, -Q], [Q, P]]
  float P = (lambda * alpha + w * beta) / d2;
  float Q = (lambda * beta - w * alpha) / d2;

  // TL^{-1} = [[P, Q], [-Q, P]] / (P^2 + Q^2)
  float det_TL = P * P + Q * Q;
  float inv_det_TL = 1.0f / det_TL;
  float TLi00 = P * inv_det_TL;
  float TLi01 = Q * inv_det_TL;
  float TLi10 = -Q * inv_det_TL;
  float TLi11 = P * inv_det_TL;

  // Compute TR = -A^{-2} * (e^{-A} - I + A) * B
  // where B = [[u2, -u1], [-u1, -u2]] (top-right of ad)
  //
  // e^{-A} - I + A = (lambda - alpha)*I + (w - beta)*J
  float c_val = lambda - alpha;   // lambda - 1 + e^{-lambda}*cos(w)
  float d_val = w - beta;         // w - e^{-lambda}*sin(w)

  // A^{-2} = ((lambda^2 - w^2)*I - 2*lambda*w*J) / d2^2
  float d4 = d2 * d2;
  float e_val = (lambda * lambda - w * w) / d4;
  float f_val = -2.0f * lambda * w / d4;

  // A^{-2} * (e^{-A} - I + A) = (e_val*c_val - f_val*d_val)*I
  //                             + (e_val*d_val + f_val*c_val)*J
  //                            =: g*I + h*J
  float g = e_val * c_val - f_val * d_val;
  float h = e_val * d_val + f_val * c_val;

  // TR = -(g*I + h*J) * B
  // where B = [[u2, -u1], [-u1, -u2]]
  // J*B = [[u1, u2], [u2, -u1]]
  // -g*B = [[-g*u2, g*u1], [g*u1, g*u2]]
  // -h*J*B = [[-h*u1, -h*u2], [-h*u2, h*u1]]
  float TR00 = -(g * u2 + h * u1);
  float TR01 = g * u1 - h * u2;
  float TR10 = g * u1 - h * u2;
  float TR11 = g * u2 + h * u1;

  // M = -TL^{-1} * TR (the top-right 2x2 block of J_r^{-1})
  float M00 = -(TLi00 * TR00 + TLi01 * TR10);
  float M01 = -(TLi00 * TR01 + TLi01 * TR11);
  float M10 = -(TLi10 * TR00 + TLi11 * TR10);
  float M11 = -(TLi10 * TR01 + TLi11 * TR11);

  // Assemble 4x4 J_r^{-1} (row-major)
  J[0]  = TLi00;  J[1]  = TLi01;  J[2]  = M00;   J[3]  = M01;
  J[4]  = TLi10;  J[5]  = TLi11;  J[6]  = M10;   J[7]  = M11;
  J[8]  = 0.0f;   J[9]  = 0.0f;   J[10] = 1.0f;  J[11] = 0.0f;
  J[12] = 0.0f;   J[13] = 0.0f;   J[14] = 0.0f;  J[15] = 1.0f;
}

Similarity2PriorFactorBatch::Similarity2PriorFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<3>* observations_ptr,
    size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      cublas_handle_(cublas_handle),
      transforms_current_(num_factors),
      transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all targets
  CudaStream stream;
  size_t num_blocks =
      (num_factors_ + kSim2PriorBlockSize - 1) / kSim2PriorBlockSize;
  inverse_sim2_kernel<<<num_blocks, kSim2PriorBlockSize, 0,
                        stream.GetStream()>>>(
      reinterpret_cast<const float*>(observations_ptr_),
      reinterpret_cast<float*>(observations_inverse_.data()),
      num_factors_);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool Similarity2PriorFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();

  // Step 1: Collect current transforms from state pointers into contiguous memory
  size_t num_blocks =
      (num_factors + kSim2PriorBlockSize - 1) / kSim2PriorBlockSize;
  collect_sim2_transforms_kernel<<<num_blocks, kSim2PriorBlockSize, 0,
                                   stream>>>(
      state_pointers, num_factors, transforms_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute T_error = T_target^{-1} * T_current using cuBLAS
  // cuBLAS column-major convention with row-major data:
  //   With CUBLAS_OP_N for both operands and row-major data:
  //   Result_row = B_row * A_row
  //   So with A=transforms_current, B=observations_inverse:
  //   Result_row = observations_inverse * transforms_current = T_inv * T_current
  auto handle = cublas_handle_.GetHandle(stream);
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(transforms_current_.data()), mat_size,
      stride,
      reinterpret_cast<const float*>(observations_inverse_.data()), mat_size,
      stride, &beta,
      reinterpret_cast<float*>(transforms_error_.data()), mat_size, stride,
      num_factors));

  // Step 3: Compute residual = Log(T_error) using Sim(2) logarithm map
  log_sim2_kernel<<<num_blocks, kSim2PriorBlockSize, 0, stream>>>(
      reinterpret_cast<const float*>(transforms_error_.data()), residuals,
      num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 4: Compute Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    jacobian_right_inverse_sim2_kernel<<<num_blocks, kSim2PriorBlockSize, 0,
                                         stream>>>(residuals, jacobians,
                                                    num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
