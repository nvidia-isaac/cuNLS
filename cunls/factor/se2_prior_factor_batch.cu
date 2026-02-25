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
#include "cunls/factor/se2_prior_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block.
constexpr size_t kSE2PriorBlockSize = 256;

/**
 * @brief CUDA kernel to compute the inverse of SE(2) transformation matrices.
 *
 * For T = [R t; 0 1], computes T^{-1} = [R^T  -R^T*t; 0  1].
 *
 * @param transforms Input transformation matrices (3x3, row-major, device pointer)
 * @param inverse_transforms Output inverse matrices (3x3, row-major, device pointer)
 * @param size Number of transforms to process
 */
__global__ void inverse_se2_kernel(const float* transforms,
                                   float* inverse_transforms, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 9;
  float* T_inv = inverse_transforms + tid * 9;

  // T = [R t; 0 1] where R is 2x2, t is 2x1
  float r00 = T[0], r01 = T[1], tx = T[2];
  float r10 = T[3], r11 = T[4], ty = T[5];

  // R^T = [[r00, r10], [r01, r11]]
  // -R^T * t = [-(r00*tx + r10*ty), -(r01*tx + r11*ty)]
  T_inv[0] = r00;   T_inv[1] = r10;   T_inv[2] = -(r00 * tx + r10 * ty);
  T_inv[3] = r01;   T_inv[4] = r11;   T_inv[5] = -(r01 * tx + r11 * ty);
  T_inv[6] = 0.0f;  T_inv[7] = 0.0f;  T_inv[8] = 1.0f;
}

/**
 * @brief CUDA kernel to collect SE(2) transforms from state pointers.
 *
 * @param state_pointers Array of state block pointers
 * @param num_factors Number of factors
 * @param transforms Output array for collected transforms
 */
__global__ void collect_se2_transforms_kernel(float const* const* state_pointers,
                                              size_t num_factors,
                                              Matrix<3>* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;

  auto transform_ptr = reinterpret_cast<const Matrix<3>*>(state_pointers[tid]);
  transforms[tid] = *transform_ptr;
}

/**
 * @brief CUDA kernel to compute the SE(2) logarithm map.
 *
 * For a 3x3 SE(2) transformation matrix T, computes the tangent vector:
 *   xi = Log(T) = [v_x, v_y, theta]
 *
 * Uses the closed-form formula from GTSAM's Pose2::Logmap.
 *
 * @param transforms Input transformation matrices (3x3, row-major, device pointer)
 * @param tangent Output tangent vectors (3D, device pointer, stride 3)
 * @param size Number of transforms to process
 */
__global__ void log_se2_kernel(const float* transforms, float* tangent,
                               size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 9;
  float* xi = tangent + tid * 3;

  float c = T[0];   // cos(theta)
  float s = T[3];   // sin(theta)
  float tx = T[2];
  float ty = T[5];

  float theta = atan2f(s, c);

  if (fabsf(theta) < 1e-3f) {
    // For small angles, V ≈ I, so Log ≈ [tx, ty, theta]
    // Note: threshold must be large enough that (1-cos(theta)) is
    // representable in single-precision float (epsilon ~1.19e-7).
    xi[0] = tx;
    xi[1] = ty;
    xi[2] = theta;
  } else {
    float c_1 = c - 1.0f;
    float det = c_1 * c_1 + s * s;  // = 2(1 - cos(theta))

    // Compute R^T * t - t
    // R^T * t = [c*tx + s*ty, -s*tx + c*ty]
    // d = R^T * t - t = [(c-1)*tx + s*ty, -s*tx + (c-1)*ty]
    float dx = c_1 * tx + s * ty;
    float dy = -s * tx + c_1 * ty;

    // p = R_PI_2 * d, where R_PI_2 rotates by 90°: (x,y) → (-y, x)
    float px = -dy;  // = s*tx + (1-c)*ty
    float py = dx;   // = (c-1)*tx + s*ty

    // v = (theta / det) * p
    float factor = theta / det;
    xi[0] = factor * px;
    xi[1] = factor * py;
    xi[2] = theta;
  }
}

/**
 * @brief CUDA kernel to compute the inverse right Jacobian of SE(2).
 *
 * For a tangent vector xi = [v_x, v_y, theta], computes J_r^{-1}(xi),
 * the inverse of the right Jacobian of SE(2).
 *
 * Uses the closed-form formula from GTSAM's Pose2::LogmapDerivative.
 *
 * @param tangent Input tangent vectors (3D, device pointer, stride 3)
 * @param jacobians Output 3x3 Jacobian matrices (device pointer, row-major, stride 9)
 * @param size Number of tangent vectors to process
 */
__global__ void jacobian_right_inverse_se2_kernel(const float* tangent,
                                                  float* jacobians,
                                                  size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* xi = tangent + tid * 3;
  float* J = jacobians + tid * 9;

  float v1 = xi[0];
  float v2 = xi[1];
  float alpha = xi[2];

  if (fabsf(alpha) > 1e-3f) {
    // Note: threshold must be large enough that (1-cos(alpha)) is
    // representable in single-precision float (epsilon ~1.19e-7).
    float alpha_inv = 1.0f / alpha;
    float half_cot_half_alpha =
        0.5f * sinf(alpha) / (1.0f - cosf(alpha));

    // Row-major 3x3 Jacobian
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
    // For small alpha, J_r^{-1} ≈ I + small correction
    J[0] = 1.0f;  J[1] = 0.0f;  J[2] = 0.5f * v2;
    J[3] = 0.0f;  J[4] = 1.0f;  J[5] = -0.5f * v1;
    J[6] = 0.0f;  J[7] = 0.0f;  J[8] = 1.0f;
  }
}

SE2PriorFactorBatch::SE2PriorFactorBatch(
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
      (num_factors_ + kSE2PriorBlockSize - 1) / kSE2PriorBlockSize;
  inverse_se2_kernel<<<num_blocks, kSE2PriorBlockSize, 0,
                       stream.GetStream()>>>(
      reinterpret_cast<const float*>(observations_ptr_),
      reinterpret_cast<float*>(observations_inverse_.data()),
      num_factors_);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SE2PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  size_t num_factors = NumFactors();

  // Step 1: Collect current transforms from state pointers into contiguous memory
  size_t num_blocks =
      (num_factors + kSE2PriorBlockSize - 1) / kSE2PriorBlockSize;
  collect_se2_transforms_kernel<<<num_blocks, kSE2PriorBlockSize, 0, stream>>>(
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

  // Step 3: Compute residual = Log(T_error) using SE(2) logarithm map
  log_se2_kernel<<<num_blocks, kSE2PriorBlockSize, 0, stream>>>(
      reinterpret_cast<const float*>(transforms_error_.data()), residuals,
      num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 4: Compute Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    jacobian_right_inverse_se2_kernel<<<num_blocks, kSE2PriorBlockSize, 0,
                                        stream>>>(residuals, jacobians,
                                                   num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
