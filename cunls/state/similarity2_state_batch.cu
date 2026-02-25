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

#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "cunls/common/helper.h"
#include "cunls/state/similarity2_state_batch.h"

namespace cunls {

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
__device__ void ComputeVCoefficients(float theta, float lambda,
                                     float& X, float& thetaY) {
  const float lambda2 = lambda * lambda;
  const float theta2 = theta * theta;

  // SE(2) or near-SE(2) case (|lambda| tiny)
  if (fabsf(lambda) < 1e-4f) {
    float A, B;
    if (theta2 > 1e-6f) {
      float s = sinf(theta);
      float c = cosf(theta);
      A = s / theta;
      B = (1.0f - c) / theta2;
    } else {
      // Taylor series for small theta
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
    // Both tiny -> identity
    X = 1.0f;
    thetaY = 0.0f;
    return;
  }

  float A, B, C;
  if (theta2 > 1e-6f) {
    float s = sinf(theta);
    float c = cosf(theta);
    A = s / theta;
    B = (1.0f - c) / theta2;
    C = (1.0f - A) / theta2;
  } else {
    // Taylor series for small theta
    A = 1.0f - theta2 / 6.0f;
    B = 0.5f - theta2 / 24.0f;
    C = 1.0f / 6.0f - theta2 / 120.0f;
  }

  float alpha = lambda2 / d2;
  float s_inv = expf(-lambda);
  X = alpha * (1.0f - s_inv) / lambda + (1.0f - alpha) * (A - lambda * B);
  float Y = alpha * (s_inv - 1.0f + lambda) / lambda2 +
            (1.0f - alpha) * (B - lambda * C);
  thetaY = theta * Y;
}

/**
 * @brief CUDA kernel that computes the Sim(2) exponential map for a batch of tangent vectors.
 *
 * For each tangent vector [u_x, u_y, theta, lambda], computes the 3x3 transformation
 * matrix in row-major order:
 *   [cos(theta), -sin(theta), tx]
 *   [sin(theta),  cos(theta), ty]
 *   [    0,           0,     e^{-lambda}]
 *
 * where [tx, ty] = V(theta, lambda) * [u_x, u_y].
 *
 * @param tangent Input tangent vectors (4D, device pointer, stride 4)
 * @param transforms Output 3x3 transformation matrices (device pointer, row-major,
 *                   9 floats per matrix)
 * @param size Number of tangent vectors to process
 */
__global__ void ExpSim2Kernel(const float* tangent, float* transforms,
                              size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const float* xi = tangent + idx * 4;
  float ux = xi[0];
  float uy = xi[1];
  float theta = xi[2];
  float lambda = xi[3];

  float c = cosf(theta);
  float s = sinf(theta);
  float inv_scale = expf(-lambda);  // 1/s = e^{-lambda}

  // Compute V-matrix coefficients
  float X, thetaY;
  ComputeVCoefficients(theta, lambda, X, thetaY);

  // t = V * u, where V = [[X, -thetaY], [thetaY, X]]
  float tx = X * ux - thetaY * uy;
  float ty = thetaY * ux + X * uy;

  // Row-major 3x3 Sim(2) matrix [R t; 0 0 1/s]
  float* T = transforms + idx * 9;
  T[0] = c;     T[1] = -s;    T[2] = tx;
  T[3] = s;     T[4] = c;     T[5] = ty;
  T[6] = 0.0f;  T[7] = 0.0f;  T[8] = inv_scale;
}

/** @copydoc Similarity2StateBatch::ApplyUpdate */
void Similarity2StateBatch::ApplyUpdate(const float* x,
                                                 const float* delta,
                                                 float* result,
                                                 bool invert_delta,
                                                 cudaStream_t stream) {
  size_t num_transforms = NumStateBlocks();

  dvector<Matrix<3>> delta_transforms_(num_transforms);

  dvector<float> tangents_(num_transforms * 4);

  auto tangents_ptr = reinterpret_cast<const float*>(tangents_.data());
  auto delta_transforms_ptr =
      reinterpret_cast<float*>(delta_transforms_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> tangents_device_ptr(tangents_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_transforms * 4,
               tangents_device_ptr);

  // Negate tangent vectors if computing Exp(-delta) instead of Exp(delta)
  if (invert_delta) {
    thrust::transform(stream_policy, tangents_device_ptr,
                      tangents_device_ptr + tangents_.size(),
                      tangents_device_ptr, thrust::negate<float>());
  }

  // Compute update matrices: delta_transforms = Exp(±delta)
  constexpr int block_size = 256;
  int num_blocks = (num_transforms + block_size - 1) / block_size;
  ExpSim2Kernel<<<num_blocks, block_size, 0, stream>>>(
      tangents_ptr, delta_transforms_ptr, num_transforms);

  auto handle = cublas_handle_.GetHandle(stream);

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr int mat_size = 3;
  constexpr int stride = 9;

  // Perform batched matrix multiplication: result = x * Exp(±delta)
  // Note: cuBLAS uses column-major, but CUBLAS_OP_N for both operands computes
  // the equivalent of row-major right-multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_transforms_ptr, mat_size, stride, x, mat_size, stride, &beta,
      result, mat_size, stride, num_transforms));
}

/**
 * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
 *
 * Computes the right-multiplication update for Sim(2) transformations.
 * First computes the update matrix Exp(delta) from the tangent vector,
 * then performs batched matrix multiplication using cuBLAS.
 *
 * Note: cuBLAS uses column-major storage, but our matrices are row-major.
 * For right-multiplication (x * update), we use CUBLAS_OP_N for both operands.
 * cuBLAS interprets the matrices as column-major, so this computes the
 * equivalent of the desired row-major result.
 *
 * @param x Input transformation matrices (device pointer, row-major)
 * @param delta Tangent space updates (4D vectors [u_x, u_y, theta, lambda],
 *              device pointer)
 * @param x_plus_delta Output transformation matrices (device pointer,
 *                     row-major)
 * @param stream CUDA stream for asynchronous execution
 */
void Similarity2StateBatch::Plus(const float* x, const float* delta,
                                          float* x_plus_delta,
                                          cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
