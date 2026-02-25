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
#include "cunls/state/similarity3_state_batch.h"

namespace cunls {

/**
 * @brief CUDA kernel that computes the Sim(3) exponential map for a batch
 *        of tangent vectors.
 *
 * For each tangent vector [w1,w2,w3, u1,u2,u3, lambda], computes the 4x4
 * Sim(3) matrix in row-major order:
 *   [R00 R01 R02 tx]
 *   [R10 R11 R12 ty]
 *   [R20 R21 R22 tz]
 *   [ 0   0   0  1/s]
 *
 * where R = Exp_SO3(w), [tx,ty,tz] = V(w,lambda)*[u1,u2,u3], s = exp(lambda).
 *
 * The V-matrix for Sim(3) is V = P*I + Q*W + R_c*W*W  (Eade, Lie Groups).
 *
 * @param tangent  Input tangent vectors (7D, stride 7)
 * @param transforms  Output 4x4 matrices (stride 16)
 * @param size  Number of elements
 */
__global__ void ExpSim3Kernel(const float* tangent, float* transforms,
                              size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const float* xi = tangent + idx * 7;
  float w1 = xi[0], w2 = xi[1], w3 = xi[2];
  float u1 = xi[3], u2 = xi[4], u3 = xi[5];
  float lambda = xi[6];

  float theta2 = w1 * w1 + w2 * w2 + w3 * w3;
  float theta = sqrtf(theta2);

  // --- SO(3) coefficients ---
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

  // --- Sim(3) V-matrix coefficients ---
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

  // --- Build R via Rodrigues ---
  float ct_val = 1.0f - A2 * theta2;  // = cos(theta) for non-tiny theta
  // R = ct*I + A1*W + A2*ww^T
  float* T = transforms + idx * 16;
  T[0]  = ct_val + A2 * w1 * w1;
  T[1]  = A2 * w1 * w2 - A1 * w3;
  T[2]  = A2 * w1 * w3 + A1 * w2;
  T[4]  = A2 * w2 * w1 + A1 * w3;
  T[5]  = ct_val + A2 * w2 * w2;
  T[6]  = A2 * w2 * w3 - A1 * w1;
  T[8]  = A2 * w3 * w1 - A1 * w2;
  T[9]  = A2 * w3 * w2 + A1 * w1;
  T[10] = ct_val + A2 * w3 * w3;

  // --- t = V * u ---
  // V = P*I + Q*W + R_c*W^2 = (P - R_c*theta2)*I + Q*W + R_c*ww^T
  float diag = P_c - R_c * theta2;
  // V*u  (3-component)
  // V = diag*I + Q*skew(w) + R_c * w*w^T
  float dot_wu = w1 * u1 + w2 * u2 + w3 * u3;
  // cross w x u
  float cx = w2 * u3 - w3 * u2;
  float cy = w3 * u1 - w1 * u3;
  float cz = w1 * u2 - w2 * u1;
  float tx = diag * u1 + Q_c * cx + R_c * w1 * dot_wu;
  float ty = diag * u2 + Q_c * cy + R_c * w2 * dot_wu;
  float tz = diag * u3 + Q_c * cz + R_c * w3 * dot_wu;

  T[3]  = tx;
  T[7]  = ty;
  T[11] = tz;

  // --- Bottom row ---
  float inv_s = expf(-lambda);
  T[12] = 0.0f;
  T[13] = 0.0f;
  T[14] = 0.0f;
  T[15] = inv_s;
}

/** @copydoc Similarity3StateBatch::ApplyUpdate */
void Similarity3StateBatch::ApplyUpdate(const float* x,
                                                 const float* delta,
                                                 float* result,
                                                 bool invert_delta,
                                                 cudaStream_t stream) {
  size_t num_transforms = NumStateBlocks();

  dvector<Matrix<4>> delta_transforms_(num_transforms);
  dvector<float> tangents_(num_transforms * 7);

  auto tangents_ptr = reinterpret_cast<const float*>(tangents_.data());
  auto delta_transforms_ptr =
      reinterpret_cast<float*>(delta_transforms_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> tangents_device_ptr(tangents_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_transforms * 7,
               tangents_device_ptr);

  if (invert_delta) {
    thrust::transform(stream_policy, tangents_device_ptr,
                      tangents_device_ptr + tangents_.size(),
                      tangents_device_ptr, thrust::negate<float>());
  }

  constexpr int block_size = 256;
  int num_blocks = (num_transforms + block_size - 1) / block_size;
  ExpSim3Kernel<<<num_blocks, block_size, 0, stream>>>(
      tangents_ptr, delta_transforms_ptr, num_transforms);

  auto handle = cublas_handle_.GetHandle(stream);
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;
  constexpr int stride = 16;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_transforms_ptr, mat_size, stride, x, mat_size, stride, &beta,
      result, mat_size, stride, num_transforms));
}

void Similarity3StateBatch::Plus(const float* x, const float* delta,
                                          float* x_plus_delta,
                                          cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
