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

#include "cunls/common/helper.h"
#include "cunls/state/similarity2_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

/** Same logic as ComputeVCoeffsSim2 in sim_lie_math.cu (inlined for fusion). */
__device__ void ComputeVCoeffsSim2Inline(float theta, float lambda, float &X,
                                         float &thetaY) {
  const float lambda2 = lambda * lambda;
  const float theta2 = theta * theta;

  if (fabsf(lambda) < 1e-4f) {
    float A, B;
    if (theta2 > 1e-6f) {
      const float sv = sinf(theta);
      const float cv = cosf(theta);
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
    const float sv = sinf(theta);
    const float cv = cosf(theta);
    A = sv / theta;
    B = (1.0f - cv) / theta2;
    C = (1.0f - A) / theta2;
  } else {
    A = 1.0f - theta2 / 6.0f;
    B = 0.5f - theta2 / 24.0f;
    C = 1.0f / 6.0f - theta2 / 120.0f;
  }

  const float alpha_coeff = lambda2 / d2;
  const float s_inv = expf(-lambda);
  X = alpha_coeff * (1.0f - s_inv) / lambda +
      (1.0f - alpha_coeff) * (A - lambda * B);
  const float Y = alpha_coeff * (s_inv - 1.0f + lambda) / lambda2 +
                  (1.0f - alpha_coeff) * (B - lambda * C);
  thetaY = theta * Y;
}

__global__ void sim2_apply_update_fused_kernel(const float *__restrict__ x,
                                               const float *__restrict__ delta,
                                               float *__restrict__ result,
                                               int n, bool negate_delta) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  const float *xi = delta + tid * 4;
  float ux = xi[0], uy = xi[1], theta = xi[2], lambda = xi[3];
  if (negate_delta) {
    ux = -ux;
    uy = -uy;
    theta = -theta;
    lambda = -lambda;
  }

  const float c = cosf(theta);
  const float s = sinf(theta);
  const float inv_scale = expf(-lambda);

  float X, thetaY;
  ComputeVCoeffsSim2Inline(theta, lambda, X, thetaY);
  const float tx = X * ux - thetaY * uy;
  const float ty = thetaY * ux + X * uy;

  const float e00 = c, e01 = -s, e02 = tx;
  const float e10 = s, e11 = c, e12 = ty;
  const float e20 = 0.0f, e21 = 0.0f, e22 = inv_scale;

  const float *xm = x + tid * 9;
  float *rm = result + tid * 9;

#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const float xr0 = xm[r * 3 + 0];
    const float xr1 = xm[r * 3 + 1];
    const float xr2 = xm[r * 3 + 2];
    rm[r * 3 + 0] = xr0 * e00 + xr1 * e10 + xr2 * e20;
    rm[r * 3 + 1] = xr0 * e01 + xr1 * e11 + xr2 * e21;
    rm[r * 3 + 2] = xr0 * e02 + xr1 * e12 + xr2 * e22;
  }
}

} // namespace

Similarity2StateBatch::Similarity2StateBatch(cuBLASHandle &cublas_handle,
                                             const float *device_ptr,
                                             size_t num_blocks)
    : Base(device_ptr, num_blocks), cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks), tangents_(num_blocks * 4) {}

Similarity2StateBatch::Similarity2StateBatch(
    cuBLASHandle &cublas_handle, const float *device_ptr, size_t num_blocks,
    const int *device_constant_state_ids, size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle), delta_transforms_(num_blocks),
      tangents_(num_blocks * 4) {}

void Similarity2StateBatch::ApplyUpdate(const float *x, const float *delta,
                                        float *result, bool invert_delta,
                                        cudaStream_t stream) {
  const int num_transforms = static_cast<int>(NumStateBlocks());
  const int grid = (num_transforms + kBlockSize - 1) / kBlockSize;
  sim2_apply_update_fused_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, delta, result, num_transforms, invert_delta);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  static_cast<void>(cublas_handle_);
}

void Similarity2StateBatch::Plus(const float *x, const float *delta,
                                 float *x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

} // namespace cunls
