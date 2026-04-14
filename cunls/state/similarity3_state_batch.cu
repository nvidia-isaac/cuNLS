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

#include "cunls/common/helper.h"
#include "cunls/state/similarity3_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

/** Fused Sim(3) Exp + row-major multiply: result = x * Exp(xi), layout matches
 *  exp_sim3_kernel + former strided batched GEMM. */
__global__ void sim3_apply_update_fused_kernel(const float* __restrict__ x,
                                               const float* __restrict__ delta,
                                               float* __restrict__ result, int n,
                                               bool negate_delta) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  const float* xi = delta + tid * 7;
  float w1 = xi[0], w2 = xi[1], w3 = xi[2];
  float u1 = xi[3], u2 = xi[4], u3 = xi[5];
  float lambda = xi[6];
  if (negate_delta) {
    w1 = -w1;
    w2 = -w2;
    w3 = -w3;
    u1 = -u1;
    u2 = -u2;
    u3 = -u3;
    lambda = -lambda;
  }

  const float theta2 = w1 * w1 + w2 * w2 + w3 * w3;
  const float theta = sqrtf(theta2);

  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    const float st = sinf(theta), ct = cosf(theta);
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

  const float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    const float e = expf(-lambda);
    P_c = (1.0f - e) / lambda;
    const float alpha = lambda2 / (lambda2 + theta2);
    const float beta = (e - 1.0f + lambda) / lambda2;
    const float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    const float one_m_alpha = 1.0f - alpha;
    Q_c = alpha * beta + one_m_alpha * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_alpha * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  const float ct_val = 1.0f - A2 * theta2;
  const float e00 = ct_val + A2 * w1 * w1;
  const float e01 = A2 * w1 * w2 - A1 * w3;
  const float e02 = A2 * w1 * w3 + A1 * w2;
  const float e10 = A2 * w2 * w1 + A1 * w3;
  const float e11 = ct_val + A2 * w2 * w2;
  const float e12 = A2 * w2 * w3 - A1 * w1;
  const float e20 = A2 * w3 * w1 - A1 * w2;
  const float e21 = A2 * w3 * w2 + A1 * w1;
  const float e22 = ct_val + A2 * w3 * w3;

  const float diag = P_c - R_c * theta2;
  const float dot_wu = w1 * u1 + w2 * u2 + w3 * u3;
  const float cx = w2 * u3 - w3 * u2;
  const float cy = w3 * u1 - w1 * u3;
  const float cz = w1 * u2 - w2 * u1;
  const float e03 = diag * u1 + Q_c * cx + R_c * w1 * dot_wu;
  const float e13 = diag * u2 + Q_c * cy + R_c * w2 * dot_wu;
  const float e23 = diag * u3 + Q_c * cz + R_c * w3 * dot_wu;

  const float e33 = expf(-lambda);

  const float* xm = x + tid * 16;
  float* rm = result + tid * 16;

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float xr0 = xm[r * 4 + 0];
    const float xr1 = xm[r * 4 + 1];
    const float xr2 = xm[r * 4 + 2];
    const float xr3 = xm[r * 4 + 3];
    rm[r * 4 + 0] = xr0 * e00 + xr1 * e10 + xr2 * e20;
    rm[r * 4 + 1] = xr0 * e01 + xr1 * e11 + xr2 * e21;
    rm[r * 4 + 2] = xr0 * e02 + xr1 * e12 + xr2 * e22;
    rm[r * 4 + 3] = xr0 * e03 + xr1 * e13 + xr2 * e23 + xr3 * e33;
  }
}

}  // namespace

Similarity3StateBatch::Similarity3StateBatch(cuBLASHandle& cublas_handle,
                                           const float* device_ptr,
                                           size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      tangents_(num_blocks * 7) {}

Similarity3StateBatch::Similarity3StateBatch(
    cuBLASHandle& cublas_handle, const float* device_ptr, size_t num_blocks,
    const int* device_constant_state_ids, size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      tangents_(num_blocks * 7) {}

void Similarity3StateBatch::ApplyUpdate(const float* x, const float* delta,
                                        float* result, bool invert_delta,
                                        cudaStream_t stream) {
  const int num_transforms = static_cast<int>(NumStateBlocks());
  const int grid = (num_transforms + kBlockSize - 1) / kBlockSize;
  sim3_apply_update_fused_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, delta, result, num_transforms, invert_delta);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  static_cast<void>(cublas_handle_);
}

void Similarity3StateBatch::Plus(const float* x, const float* delta,
                                 float* x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
