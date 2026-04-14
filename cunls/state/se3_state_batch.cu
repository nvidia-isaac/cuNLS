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
#include "cunls/state/se3_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

/**
 * Fused SE(3) Plus: out[i] = x[i] * Exp(delta[i]) with 4x4 matrices in row-major
 * (16 floats per matrix). Exp uses the SO(3) left Jacobian for translation and
 * Rodrigues for rotation, matching the former ComputeExpSE3 + batched GEMM path.
 */
__global__ void se3_plus_fused_kernel(const float* __restrict__ x,
                                      const float* __restrict__ delta,
                                      float* __restrict__ out, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  const float* d = delta + tid * 6;
  float w0 = d[0], w1 = d[1], w2 = d[2];
  float v0 = d[3], v1 = d[4], v2 = d[5];

  const float theta2 = w0 * w0 + w1 * w1 + w2 * w2;
  const float theta = sqrtf(theta2);
  float A, B, C;
  if (theta2 < 1e-6f) {
    A = 1.0f - theta2 / 6.0f;
    B = 0.5f - theta2 / 24.0f;
    C = 1.0f / 6.0f - theta2 / 120.0f;
  } else {
    const float st = sinf(theta), ct = cosf(theta);
    A = st / theta;
    B = (1.0f - ct) / theta2;
    C = (1.0f - A) / theta2;
  }

  const float j00 = 1.0f - C * (w1 * w1 + w2 * w2);
  const float j01 = C * w0 * w1 - B * w2;
  const float j02 = C * w0 * w2 + B * w1;
  const float j10 = C * w0 * w1 + B * w2;
  const float j11 = 1.0f - C * (w0 * w0 + w2 * w2);
  const float j12 = C * w1 * w2 - B * w0;
  const float j20 = C * w0 * w2 - B * w1;
  const float j21 = C * w1 * w2 + B * w0;
  const float j22 = 1.0f - C * (w0 * w0 + w1 * w1);

  const float tx = j00 * v0 + j01 * v1 + j02 * v2;
  const float ty = j10 * v0 + j11 * v1 + j12 * v2;
  const float tz = j20 * v0 + j21 * v1 + j22 * v2;

  const float r00 = 1.0f - B * (w1 * w1 + w2 * w2);
  const float r01 = B * w0 * w1 - A * w2;
  const float r02 = B * w0 * w2 + A * w1;
  const float r10 = B * w0 * w1 + A * w2;
  const float r11 = 1.0f - B * (w0 * w0 + w2 * w2);
  const float r12 = B * w1 * w2 - A * w0;
  const float r20 = B * w0 * w2 - A * w1;
  const float r21 = B * w1 * w2 + A * w0;
  const float r22 = 1.0f - B * (w0 * w0 + w1 * w1);

  const float* xm = x + tid * 16;
  float* om = out + tid * 16;

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float xr0 = xm[r * 4 + 0];
    const float xr1 = xm[r * 4 + 1];
    const float xr2 = xm[r * 4 + 2];
    const float xr3 = xm[r * 4 + 3];
    om[r * 4 + 0] = xr0 * r00 + xr1 * r10 + xr2 * r20;
    om[r * 4 + 1] = xr0 * r01 + xr1 * r11 + xr2 * r21;
    om[r * 4 + 2] = xr0 * r02 + xr1 * r12 + xr2 * r22;
    om[r * 4 + 3] = xr0 * tx + xr1 * ty + xr2 * tz + xr3;
  }
}

}  // namespace

SE3StateBatch::SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 6) {}

SE3StateBatch::SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 6) {}

void SE3StateBatch::Plus(const float* x, const float* delta, float* x_plus_delta,
                         cudaStream_t stream) {
  const int num_transforms = static_cast<int>(NumStateBlocks());
  const int grid = (num_transforms + kBlockSize - 1) / kBlockSize;
  se3_plus_fused_kernel<<<grid, kBlockSize, 0, stream>>>(x, delta, x_plus_delta,
                                                         num_transforms);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  static_cast<void>(cublas_handle_);
}

}  // namespace cunls
