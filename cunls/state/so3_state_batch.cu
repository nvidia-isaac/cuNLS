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
#include "cunls/state/so3_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

__global__ void fused_so3_plus_kernel(const float* __restrict__ x,
                                      const float* __restrict__ delta,
                                      float* __restrict__ result, int n,
                                      bool negate_delta) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const float* d = delta + tid * 3;
  float w0 = d[0];
  float w1 = d[1];
  float w2 = d[2];
  if (negate_delta) {
    w0 = -w0;
    w1 = -w1;
    w2 = -w2;
  }
  float theta2 = w0 * w0 + w1 * w1 + w2 * w2;
  float theta = sqrtf(theta2);
  float A;
  float B;
  if (theta2 < 1e-6f) {
    A = 1.0f - theta2 / 6.0f;
    B = 0.5f - theta2 / 24.0f;
  } else {
    float st = sinf(theta);
    float ct = cosf(theta);
    A = st / theta;
    B = (1.0f - ct) / theta2;
  }
  float e00 = 1.0f - B * (w1 * w1 + w2 * w2);
  float e11 = 1.0f - B * (w0 * w0 + w2 * w2);
  float e22 = 1.0f - B * (w0 * w0 + w1 * w1);
  float e01 = B * w0 * w1 - A * w2;
  float e10 = B * w0 * w1 + A * w2;
  float e02 = B * w0 * w2 + A * w1;
  float e20 = B * w0 * w2 - A * w1;
  float e12 = B * w1 * w2 - A * w0;
  float e21 = B * w1 * w2 + A * w0;

  const float* X = x + tid * 9;
  float* R = result + tid * 9;
  float x0 = X[0];
  float x1 = X[1];
  float x2 = X[2];
  float x3 = X[3];
  float x4 = X[4];
  float x5 = X[5];
  float x6 = X[6];
  float x7 = X[7];
  float x8 = X[8];

  R[0] = x0 * e00 + x1 * e10 + x2 * e20;
  R[1] = x0 * e01 + x1 * e11 + x2 * e21;
  R[2] = x0 * e02 + x1 * e12 + x2 * e22;
  R[3] = x3 * e00 + x4 * e10 + x5 * e20;
  R[4] = x3 * e01 + x4 * e11 + x5 * e21;
  R[5] = x3 * e02 + x4 * e12 + x5 * e22;
  R[6] = x6 * e00 + x7 * e10 + x8 * e20;
  R[7] = x6 * e01 + x7 * e11 + x8 * e21;
  R[8] = x6 * e02 + x7 * e12 + x8 * e22;
}

void LaunchFusedSo3Plus(cudaStream_t stream, const float* x, const float* delta,
                        float* result, size_t num_blocks, bool negate_delta) {
  int n = static_cast<int>(num_blocks);
  if (n <= 0) {
    return;
  }
  int grid = (n + kBlockSize - 1) / kBlockSize;
  fused_so3_plus_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, delta, result, n, negate_delta);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

SO3StateBatch::SO3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      twists_(num_blocks * 3) {}

SO3StateBatch::SO3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      twists_(num_blocks * 3) {}

void SO3StateBatch::ApplyUpdate(const float* x, const float* delta,
                                float* result, bool invert_delta,
                                cudaStream_t stream) {
  LaunchFusedSo3Plus(stream, x, delta, result, NumStateBlocks(), invert_delta);
}

void SO3StateBatch::Plus(const float* x, const float* delta,
                         float* x_plus_delta, cudaStream_t stream) {
  LaunchFusedSo3Plus(stream, x, delta, x_plus_delta, NumStateBlocks(), false);
}

}  // namespace cunls
