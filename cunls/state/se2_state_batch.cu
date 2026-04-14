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
#include "cunls/state/se2_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

__global__ void fused_se2_plus_kernel(const float *__restrict__ x,
                                      const float *__restrict__ delta,
                                      float *__restrict__ result, int n,
                                      bool negate_delta) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  const float *d = delta + tid * 3;
  float vx = d[0];
  float vy = d[1];
  float w = d[2];
  if (negate_delta) {
    vx = -vx;
    vy = -vy;
    w = -w;
  }
  float c = cosf(w);
  float s = sinf(w);
  float tx;
  float ty;
  if (fabsf(w) < 1e-3f) {
    tx = vx;
    ty = vy;
  } else {
    float sw = s / w;
    float cw = (1.0f - c) / w;
    tx = vx * sw - vy * cw;
    ty = vx * cw + vy * sw;
  }

  const float *X = x + tid * 9;
  float *R = result + tid * 9;
  float x0 = X[0];
  float x1 = X[1];
  float x2 = X[2];
  float x3 = X[3];
  float x4 = X[4];
  float x5 = X[5];
  float x6 = X[6];
  float x7 = X[7];
  float x8 = X[8];

  R[0] = x0 * c + x1 * s;
  R[1] = x0 * (-s) + x1 * c;
  R[2] = x0 * tx + x1 * ty + x2;
  R[3] = x3 * c + x4 * s;
  R[4] = x3 * (-s) + x4 * c;
  R[5] = x3 * tx + x4 * ty + x5;
  R[6] = x6 * c + x7 * s;
  R[7] = x6 * (-s) + x7 * c;
  R[8] = x6 * tx + x7 * ty + x8;
}

void LaunchFusedSe2Plus(cudaStream_t stream, const float *x, const float *delta,
                        float *result, size_t num_blocks, bool negate_delta) {
  int n = static_cast<int>(num_blocks);
  if (n <= 0) {
    return;
  }
  int grid = (n + kBlockSize - 1) / kBlockSize;
  fused_se2_plus_kernel<<<grid, kBlockSize, 0, stream>>>(x, delta, result, n,
                                                         negate_delta);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace

SE2StateBatch::SE2StateBatch(cuBLASHandle &cublas_handle,
                             const float *device_ptr, size_t num_blocks)
    : Base(device_ptr, num_blocks), cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks), tangents_(num_blocks * 3) {}

SE2StateBatch::SE2StateBatch(cuBLASHandle &cublas_handle,
                             const float *device_ptr, size_t num_blocks,
                             const int *device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle), delta_transforms_(num_blocks),
      tangents_(num_blocks * 3) {}

void SE2StateBatch::ApplyUpdate(const float *x, const float *delta,
                                float *result, bool invert_delta,
                                cudaStream_t stream) {
  LaunchFusedSe2Plus(stream, x, delta, result, NumStateBlocks(), invert_delta);
}

void SE2StateBatch::Plus(const float *x, const float *delta,
                         float *x_plus_delta, cudaStream_t stream) {
  LaunchFusedSe2Plus(stream, x, delta, x_plus_delta, NumStateBlocks(), false);
}

} // namespace cunls
