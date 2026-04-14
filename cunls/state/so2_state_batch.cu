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
#include "cunls/state/so2_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

__global__ void fused_so2_plus_kernel(const float* __restrict__ x,
                                      const float* __restrict__ delta,
                                      float* __restrict__ result, int n,
                                      bool negate_delta) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }
  float theta = delta[tid];
  if (negate_delta) {
    theta = -theta;
  }
  float c = cosf(theta);
  float s = sinf(theta);
  const float* X = x + tid * 4;
  float* R = result + tid * 4;
  float x0 = X[0];
  float x1 = X[1];
  float x2 = X[2];
  float x3 = X[3];
  R[0] = x0 * c + x1 * s;
  R[1] = x0 * (-s) + x1 * c;
  R[2] = x2 * c + x3 * s;
  R[3] = x2 * (-s) + x3 * c;
}

void LaunchFusedSo2Plus(cudaStream_t stream, const float* x, const float* delta,
                        float* result, size_t num_blocks, bool negate_delta) {
  int n = static_cast<int>(num_blocks);
  if (n <= 0) {
    return;
  }
  int grid = (n + kBlockSize - 1) / kBlockSize;
  fused_so2_plus_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, delta, result, n, negate_delta);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

SO2StateBatch::SO2StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      angles_(num_blocks) {}

SO2StateBatch::SO2StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      angles_(num_blocks) {}

void SO2StateBatch::ApplyUpdate(const float* x, const float* delta,
                                float* result, bool invert_delta,
                                cudaStream_t stream) {
  LaunchFusedSo2Plus(stream, x, delta, result, NumStateBlocks(), invert_delta);
}

void SO2StateBatch::Plus(const float* x, const float* delta,
                         float* x_plus_delta, cudaStream_t stream) {
  LaunchFusedSo2Plus(stream, x, delta, x_plus_delta, NumStateBlocks(), false);
}

}  // namespace cunls
