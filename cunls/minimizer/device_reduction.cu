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
#include "cunls/minimizer/device_reduction.h"

namespace cunls {

constexpr int kReduceBlockSize = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

// ---------------------------------------------------------------------------
// Pass 1: each block reduces a tile of the input into a single partial sum.
// Three modes are supported via template parameter:
//   Mode=0 : sum          input[i]
//   Mode=1 : dot-product  a[i]*b[i]
//   Mode=2 : weighted dot a[i]*w[i]*b[i]
// ---------------------------------------------------------------------------
template <int Mode>
__global__ void reduce_pass1_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    const float* __restrict__ w, int n,
                                    float* __restrict__ partials) {
  __shared__ float smem[kReduceBlockSize / kWarpSize];

  int tid = threadIdx.x;
  int gid = blockIdx.x * kReduceBlockSize + tid;
  int stride = gridDim.x * kReduceBlockSize;

  float sum = 0.0f;
  for (int i = gid; i < n; i += stride) {
    if constexpr (Mode == 0) {
      sum += a[i];
    } else if constexpr (Mode == 1) {
      sum += a[i] * b[i];
    } else {
      sum += a[i] * w[i] * b[i];
    }
  }

  sum = warp_reduce_sum(sum);

  int lane = tid & (kWarpSize - 1);
  int warp_id = tid / kWarpSize;
  if (lane == 0) smem[warp_id] = sum;
  __syncthreads();

  constexpr int num_warps = kReduceBlockSize / kWarpSize;
  if (tid < num_warps) {
    sum = smem[tid];
    sum = warp_reduce_sum(sum);
    if (tid == 0) partials[blockIdx.x] = sum;
  }
}

// ---------------------------------------------------------------------------
// Pass 2: reduce the partial sums from pass 1 into a single scalar.
// Launched with 1 block.
// ---------------------------------------------------------------------------
__global__ void reduce_pass2_kernel(const float* __restrict__ partials,
                                    int num_partials,
                                    float* __restrict__ output) {
  __shared__ float smem[kReduceBlockSize / kWarpSize];

  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < num_partials; i += kReduceBlockSize)
    sum += partials[i];

  sum = warp_reduce_sum(sum);

  int lane = tid & (kWarpSize - 1);
  int warp_id = tid / kWarpSize;
  if (lane == 0) smem[warp_id] = sum;
  __syncthreads();

  constexpr int num_warps = kReduceBlockSize / kWarpSize;
  if (tid < num_warps) {
    sum = smem[tid];
    sum = warp_reduce_sum(sum);
    if (tid == 0) *output = sum;
  }
}

static int ComputeGridSize(size_t n) {
  int blocks = static_cast<int>((n + kReduceBlockSize - 1) / kReduceBlockSize);
  constexpr int kMaxBlocks = 1024;
  return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

size_t ReducePartialCount(size_t n) {
  return static_cast<size_t>(ComputeGridSize(n));
}

void ReduceSumToDevice(cudaStream_t stream, const float* input, size_t n,
                       float* d_output, float* d_partials) {
  if (n == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
    return;
  }
  int grid = ComputeGridSize(n);
  reduce_pass1_kernel<0>
      <<<grid, kReduceBlockSize, 0, stream>>>(input, nullptr, nullptr,
                                              static_cast<int>(n), d_partials);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  reduce_pass2_kernel<<<1, kReduceBlockSize, 0, stream>>>(d_partials, grid,
                                                          d_output);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void DotProductToDevice(cudaStream_t stream, const float* a, const float* b,
                        size_t n, float* d_output, float* d_partials) {
  if (n == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
    return;
  }
  int grid = ComputeGridSize(n);
  reduce_pass1_kernel<1>
      <<<grid, kReduceBlockSize, 0, stream>>>(a, b, nullptr,
                                              static_cast<int>(n), d_partials);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  reduce_pass2_kernel<<<1, kReduceBlockSize, 0, stream>>>(d_partials, grid,
                                                          d_output);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void WeightedDotProductToDevice(cudaStream_t stream, const float* a,
                                const float* w, const float* b, size_t n,
                                float* d_output, float* d_partials) {
  if (n == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(d_output, 0, sizeof(float), stream));
    return;
  }
  int grid = ComputeGridSize(n);
  reduce_pass1_kernel<2>
      <<<grid, kReduceBlockSize, 0, stream>>>(a, w, b, static_cast<int>(n),
                                              d_partials);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  reduce_pass2_kernel<<<1, kReduceBlockSize, 0, stream>>>(d_partials, grid,
                                                          d_output);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
