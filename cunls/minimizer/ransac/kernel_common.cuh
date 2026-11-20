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

#pragma once

/**
 * @file kernel_common.cuh
 * @brief Device helpers shared by the RANSAC kernels: deterministic warp,
 * block and group reductions.
 */

#include <cuda_runtime.h>

#include <cstddef>

#include "cunls/minimizer/ransac/ransac_kernels.h"

namespace cunls {
namespace ransac_internal {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

/** @brief Grid size for `count` threads in blocks of `block`. */
inline int GridFor(size_t count, int block = 256) {
  return static_cast<int>((count + block - 1) / block);
}

/** @brief Sum over a warp; fixed butterfly order, so deterministic. Result in every lane. */
template <typename T>
__device__ __forceinline__ T WarpSum(T value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

/**
 * @brief Arg-min over a warp of (value, index); ties go to the lower index.
 * Result in every lane.
 */
__device__ __forceinline__ void WarpArgMin(float &value, int &index) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const float other_value = __shfl_xor_sync(kFullMask, value, offset);
    const int other_index = __shfl_xor_sync(kFullMask, index, offset);
    if (other_value < value || (other_value == value && other_index < index)) {
      value = other_value;
      index = other_index;
    }
  }
}

/**
 * @brief Sum over a thread block (blockDim a multiple of 32, <= 1024): warp
 * sums, then warp 0 sums the per-warp partials. Deterministic. Result valid in
 * every thread. `scratch` needs blockDim / 32 entries.
 */
template <typename T>
__device__ T BlockSum(T value, T *scratch) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;
  value = WarpSum(value);
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  T total = lane < num_warps ? scratch[lane] : T(0);
  total = WarpSum(total);
  __syncthreads();
  return total;
}

/** @brief Synchronizes a group: one warp (kGroup == kWarpSize) or the whole block. */
template <int kGroup>
__device__ __forceinline__ void GroupSync() {
  if (kGroup == kWarpSize) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

/**
 * @brief Sum over a group: WarpSum for a warp, BlockSum for a block
 * (deterministic). `scratch` as for BlockSum.
 */
template <int kGroup>
__device__ __forceinline__ float GroupSum(float value, float *scratch) {
  return kGroup == kWarpSize ? WarpSum(value) : BlockSum(value, scratch);
}

}  // namespace ransac_internal
}  // namespace cunls
