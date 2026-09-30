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
 * @brief Device helpers shared by the RANSAC kernels: deterministic warp and
 * block reductions, and item resolution for a slot.
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

/** @brief Max over a warp. Result in every lane. */
__device__ __forceinline__ float WarpMax(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
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

/**
 * @brief Locates item `e` of a slot: the view it comes from and the factor
 * index within that view. Returns false when the item contributes nothing
 * (masked out, or a sample outside every wave view).
 */
__device__ inline bool ResolveItem(const SlotItems &items, int slot, int e, int &view_out,
                                   int &factor_out) {
  if (e < items.sample_size) {
    const int u = items.samples[static_cast<size_t>(slot) * items.sample_size + e];
    for (int v = 0; v < items.num_views; ++v) {
      const BatchView &view = items.views[v];
      if (view.kind == kViewWave && u >= view.u_offset && u < view.u_offset + view.num_factors) {
        view_out = v;
        factor_out = u - view.u_offset;
        return true;
      }
    }
    return false;
  }
  int rest = e - items.sample_size;
  for (int v = 0; v < items.num_views; ++v) {
    const BatchView &view = items.views[v];
    if (view.kind == kViewWave) {
      continue;
    }
    if (rest < view.num_factors) {
      if (view.kind == kViewPerSlotMasked &&
          items.mask[static_cast<size_t>(slot) * items.mask_stride + view.u_offset + rest] == 0) {
        return false;
      }
      view_out = v;
      factor_out = rest;
      return true;
    }
    rest -= view.num_factors;
  }
  return false;
}

/** @brief Buffer copy (slot, or wave for wave views) holding a slot's rows of a view. */
__device__ __forceinline__ size_t BufferIndex(const SlotItems &items, const BatchView &view,
                                              int slot) {
  return static_cast<size_t>(view.kind == kViewWave ? slot / items.hyp_per_wave : slot);
}

/** @brief Block of the factor column `col` (blocks are contiguous and ordered). */
__device__ __forceinline__ int BlockOfColumn(const BatchView &view, int col) {
  int blk = 0;
  while (blk + 1 < view.nb && col >= view.block_col_off[blk + 1]) {
    ++blk;
  }
  return blk;
}

}  // namespace ransac_internal
}  // namespace cunls
