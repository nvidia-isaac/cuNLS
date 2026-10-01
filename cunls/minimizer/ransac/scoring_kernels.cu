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

/**
 * @file scoring_kernels.cu
 * @brief Hypothesis scoring (one thread per factor; large slots split across
 * blocks, partials added in split order) and top-M selection (one block,
 * warp-shuffle arg-min passes).
 */

#include <climits>
#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/minimizer/ransac/kernel_common.cuh"

namespace cunls {
namespace ransac_internal {

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxTop = 64;

/** True if the factor's Jacobian has a non-zero entry on a free (non-constant) block. */
__device__ bool IsInformative(const BatchView &view, int local_slot, int f) {
  const float *jac = view.jac + static_cast<size_t>(local_slot) * view.stride_jac +
                     static_cast<size_t>(f) * view.m * view.n;
  const int *local = view.local_col + static_cast<size_t>(f) * view.nb;
  for (int blk = 0; blk < view.nb; ++blk) {
    if (local[blk] < 0) {
      continue;
    }
    const int first = view.block_col_off[blk];
    for (int row = 0; row < view.m; ++row) {
      for (int c = first; c < first + view.block_size[blk]; ++c) {
        if (jac[row * view.n + c] != 0.f) {
          return true;
        }
      }
    }
  }
  return false;
}

/** Squared raw residual norm of sampled factor u; +inf when not usable. */
__device__ float SquaredError(const ScoreInputs &in, int local_slot, int u, const BatchView *&view,
                              int &f) {
  int v = 0;
  while (v + 1 < in.num_sampled && u >= in.sampled[v + 1].u_offset) {
    ++v;
  }
  view = &in.sampled[v];
  f = u - view->u_offset;
  const float *r = view->res + static_cast<size_t>(local_slot) * view->stride_res +
                   static_cast<size_t>(f) * view->m;
  float e = 0.f;
  for (int k = 0; k < view->m; ++k) {
    e = fmaf(r[k], r[k], e);
  }
  if (in.require_informative && !IsInformative(*view, local_slot, f)) {
    e = INFINITY;
  }
  return e;
}

/** Factors of one slot handled per block; more blocks per slot beyond this. */
constexpr int kScoreFactorsPerBlock = 4096;
constexpr int kMaxScoreSplits = 64;
constexpr int kScorePartials = 4;  ///< msac, msac bound, inlier count, always-on cost.

int ScoreSplits(int total_sampled) {
  const int splits = (total_sampled + kScoreFactorsPerBlock - 1) / kScoreFactorsPerBlock;
  return splits < 1 ? 1 : (splits > kMaxScoreSplits ? kMaxScoreSplits : splits);
}

/**
 * Block (split, local slot) scores a contiguous range of sampled factors and
 * writes its partial sums; split 0 also adds the always-on cost.
 */
__device__ __forceinline__ int GlobalSlot(int slot_offset, const int *slot_index, int local) {
  return slot_index != nullptr ? slot_index[local] : slot_offset + local;
}

__global__ void ScorePartialKernel(ScoreInputs in, int slot_offset, const int *slot_index,
                                   int num_slots, uint8_t *mask, float *partials) {
  __shared__ float scratch[kBlockThreads / kWarpSize];
  const int split = blockIdx.x;
  const int local = blockIdx.y;
  const int per_split = (in.total_sampled + gridDim.x - 1) / gridDim.x;
  const int end = min(in.total_sampled, (split + 1) * per_split);
  float msac = 0.f, bound = 0.f, count = 0.f, always_on = 0.f;
  for (int u = split * per_split + threadIdx.x; u < end; u += blockDim.x) {
    const BatchView *view = nullptr;
    int f = 0;
    const float e = SquaredError(in, local, u, view, f);
    const bool inlier = isfinite(e) && e <= view->tau_sq;
    msac += (isfinite(e) && e < view->tau_sq) ? e : view->tau_sq;
    bound += view->tau_sq;
    count += inlier ? 1.f : 0.f;
    if (mask != nullptr) {
      mask[static_cast<size_t>(local) * in.total_sampled + u] = inlier ? 1 : 0;
    }
  }
  for (int v = 0; split == 0 && v < in.num_always_on; ++v) {
    const BatchView &view = in.always_on[v];
    for (int f = threadIdx.x; f < view.num_factors; f += blockDim.x) {
      always_on += view.cost[static_cast<size_t>(GlobalSlot(slot_offset, slot_index, local)) *
                                 view.stride_cost +
                             f];
    }
  }
  const float sums[kScorePartials] = {BlockSum(msac, scratch), BlockSum(bound, scratch),
                                      BlockSum(count, scratch), BlockSum(always_on, scratch)};
  if (threadIdx.x < kScorePartials) {
    partials[(static_cast<size_t>(split) * num_slots + local) * kScorePartials + threadIdx.x] =
        sums[threadIdx.x];
  }
}

/** Adds the partials of each slot in split order and turns them into a score. */
__global__ void ScoreFinalizeKernel(ScoreInputs in, int slot_offset, const int *slot_index,
                                    int num_slots, int splits, const float *partials,
                                    const int *valid, float *score, int *inliers) {
  const int local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= num_slots) {
    return;
  }
  float sum[kScorePartials] = {};
  for (int s = 0; s < splits; ++s) {
    for (int k = 0; k < kScorePartials; ++k) {
      sum[k] += partials[(static_cast<size_t>(s) * num_slots + local) * kScorePartials + k];
    }
  }
  const int count = static_cast<int>(sum[2]);
  const int global = GlobalSlot(slot_offset, slot_index, local);
  float s = in.rule == kScoreInlierCount ? -static_cast<float>(count) + sum[0] / (sum[1] + 1.f)
                                         : sum[0] + (in.add_always_on ? 2.f * sum[3] : 0.f);
  if (!isfinite(s) || (valid != nullptr && valid[global] == 0)) {
    s = INFINITY;
  }
  score[global] = s;
  inliers[global] = count;
}

/** Block-wide arg-min over unchosen slots; ties to the lower index. Valid in every thread. */
__device__ int BlockArgMin(const float *score, int num_slots, const unsigned *chosen,
                           float *s_value, int *s_index, float &min_value) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  float best = INFINITY;
  int index = INT_MAX;
  for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
    if (chosen[i / 32] & (1u << (i % 32))) {
      continue;
    }
    const float s = isnan(score[i]) ? INFINITY : score[i];
    if (s < best || (s == best && i < index)) {
      best = s;
      index = i;
    }
  }
  WarpArgMin(best, index);
  if (lane == 0) {
    s_value[warp] = best;
    s_index[warp] = index;
  }
  __syncthreads();
  best = lane < blockDim.x / kWarpSize ? s_value[lane] : INFINITY;
  index = lane < blockDim.x / kWarpSize ? s_index[lane] : INT_MAX;
  WarpArgMin(best, index);
  __syncthreads();
  min_value = best;
  return index;
}

/** Lowest unchosen index (all remaining scores are +inf and no slot won the arg-min). */
__device__ int FirstUnchosen(const unsigned *chosen, int num_slots) {
  for (int i = 0; i < num_slots; ++i) {
    if (!(chosen[i / 32] & (1u << (i % 32)))) {
      return i;
    }
  }
  return 0;
}

__device__ void UpdateBest(DeviceStats *stats, float value, int slot, const int *inliers) {
  stats->last_min_score = value;
  stats->improved = value < stats->best_score ? 1 : 0;
  if (stats->improved) {
    stats->best_score = value;
    stats->best_inliers = inliers[slot];
    stats->best_slot = slot;
  }
}

__global__ void SelectKernel(const float *score, const int *inliers, const int *valid,
                             int num_slots, int top_m, int *top_indices, DeviceStats *stats) {
  extern __shared__ unsigned chosen[];
  __shared__ float s_value[kBlockThreads / kWarpSize];
  __shared__ int s_index[kBlockThreads / kWarpSize];
  __shared__ int s_count[kBlockThreads / kWarpSize];
  for (int i = threadIdx.x; i < (num_slots + 31) / 32; i += blockDim.x) {
    chosen[i] = 0u;
  }
  __syncthreads();
  const int m = min(top_m, min(num_slots, kMaxTop));
  for (int pass = 0; pass < m; ++pass) {
    float value = INFINITY;
    int pick = BlockArgMin(score, num_slots, chosen, s_value, s_index, value);
    if (threadIdx.x == 0) {
      pick = pick == INT_MAX ? FirstUnchosen(chosen, num_slots) : pick;
      chosen[pick / 32] |= 1u << (pick % 32);
      top_indices[pass] = pick;
      if (pass == 0) {
        UpdateBest(stats, value, pick, inliers);
      }
    }
    __syncthreads();
  }
  if (valid != nullptr) {
    int count = 0;
    for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
      count += valid[i] ? 1 : 0;
    }
    count = BlockSum(count, s_count);
    if (threadIdx.x == 0) {
      stats->valid_total += count;
    }
  }
}

__global__ void InitStatsKernel(DeviceStats *stats) {
  stats->best_score = INFINITY;
  stats->best_inliers = 0;
  stats->improved = 0;
  stats->best_slot = 0;
  stats->valid_total = 0;
  stats->last_min_score = INFINITY;
}

}  // namespace

size_t ScoreScratchFloats(int total_sampled, int num_slots) {
  return static_cast<size_t>(ScoreSplits(total_sampled)) * num_slots * kScorePartials;
}

void LaunchScore(cudaStream_t stream, const ScoreInputs &inputs, int num_slots, int slot_offset,
                 const int *valid, float *score, int *inliers, uint8_t *mask, float *scratch,
                 const int *slot_index) {
  if (num_slots <= 0) {
    return;
  }
  const int splits = ScoreSplits(inputs.total_sampled);
  ScorePartialKernel<<<dim3(splits, num_slots), kBlockThreads, 0, stream>>>(
      inputs, slot_offset, slot_index, num_slots, mask, scratch);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  ScoreFinalizeKernel<<<GridFor(num_slots, kBlockThreads), kBlockThreads, 0, stream>>>(
      inputs, slot_offset, slot_index, num_slots, splits, scratch, valid, score, inliers);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchSelect(cudaStream_t stream, const float *score, const int *inliers, const int *valid,
                  int num_slots, int top_m, int *top_indices, DeviceStats *stats) {
  const size_t smem = static_cast<size_t>((num_slots + 31) / 32) * sizeof(unsigned);
  SelectKernel<<<1, kBlockThreads, smem, stream>>>(score, inliers, valid, num_slots, top_m,
                                                   top_indices, stats);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchInitStats(cudaStream_t stream, DeviceStats *stats) {
  InitStatsKernel<<<1, 1, 0, stream>>>(stats);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace ransac_internal
}  // namespace cunls
