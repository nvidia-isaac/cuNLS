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
 * @file normal_equations_kernels.cu
 * @brief Per-slot normal equations and cost sums.
 *
 * A group of threads (a warp, or a whole block) processes a slot's items in
 * tiles. Each tile's Jacobian rows are gathered into the slot's local columns
 * in shared memory (a gather: no atomics). H = J^T J and g = -J^T r are then
 * accumulated in 4x4 register tiles: each thread owns one H tile of the upper
 * triangle (or one 4-entry slice of g) and a fixed subset of the rows.
 *
 * Slots with many rows are split across several blocks ("splits"); partial
 * sums go to a scratch buffer and a second kernel adds them in split order.
 * Every partition is fixed by the problem size, so results are bitwise
 * reproducible.
 */

#include <algorithm>

#include "cunls/common/helper.h"
#include "cunls/minimizer/ransac/kernel_common.cuh"

namespace cunls {
namespace ransac_internal {

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWarpGroupWords = 1024;  ///< Shared words per warp group.
constexpr int kMaxTileRows = 256;      ///< Rows per tile: keeps enough blocks per slot.
constexpr int kMaxSplits = 64;         ///< Blocks per slot (bounds the scratch buffer).
constexpr int kMaxSharedViews = 16;    ///< Views copied to shared memory.

__host__ __device__ inline int Padded(int dim) { return (dim + 3) & ~3; }
__host__ __device__ inline int TileCount(int dim) { return Padded(dim) / 4; }

/** Work units: upper-triangle 4x4 H tiles, then 4-entry g slices. */
__host__ __device__ inline int WorkUnits(int dim) {
  const int t = TileCount(dim);
  return t * (t + 1) / 2 + t;
}

/** Items staged per tile: bounded by the shared budget and by kMaxTileRows. */
__host__ __device__ inline int TileItems(int words, int dim, int m_max) {
  const int by_words = words / NormalEquationsItemWords(m_max, dim);
  const int by_rows = kMaxTileRows / m_max;
  const int items = by_words < by_rows ? by_words : by_rows;
  return items > 0 ? items : 1;
}

/** Launch geometry shared by the launcher, the scratch-size query and the kernel. */
struct Geometry {
  bool warp = false;
  int group_words = 0;
  int tile_items = 0;
  int splits = 1;       ///< Blocks per slot (block groups only).
  int split_items = 0;  ///< Items per block: a multiple of tile_items.
};

Geometry MakeGeometry(const SlotItems &items, int dim, SlotGroup requested) {
  Geometry g;
  const bool fits_warp =
      WorkUnits(dim) <= kWarpSize && NormalEquationsItemWords(items.m_max, dim) <= kWarpGroupWords;
  const bool few_rows = items.items_per_slot * items.m_max <= 256;
  g.warp =
      fits_warp && (requested == SlotGroup::kWarp || (requested == SlotGroup::kAuto && few_rows));
  g.group_words = g.warp ? kWarpGroupWords : kBlockGroupWords;
  g.tile_items = TileItems(g.group_words, dim, std::max(items.m_max, 1));
  const int tiles = std::max(1, (items.items_per_slot + g.tile_items - 1) / g.tile_items);
  g.splits = g.warp ? 1 : std::min(tiles, kMaxSplits);
  g.split_items = ((tiles + g.splits - 1) / g.splits) * g.tile_items;
  return g;
}

template <int kGroup>
__device__ __forceinline__ void GroupSync() {
  if (kGroup == kWarpSize) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

template <int kGroup>
__device__ __forceinline__ float GroupSum(float value, float *scratch) {
  return kGroup == kWarpSize ? WarpSum(value) : BlockSum(value, scratch);
}

/** Shared-memory layout of one group's tile. */
struct Stage {
  float *jac;       ///< rows x padded dim, local columns (zero padded).
  float *res;       ///< rows.
  int *item_view;   ///< items; -1 = empty.
  int *item_row;    ///< items: row in the slot's part of the view's buffers.
  int *item_local;  ///< items x kMaxBlocksPerFactor local columns (-1 = constant).
  int ld;           ///< Padded dim.
};

__device__ inline Stage MapStage(float *base, int items, int m_max, int dim) {
  Stage s;
  s.ld = Padded(dim);
  s.jac = base;
  s.res = s.jac + items * m_max * s.ld;
  s.item_view = reinterpret_cast<int *>(s.res + items * m_max);
  s.item_row = s.item_view + items;
  s.item_local = s.item_row + items;
  return s;
}

/**
 * Resolves the tile's items and caches their local columns; thread t also
 * sums the costs of items t, t + G, ...
 */
template <int kGroup>
__device__ void ResolveTile(const SlotItems &items, int slot, int first, int last, int count,
                            const Stage &st, int t, float &cost_acc) {
  for (int k = t; k < count; k += kGroup) {
    const int e = first + k;
    ItemRef ref{-1, 0, 0};
    if (e >= last || !ResolveItem(items, slot, e, ref)) {
      ref.view = -1;
    } else {
      const BatchView &view = items.views[ref.view];
      const int *local = view.local_col + static_cast<size_t>(ref.factor) * view.nb;
      for (int blk = 0; blk < view.nb; ++blk) {
        st.item_local[k * kMaxBlocksPerFactor + blk] = local[blk];
      }
      if (view.cost != nullptr) {
        cost_acc += view.cost[static_cast<size_t>(slot) * view.stride_cost + ref.row];
      }
    }
    st.item_view[k] = ref.view;
    st.item_row[k] = ref.row;
  }
}

/**
 * Gathers `count` items' Jacobian rows into local columns. Every (row, local
 * column) entry is computed independently by one thread as the sum of the
 * factor columns that map to it (more than one only when a factor references
 * the same state twice), so loads are independent and no atomics are needed.
 */
template <int kGroup>
__device__ void GatherTile(const SlotItems &items, int slot, int count, const Stage &st, int t) {
  const int m_max = items.m_max;
  for (int idx = t; idx < count * m_max * st.ld; idx += kGroup) {
    const int r = idx / st.ld;
    const int a = idx - r * st.ld;
    const int k = r / m_max;
    const int row = r - k * m_max;
    const int v = st.item_view[k];
    const BatchView *view = v >= 0 ? &items.views[v] : nullptr;
    const bool live = view != nullptr && row < view->m;
    float value = 0.f;
    if (live) {
      const size_t item_row = st.item_row[k];
      const size_t slot_index = static_cast<size_t>(slot);
      const float *jrow =
          view->jac + slot_index * view->stride_jac + (item_row * view->m + row) * view->n;
      const int *local = st.item_local + k * kMaxBlocksPerFactor;
      for (int blk = 0; blk < view->nb; ++blk) {
        const int offset = a - local[blk];
        if (local[blk] >= 0 && offset >= 0 && offset < view->block_size[blk]) {
          value += jrow[view->block_col_off[blk] + offset];
        }
      }
      if (a == 0) {
        st.res[r] = view->res[slot_index * view->stride_res + item_row * view->m + row];
      }
    } else if (a == 0) {
      st.res[r] = 0.f;
    }
    st.jac[idx] = value;
  }
}

/** A thread's work unit: an upper H tile (ti <= tj) or a g slice (tj = -1). */
struct Unit {
  int ti = 0;
  int tj = -1;
};

__device__ inline Unit UnitAt(int u, int dim) {
  const int tiles = TileCount(dim);
  const int upper = tiles * (tiles + 1) / 2;
  Unit unit;
  if (u >= upper) {
    unit.ti = u - upper;
    return unit;
  }
  while (u >= tiles - unit.ti) {
    u -= tiles - unit.ti;
    ++unit.ti;
  }
  unit.tj = unit.ti + u;
  return unit;
}

/** acc += J[:, ti]^T J[:, tj] (or -J[:, ti]^T r) over rows row_group, row_group + stride, ... */
__device__ void AccumulateTile(const Stage &st, int rows, const Unit &unit, int row_group,
                               int stride, float acc[16]) {
  for (int r = row_group; r < rows; r += stride) {
    const float4 a = *reinterpret_cast<const float4 *>(st.jac + r * st.ld + 4 * unit.ti);
    const float av[4] = {a.x, a.y, a.z, a.w};
    if (unit.tj < 0) {
      const float nr = -st.res[r];
      for (int i = 0; i < 4; ++i) acc[i] = fmaf(av[i], nr, acc[i]);
      continue;
    }
    const float4 b = *reinterpret_cast<const float4 *>(st.jac + r * st.ld + 4 * unit.tj);
    const float bv[4] = {b.x, b.y, b.z, b.w};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) acc[i * 4 + j] = fmaf(av[i], bv[j], acc[i * 4 + j]);
    }
  }
}

/** Where a (split, slot) writes H, g and cost. */
struct Outputs {
  float *h;
  float *g;
  float *cost;
  size_t stride_h;
  size_t stride_g;
  size_t stride_cost;
};

/** Writes element i (row-major 4x4, or 4-slice for g) of a unit's sum. */
__device__ void WriteElement(const Outputs &out, size_t index, int dim, const Unit &unit, int i,
                             float value) {
  if (unit.tj < 0) {
    const int a = 4 * unit.ti + i;
    if (i < 4 && a < dim) {
      out.g[index * out.stride_g + a] = value;
    }
    return;
  }
  const int a = 4 * unit.ti + i / 4;
  const int b = 4 * unit.tj + i % 4;
  if (a < dim && b < dim) {
    float *h = out.h + index * out.stride_h;
    h[a * dim + b] = value;
    h[b * dim + a] = value;
  }
}

/**
 * Adds the row-group partials of every (unit, element) in row-group order
 * (parallel over elements) and writes H / g for output `index`.
 */
template <int kGroup>
__device__ void ReduceAndWrite(float *partial, const float acc[16], bool owns, int row_group,
                               int row_groups, int units, int t, int dim, size_t index,
                               const Outputs &out) {
  if (owns) {
    for (int i = 0; i < 16; ++i) partial[(row_group * units + t % units) * 16 + i] = acc[i];
  }
  GroupSync<kGroup>();
  for (int v = t; v < units * 16; v += kGroup) {
    float sum = 0.f;
    for (int rg = 0; rg < row_groups; ++rg) {
      sum += partial[rg * units * 16 + v];
    }
    WriteElement(out, index, dim, UnitAt(v / 16, dim), v % 16, sum);
  }
}

/** Copies the views to shared memory when they fit; returns the items to use. */
__device__ SlotItems WithSharedViews(const SlotItems &items, BatchView *shared) {
  SlotItems local = items;
  if (items.num_views <= kMaxSharedViews) {
    for (int v = threadIdx.x; v < items.num_views; v += blockDim.x) shared[v] = items.views[v];
    local.views = shared;
  }
  __syncthreads();
  return local;
}

template <int kGroup>
__global__ void NormalEquationsKernel(SlotItems global_items, int num_slots, int dim,
                                      int group_words, int tile_items, int split_items,
                                      Outputs out) {
  extern __shared__ __align__(16) float smem[];
  __shared__ float scratch[kBlockThreads / kWarpSize];
  __shared__ BatchView shared_views[kMaxSharedViews];
  const SlotItems items = WithSharedViews(global_items, shared_views);
  const int group = threadIdx.x / kGroup;
  const int t = threadIdx.x % kGroup;
  const int slot = kGroup == kWarpSize ? blockIdx.x * (blockDim.x / kWarpSize) + group
                                       : static_cast<int>(blockIdx.y);
  const int split = kGroup == kWarpSize ? 0 : static_cast<int>(blockIdx.x);
  if (slot >= num_slots) {
    return;  // only warp groups can be out of range; they never sync with others
  }
  const int begin = split * split_items;
  const int end = min(items.items_per_slot, begin + split_items);
  float *base = smem + group * group_words;
  const Stage st = MapStage(base, tile_items, items.m_max, dim);

  const int units = WorkUnits(dim);
  const int row_groups = max(1, kGroup / units);
  const int row_group = t / units;
  const bool owns = row_group < row_groups;
  const Unit unit = UnitAt(owns ? t % units : 0, dim);
  float acc[16] = {};
  float cost_acc = 0.f;

  for (int first = begin; first < end; first += tile_items) {
    const int count = min(tile_items, end - first);
    ResolveTile<kGroup>(items, slot, first, end, count, st, t, cost_acc);
    GroupSync<kGroup>();
    GatherTile<kGroup>(items, slot, count, st, t);
    GroupSync<kGroup>();
    if (owns) {
      AccumulateTile(st, count * items.m_max, unit, row_group, row_groups, acc);
    }
    GroupSync<kGroup>();
  }

  ReduceAndWrite<kGroup>(base, acc, owns, row_group, row_groups, units, t, dim,
                         static_cast<size_t>(split) * num_slots + slot, out);
  const float total_cost = GroupSum<kGroup>(cost_acc, scratch);
  if (t == 0 && out.cost != nullptr) {
    out.cost[(static_cast<size_t>(split) * num_slots + slot) * out.stride_cost] = total_cost;
  }
}

/** Adds the split partials of every (slot, entry) in split order. */
__global__ void SumSplitsKernel(const float *partials, int splits, int num_slots, int entries,
                                int dim, float *hessian, float *gradient, float *cost) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(num_slots) * entries) {
    return;
  }
  const size_t slot = idx / entries;
  const int e = static_cast<int>(idx % entries);
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) {
    sum += partials[(static_cast<size_t>(s) * num_slots + slot) * entries + e];
  }
  if (e < dim * dim) {
    hessian[slot * dim * dim + e] = sum;
  } else if (e < dim * dim + dim) {
    gradient[slot * dim + e - dim * dim] = sum;
  } else if (cost != nullptr) {
    cost[slot] = sum;
  }
}

constexpr int kCostItemsPerBlock = 4096;

/** Blocks per slot of the cost kernel (block groups only). */
int CostSplits(const SlotItems &items) {
  const int splits = (items.items_per_slot + kCostItemsPerBlock - 1) / kCostItemsPerBlock;
  return std::max(1, std::min(splits, kMaxSplits));
}

/**
 * Sums item costs over [split range) of one slot; writes the slot's cost
 * directly (one split) or a partial (several splits).
 */
template <int kGroup>
__global__ void SlotCostKernel(SlotItems items, int num_slots, int splits, float *out) {
  __shared__ float scratch[kBlockThreads / kWarpSize];
  const int group = threadIdx.x / kGroup;
  const int t = threadIdx.x % kGroup;
  const int slot = kGroup == kWarpSize ? blockIdx.x * (blockDim.x / kGroup) + group
                                       : static_cast<int>(blockIdx.y);
  const int split = kGroup == kWarpSize ? 0 : static_cast<int>(blockIdx.x);
  if (slot >= num_slots) {
    return;  // only warp groups can be out of range
  }
  const int per_split = (items.items_per_slot + splits - 1) / splits;
  const int end = min(items.items_per_slot, (split + 1) * per_split);
  float sum = 0.f;
  for (int e = split * per_split + t; e < end; e += kGroup) {
    ItemRef ref;
    if (ResolveItem(items, slot, e, ref)) {
      sum += items.views[ref.view]
                 .cost[static_cast<size_t>(slot) * items.views[ref.view].stride_cost + ref.row];
    }
  }
  const float total = GroupSum<kGroup>(sum, scratch);
  if (t == 0) {
    out[static_cast<size_t>(split) * num_slots + slot] = total;
  }
}

/** cost[slot] = sum of the split partials in split order. */
__global__ void SumCostSplitsKernel(const float *partials, int splits, int num_slots, float *cost) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots) {
    return;
  }
  float sum = 0.f;
  for (int s = 0; s < splits; ++s) {
    sum += partials[static_cast<size_t>(s) * num_slots + slot];
  }
  cost[slot] = sum;
}

}  // namespace

size_t NormalEquationsScratchFloats(const SlotItems &items, int num_slots, int dim,
                                    SlotGroup group) {
  const Geometry g = MakeGeometry(items, dim, group);
  if (g.splits <= 1) {
    return 0;
  }
  return static_cast<size_t>(g.splits) * num_slots * (dim * dim + dim + 1);
}

void LaunchNormalEquations(cudaStream_t stream, const SlotItems &items, int num_slots, int dim,
                           float *hessian, float *gradient, float *cost, float *scratch,
                           SlotGroup group) {
  if (num_slots <= 0) {
    return;
  }
  const Geometry g = MakeGeometry(items, dim, group);
  const size_t entries = static_cast<size_t>(dim) * dim + dim + 1;
  Outputs out{hessian, gradient, cost, static_cast<size_t>(dim) * dim, static_cast<size_t>(dim), 1};
  if (g.splits > 1) {
    out =
        Outputs{scratch, scratch + dim * dim, scratch + dim * dim + dim, entries, entries, entries};
  }
  if (g.warp) {
    constexpr int kGroups = kBlockThreads / kWarpSize;
    const size_t smem = static_cast<size_t>(kGroups) * g.group_words * sizeof(float);
    NormalEquationsKernel<kWarpSize><<<GridFor(num_slots, kGroups), kBlockThreads, smem, stream>>>(
        items, num_slots, dim, g.group_words, g.tile_items, g.split_items, out);
  } else {
    const dim3 grid(g.splits, num_slots);
    const size_t smem = static_cast<size_t>(g.group_words) * sizeof(float);
    NormalEquationsKernel<kBlockThreads><<<grid, kBlockThreads, smem, stream>>>(
        items, num_slots, dim, g.group_words, g.tile_items, g.split_items, out);
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (g.splits > 1) {
    SumSplitsKernel<<<GridFor(num_slots * entries), kBlockThreads, 0, stream>>>(
        scratch, g.splits, num_slots, static_cast<int>(entries), dim, hessian, gradient, cost);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

size_t SlotCostScratchFloats(const SlotItems &items, int num_slots) {
  const int splits = items.items_per_slot <= 256 ? 1 : CostSplits(items);
  return splits > 1 ? static_cast<size_t>(splits) * num_slots : 0;
}

void LaunchSlotCost(cudaStream_t stream, const SlotItems &items, int num_slots, float *cost,
                    float *scratch, SlotGroup group) {
  if (num_slots <= 0) {
    return;
  }
  const bool warp =
      group == SlotGroup::kWarp || (group == SlotGroup::kAuto && items.items_per_slot <= 256);
  if (warp) {
    constexpr int kGroups = kBlockThreads / kWarpSize;
    SlotCostKernel<kWarpSize>
        <<<GridFor(num_slots, kGroups), kBlockThreads, 0, stream>>>(items, num_slots, 1, cost);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
  const int splits = scratch != nullptr ? CostSplits(items) : 1;
  SlotCostKernel<kBlockThreads><<<dim3(splits, num_slots), kBlockThreads, 0, stream>>>(
      items, num_slots, splits, splits > 1 ? scratch : cost);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (splits > 1) {
    SumCostSplitsKernel<<<GridFor(num_slots, kBlockThreads), kBlockThreads, 0, stream>>>(
        scratch, splits, num_slots, cost);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

}  // namespace ransac_internal
}  // namespace cunls
