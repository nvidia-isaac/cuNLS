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

#include <algorithm>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/block_sparse_cholesky_solver.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;
constexpr int kW = internal::kCholeskyPanelWidth;
constexpr int kR = internal::kCholeskyTileRows;
constexpr int kPad = kW + 1;  // shared-memory row stride: conflict-free column access

/**
 * A pivot within this fraction of its original diagonal entry (either sign)
 * is a rank deficiency, not a failure: about 100 float epsilons, the
 * round-off floor of a pivot whose diagonal entry is cancelled by the
 * updates (free gauges land near 1e-6). Its unknown is fixed at zero (a zero
 * column with a zero reciprocal), which solves a consistent semidefinite
 * system along a free gauge (Gauss-Newton without an anchored state).
 * Regular pivots can be far below their diagonal entries (strongly coupled
 * states, e.g. motion priors: 1e-4), so the threshold stays tight.
 */
constexpr float kZeroPivot = 1e-5f;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/** factor[dst[e]] = values[e] for the nonzeros with dst[e] >= 0 (the lower triangle of PAPᵀ). */
__global__ void AssembleKernel(int nnz, const int *dst, const float *values, float *factor) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < nnz) {
    const int d = dst[e];
    if (d >= 0) factor[d] = values[e];
  }
}

/** One warp per row: flag[i] = 1 if row i starts a supervariable (its pattern differs from row i -
 * 1). */
__global__ void SupervariableKernel(int n, const int *row_offsets, const int *col_ids, int *flag) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  bool differs = i == 0;
  if (i > 0) {
    const int a = row_offsets[i - 1], b = row_offsets[i], c = row_offsets[i + 1];
    if (c - b != b - a) {
      differs = true;
    } else {
      bool d = false;
      for (int k = lane; k < b - a; k += 32) d = d || col_ids[a + k] != col_ids[b + k];
      differs = __any_sync(0xffffffffu, d);
    }
  }
  if (lane == 0) flag[i] = differs;
}

/**
 * One warp per supervariable v: its neighbors, from the pattern of its first
 * row (column indices ascending, so a neighbor's columns are consecutive).
 * Counts them when `adjacency` is null, else writes them at offsets[v].
 */
__global__ void AdjacencyKernel(int nb, const int *first_row, const int *row_offsets,
                                const int *col_ids, const int *supervariable, const int *offsets,
                                int *count, int *adjacency) {
  const int v = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (v >= nb) return;
  const int r = first_row[v], begin = row_offsets[r], end = row_offsets[r + 1];
  int written = 0;
  for (int base = begin; base < end; base += 32) {
    const int e = base + lane;
    bool keep = false;
    int u = -1;
    if (e < end) {
      u = supervariable[col_ids[e]];
      keep = u != v && (e == begin || u != supervariable[col_ids[e - 1]]);
    }
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (adjacency != nullptr && keep) {
      adjacency[offsets[v] + written + __popc(mask & ((1u << lane) - 1))] = u;
    }
    written += __popc(mask);
  }
  if (adjacency == nullptr && lane == 0) count[v] = written;
}

/**
 * One warp per row i of A: the factor offset of every nonzero (i, j) whose
 * permuted position (i', j') is in the lower triangle, else -1. The row i'
 * is found in the panel of column j' by arithmetic (its own columns) or by
 * binary search in its below rows. diagonal_index[i'] = the nonzero (i, i).
 */
__global__ void AssemblyMapKernel(int n, const int *row_offsets, const int *col_ids,
                                  const int *iperm, const int *col_panel, const int4 *info,
                                  const int64_t *offset, const int *below_rows, int *dst,
                                  int *diagonal_index) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  const int ip = iperm[i];
  for (int e = row_offsets[i] + lane; e < row_offsets[i + 1]; e += 32) {
    const int jp = iperm[col_ids[e]];
    int d = -1;
    if (ip >= jp) {
      const int p = col_panel[jp];
      const int4 panel = info[p];  // col, width, height, rows offset
      int pos;
      if (ip < panel.x + panel.y) {
        pos = ip - panel.x;
      } else {
        const int *rows = below_rows + panel.w;
        int lo = 0, hi = panel.z - panel.y;
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (rows[mid] < ip) {
            lo = mid + 1;
          } else {
            hi = mid;
          }
        }
        pos = panel.y + lo;
      }
      d = static_cast<int>(offset[p] + static_cast<int64_t>(jp - panel.x) * panel.z + pos);
      if (ip == jp) diagonal_index[ip] = e;
    }
    dst[e] = d;
  }
}

/**
 * The relative map: for run r (one warp), the positions in its target panel
 * of the source's below rows from the run on (the target's own columns by
 * arithmetic, its below rows by binary search).
 */
__global__ void RelmapKernel(int num_runs, const int4 *runs, const int *sources, const int4 *info,
                             const int *below_rows, int *relmap) {
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (r >= num_runs) return;
  const int4 run = runs[r];  // target, begin, end, relmap offset
  const int4 source = info[sources[r]], target = info[run.x];
  const int *rows = below_rows + source.w, *target_rows = below_rows + target.w;
  const int count = source.z - source.y, target_count = target.z - target.y;
  for (int i = run.y + lane; i < count; i += 32) {
    const int row = rows[i];
    int pos;
    if (row < target.x + target.y) {
      pos = row - target.x;
    } else {
      int lo = 0, hi = target_count;
      while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (target_rows[mid] < row) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      pos = target.y + lo;
    }
    relmap[run.w + i - run.y] = pos;
  }
}

/**
 * out[i] = in[perm[i]], and original[i] = the diagonal entry of A at permuted
 * column i (values[diagonal_index[i]], 0 without one).
 */
__global__ void PermuteKernel(int n, const int *perm, const float *in, float *out,
                              const int *diagonal_index, const float *values, float *original) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[perm[i]];
    const int e = diagonal_index[i];
    original[i] = e >= 0 ? values[e] : 0.f;
  }
}

/** out[perm[i]] = in[i]. */
__global__ void UnpermuteKernel(int n, const int *perm, const float *in, float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[perm[i]] = in[i];
}

/**
 * One CTA per (panel, 64 below rows) of a level: factors the stacked panel
 * [D; B; y_pᵀ] (D the diagonal block, B the CTA's rows below it, y_p the
 * panel's part of the forward-substitution vector): L_D = chol(D), B L_D^{-T}
 * and, as the last row, z_pᵀ = y_pᵀ L_D^{-T} (the forward substitution of the
 * panel comes out of the row solve). Then y[rows of B] -= (B L_D^{-T}) z_p.
 * Every CTA of a panel factors D redundantly; the one with row_begin 0
 * stores L_D (row-major, in `diagonal`, reciprocals on its diagonal) and z_p,
 * each stores its rows of B L_D^{-T} in place. y_p is final: the panels
 * below finished in earlier levels.
 *
 * Blocked by 16 columns: one warp factors the 16 x 16 diagonal block with
 * shuffles (a lane per row), a thread per row solves the rows below it, and
 * the trailing columns take the rank-16 update, 3 x 7 outputs per thread
 * with the 16 multipliers of its columns in registers. The upper triangle of
 * D takes garbage that is never read.
 */
__global__ void __launch_bounds__(kThreads)
    FactorKernel(const int2 *items, const int4 *info, const int64_t *offset,
                 const int64_t *diagonal_offset, const int *below_rows, const float *original,
                 float *factor, float *diagonal, float *y, float *z, int *failure) {
  constexpr int kRows = kW + kR;  // stacked rows, plus y_pᵀ
  constexpr int kB = 16;          // column block
  __shared__ float tile[kRows + 1][kPad];
  __shared__ float reciprocal[kW];                // 1 / L_D(k, k)
  __shared__ __align__(16) float column[kB + 4];  // current column; [kB]: next diagonal entry
  __shared__ __align__(16) float limit[kW];       // zero-pivot limits (kZeroPivot)
  __shared__ float partial[kThreads / kR][kR];
  const int2 item = items[blockIdx.x];
  const int4 panel_info = info[item.x];
  const int col = panel_info.x, w = panel_info.y, h = panel_info.z;
  const int below = max(0, min(kR, h - w - item.y));
  const int rows = w + below + 1;  // y_pᵀ last
  float *panel = factor + offset[item.x];
  const int t = threadIdx.x, lane = t % 32;
  {
    // Lanes along rows (coalesced), all loads in flight before the stores.
    const int i = t % kRows, j0 = t / kRows;
    const float *src = panel + (i < w ? i : item.y + i);
    float v[kW / 2];
#pragma unroll
    for (int m = 0; m < kW / 2; ++m) {
      const int j = j0 + 2 * m;
      v[m] = i < rows - 1 && j < w ? src[j * h] : 0.f;
    }
    if (i < rows - 1) {
#pragma unroll
      for (int m = 0; m < kW / 2; ++m) tile[i][j0 + 2 * m] = v[m];
    }
  }
  if (t < w) tile[rows - 1][t] = y[col + t];
  if (t < kW) limit[t] = t < w ? kZeroPivot * original[col + t] : 0.f;
  __syncthreads();
  for (int kb = 0; kb < w; kb += kB) {
    const int bw = min(kB, w - kb);
    if (t < 32) {
      // 16 x 16 diagonal block: lane i holds row kb + i.
      float r[kB];
#pragma unroll
      for (int c = 0; c < kB; ++c) r[c] = lane < bw && c <= lane ? tile[kb + lane][kb + c] : 0.f;
      // Column c is published through shared memory and read back as
      // broadcast vector loads (a shuffle per element serializes on its
      // latency); with it, the not yet updated next diagonal entry, so every
      // lane computes the next pivot itself.
      float pivot = __shfl_sync(0xffffffffu, r[0], 0);
      float lim[kB];
#pragma unroll
      for (int q = 0; q < kB / 4; ++q) {
        const float4 v = reinterpret_cast<const float4 *>(limit + kb)[q];
        lim[4 * q] = v.x;
        lim[4 * q + 1] = v.y;
        lim[4 * q + 2] = v.z;
        lim[4 * q + 3] = v.w;
      }
#pragma unroll
      for (int c = 0; c < kB; ++c) {
        if (c < bw) {
          // A vanishing pivot (rank deficiency) fixes its unknown: zero column, zero reciprocal.
          const bool zero = fabsf(pivot) <= lim[c];
          float scale;
          asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(pivot));
          if (zero) scale = 0.f;
          const float l = lane >= c ? r[c] * scale : 0.f;
          if (lane < kB) column[lane] = l;
          if (c + 1 < kB && lane == c + 1) column[kB] = r[c + 1];
          __syncwarp();
          float lc[kB];
#pragma unroll
          for (int q = 0; q < kB / 4; ++q) {
            const float4 v = reinterpret_cast<const float4 *>(column)[q];
            lc[4 * q] = v.x;
            lc[4 * q + 1] = v.y;
            lc[4 * q + 2] = v.z;
            lc[4 * q + 3] = v.w;
          }
#pragma unroll
          for (int j = c + 1; j < kB; ++j) r[j] -= l * lc[j];  // zero above the diagonal
          if (lane >= c) r[c] = l;
          if (lane == c) {
            reciprocal[kb + c] = scale;
            if (!zero && (!(pivot > 0.f) || !isfinite(pivot))) *failure = 1;
          }
          if (c + 1 < kB) pivot = column[kB] - lc[c + 1] * lc[c + 1];
          __syncwarp();
        }
      }
      if (lane < bw) {
#pragma unroll
        for (int c = 0; c < kB; ++c) {
          if (c <= lane) tile[kb + lane][kb + c] = r[c];
        }
      }
    }
    __syncthreads();
    // Rows below the block: x L_blockᵀ = row, one thread per row.
    const int first = kb + bw;
    if (t < rows - first) {
      const int i = first + t;
      float x[kB];
#pragma unroll
      for (int c = 0; c < kB; ++c) {
        if (c < bw) {
          float acc0 = tile[i][kb + c], acc1 = 0.f;  // two chains
#pragma unroll
          for (int j = 0; j + 1 < c; j += 2) {
            acc0 -= x[j] * tile[kb + c][kb + j];
            acc1 -= x[j + 1] * tile[kb + c][kb + j + 1];
          }
          if (c % 2 == 1) acc0 -= x[c - 1] * tile[kb + c][kb + c - 1];
          x[c] = (acc0 + acc1) * reciprocal[kb + c];
          tile[i][kb + c] = x[c];
        }
      }
    }
    __syncthreads();
    // Trailing update of columns first..w-1, rows first..rows-1.
    if (first < w) {
      const int jj = t % 16, ii = t / 16;
      float lj[3][kB];
#pragma unroll
      for (int q = 0; q < 3; ++q) {
        const int j = first + jj + 16 * q;
#pragma unroll
        for (int c = 0; c < kB; ++c) lj[q][c] = j < w && c < bw ? tile[j][kb + c] : 0.f;
      }
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const int i = first + ii + 16 * m;
        if (i < rows) {
          float li[kB];
#pragma unroll
          for (int c = 0; c < kB; ++c) li[c] = tile[i][kb + c];
#pragma unroll
          for (int q = 0; q < 3; ++q) {
            const int j = first + jj + 16 * q;
            float acc = 0.f;
#pragma unroll
            for (int c = 0; c < kB; ++c) acc += li[c] * lj[q][c];
            if (j < w) tile[i][j] -= acc;
          }
        }
      }
    }
    __syncthreads();
  }
  const float *zp = tile[rows - 1];
  if (item.y == 0) {
    if (t < w) z[col + t] = zp[t];
    float *d = diagonal + diagonal_offset[item.x];
    const int c = t % kW;
#pragma unroll 4
    for (int i = t / kW; i < w; i += kThreads / kW) {
      if (c < w) d[i * w + c] = c < i ? tile[i][c] : c == i ? reciprocal[c] : 0.f;
    }
  }
  if (below > 0) {
    const int r = t % kR, q = t / kR;
    float *dst = panel + w + item.y + r;
    float acc = 0.f;
#pragma unroll 4
    for (int c = q; c < w; c += kThreads / kR) {
      if (r < below) {
        const float v = tile[w + r][c];
        dst[c * h] = v;
        acc += v * zp[c];
      }
    }
    partial[q][r] = acc;
    __syncthreads();
    if (t < below) {
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < kThreads / kR; ++s) sum += partial[s][t];
      atomicAdd(y + below_rows[panel_info.w + item.y + t], -sum);
    }
  }
}

/** Loads the row-major w x w L_D of a panel into shared memory (whole CTA). */
__device__ void LoadDiagonal(const float *d, int w, float (*ld)[kPad]) {
  const int c = threadIdx.x % kW, i0 = threadIdx.x / kW;
  float v[kW * kW / kThreads];
#pragma unroll
  for (int m = 0; m < kW * kW / kThreads; ++m) {
    const int i = i0 + m * (kThreads / kW);
    v[m] = i < w && c < w ? d[i * w + c] : 0.f;
  }
#pragma unroll
  for (int m = 0; m < kW * kW / kThreads; ++m) ld[i0 + m * (kThreads / kW)][c] = v[m];
}

/**
 * One warp: v := L_D^{-1} v (forward) or L_D^{-T} v (backward), v in shared
 * memory; ld holds L_D with the reciprocals of its diagonal on the diagonal.
 */
template <bool kTranspose>
__device__ void WarpTriangularSolve(const float (*ld)[kPad], int w, float *v) {
  const int lane = threadIdx.x % 32;
  float lo = lane < w ? v[lane] : 0.f, hi = lane + 32 < w ? v[lane + 32] : 0.f;
  for (int s = 0; s < w; ++s) {
    const int k = kTranspose ? w - 1 - s : s;
    const bool owner_hi = k >= 32;
    if (lane == k % 32) {
      if (owner_hi) {
        hi *= ld[k][k];
      } else {
        lo *= ld[k][k];
      }
    }
    const float xk = __shfl_sync(0xffffffffu, owner_hi ? hi : lo, k % 32);
    if (kTranspose) {
      if (lane < k) lo -= ld[k][lane] * xk;
      if (lane + 32 < k) hi -= ld[k][lane + 32] * xk;
    } else {
      if (lane > k && lane < w) lo -= ld[lane][k] * xk;
      if (lane + 32 > k && lane + 32 < w) hi -= ld[lane + 32][k] * xk;
    }
  }
  if (lane < w) v[lane] = lo;
  if (lane + 32 < w) v[lane + 32] = hi;
}

/**
 * One CTA per (run, 64 rows): the update -L_b[rows] L_b[run]ᵀ of a panel
 * into the target panel of the run (rows at or below the run, columns of the
 * run), lower triangle of the target's diagonal block only. Each thread
 * computes 4 x 4 outputs from float4 shared loads (rows 4 tx.., columns
 * 4 ty..); the tile then goes through shared memory so the read-modify-write
 * of the target runs along its rows (coalesced).
 */
template <bool kAtomic>
__global__ void __launch_bounds__(kThreads)
    UpdateKernel(const int4 *items, const int4 *runs, const int *relmap, const int4 *info,
                 const int64_t *offset, float *factor) {
  __shared__ __align__(16) float as[kW][kR];
  __shared__ __align__(16) float bs[kW][kR];  // 32 KB in all: 3 CTAs per SM on sm_86
  const int4 item = items[blockIdx.x];
  const int4 panel_info = info[item.x];
  const int w = panel_info.y, h = panel_info.z;
  const int4 run = runs[item.y];
  const int rows = min(kR, h - w - item.z), cols = run.z - run.y;
  const float *src = factor + offset[item.x] + w;
  const int t = threadIdx.x;
  {
    const int i = t % kR, k0 = t / kR;
    float va[kW * kR / kThreads], vb[kW * kR / kThreads];
#pragma unroll
    for (int m = 0; m < kW * kR / kThreads; ++m) {
      const int k = k0 + m * (kThreads / kR);
      va[m] = i < rows && k < w ? src[k * h + item.z + i] : 0.f;
      vb[m] = i < cols && k < w ? src[k * h + run.y + i] : 0.f;
    }
#pragma unroll
    for (int m = 0; m < kW * kR / kThreads; ++m) {
      as[k0 + m * (kThreads / kR)][i] = va[m];
      bs[k0 + m * (kThreads / kR)][i] = vb[m];
    }
  }
  __syncthreads();
  const int tx = t % 16, ty = t / 16;
  float acc[4][4] = {};
  for (int k = 0; k < w; ++k) {
    const float4 a = reinterpret_cast<const float4 *>(as[k])[tx];
    const float4 b = reinterpret_cast<const float4 *>(bs[k])[ty];
    const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
      for (int n = 0; n < 4; ++n) acc[m][n] += av[m] * bv[n];
    }
  }
  __syncthreads();
  // as := the tile, column-major (as[j][i]).
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    reinterpret_cast<float4 *>(as[4 * ty + n])[tx] =
        make_float4(acc[0][n], acc[1][n], acc[2][n], acc[3][n]);
  }
  __syncthreads();
  const int4 target = info[run.x];
  float *dst = factor + offset[run.x];
  const int lane = t % 32, warp = t / 32;
  const int *rel = relmap + run.w + item.z - run.y;
  int row_pos[kR / 32];
#pragma unroll
  for (int half = 0; half < kR / 32; ++half) {
    const int i = lane + 32 * half;
    row_pos[half] = i < rows ? rel[i] : -1;
  }
  for (int j = warp; j < cols; j += kThreads / 32) {
    const int col_pos = relmap[run.w + j];
    float *column = dst + static_cast<int64_t>(col_pos) * target.z;
#pragma unroll
    for (int half = 0; half < kR / 32; ++half) {
      if (row_pos[half] >= col_pos) {
        const float v = as[j][lane + 32 * half];
        if (kAtomic) {
          atomicAdd(column + row_pos[half], -v);
        } else {
          column[row_pos[half]] -= v;
        }
      }
    }
  }
}

/**
 * Backward substitution, one CTA per (panel, 64 below rows) of a level
 * (levels in reverse): each CTA adds its rows' L_b[rows]ᵀ x[rows] to
 * accumulator[cols]; the last CTA of the panel then solves
 * x_p = L_D^{-T} (z_p - accumulator) and resets the accumulator and counter.
 */
__global__ void __launch_bounds__(kThreads)
    BackwardKernel(const int2 *items, const int4 *info, const int64_t *offset,
                   const int64_t *diagonal_offset, const int *below_rows, const float *factor,
                   const float *diagonal, float *x, float *accumulator, int *counter) {
  __shared__ float ld[kW][kPad];
  __shared__ float v[kW];
  __shared__ bool last;
  const int2 item = items[blockIdx.x];
  const int4 panel_info = info[item.x];
  const int col = panel_info.x, w = panel_info.y, h = panel_info.z;
  const int t = threadIdx.x, lane = t % 32;
  const int below = min(kR, h - w - item.y);
  const int tiles = max(1, (h - w + kR - 1) / kR);
  if (below > 0) {
    const float *l = factor + offset[item.x] + w + item.y;
    const int r = t % kR, q = t / kR;
    const float xr = r < below ? x[below_rows[panel_info.w + item.y + r]] : 0.f;
    constexpr int kPerThread = kW / (kThreads / kR);
    float v[kPerThread];
#pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
      const int k = q + m * (kThreads / kR);
      v[m] = r < below && k < w ? l[k * h + r] : 0.f;
    }
#pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
      const int k = q + m * (kThreads / kR);
      float acc = v[m] * xr;
#pragma unroll
      for (int s = 16; s > 0; s /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, s);
      if (lane == 0 && k < w) atomicAdd(accumulator + col + k, acc);
    }
  }
  if (tiles > 1) {
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(counter + item.x, 1) == tiles - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
  } else {
    __syncthreads();
  }
  LoadDiagonal(diagonal + diagonal_offset[item.x], w, ld);
  if (t < w) {
    v[t] = x[col + t] - __ldcg(accumulator + col + t);
    accumulator[col + t] = 0.f;
  }
  if (t == 0) counter[item.x] = 0;
  __syncthreads();
  if (t < 32) WarpTriangularSolve<true>(ld, w, v);
  __syncthreads();
  if (t < w) x[col + t] = v[t];
}

}  // namespace

BlockSparseCholeskySolver::BlockSparseCholeskySolver() : failure_(1), failure_host_(1) {}

bool BlockSparseCholeskySolver::Initialize(cudaStream_t stream, const Problem & /*problem*/,
                                           const CSRSparseMatrix &spd_matrix,
                                           const dvector<float> &rhs, dvector<float> &result) {
  const size_t n = spd_matrix.NumRows();
  if (rhs.size() != n || result.size() != n) {
    LogError("BlockSparseCholeskySolver: matrix size {} does not match rhs {} / result {}", n,
             rhs.size(), result.size());
    return false;
  }
  // Supervariables and their graph on the device; only the (small) graph comes to the host.
  const int nnz = static_cast<int>(spd_matrix.col_ids.size());
  std::vector<int> start{0}, adjacency_offsets{0}, adjacency;
  dvector<int> supervariable(n);
  if (n > 0) {
    const int *ro = spd_matrix.row_offsets.data(), *ci = spd_matrix.col_ids.data();
    dvector<int> flag(n);
    SupervariableKernel<<<Blocks(32 * n), kThreads, 0, stream>>>(static_cast<int>(n), ro, ci,
                                                                 flag.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    std::vector<int> h_flag(n), h_supervariable(n);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    flag.CopyToHost(h_flag.data(), n);
    for (size_t i = 0; i < n; ++i) {
      if (i > 0 && h_flag[i]) start.push_back(static_cast<int>(i));
      h_supervariable[i] = static_cast<int>(start.size()) - 1;
    }
    const int nb = static_cast<int>(start.size());
    start.push_back(static_cast<int>(n));
    supervariable.CopyFromHost(h_supervariable.data(), n);
    dvector<int> first_row(std::vector<int>(start.begin(), start.end() - 1)), count(nb);
    AdjacencyKernel<<<Blocks(32 * static_cast<size_t>(nb)), kThreads, 0, stream>>>(
        nb, first_row.data(), ro, ci, supervariable.data(), nullptr, count.data(), nullptr);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    std::vector<int> h_count(nb);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    count.CopyToHost(h_count.data(), nb);
    for (int v = 0; v < nb; ++v) adjacency_offsets.push_back(adjacency_offsets.back() + h_count[v]);
    adjacency.resize(adjacency_offsets.back());
    if (!adjacency.empty()) {
      dvector<int> offsets(adjacency_offsets), d_adjacency(adjacency.size());
      AdjacencyKernel<<<Blocks(32 * static_cast<size_t>(nb)), kThreads, 0, stream>>>(
          nb, first_row.data(), ro, ci, supervariable.data(), offsets.data(), nullptr,
          d_adjacency.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      d_adjacency.CopyToHost(adjacency.data(), adjacency.size());
    }
  }
  plan_ = internal::AnalyzeCholesky(static_cast<int>(n), start, adjacency_offsets, adjacency);
  const auto &plan = plan_;
  const size_t np = plan.panels.size();
  std::vector<int4> info(np);
  std::vector<int64_t> offset(np), diagonal_offset(np);
  for (size_t p = 0; p < np; ++p) {
    const auto &panel = plan.panels[p];
    info[p] = make_int4(panel.col, panel.width, panel.height, panel.rows_offset);
    offset[p] = panel.offset;
    diagonal_offset[p] = panel.diagonal_offset;
  }
  std::vector<int4> runs(plan.runs.size());
  for (size_t i = 0; i < runs.size(); ++i) {
    const auto &run = plan.runs[i];
    runs[i] = make_int4(run.target, run.begin, run.end, run.relmap_offset);
  }
  std::vector<int2> factor_items(plan.factor_items.size());
  for (size_t i = 0; i < factor_items.size(); ++i) {
    factor_items[i] = make_int2(plan.factor_items[i].panel, plan.factor_items[i].row_begin);
  }
  std::vector<int4> update_items(plan.update_items.size());
  for (size_t i = 0; i < update_items.size(); ++i) {
    const auto &item = plan.update_items[i];
    update_items[i] = make_int4(item.panel, item.run, item.row_begin, 0);
  }
  auto upload = [](auto &device, const auto &host) {
    device.resize(host.size());
    if (!host.empty()) device.CopyFromHost(host.data(), host.size());
  };
  upload(perm_, plan.perm);
  upload(panel_info_, info);
  upload(panel_offset_, offset);
  upload(diagonal_offset_, diagonal_offset);
  upload(below_rows_, plan.below_rows);
  upload(runs_, runs);
  upload(factor_items_, factor_items);
  upload(update_items_, update_items);
  // The assembly map: where each nonzero of A goes in the factor.
  assembly_dst_.resize(nnz);
  diagonal_index_.resize(n);
  original_.resize(n);
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(diagonal_index_.data(), 0xFF, n * sizeof(int), stream));
    std::vector<int> iperm(n), col_panel(n);
    for (size_t c = 0; c < n; ++c) iperm[plan.perm[c]] = static_cast<int>(c);
    for (size_t p = 0; p < np; ++p) {
      std::fill(col_panel.begin() + plan.panels[p].col,
                col_panel.begin() + plan.panels[p].col + plan.panels[p].width, static_cast<int>(p));
    }
    dvector<int> d_iperm(iperm), d_col_panel(col_panel);
    AssemblyMapKernel<<<Blocks(32 * n), kThreads, 0, stream>>>(
        static_cast<int>(n), spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(),
        d_iperm.data(), d_col_panel.data(), panel_info_.data(), panel_offset_.data(),
        below_rows_.data(), assembly_dst_.data(), diagonal_index_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));  // before the temporaries go
  }
  // The relative map of the updates.
  relmap_.resize(plan.relmap_size);
  if (!plan.runs.empty()) {
    std::vector<int> sources(plan.runs.size());
    for (size_t i = 0; i < sources.size(); ++i) sources[i] = plan.runs[i].source;
    dvector<int> d_sources(sources);
    RelmapKernel<<<Blocks(32 * plan.runs.size()), kThreads, 0, stream>>>(
        static_cast<int>(plan.runs.size()), runs_.data(), d_sources.data(), panel_info_.data(),
        below_rows_.data(), relmap_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));  // before the temporaries go
  }
  factor_.resize(plan.factor_size);
  diagonal_.resize(plan.diagonal_size);
  work_.resize(n);
  solution_.resize(n);
  accumulator_.resize(n);
  counter_.resize(np);
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(accumulator_.data(), 0, n * sizeof(float), stream));
  }
  if (np > 0) THROW_ON_CUDA_ERROR(cudaMemsetAsync(counter_.data(), 0, np * sizeof(int), stream));
  LogMessage(
      "BlockSparseCholesky: n {}, {} supervariables, {} ordering, {} supernodes, {} panels, {} "
      "levels, factor {:.2f} M entries, {:.3f} GFLOP",
      n, plan.num_supervariables, plan.ordering, plan.num_supernodes, np, plan.num_levels,
      plan.factor_size / 1e6, plan.flops / 1e9);
  return true;
}

bool BlockSparseCholeskySolver::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                                      const dvector<float> &rhs, dvector<float> &result) {
  const auto &plan = plan_;
  const int n = plan.n;
  if (n == 0) return true;
  if (safety_checks_) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(failure_.data(), 0, sizeof(int), stream));
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(factor_.data(), 0, factor_.size() * sizeof(float), stream));
  const int nnz = static_cast<int>(assembly_dst_.size());
  AssembleKernel<<<Blocks(nnz), kThreads, 0, stream>>>(nnz, assembly_dst_.data(),
                                                       spd_matrix.values.data(), factor_.data());
  PermuteKernel<<<Blocks(n), kThreads, 0, stream>>>(n, perm_.data(), rhs.data(), work_.data(),
                                                    diagonal_index_.data(),
                                                    spd_matrix.values.data(), original_.data());
  for (int l = 0; l < plan.num_levels; ++l) {
    const int f0 = plan.factor_offsets[l], f1 = plan.factor_offsets[l + 1];
    FactorKernel<<<f1 - f0, kThreads, 0, stream>>>(
        factor_items_.data() + f0, panel_info_.data(), panel_offset_.data(),
        diagonal_offset_.data(), below_rows_.data(), original_.data(), factor_.data(),
        diagonal_.data(), work_.data(), solution_.data(), failure_.data());
    const int u0 = plan.update_offsets[l], u1 = plan.update_offsets[l + 1];
    if (u1 > u0) {
      if (plan.update_exclusive[l]) {
        UpdateKernel<false><<<u1 - u0, kThreads, 0, stream>>>(
            update_items_.data() + u0, runs_.data(), relmap_.data(), panel_info_.data(),
            panel_offset_.data(), factor_.data());
      } else {
        UpdateKernel<true><<<u1 - u0, kThreads, 0, stream>>>(
            update_items_.data() + u0, runs_.data(), relmap_.data(), panel_info_.data(),
            panel_offset_.data(), factor_.data());
      }
    }
  }
  for (int l = plan.num_levels - 1; l >= 0; --l) {
    const int f0 = plan.factor_offsets[l], f1 = plan.factor_offsets[l + 1];
    BackwardKernel<<<f1 - f0, kThreads, 0, stream>>>(
        factor_items_.data() + f0, panel_info_.data(), panel_offset_.data(),
        diagonal_offset_.data(), below_rows_.data(), factor_.data(), diagonal_.data(),
        solution_.data(), accumulator_.data(), counter_.data());
  }
  UnpermuteKernel<<<Blocks(n), kThreads, 0, stream>>>(n, perm_.data(), solution_.data(),
                                                      result.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (safety_checks_) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(failure_host_.data(), failure_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    if (failure_host_[0] != 0) {
      LogError("BlockSparseCholeskySolver: the matrix is not positive definite");
      return false;
    }
  }
  return true;
}

}  // namespace cunls
