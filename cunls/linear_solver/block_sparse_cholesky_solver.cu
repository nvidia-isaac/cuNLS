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
#include <climits>
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

using Panel = internal::CholeskyPlan::Panel;
using Run = internal::CholeskyPlan::Run;
using Task = internal::CholeskyPlan::Task;
using Launch = internal::CholeskyPlan::Launch;
using Phase = internal::CholeskyPlan::Phase;

constexpr unsigned kFullMask = 0xffffffffu;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/**
 * The device arrays of the numeric phase, read or written by its kernels: one struct, so every
 * kernel takes (Args, its tasks, their count) and the solve launches them from one place.
 */
struct Args {
  const Panel *panels;    ///< The plan's panels.
  const int *below_rows;  ///< Below rows of all panels.
  const Run *runs;        ///< Update runs of all panels.
  const int *relmap;      ///< Target positions of the runs' rows.
  const int *perm;        ///< Permuted column -> original column.
  const int *row_active;  ///< SetRowActivity(), or null for all rows.
  const float *original;  ///< Per column of an updated panel: A's diagonal entry.
  float *factor;          ///< The panels of L.
  float *diagonal;        ///< The panels' L_D (column-major, reciprocals on the diagonal).
  float *y;               ///< Permuted right-hand side, reduced by the forward substitution.
  float *x;               ///< Per column z = L^{-1} b, then the solution.
  float *accumulator;     ///< Backward substitution: L_bᵀ x per column (zero between solves).
  int *counter;           ///< Backward substitution: finished CTAs per panel (zero between).
  int *failure;           ///< Set by a non-positive or non-finite pivot.
};

/**
 * Whether any column of a panel is a needed row (SparseLinearSolver::SetRowActivity; always
 * without row activity). Every thread of the CTA (kWarp false) or of the warp calls it.
 */
template <bool kWarp>
__device__ bool PanelActive(const Panel &panel, const Args &args) {
  if (args.row_active == nullptr) return true;
  const int i = kWarp ? threadIdx.x % 32 : threadIdx.x;
  const bool active = i < panel.width && args.row_active[args.perm[panel.col + i]] != 0;
  return kWarp ? __any_sync(kFullMask, active) : __syncthreads_or(active);
}

/**
 * Programmatic dependent launch (sm_90 and newer; SolveStoredMatrix launches the program's
 * kernels with it): a kernel starts while its predecessor still runs and reads in its prologue
 * only what that predecessor does not write (the plan, the relative map, and in the backward
 * pass the factor). Here it waits for the predecessor to finish. Before sm_90 the kernels run
 * in order and this does nothing.
 */
__device__ __forceinline__ void AwaitPredecessor() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

/**
 * Lets the successor start its prologue (programmatic dependent launch), once the CTA's main
 * work is done (only its final writes remain): its CTAs then wait next to this kernel's tail,
 * not next to all of its work. Always after AwaitPredecessor(), so older kernels have
 * finished when a prologue runs.
 */
__device__ __forceinline__ void StartSuccessor() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
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
 * Position of permuted row `row` (at or after the panel's first column) in a
 * panel: its own columns by arithmetic, its below rows by binary search.
 */
__device__ int PanelRowPosition(const Panel &panel, const int *below_rows, int row) {
  if (row < panel.col + panel.width) return row - panel.col;
  const int *rows = below_rows + panel.rows_offset;
  int lo = 0, hi = panel.height - panel.width;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (rows[mid] < row) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return panel.width + lo;
}

/**
 * One warp per row i of A: the factor offset of every nonzero (i, j) whose
 * permuted position (i', j') is in the lower triangle, else -1. The row i'
 * is found in the panel of column j' (PanelRowPosition). diagonal_index[i'] =
 * the nonzero (i, i).
 */
template <typename Index>
__global__ void AssemblyMapKernel(int n, const int *row_offsets, const int *col_ids,
                                  const int *iperm, const int *col_panel, const Panel *panels,
                                  const int *below_rows, Index *dst, int *diagonal_index) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return;
  const int ip = iperm[i];
  for (int e = row_offsets[i] + lane; e < row_offsets[i + 1]; e += 32) {
    const int jp = iperm[col_ids[e]];
    Index d = -1;
    if (ip >= jp) {
      const int p = col_panel[jp];
      const Panel panel = panels[p];
      const int pos = PanelRowPosition(panel, below_rows, ip);
      d = static_cast<Index>(panel.offset + static_cast<int64_t>(jp - panel.col) * panel.height +
                             pos);
      if (ip == jp) diagonal_index[ip] = e;
    }
    dst[e] = d;
  }
}

/**
 * The relative map: for run r (one warp), the positions in its target panel
 * of the source's below rows from the run on (PanelRowPosition).
 */
__global__ void RelmapKernel(int num_runs, const Run *runs, const Panel *panels,
                             const int *below_rows, int *relmap) {
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (r >= num_runs) return;
  const Run run = runs[r];
  const Panel source = panels[run.source], target = panels[run.target];
  const int *rows = below_rows + source.rows_offset;
  const int count = source.height - source.width;
  for (int i = run.begin + lane; i < count; i += 32) {
    relmap[run.relmap_offset + i - run.begin] = PanelRowPosition(target, below_rows, rows[i]);
  }
}

/**
 * out[i] = in[perm[i]]; with `diagonal_slot`, also original[i] = the diagonal entry of A at
 * permuted column i, read from the assembled factor (factor[diagonal_slot[i]], 0 without one)
 * before the factorization updates it.
 */
__global__ void PermuteKernel(int n, const int *perm, const float *in, float *out,
                              const int64_t *diagonal_slot, const float *factor, float *original) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[perm[i]];
    if (diagonal_slot != nullptr) {
      const int64_t d = diagonal_slot[i];
      original[i] = d >= 0 ? factor[d] : 0.f;
    }
  }
}

/**
 * diagonal_slot[i] = the factor offset of A's diagonal entry at permuted column i, or -1 for
 * none and for the columns of leaf panels (which read it from their own storage).
 */
template <typename Index>
__global__ void DiagonalSlotKernel(int n, const int *diagonal_index, const Index *dst,
                                   const int *col_panel, const Panel *panels,
                                   int64_t *diagonal_slot) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int e = diagonal_index[i];
    diagonal_slot[i] = e >= 0 && !panels[col_panel[i]].leaf ? static_cast<int64_t>(dst[e]) : -1;
  }
}

/** out[perm[i]] = in[i]. */
__global__ void UnpermuteKernel(int n, const int *perm, const float *in, float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[perm[i]] = in[i];
}

/** Loads the column-major w x w L_D of a panel into shared memory as ld[row][column] (whole CTA).
 */
__device__ void LoadDiagonal(const float *d, int w, float (*ld)[kPad]) {
  const int i = threadIdx.x % kW, c0 = threadIdx.x / kW;
  float v[kW * kW / kThreads];
#pragma unroll
  for (int m = 0; m < kW * kW / kThreads; ++m) {
    const int c = c0 + m * (kThreads / kW);
    v[m] = i < w && c < w ? d[c * w + i] : 0.f;
  }
#pragma unroll
  for (int m = 0; m < kW * kW / kThreads; ++m) ld[i][c0 + m * (kThreads / kW)] = v[m];
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
 * One CTA per (panel, 64 below rows) of a level: factors the stacked panel
 * [D; B; y_pᵀ] (D the diagonal block, B the CTA's rows below it, y_p the
 * panel's part of the forward-substitution vector): L_D = chol(D), B L_D^{-T}
 * and, as the last row, z_pᵀ = y_pᵀ L_D^{-T} (the forward substitution of the
 * panel comes out of the row solve). Then y[rows of B] -= (B L_D^{-T}) z_p.
 * Every CTA of a panel factors D redundantly; the one with row_begin 0
 * stores L_D (column-major, in `diagonal`, reciprocals on its diagonal) and z_p,
 * each stores its rows of B L_D^{-T} in place. y_p is final: the panels
 * below finished in earlier levels. A root (no rows below) instead solves
 * L_Dᵀ x_p = z_p right away and stores x_p: it has no backward task.
 *
 * Blocked by 16 columns: one warp factors the 16 x 16 diagonal block in
 * groups of 4 columns (a lane per row), a thread per row solves the rows below it, and
 * the trailing columns take the rank-16 update, 3 x 7 outputs per thread
 * with the 16 multipliers of its columns in registers. The upper triangle of
 * D takes garbage that is never read.
 */
__global__ void __launch_bounds__(kThreads) FactorKernel(Args args, const Task *tasks, int) {
  constexpr int kRows = kW + kR;  // stacked rows, plus y_pᵀ
  constexpr int kB = 16;          // column block
  __shared__ float tile[kRows + 1][kPad];
  __shared__ float reciprocal[kW];              // 1 / L_D(k, k)
  __shared__ __align__(16) float corner[4][4];  // the 4 x 4 diagonal block of a column group
  __shared__ __align__(16) float group[4][kB];  // the 4 columns of a column group
  __shared__ __align__(16) float limit[kW];     // zero-pivot limits (kZeroPivot)
  __shared__ float partial[kThreads / kR][kR];
  const Task task = tasks[blockIdx.x];
  const Panel panel_info = args.panels[task.index];
  const int col = panel_info.col, w = panel_info.width, h = panel_info.height;
  const bool root = h == w;
  const int below = max(0, min(kR, h - w - task.row_begin));
  const int rows = w + below + 1;  // y_pᵀ last
  float *panel = args.factor + panel_info.offset;
  const int t = threadIdx.x, lane = t % 32;
  if (!PanelActive<false>(panel_info, args)) {
    // An idle panel is not factored; a root gets its zero solution here.
    if (root && t < w) args.x[col + t] = 0.f;
    return;
  }
  AwaitPredecessor();
  {
    // Lanes along rows (coalesced), all loads in flight before the stores.
    const int i = t % kRows, j0 = t / kRows;
    const float *src = panel + (i < w ? i : task.row_begin + i);
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
  if (t < w) tile[rows - 1][t] = args.y[col + t];
  if (t < kW) {
    // A's diagonal: in the panel itself while no update has reached it (a leaf).
    const float a_tt = t >= w ? 0.f : panel_info.leaf ? panel[t * h + t] : args.original[col + t];
    limit[t] = kZeroPivot * a_tt;
  }
  __syncthreads();
  for (int kb = 0; kb < w; kb += kB) {
    const int bw = min(kB, w - kb);
    if (t < 32) {
      // 16 x 16 diagonal block: lane i holds row kb + i.
      float r[kB];
#pragma unroll
      for (int c = 0; c < kB; ++c) r[c] = lane < bw && c <= lane ? tile[kb + lane][kb + c] : 0.f;
      float lim[kB];
#pragma unroll
      for (int q = 0; q < kB / 4; ++q) {
        const float4 v = reinterpret_cast<const float4 *>(limit + kb)[q];
        lim[4 * q] = v.x;
        lim[4 * q + 1] = v.y;
        lim[4 * q + 2] = v.z;
        lim[4 * q + 3] = v.w;
      }
      // Groups of 4 columns, two warp round trips each: the owner lanes
      // publish the group's 4 x 4 diagonal block, every lane factors it in
      // registers and solves its own row against it, then the 4 columns are
      // published and every row takes the rank-4 update. Lane c keeps the
      // reciprocal and the failure of column c (written after the loop).
      float own_scale = 0.f;
      bool own_failure = false;
#pragma unroll
      for (int c0 = 0; c0 < kB; c0 += 4) {
        if (c0 < bw) {
          if (lane >= c0 && lane < c0 + 4) {
#pragma unroll
            for (int b = 0; b < 4; ++b) corner[lane - c0][b] = r[c0 + b];
          }
          __syncwarp();
          float a[4][4];
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            const float4 v = reinterpret_cast<const float4 *>(corner[x])[0];
            a[x][0] = v.x;
            a[x][1] = v.y;
            a[x][2] = v.z;
            a[x][3] = v.w;
          }
          // The group's block: L (lower), scales 1 / L(b, b); a vanishing
          // pivot (rank deficiency) fixes its unknown: zero scale.
          float lg[4][4], scale[4];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            float pivot = a[b][b];
#pragma unroll
            for (int q = 0; q < b; ++q) pivot -= lg[b][q] * lg[b][q];
            const bool zero = fabsf(pivot) <= lim[c0 + b];
            asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(scale[b]) : "f"(pivot));
            if (zero) scale[b] = 0.f;
            const bool own = lane == c0 + b;
            own_scale = own ? scale[b] : own_scale;
            own_failure |= own && !zero && (!(pivot > 0.f) || !isfinite(pivot));
#pragma unroll
            for (int x = b + 1; x < 4; ++x) {
              float v = a[x][b];
#pragma unroll
              for (int q = 0; q < b; ++q) v -= lg[x][q] * lg[b][q];
              lg[x][b] = v * scale[b];
            }
          }
          // The lane's own row of the group's columns (zero above the diagonal).
          float xv[4];
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            float v = r[c0 + b];
#pragma unroll
            for (int q = 0; q < b; ++q) v -= xv[q] * lg[b][q];
            xv[b] = lane >= c0 + b ? v * scale[b] : 0.f;
          }
          if (lane < kB) {
#pragma unroll
            for (int b = 0; b < 4; ++b) group[b][lane] = xv[b];
          }
          __syncwarp();
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            float lc[kB];
#pragma unroll
            for (int q = c0 / 4 + 1; q < kB / 4; ++q) {
              const float4 v = reinterpret_cast<const float4 *>(group[b])[q];
              lc[4 * q] = v.x;
              lc[4 * q + 1] = v.y;
              lc[4 * q + 2] = v.z;
              lc[4 * q + 3] = v.w;
            }
#pragma unroll
            for (int j = c0 + 4; j < kB; ++j) r[j] -= xv[b] * lc[j];
            if (lane >= c0 + b) r[c0 + b] = xv[b];
          }
        }
      }
      if (lane < bw) reciprocal[kb + lane] = own_scale;
      if (own_failure) *args.failure = 1;
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
  StartSuccessor();
  float *zp = tile[rows - 1];
  if (root) {
    // Nothing below the panel: its backward step L_Dᵀ x_p = z_p needs nothing from other
    // panels, so it is taken here (L_D with the reciprocals on its diagonal, in the tile).
    if (t < w) tile[t][t] = reciprocal[t];
    __syncthreads();
    if (t < 32) WarpTriangularSolve<true>(tile, w, zp);
    __syncthreads();
    if (t < w) args.x[col + t] = zp[t];
    return;
  }
  if (task.row_begin == 0) {
    if (t < w) args.x[col + t] = zp[t];
    // L_D, column-major.
    float *d = args.diagonal + panel_info.diagonal_offset;
    const int i = t % kW;
#pragma unroll 4
    for (int c = t / kW; c < w; c += kThreads / kW) {
      if (i < w) d[c * w + i] = i > c ? tile[i][c] : i == c ? reciprocal[c] : 0.f;
    }
  }
  if (below > 0) {
    const int r = t % kR, q = t / kR;
    float *dst = panel + w + task.row_begin + r;
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
      atomicAdd(args.y + args.below_rows[panel_info.rows_offset + task.row_begin + t], -sum);
    }
  }
}

/**
 * One CTA per (run, 64 rows): the update -L_b[rows] L_b[run]ᵀ of a panel
 * into the target panel of the run (rows at or below the run, columns of the
 * run), lower triangle of the target's diagonal block only. Each thread
 * computes 4 x 4 outputs from float4 shared loads (rows 4 tx.., columns
 * 4 ty..), a warp the same 8 columns, so the warps past the run's columns
 * (a short run) skip the product; the tile then goes through shared memory
 * so the reductions into
 * the target run along its rows (coalesced). Reductions (atomic adds without
 * a result: no load on the critical path), which also let two sources of a
 * level update one target; with one source per target the result is exact
 * and deterministic all the same.
 */
__global__ void __launch_bounds__(kThreads) UpdateKernel(Args args, const Task *tasks, int) {
  __shared__ __align__(16) float as[kW][kR];
  __shared__ __align__(16) float bs[kW][kR];  // 32 KB in all: 3 CTAs per SM on sm_86
  constexpr int kWarps = kThreads / 32;
  const Task task = tasks[blockIdx.x];
  const Run run = args.runs[task.index];
  const Panel source = args.panels[run.source];
  // An idle source updates only panels of its own (idle) subproblem.
  if (!PanelActive<false>(source, args)) return;
  const int w = source.width, h = source.height;
  const int rows = min(kR, h - w - task.row_begin), cols = run.end - run.begin;
  const float *src = args.factor + source.offset + w;
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  // Target positions (relative map): of the lane's rows, and of the warp's columns of the run.
  const Panel target = args.panels[run.target];
  const int *rel = args.relmap + run.relmap_offset + task.row_begin - run.begin;
  int row_pos[kR / 32], col_pos[kW / kWarps];
#pragma unroll
  for (int half = 0; half < kR / 32; ++half) {
    const int i = lane + 32 * half;
    row_pos[half] = i < rows ? rel[i] : -1;
  }
#pragma unroll
  for (int c = 0; c < kW / kWarps; ++c) {
    const int j = warp + kWarps * c;
    col_pos[c] = j < cols ? args.relmap[run.relmap_offset + j] : 0;
  }
  AwaitPredecessor();
  {
    const int i = t % kR, k0 = t / kR;
    float va[kW * kR / kThreads], vb[kW * kR / kThreads];
#pragma unroll
    for (int m = 0; m < kW * kR / kThreads; ++m) {
      const int k = k0 + m * (kThreads / kR);
      va[m] = i < rows && k < w ? src[k * h + task.row_begin + i] : 0.f;
      vb[m] = i < cols && k < w ? src[k * h + run.begin + i] : 0.f;
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
  const int k_end = 8 * (t / 32) < cols ? w : 0;  // the same for the whole warp
  for (int k = 0; k < k_end; ++k) {
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
  StartSuccessor();
  // as := the tile, column-major (as[j][i]).
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    reinterpret_cast<float4 *>(as[4 * ty + n])[tx] =
        make_float4(acc[0][n], acc[1][n], acc[2][n], acc[3][n]);
  }
  __syncthreads();
  float *dst = args.factor + target.offset;
#pragma unroll
  for (int c = 0; c < kW / kWarps; ++c) {
    const int j = warp + kWarps * c;
    if (j >= cols) break;  // the whole warp
    float *column = dst + static_cast<int64_t>(col_pos[c]) * target.height;
#pragma unroll
    for (int half = 0; half < kR / 32; ++half) {
      if (row_pos[half] >= col_pos[c]) {
        atomicAdd(column + row_pos[half], -as[j][lane + 32 * half]);
      }
    }
  }
}

/**
 * Backward substitution, one CTA per (panel, 64 below rows) of a level
 * (levels in reverse; a root was solved by its factor task): each CTA sums
 * its rows' L_b[rows]ᵀ x[rows] in shared memory, then solves
 * x_p = L_D^{-T} (z_p - sum). The CTAs of a panel taller than one tile add
 * their sums up in `accumulator`, and the last one solves (and resets the
 * accumulator and counter). L_D does not depend on the solution, so a
 * single-tile panel loads it before anything else.
 */
__global__ void __launch_bounds__(kThreads) BackwardKernel(Args args, const Task *tasks, int) {
  __shared__ float ld[kW][kPad];
  __shared__ float v[kW];  // L_b[rows]ᵀ x[rows], then the panel's solution
  __shared__ bool last;
  const Task task = tasks[blockIdx.x];
  const Panel panel = args.panels[task.index];
  const int col = panel.col, w = panel.width, h = panel.height;
  const int t = threadIdx.x, lane = t % 32;
  float *x = args.x;
  if (!PanelActive<false>(panel, args)) {
    // Idle panel (not factored): a zero solution, so nothing stale is read.
    if (task.row_begin == 0 && t < w) x[col + t] = 0.f;
    return;
  }
  const int below = min(kR, h - w - task.row_begin);
  const int tiles = (h - w + kR - 1) / kR;  // a root (no rows below) has no backward task
  if (tiles == 1) LoadDiagonal(args.diagonal + panel.diagonal_offset, w, ld);
  if (t < kW) v[t] = 0.f;
  {
    const float *l = args.factor + panel.offset + w + task.row_begin;
    const int r = t % kR, q = t / kR;
    const int row = r < below ? args.below_rows[panel.rows_offset + task.row_begin + r] : -1;
    constexpr int kPerThread = kW / (kThreads / kR);
    float lr[kPerThread];
#pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
      const int k = q + m * (kThreads / kR);
      lr[m] = r < below && k < w ? l[k * h + r] : 0.f;
    }
    AwaitPredecessor();
    const float xr = row >= 0 ? x[row] : 0.f;
    __syncthreads();
#pragma unroll
    for (int m = 0; m < kPerThread; ++m) {
      const int k = q + m * (kThreads / kR);
      float acc = lr[m] * xr;
#pragma unroll
      for (int s = 16; s > 0; s /= 2) acc += __shfl_xor_sync(kFullMask, acc, s);
      if (lane == 0 && k < w) atomicAdd(v + k, acc);
    }
  }
  __syncthreads();
  if (tiles > 1) {
    if (t < w) atomicAdd(args.accumulator + col + t, v[t]);
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(args.counter + task.index, 1) == tiles - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    LoadDiagonal(args.diagonal + panel.diagonal_offset, w, ld);
    if (t < w) {
      v[t] = __ldcg(args.accumulator + col + t);
      args.accumulator[col + t] = 0.f;
    }
    if (t == 0) args.counter[task.index] = 0;
    __syncthreads();
  }
  StartSuccessor();
  if (t < w) v[t] = x[col + t] - v[t];
  __syncthreads();
  if (t < 32) WarpTriangularSolve<true>(ld, w, v);
  __syncthreads();
  if (t < w) x[col + t] = v[t];
}

/// Small panels (or runs) per CTA of the warp kernels below, one warp each.
constexpr int kWarpsPerBlock = 4;
static_assert(internal::kCholeskySmallPanelRows == 32, "the warp kernels hold a row per lane");

unsigned WarpBlocks(int count) {
  return static_cast<unsigned>((count + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

/**
 * Factors one small panel of at most W columns with one warp (the warp version of
 * FactorKernel). Lane i holds row i of the stacked panel [D; B] (diagonal block, then the
 * rows below it) from the current column on, shifted down after every column, so the
 * column loop stays rolled (a fully unrolled loop nest is bound by instruction fetch when
 * few warps run it). Column j takes its pivot from lane j and the multipliers L(j + l, j)
 * by shuffles, then every row takes the rank-1 update, predicated rather than branched so
 * the shuffles issue back to back. The lanes of the diagonal-block rows also hold the
 * panel's part y_p of the forward substitution and update it like a column:
 * z_j = y_j / L(j, j), y_i -= L(i, j) z_j. The zero-pivot test compares against A's
 * diagonal: from `original`, or from the panel itself for a leaf (no update reached it).
 *
 * A panel with rows below stores each column's results as soon as they are final: its
 * column of L_D (column-major for small panels, one coalesced store; the reciprocal on the
 * diagonal), of B L_D^{-T} (in place) and z_j; at the end y[rows of B] -= (B L_D^{-T}) z_p.
 * A root (no rows below) needs neither: its columns of L go to the warp's shared memory
 * `l_root` (W x (W + 1), row k at k (W + 1): conflict-free by row and by column), and the
 * warp solves L_Dᵀ x = z_p at the end, writing x instead of z (its backward substitution
 * needs nothing from other panels, so none follows).
 */
template <int W>
__device__ void FactorPanelByWarp(const Panel &info, const Args &args, float *l_root) {
  const int col = info.col, w = info.width, h = info.height;
  const bool root = h == w;
  const int lane = threadIdx.x % 32;  // the row
  float *panel = args.factor + info.offset;
  float *z = args.x;
  float s[W];  // the lane's row from column j on
#pragma unroll
  for (int c = 0; c < W; ++c) {
    s[c] = c < w && lane < h ? panel[static_cast<int64_t>(c) * h + lane] : 0.f;
  }
  // A's diagonal entry of the lane's row: in the panel itself while no update has reached it.
  float a_ii = 0.f;
  if (info.leaf) {
#pragma unroll
    for (int c = 0; c < W; ++c) a_ii = c == lane ? s[c] : a_ii;
  } else if (lane < w) {
    a_ii = args.original[col + lane];
  }
  const float limit = kZeroPivot * a_ii;
  float y_own = lane < w ? args.y[col + lane] : 0.f;
  float below_dot = 0.f;  // a row below: its row of B L_D^{-T} times z_p
  float own_scale = 0.f, own_z = 0.f;
  bool failed = false;
#pragma unroll 1
  for (int j = 0; j < w; ++j) {
    const float pivot = __shfl_sync(kFullMask, s[0], j);
    const bool zero = fabsf(pivot) <= __shfl_sync(kFullMask, limit, j);
    float scale;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(pivot));
    if (zero) scale = 0.f;  // a rank deficiency fixes the unknown (zero column)
    const float zj = __shfl_sync(kFullMask, y_own, j) * scale;
    const float lij = lane > j ? s[0] * scale : 0.f;
    if (lane == j) {
      failed = !zero && (!(pivot > 0.f) || !isfinite(pivot));
      own_scale = scale;
      own_z = zj;
    }
    if (lane < w) y_own -= lij * zj;
    if (root) {
      if (lane < W) l_root[lane * (W + 1) + j] = lij;  // L(lane, j), 0 for lane <= j
    } else {
      if (lane < w) {
        float *d = args.diagonal + info.diagonal_offset;
        d[j * w + lane] = lane > j ? lij : lane == j ? scale : 0.f;  // column j, coalesced
      } else if (lane < h) {
        panel[static_cast<int64_t>(j) * h + lane] = lij;
        below_dot += lij * zj;
      }
      if (lane == 0) z[col + j] = zj;
    }
#pragma unroll
    for (int l = 1; l < W; ++l) {
      const float llj = __shfl_sync(kFullMask, lij, (j + l) & 31);  // L(j + l, j)
      if (j + l < w && lane >= j + l) s[l] -= lij * llj;
    }
#pragma unroll
    for (int l = 1; l < W; ++l) s[l - 1] = s[l];
  }
  if (failed) *args.failure = 1;
  if (root) {
    // L_Dᵀ x = z_p, lane c for x_c (L(k, c) from l_root; columns past the width never read).
    __syncwarp();
    float t = own_z, own_x = 0.f;
#pragma unroll
    for (int k = W - 1; k >= 0; --k) {
      const float xk = __shfl_sync(kFullMask, t * own_scale, k);
      if (lane == k) own_x = xk;
      if (lane < k && k < w) t -= l_root[k * (W + 1) + lane] * xk;
    }
    if (lane < w) z[col + lane] = own_x;
  } else if (lane >= w && lane < h) {
    atomicAdd(args.y + args.below_rows[info.rows_offset + lane - w], -below_dot);
  }
}

/**
 * FactorKernel for small panels, one warp each (FactorPanelByWarp, sized by the panel's width
 * up to W, the widest of the launch: the kernel's registers): an idle panel (row_active) is
 * skipped, an idle root gets its zero solution.
 */
template <int W>
__global__ void __launch_bounds__(32 * kWarpsPerBlock)
    FactorKernelByWarp(Args args, const Task *tasks, int count) {
  __shared__ float l_root[kWarpsPerBlock][W * (W + 1)];  // a root's columns of L, per warp
  const int item = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
  if (item >= count) return;  // the whole warp
  const Panel info = args.panels[tasks[item].index];
  if (!PanelActive<true>(info, args)) {
    const int lane = threadIdx.x % 32;
    if (info.height == info.width && lane < info.width) args.x[info.col + lane] = 0.f;
    return;
  }
  AwaitPredecessor();
  float *own = l_root[threadIdx.x / 32];
  if (W == 8 || info.width <= 8) {
    FactorPanelByWarp<8>(info, args, own);
  } else if (W == 16 || info.width <= 16) {
    FactorPanelByWarp<16>(info, args, own);
  } else {
    FactorPanelByWarp<32>(info, args, own);
  }
  StartSuccessor();
}

/**
 * The update of one run of a small panel of at most W columns (the warp version of
 * UpdateKernel): lane i holds below row i of the source's L_b and its target position;
 * per column of the run, that row's L_b and position come by shuffles (no memory access
 * in the loop), and every row at or after the run subtracts its product from its target
 * entry (the lower triangle of the target's diagonal block only), by a reduction as in
 * UpdateKernel. L_b is zero past the panel's width, so the products need no guard.
 */
template <int W>
__device__ void UpdateRunByWarp(const Run &run, const Args &args) {
  const Panel source = args.panels[run.source];
  const int w = source.width, h = source.height, below = h - w;
  const int lane = threadIdx.x % 32;  // the below row
  // Target position of the lane's row (relmap from the run's first row on), -1 before it.
  const int row_pos =
      lane >= run.begin && lane < below ? args.relmap[run.relmap_offset + lane - run.begin] : -1;
  const Panel target = args.panels[run.target];
  float *dst = args.factor + target.offset;
  AwaitPredecessor();
  const float *src = args.factor + source.offset + w;
  float lb[W];
#pragma unroll
  for (int k = 0; k < W; ++k) {
    lb[k] = k < w && lane < below ? src[static_cast<int64_t>(k) * h + lane] : 0.f;
  }
  for (int j = run.begin; j < run.end; ++j) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < W; ++k) acc += lb[k] * __shfl_sync(kFullMask, lb[k], j);
    const int col_pos = __shfl_sync(kFullMask, row_pos, j);
    if (row_pos >= col_pos) {
      atomicAdd(dst + static_cast<int64_t>(col_pos) * target.height + row_pos, -acc);
    }
  }
}

/**
 * UpdateKernel for the runs of small panels, one warp per run (UpdateRunByWarp, sized by the
 * source's width up to W, the widest of the launch).
 */
template <int W>
__global__ void __launch_bounds__(32 * kWarpsPerBlock)
    UpdateKernelByWarp(Args args, const Task *tasks, int count) {
  const int item = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
  if (item >= count) return;  // the whole warp
  const Run run = args.runs[tasks[item].index];
  const Panel source = args.panels[run.source];
  // An idle source updates only panels of its own (idle) subproblem.
  if (!PanelActive<true>(source, args)) return;
  if (W == 8 || source.width <= 8) {
    UpdateRunByWarp<8>(run, args);
  } else if (W == 16 || source.width <= 16) {
    UpdateRunByWarp<16>(run, args);
  } else {
    UpdateRunByWarp<32>(run, args);
  }
  StartSuccessor();
}

/**
 * The backward substitution of one small panel of at most W columns (the warp version
 * of BackwardKernel, no accumulator: the warp holds the whole panel). Lane c loads column
 * c of L_b and of L_D and the lanes the solution at the rows below, all before any
 * arithmetic; then lane c takes z_c - L_b[:, c]ᵀ x[rows below], and the warp solves with
 * L_Dᵀ by shuffles. L_b and L_D are zero past the panel's rows and width, so neither step
 * needs a guard.
 */
template <int W>
__device__ void BackwardPanelByWarp(const Panel &info, const Args &args) {
  const int col = info.col, w = info.width, h = info.height, below = h - w;
  const int lane = threadIdx.x % 32;  // the column
  float *x = args.x;
  const int row = lane < below ? args.below_rows[info.rows_offset + lane] : -1;
  const bool own = lane < w;
  const float *lb = args.factor + info.offset + static_cast<int64_t>(own ? lane : 0) * h + w;
  const float *d = args.diagonal + info.diagonal_offset + (own ? lane : 0) * w;  // column-major
  float l_col[32 - 1];  // L_b(r, lane): at most 31 rows below (w >= 1)
  float d_col[W];       // L_D(k, lane), 1 / L_D at k = lane
#pragma unroll
  for (int r = 0; r < 32 - 1; ++r) l_col[r] = own && r < below ? lb[r] : 0.f;
#pragma unroll
  for (int k = 0; k < W; ++k) d_col[k] = own && k < w ? d[k] : 0.f;
  AwaitPredecessor();
  const float x_below = row >= 0 ? x[row] : 0.f;
  float t = own ? x[col + lane] : 0.f;
#pragma unroll
  for (int r = 0; r < 32 - 1; ++r) t -= l_col[r] * __shfl_sync(kFullMask, x_below, r);
  float own_x = 0.f;
#pragma unroll
  for (int k = W - 1; k >= 0; --k) {
    const float xk = __shfl_sync(kFullMask, t * d_col[k], k);
    if (lane == k) own_x = xk;
    if (lane < k) t -= d_col[k] * xk;
  }
  if (own) x[col + lane] = own_x;
}

/**
 * BackwardKernel for small panels with rows below, one warp each (BackwardPanelByWarp, sized
 * by the panel's width up to W, the widest of the launch).
 */
template <int W>
__global__ void __launch_bounds__(32 * kWarpsPerBlock)
    BackwardKernelByWarp(Args args, const Task *tasks, int count) {
  const int item = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
  if (item >= count) return;  // the whole warp
  const Panel info = args.panels[tasks[item].index];
  if (!PanelActive<true>(info, args)) {
    // Idle panel (not factored): a zero solution.
    if (threadIdx.x % 32 < info.width) args.x[info.col + threadIdx.x % 32] = 0.f;
    return;
  }
  if (W == 8 || info.width <= 8) {
    BackwardPanelByWarp<8>(info, args);
  } else if (W == 16 || info.width <= 16) {
    BackwardPanelByWarp<16>(info, args);
  } else {
    BackwardPanelByWarp<32>(info, args);
  }
  StartSuccessor();
}

using KernelFn = void (*)(Args, const Task *, int);

/** The kernel of a launch: its phase and executor (a CTA per task, or a warp of a width). */
KernelFn KernelOf(const Launch &launch) {
  const int w = launch.warp_width;
  switch (launch.phase) {
    case Phase::kFactor:
      return w == 0    ? FactorKernel
             : w == 8  ? FactorKernelByWarp<8>
             : w == 16 ? FactorKernelByWarp<16>
                       : FactorKernelByWarp<32>;
    case Phase::kUpdate:
      return w == 0    ? UpdateKernel
             : w == 8  ? UpdateKernelByWarp<8>
             : w == 16 ? UpdateKernelByWarp<16>
                       : UpdateKernelByWarp<32>;
    case Phase::kBackward:
      return w == 0    ? BackwardKernel
             : w == 8  ? BackwardKernelByWarp<8>
             : w == 16 ? BackwardKernelByWarp<16>
                       : BackwardKernelByWarp<32>;
  }
  return nullptr;
}

/** The supervariables of a symmetric pattern and their adjacency (no self loops). */
struct SupervariableGraph {
  std::vector<int> start{0};              ///< Per supervariable: first row; plus n.
  std::vector<int> adjacency_offsets{0};  ///< Per supervariable: its neighbors [offset, next).
  std::vector<int> adjacency;             ///< Neighbor supervariables.
};

/**
 * Finds the supervariables (consecutive rows with identical patterns) and
 * their graph on the device; only the graph comes to the host.
 */
SupervariableGraph FindSupervariableGraph(cudaStream_t stream, const CSRSparseMatrix &matrix) {
  SupervariableGraph graph;
  const size_t n = matrix.NumRows();
  if (n == 0) return graph;
  const int *ro = matrix.row_offsets.data(), *ci = matrix.col_ids.data();
  dvector<int> flag(n);
  SupervariableKernel<<<Blocks(32 * n), kThreads, 0, stream>>>(static_cast<int>(n), ro, ci,
                                                               flag.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  std::vector<int> h_flag(n), h_supervariable(n);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  flag.CopyToHost(h_flag.data(), n);
  for (size_t i = 0; i < n; ++i) {
    if (i > 0 && h_flag[i]) graph.start.push_back(static_cast<int>(i));
    h_supervariable[i] = static_cast<int>(graph.start.size()) - 1;
  }
  const int nb = static_cast<int>(graph.start.size());
  graph.start.push_back(static_cast<int>(n));
  dvector<int> supervariable(h_supervariable);
  dvector<int> first_row(std::vector<int>(graph.start.begin(), graph.start.end() - 1)), count(nb);
  AdjacencyKernel<<<Blocks(32 * static_cast<size_t>(nb)), kThreads, 0, stream>>>(
      nb, first_row.data(), ro, ci, supervariable.data(), nullptr, count.data(), nullptr);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  std::vector<int> h_count(nb);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  count.CopyToHost(h_count.data(), nb);
  for (int v = 0; v < nb; ++v) {
    graph.adjacency_offsets.push_back(graph.adjacency_offsets.back() + h_count[v]);
  }
  graph.adjacency.resize(graph.adjacency_offsets.back());
  if (!graph.adjacency.empty()) {
    dvector<int> offsets(graph.adjacency_offsets), d_adjacency(graph.adjacency.size());
    AdjacencyKernel<<<Blocks(32 * static_cast<size_t>(nb)), kThreads, 0, stream>>>(
        nb, first_row.data(), ro, ci, supervariable.data(), offsets.data(), nullptr,
        d_adjacency.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    d_adjacency.CopyToHost(graph.adjacency.data(), graph.adjacency.size());
  }
  return graph;
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
  const SupervariableGraph graph = FindSupervariableGraph(stream, spd_matrix);
  plan_ = internal::AnalyzeCholesky(static_cast<int>(n), graph.start, graph.adjacency_offsets,
                                    graph.adjacency);
  const auto &plan = plan_;
  const size_t np = plan.panels.size();
  perm_ = dvector<int>(plan.perm);
  panels_ = dvector<Panel>(plan.panels);
  below_rows_ = dvector<int>(plan.below_rows);
  runs_ = dvector<Run>(plan.runs);
  tasks_ = dvector<Task>(plan.tasks);

  // Where each nonzero of A goes in the factor, and the target positions of the updates.
  // 32-bit factor offsets unless the factor is larger (then 64-bit, twice the map's memory).
  wide_assembly_ = plan.factor_size > INT_MAX;
  assembly_dst_.resize(wide_assembly_ ? 0 : spd_matrix.col_ids.size());
  assembly_dst64_.resize(wide_assembly_ ? spd_matrix.col_ids.size() : 0);
  if (n > 0) {
    dvector<int> diagonal_index(n);  // per permuted column: its diagonal nonzero of A, or -1
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(diagonal_index.data(), 0xFF, n * sizeof(int), stream));
    std::vector<int> iperm(n), col_panel(n);
    for (size_t c = 0; c < n; ++c) iperm[plan.perm[c]] = static_cast<int>(c);
    for (size_t p = 0; p < np; ++p) {
      const auto &panel = plan.panels[p];
      std::fill(col_panel.begin() + panel.col, col_panel.begin() + panel.col + panel.width,
                static_cast<int>(p));
    }
    dvector<int> d_iperm(iperm), d_col_panel(col_panel);
    if (wide_assembly_) {
      AssemblyMapKernel<<<Blocks(32 * n), kThreads, 0, stream>>>(
          static_cast<int>(n), spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(),
          d_iperm.data(), d_col_panel.data(), panels_.data(), below_rows_.data(),
          assembly_dst64_.data(), diagonal_index.data());
    } else {
      AssemblyMapKernel<<<Blocks(32 * n), kThreads, 0, stream>>>(
          static_cast<int>(n), spd_matrix.row_offsets.data(), spd_matrix.col_ids.data(),
          d_iperm.data(), d_col_panel.data(), panels_.data(), below_rows_.data(),
          assembly_dst_.data(), diagonal_index.data());
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    // Where A's diagonal entry of each column of an updated panel lands (the zero-pivot test
    // reads a copy; a leaf panel reads its own). None: no copy at all.
    const bool updated = std::any_of(plan.panels.begin(), plan.panels.end(),
                                     [](const Panel &panel) { return !panel.leaf; });
    diagonal_slot_.resize(updated ? n : 0);
    if (updated && wide_assembly_) {
      DiagonalSlotKernel<<<Blocks(n), kThreads, 0, stream>>>(
          static_cast<int>(n), diagonal_index.data(), assembly_dst64_.data(), d_col_panel.data(),
          panels_.data(), diagonal_slot_.data());
    } else if (updated) {
      DiagonalSlotKernel<<<Blocks(n), kThreads, 0, stream>>>(
          static_cast<int>(n), diagonal_index.data(), assembly_dst_.data(), d_col_panel.data(),
          panels_.data(), diagonal_slot_.data());
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));  // before the temporaries go
  }
  relmap_.resize(plan.relmap_size);
  if (!plan.runs.empty()) {
    RelmapKernel<<<Blocks(32 * plan.runs.size()), kThreads, 0, stream>>>(
        static_cast<int>(plan.runs.size()), runs_.data(), panels_.data(), below_rows_.data(),
        relmap_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  factor_.resize(plan.factor_size);
  diagonal_.resize(plan.diagonal_size);
  original_.resize(n);
  work_.resize(n);
  solution_.resize(n);
  accumulator_.resize(n);
  counter_.resize(np);
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(accumulator_.data(), 0, n * sizeof(float), stream));
  }
  if (np > 0) THROW_ON_CUDA_ERROR(cudaMemsetAsync(counter_.data(), 0, np * sizeof(int), stream));
  LogMessage(
      "BlockSparseCholesky: n {}, {} supervariables, {} ordering, {} supernodes, {} panels ({} "
      "small), {} levels, {} launches, factor {:.2f} M entries, relative map {:.2f} M entries, "
      "{:.3f} GFLOP",
      n, plan.num_supervariables, plan.ordering, plan.num_supernodes, np, plan.num_small_panels,
      plan.num_levels, plan.program.size(), plan.factor_size / 1e6, plan.relmap_size / 1e6,
      plan.flops / 1e9);
  return true;
}

SolverMatrixStorage BlockSparseCholeskySolver::MatrixStorage() {
  if (wide_assembly_) return {factor_.data(), nullptr, assembly_dst64_.data()};
  return {factor_.data(), assembly_dst_.data(), nullptr};
}

void BlockSparseCholeskySolver::ClearMatrixStorage(cudaStream_t stream) {
  if (factor_.size() == 0) return;
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(factor_.data(), 0, factor_.size() * sizeof(float), stream));
}

bool BlockSparseCholeskySolver::SolveStoredMatrix(cudaStream_t stream, const dvector<float> &rhs,
                                                  dvector<float> &result) {
  const auto &plan = plan_;
  const int n = plan.n;
  if (n == 0) return true;
  if (safety_checks_) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(failure_.data(), 0, sizeof(int), stream));
  }
  // The permuted right-hand side, and A's diagonal (zero-pivot test) before any update.
  PermuteKernel<<<Blocks(n), kThreads, 0, stream>>>(
      n, perm_.data(), rhs.data(), work_.data(),
      diagonal_slot_.size() > 0 ? diagonal_slot_.data() : nullptr, factor_.data(),
      original_.data());
  // The plan's program: factorization and forward substitution level by level, then the
  // backward substitution (see internal::CholeskyPlan).
  const Args args{panels_.data(),   below_rows_.data(), runs_.data(),     relmap_.data(),
                  perm_.data(),     row_active_,        original_.data(), factor_.data(),
                  diagonal_.data(), work_.data(),       solution_.data(), accumulator_.data(),
                  counter_.data(),  failure_.data()};
  // Each kernel's prologue may overlap its predecessor (AwaitPredecessor; sm_90 and newer).
  cudaLaunchAttribute overlap = {};
  overlap.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  overlap.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.stream = stream;
  config.attrs = &overlap;
  config.numAttrs = 1;
  for (const Launch &launch : plan.program) {
    const bool warp = launch.warp_width != 0;
    config.gridDim = dim3(warp ? WarpBlocks(launch.num_tasks) : launch.num_tasks);
    config.blockDim = dim3(warp ? 32 * kWarpsPerBlock : kThreads);
    THROW_ON_CUDA_ERROR(cudaLaunchKernelEx(&config, KernelOf(launch), args,
                                           tasks_.data() + launch.first_task, launch.num_tasks));
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
