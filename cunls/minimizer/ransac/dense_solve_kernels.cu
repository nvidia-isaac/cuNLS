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
 * @file dense_solve_kernels.cu
 * @brief Per-slot damped dense solve (dim <= 64).
 *
 * A group of threads owns one slot: a warp for dim <= 32, a 128-thread block
 * above that. The system is Jacobi-equilibrated, factored as
 * P S A S P^T = L D L^T (symmetric diagonal pivoting for LDLT; no pivoting and
 * positive pivots for Cholesky), and solved with column sweeps. Trailing
 * updates are element-wise over the group; the pivot search and the
 * triangular substitutions run on one warp (dim <= 64) to avoid block syncs.
 */

#include <algorithm>
#include <climits>
#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/minimizer/ransac/kernel_common.cuh"

namespace cunls {
namespace ransac_internal {

namespace {

constexpr int kBlockThreads = 128;  ///< Block groups (dim > 32): fewer threads, cheaper syncs.
constexpr int kMaxWarpsPerBlock = 8;  ///< Warp groups per block (and warps of a block group).
constexpr int kWarpDimLimit = 32;         ///< Largest dim solved by a warp group.
constexpr int kMaxWordsPerBlock = 12288;  ///< 48 KiB of shared memory.
constexpr float kPivotTolerance = 1e-6f;  ///< On the equilibrated matrix (unit diagonal).

template <int kGroup>
__device__ __forceinline__ void GroupSync() {
  if (kGroup == kWarpSize) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

template <int kGroup>
__device__ __forceinline__ bool GroupAll(bool predicate) {
  return kGroup == kWarpSize ? __all_sync(kFullMask, predicate) != 0
                             : __syncthreads_and(predicate) != 0;
}

/** Group sum; block groups reduce the per-warp sums in warp order (deterministic). */
template <int kGroup>
__device__ float GroupSum(float value, float *scratch) {
  return kGroup == kWarpSize ? WarpSum(value) : BlockSum(value, scratch);
}

/** One group's shared-memory workspace. */
struct Workspace {
  float *a;  ///< dim x ld matrix (ld = dim + 1 avoids bank conflicts).
  float *rhs;
  float *x;
  float *d;
  float *scale;
  int *perm;
  int ld;
};

__host__ __device__ inline int WordsPerGroup(int dim) { return dim * (dim + 1) + 5 * dim; }

__device__ inline Workspace MapWorkspace(float *base, int dim) {
  Workspace w;
  w.ld = dim + 1;
  w.a = base;
  w.rhs = w.a + dim * w.ld;
  w.x = w.rhs + dim;
  w.d = w.x + dim;
  w.scale = w.d + dim;
  w.perm = reinterpret_cast<int *>(w.scale + dim);
  return w;
}

/**
 * Loads H and g, applies damping and equilibrates: A = S (H + damping) S,
 * rhs = S g with S = diag(A)^-1/2 (0 for non-positive diagonals).
 * @return false if the system is not finite.
 */
template <int kGroup>
__device__ bool LoadSystem(const Workspace &w, int dim, const float *h, const float *g,
                           const float *lambda, int t) {
  bool finite = true;
  const int lane = t % kWarpSize;
  for (int i = t / kWarpSize; i < dim; i += kGroup / kWarpSize) {
    for (int j = lane; j < dim; j += kWarpSize) {
      const float v = h[i * dim + j];
      finite = finite && isfinite(v);
      w.a[i * w.ld + j] = v;
    }
  }
  GroupSync<kGroup>();
  for (int i = t; i < dim; i += kGroup) {
    float di = w.a[i * w.ld + i];
    di = lambda != nullptr ? di + *lambda * fmaxf(di, 1e-6f) : di * (1.f + 1e-6f);
    w.a[i * w.ld + i] = di;
    w.scale[i] = (di > 0.f && isfinite(di)) ? rsqrtf(di) : 0.f;
    w.rhs[i] = g[i] * w.scale[i];
    w.perm[i] = i;
    finite = finite && isfinite(g[i]);
  }
  GroupSync<kGroup>();
  for (int i = t / kWarpSize; i < dim; i += kGroup / kWarpSize) {
    for (int j = lane; j < dim; j += kWarpSize) {
      w.a[i * w.ld + j] *= w.scale[i] * w.scale[j];
    }
  }
  return GroupAll<kGroup>(finite);  // also synchronizes the group
}

/**
 * Row in [k, dim) with the largest |diagonal| (ties to the lower row),
 * computed by one warp. Valid in every lane of that warp.
 */
__device__ int WarpPivotRow(const Workspace &w, int dim, int k, int lane) {
  float best = INFINITY;
  int row = INT_MAX;
  for (int i = k + lane; i < dim; i += kWarpSize) {
    const float v = -fabsf(w.a[i * w.ld + i]);
    if (v < best) {
      best = v;
      row = i;
    }
  }
  WarpArgMin(best, row);
  return row;
}

/**
 * Symmetric swap of rows / columns k and p in one pass: rows k, p swap every
 * column j (L entries included), columns k, p swap at every other row, the two
 * diagonal entries swap, and the symmetric pair a[k][p] = a[p][k] stays.
 */
template <int kGroup>
__device__ void SymmetricSwap(const Workspace &w, int dim, int k, int p, int t) {
  for (int j = t; j < dim; j += kGroup) {
    float *rk = w.a + k * w.ld;
    float *rp = w.a + p * w.ld;
    if (j == k) {
      const float tmp = rk[k];
      rk[k] = rp[p];
      rp[p] = tmp;
      const int pk = w.perm[k];
      w.perm[k] = w.perm[p];
      w.perm[p] = pk;
    } else if (j != p) {
      float tmp = rk[j];
      rk[j] = rp[j];
      rp[j] = tmp;
      float *rj = w.a + j * w.ld;
      tmp = rj[k];
      rj[k] = rj[p];
      rj[p] = tmp;
    }
  }
}

/**
 * In-place L D L^T. L is stored below the diagonal, D in w.d. Three group
 * syncs per column: after the swap, after the trailing update, and after
 * scaling the column (warp 0 finds the next pivot meanwhile).
 * @return the numerical rank (LDLT), or -1 when Cholesky meets a non-positive pivot.
 */
template <int kGroup>
__device__ int Factorize(const Workspace &w, int dim, bool pivoting, int t, int *s_pivot) {
  if (pivoting && t < kWarpSize) {
    const int p = WarpPivotRow(w, dim, 0, t);
    if (t == 0) *s_pivot = p;
  }
  GroupSync<kGroup>();
  for (int k = 0; k < dim; ++k) {
    if (pivoting && *s_pivot != k) {
      SymmetricSwap<kGroup>(w, dim, k, *s_pivot, t);
      GroupSync<kGroup>();
    }
    const float piv = w.a[k * w.ld + k];
    if (pivoting ? !(fabsf(piv) > kPivotTolerance) : !(piv > kPivotTolerance)) {
      return pivoting ? k : -1;
    }
    const float inv = 1.f / piv;
    const float *row_k = w.a + k * w.ld;
    const int lane = t % kWarpSize;
#pragma unroll 4
    for (int i = k + 1 + t / kWarpSize; i < dim; i += kGroup / kWarpSize) {  // warps over rows
      float *row_i = w.a + i * w.ld;
      const float l = row_i[k] * inv;
      for (int j = k + 1 + lane; j < dim; j += kWarpSize) {  // lanes over columns
        row_i[j] -= l * row_k[j];
      }
    }
    GroupSync<kGroup>();
    for (int i = k + 1 + t; i < dim; i += kGroup) {
      w.a[i * w.ld + k] *= inv;
    }
    if (t == 0) {
      w.d[k] = piv;
    }
    if (pivoting && k + 1 < dim && t < kWarpSize) {
      const int p = WarpPivotRow(w, dim, k + 1, t);
      if (t == 0) *s_pivot = p;
    }
    GroupSync<kGroup>();
  }
  return dim;
}

/**
 * Solves with the factorization on one warp (dim <= 64: two rows per lane);
 * the unscaled, unpermuted step ends up in w.rhs.
 */
template <int kGroup>
__device__ void Substitute(const Workspace &w, int dim, int rank, int t) {
  if (t < kWarpSize) {
    const int lane = t;
    for (int i = lane; i < dim; i += kWarpSize) w.x[i] = w.rhs[w.perm[i]];
    __syncwarp();
    for (int k = 0; k < rank; ++k) {  // L y = P b
      const float yk = w.x[k];
      for (int i = k + 1 + lane; i < rank; i += kWarpSize) w.x[i] -= w.a[i * w.ld + k] * yk;
      __syncwarp();
    }
    for (int i = lane; i < dim; i += kWarpSize) {  // D z = y; the null space gets 0
      w.x[i] = i < rank ? w.x[i] / w.d[i] : 0.f;
    }
    __syncwarp();
    for (int k = rank - 1; k >= 0; --k) {  // L^T x = z
      const float xk = w.x[k];
      for (int i = lane; i < k; i += kWarpSize) w.x[i] -= w.a[k * w.ld + i] * xk;
      __syncwarp();
    }
    for (int i = lane; i < dim; i += kWarpSize) {
      const int j = w.perm[i];
      w.rhs[j] = w.x[i] * w.scale[j];
    }
  }
  GroupSync<kGroup>();
}

/** Where one slot's results go. */
struct SlotOutputs {
  float *delta;
  float *predicted;
  float *step_sq;
  int *solve_ok;
};

/** Writes the step, its predicted decrease (undamped H) and |step|^2. */
template <int kGroup>
__device__ void WriteStep(const Workspace &w, int dim, bool ok, const float *h, const float *g,
                          const SlotOutputs &out, int t, float *scratch) {
  float gx = 0.f, xhx = 0.f, sq = 0.f;
  bool finite = true;
  for (int i = t; i < dim; i += kGroup) {
    const float xi = ok ? w.rhs[i] : 0.f;
    float hx = 0.f;
    for (int j = 0; j < dim; ++j) {
      hx = fmaf(h[i * dim + j], ok ? w.rhs[j] : 0.f, hx);
    }
    gx = fmaf(g[i], xi, gx);
    xhx = fmaf(xi, hx, xhx);
    sq = fmaf(xi, xi, sq);
    finite = finite && isfinite(xi);
  }
  ok = ok && GroupAll<kGroup>(finite);
  gx = GroupSum<kGroup>(gx, scratch);
  xhx = GroupSum<kGroup>(xhx, scratch);
  sq = GroupSum<kGroup>(sq, scratch);
  for (int i = t; i < dim; i += kGroup) {
    out.delta[i] = ok ? w.rhs[i] : 0.f;
  }
  if (t == 0) {
    *out.predicted = ok ? gx - 0.5f * xhx : 0.f;
    *out.step_sq = ok ? sq : 0.f;
    *out.solve_ok = ok ? 1 : 0;
  }
}

template <int kGroup>
__global__ void SolveKernel(int num_slots, int dim, int solver, const float *hessian,
                            const float *gradient, const float *lambda, const int *active,
                            float *delta, float *predicted, float *step_sq, int *solve_ok) {
  extern __shared__ float smem[];
  __shared__ float s_scratch[kMaxWarpsPerBlock];  // block-group reductions
  __shared__ int s_pivot[kMaxWarpsPerBlock];      // one per group
  const int group = threadIdx.x / kGroup;
  const int t = threadIdx.x % kGroup;
  const int slot = blockIdx.x * (blockDim.x / kGroup) + group;
  if (slot >= num_slots) {
    return;  // only warp groups can be out of range; they never sync with others
  }
  const float *h = hessian + static_cast<size_t>(slot) * dim * dim;
  const float *g = gradient + static_cast<size_t>(slot) * dim;
  const Workspace w = MapWorkspace(smem + group * WordsPerGroup(dim), dim);
  const bool is_active = active == nullptr || active[slot] != 0;  // uniform per group

  bool ok = is_active && LoadSystem<kGroup>(w, dim, h, g,
                                            lambda != nullptr ? lambda + slot : nullptr, t);
  if (ok) {
    const int rank = Factorize<kGroup>(w, dim, solver == kSolveLDLT, t, s_pivot + group);
    ok = rank >= 0;
    if (ok) {
      Substitute<kGroup>(w, dim, rank, t);
    }
  }
  const SlotOutputs out{delta + static_cast<size_t>(slot) * dim, predicted + slot, step_sq + slot,
                        solve_ok + slot};
  WriteStep<kGroup>(w, dim, ok, h, g, out, t, s_scratch);
  if (!is_active && t == 0) {
    solve_ok[slot] = 1;  // nothing was solved, nothing failed
  }
}

}  // namespace

void LaunchSolve(cudaStream_t stream, int num_slots, int dim, SolverKind solver,
                 const float *hessian, const float *gradient, const float *lambda,
                 const int *active, float *delta, float *predicted, float *step_sq,
                 int *solve_ok) {
  if (num_slots <= 0) {
    return;
  }
  const int words = WordsPerGroup(dim);
  if (dim <= kWarpDimLimit) {
    const int warps = std::max(1, std::min(kMaxWarpsPerBlock, kMaxWordsPerBlock / words));
    const size_t smem = static_cast<size_t>(warps) * words * sizeof(float);
    SolveKernel<kWarpSize><<<GridFor(num_slots, warps), warps * kWarpSize, smem, stream>>>(
        num_slots, dim, solver, hessian, gradient, lambda, active, delta, predicted, step_sq,
        solve_ok);
  } else {
    const size_t smem = static_cast<size_t>(words) * sizeof(float);
    SolveKernel<kBlockThreads><<<num_slots, kBlockThreads, smem, stream>>>(
        num_slots, dim, solver, hessian, gradient, lambda, active, delta, predicted, step_sq,
        solve_ok);
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace ransac_internal
}  // namespace cunls
