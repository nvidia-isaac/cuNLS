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
 * @file block_sparse_pcg_solver.cu
 *
 * Implementation of the block-Jacobi preconditioned conjugate gradient
 * solver declared in @ref block_sparse_pcg_solver.h.
 *
 * ## Math summary
 *
 * Given a symmetric positive (semi-)definite matrix `H ∈ R^{N×N}` in CSR
 * format and a right-hand side `b ∈ R^N`, the algorithm computes a
 * sequence of approximations `x_k` minimizing the H-norm error
 * `||x - x_*||_H` over the Krylov subspace
 * `K_k(M^{-1} H, M^{-1} r_0)` where `M` is the block-Jacobi
 * preconditioner.
 *
 * Block-Jacobi preconditioner.  Define a partition of the index set
 * `{0,...,N-1}` into contiguous diagonal blocks `B_1, B_2, ...`.  The
 * preconditioner is the block-diagonal matrix whose restriction to
 * `B_i × B_i` equals the corresponding diagonal tile `H[B_i, B_i]`.
 * Equivalently, `M^{-1}` is block-diagonal and applying it amounts to
 * solving `|B_i|`-sized SPD systems independently per block.  Each
 * tile is factored once per @ref BlockSparsePCGSolver::Solve via LDLT
 * (`H_d = L D L^T`); applying `M^{-1}` is then a triangular solve plus
 * diagonal scaling.
 *
 * The PCG recurrence (Saad, *Iterative Methods for Sparse Linear
 * Systems*, 2nd ed., Algorithm 9.1):
 *   r_0     = b - H x_0
 *   z_0     = M^{-1} r_0
 *   p_0     = z_0
 *   rz_0    = <r_0, z_0>
 *   for k = 0, 1, ...:
 *     q_k     = H p_k                       // SpMV
 *     alpha_k = rz_k / <p_k, q_k>           // scalar
 *     x_{k+1} = x_k + alpha_k p_k           // axpy
 *     r_{k+1} = r_k - alpha_k q_k           // axpy
 *     z_{k+1} = M^{-1} r_{k+1}              // block-Jacobi apply
 *     rz_{k+1}= <r_{k+1}, z_{k+1}>          // dot
 *     beta_k  = rz_{k+1} / rz_k             // scalar
 *     p_{k+1} = z_{k+1} + beta_k p_k        // axpy
 *
 * We initialize `x_0 = 0`, which makes `r_0 = b` (saves one SpMV).
 *
 * ## Sync strategy
 *
 * The inner loop computes both alpha and beta on the device (single
 * thread kernels reading the dot-product slots), so the host never has
 * to read them.  Convergence is checked every
 * @ref BlockSparsePCGOptions::check_period iterations by copying
 * `||r_k||^2` (one float) from device to host.  This keeps the host
 * roughly `check_period` iterations ahead of the GPU, which is the
 * sweet spot — frequent syncs (period 1) re-stall the launch queue,
 * infrequent syncs (period > 8) waste work after the residual already
 * dipped below the tolerance.
 *
 * ## Variable block sizes
 *
 * Each segment in @ref BlockSparsePCGOptions::block_layout gets its
 * own templated kernel launch (one per `Factor` and one per `Apply`).
 * Per-segment dispatch is a small host-side branch on the templated
 * block-size specialization (B ∈ {1, 2, 3, 4, 6, 7, 15}) and falls
 * back to a runtime-B generic kernel for any other value.  This keeps
 * the hot kernels register-resident for the common cases (SE3 = 6,
 * Vector<3> = 3) while still supporting arbitrary user state batches.
 */

#include "cunls/linear_solver/block_sparse_pcg_solver.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch.h"

namespace cunls {

namespace {

constexpr int kMaxBlockSize = 16;

// =============================================================================
// Device kernels — preconditioner construction (`Factor`)
// =============================================================================

/**
 * @brief Factors one segment of block-diagonal tiles, in place into the
 *        factor buffer.
 *
 * One CTA per tile.  Threads cooperatively:
 *   1. Gather the dense `B × B` tile from `H`'s CSR rows
 *      `[segment_row_start + b*B, segment_row_start + b*B + B)`,
 *      filtering column indices to the tile's column range.  Any
 *      entries outside are zero (the tile is dense in the
 *      preconditioner's view of H; CSR sparsity within the tile is
 *      irrelevant for Jacobi).
 *   2. Symmetrize numerically: `H_d := (H_d + H_d^T) / 2`.  Algebraically
 *      a no-op for `J^T J`, but cheap insurance against FP drift when
 *      LM damping or column scaling has been applied in place.
 *   3. Run a serial LDLT on thread 0:
 *      `for k in 0..B: D_{kk} = H_{kk}; L_{ik} = H_{ik}/D_{kk};
 *                     H_{ij} -= L_{ik} D_{kk} L_{jk}` for i, j > k.
 *      A pivot floor (sign-preserving) guards against singular tiles.
 *   4. Write the (lower-triangle L, diagonal D) result to the global
 *      factor buffer.
 *
 * The serial LDLT is fine for the sizes here (B ≤ 16, dominant cost is
 * the global I/O for the tile, not the O(B³) arithmetic).
 *
 * @tparam B Compile-time tile side length.
 */
template <int B>
__global__ void ExtractAndFactorBlockDiagonalsKernel(
    const int *__restrict__ row_off, const int *__restrict__ col_idx,
    const float *__restrict__ values, int row_start, int num_blocks,
    int factor_offset, float pivot_floor, float *__restrict__ factors) {
  int block_row = blockIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  __shared__ float tile[B * B];

  int tid = threadIdx.x;
  for (int i = tid; i < B * B; i += blockDim.x) {
    tile[i] = 0.f;
  }
  __syncthreads();

  // Step 1: gather dense tile from CSR.  Early-break exploits the
  // sorted-cols invariant — for SBA pose rows where the many
  // landmark columns sit after the diagonal-tile range, this turns
  // an O(nnz_per_row) scan into O(B).
  if (tid < B) {
    int global_row = row_start + block_row * B + tid;
    int start = row_off[global_row];
    int end = row_off[global_row + 1];
    int col_lo = row_start + block_row * B;
    int col_hi = col_lo + B;
    for (int k = start; k < end; ++k) {
      int c = col_idx[k];
      if (c >= col_hi) {
        break;
      }
      if (c >= col_lo) {
        tile[tid * B + (c - col_lo)] = values[k];
      }
    }
  }
  __syncthreads();

  // Step 2: numerical symmetrization.
  if (tid < B) {
    for (int j = tid + 1; j < B; ++j) {
      float a = tile[tid * B + j];
      float b = tile[j * B + tid];
      float s = 0.5f * (a + b);
      tile[tid * B + j] = s;
      tile[j * B + tid] = s;
    }
  }
  __syncthreads();

  // Step 3: serial LDLT (one thread).  Diagonal D is stored on the
  // tile's diagonal, strict lower triangle gets L (unit-diagonal
  // implicit).
  if (tid == 0) {
    for (int k = 0; k < B; ++k) {
      float d = tile[k * B + k];
      if (fabsf(d) < pivot_floor) {
        d = (d >= 0.f) ? pivot_floor : -pivot_floor;
      }
      tile[k * B + k] = d;
      float inv_d = 1.f / d;
      for (int i = k + 1; i < B; ++i) {
        float lik = tile[i * B + k] * inv_d;
        tile[i * B + k] = lik;
        for (int j = k + 1; j <= i; ++j) {
          tile[i * B + j] -= lik * tile[j * B + k] * d;
        }
      }
    }
  }
  __syncthreads();

  // Step 4: write back.
  float *out = factors + factor_offset + block_row * B * B;
  for (int i = tid; i < B * B; i += blockDim.x) {
    out[i] = tile[i];
  }
}

/** Generic-B fallback for non-templated block sizes.  Uses dynamic
 *  shared memory of size `B*B` floats.  Slightly slower than the
 *  templated version (no constant-B unrolling) but works for any
 *  block size up to @c kMaxBlockSize. */
__global__ void ExtractAndFactorGenericKernel(
    const int *__restrict__ row_off, const int *__restrict__ col_idx,
    const float *__restrict__ values, int B, int row_start, int num_blocks,
    int factor_offset, float pivot_floor, float *__restrict__ factors) {
  int block_row = blockIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  extern __shared__ float smem[];
  float *tile = smem;

  int tid = threadIdx.x;
  for (int i = tid; i < B * B; i += blockDim.x) {
    tile[i] = 0.f;
  }
  __syncthreads();

  if (tid < B) {
    int global_row = row_start + block_row * B + tid;
    int start = row_off[global_row];
    int end = row_off[global_row + 1];
    int col_lo = row_start + block_row * B;
    int col_hi = col_lo + B;
    for (int k = start; k < end; ++k) {
      int c = col_idx[k];
      if (c >= col_hi) {
        break;
      }
      if (c >= col_lo) {
        tile[tid * B + (c - col_lo)] = values[k];
      }
    }
  }
  __syncthreads();
  if (tid < B) {
    for (int j = tid + 1; j < B; ++j) {
      float a = tile[tid * B + j];
      float b = tile[j * B + tid];
      float s = 0.5f * (a + b);
      tile[tid * B + j] = s;
      tile[j * B + tid] = s;
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (int k = 0; k < B; ++k) {
      float d = tile[k * B + k];
      if (fabsf(d) < pivot_floor) {
        d = (d >= 0.f) ? pivot_floor : -pivot_floor;
      }
      tile[k * B + k] = d;
      float inv_d = 1.f / d;
      for (int i = k + 1; i < B; ++i) {
        float lik = tile[i * B + k] * inv_d;
        tile[i * B + k] = lik;
        for (int j = k + 1; j <= i; ++j) {
          tile[i * B + j] -= lik * tile[j * B + k] * d;
        }
      }
    }
  }
  __syncthreads();
  float *out = factors + factor_offset + block_row * B * B;
  for (int i = tid; i < B * B; i += blockDim.x) {
    out[i] = tile[i];
  }
}

/**
 * @brief One-thread-per-block factor for small B (≤ 6).
 *
 * Symmetric counterpart to @ref ApplyBlockJacobiPerThreadKernel.  Each
 * thread loads the dense `B × B` diagonal tile from CSR into thread-local
 * registers (no shared memory), runs the LDLT in registers, and writes
 * the factored tile back.  Packs ~256 tiles per CTA — for SBA's 1 M
 * landmark tiles this is a >200× reduction in CTA count vs the
 * one-CTA-per-tile path.
 *
 * Note: the gather still has to scan the CSR row of length `nnz/B`
 * per thread (one row per thread).  For SBA's pose / landmark rows
 * this is short (~5 entries for a landmark, ~25 for a pose) so the
 * scan stays in register loops.
 */
template <int B>
__global__ void ExtractAndFactorPerThreadKernel(
    const int *__restrict__ row_off, const int *__restrict__ col_idx,
    const float *__restrict__ values, int row_start, int num_blocks,
    int factor_offset, float pivot_floor, float *__restrict__ factors) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  // Gather the dense B×B tile into registers.
  float tile[B * B];
#pragma unroll
  for (int i = 0; i < B * B; ++i) {
    tile[i] = 0.f;
  }
  int col_lo = row_start + block_row * B;
  int col_hi = col_lo + B;
  // CSR column indices are sorted within a row, so as soon as we walk
  // past `col_hi` we are guaranteed never to see a column in the
  // tile's range again — break out of the inner loop.  Crucially, for
  // SBA's pose rows where the non-diagonal cols are the (many)
  // landmark cols sorted *after* the diagonal-tile cols, this turns a
  // per-row scan of ~hundreds of entries into a scan of ~B entries.
#pragma unroll
  for (int rr = 0; rr < B; ++rr) {
    int global_row = col_lo + rr;
    int start = row_off[global_row];
    int end = row_off[global_row + 1];
    for (int k = start; k < end; ++k) {
      int c = col_idx[k];
      if (c >= col_hi) {
        break;
      }
      if (c >= col_lo) {
        tile[rr * B + (c - col_lo)] = values[k];
      }
    }
  }
  // Symmetrize numerically.
#pragma unroll
  for (int i = 0; i < B; ++i) {
#pragma unroll
    for (int j = i + 1; j < B; ++j) {
      float s = 0.5f * (tile[i * B + j] + tile[j * B + i]);
      tile[i * B + j] = s;
      tile[j * B + i] = s;
    }
  }
  // LDLT.
#pragma unroll
  for (int k = 0; k < B; ++k) {
    float d = tile[k * B + k];
    if (fabsf(d) < pivot_floor) {
      d = (d >= 0.f) ? pivot_floor : -pivot_floor;
    }
    tile[k * B + k] = d;
    float inv_d = 1.f / d;
#pragma unroll
    for (int i = k + 1; i < B; ++i) {
      float lik = tile[i * B + k] * inv_d;
      tile[i * B + k] = lik;
#pragma unroll
      for (int j = k + 1; j < B; ++j) {
        if (j <= i) {
          tile[i * B + j] -= lik * tile[j * B + k] * d;
        }
      }
    }
  }
  float *out = factors + factor_offset + block_row * B * B;
#pragma unroll
  for (int i = 0; i < B * B; ++i) {
    out[i] = tile[i];
  }
}

/** Scalar-Jacobi extractor for B == 1: `M^{-1}[i] = 1 / H[i,i]`. */
__global__ void ExtractScalarJacobi(const int *__restrict__ row_off,
                                    const int *__restrict__ col_idx,
                                    const float *__restrict__ values,
                                    int row_start, int n, int factor_offset,
                                    float pivot_floor,
                                    float *__restrict__ factors) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  int i = row_start + idx;
  int start = row_off[i];
  int end = row_off[i + 1];
  float d = pivot_floor;
  for (int k = start; k < end; ++k) {
    if (col_idx[k] == i) {
      d = fmaxf(fabsf(values[k]), pivot_floor);
      break;
    }
  }
  factors[factor_offset + idx] = d;
}

// =============================================================================
// Device kernels — preconditioner application (`Apply`)
// =============================================================================

/**
 * @brief Computes `z[block] = M^{-1} r[block]` for one tile in a
 *        segment, where `M^{-1}` is the inverse of the LDLT factor.
 *
 * Each CTA owns a single tile.  We pull the tile's `B` residual entries
 * into shared memory, then perform three serial sweeps on thread 0:
 *   forward solve `L y = r`     (L is unit-lower triangular)
 *   diagonal scale `D w = y`    (D is stored on the tile's diagonal)
 *   back solve    `L^T z = w`
 * giving the block `M^{-1} r`.  Serial because B is at most 16; the
 * per-block arithmetic is ~3*B² FMAs, fully dominated by the global
 * factor read.
 *
 * @tparam B Compile-time tile side length.
 */
template <int B>
__global__ void ApplyBlockJacobiKernel(const float *__restrict__ factors,
                                       int factor_offset,
                                       const float *__restrict__ r,
                                       int row_start, int num_blocks,
                                       float *__restrict__ z) {
  int block_row = blockIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  __shared__ float L[B * B];
  __shared__ float v[B];

  int tid = threadIdx.x;
  if (tid < B) {
    v[tid] = r[row_start + block_row * B + tid];
  }
  for (int i = tid; i < B * B; i += blockDim.x) {
    L[i] = factors[factor_offset + block_row * B * B + i];
  }
  __syncthreads();

  if (tid == 0) {
    // Forward: L y = r (unit diagonal).
    for (int i = 1; i < B; ++i) {
      float s = v[i];
#pragma unroll
      for (int j = 0; j < B; ++j) {
        if (j < i) {
          s -= L[i * B + j] * v[j];
        }
      }
      v[i] = s;
    }
    // Diagonal: D w = y.
    for (int i = 0; i < B; ++i) {
      v[i] /= L[i * B + i];
    }
    // Back: L^T z = w.
    for (int i = B - 2; i >= 0; --i) {
      float s = v[i];
#pragma unroll
      for (int j = 0; j < B; ++j) {
        if (j > i) {
          s -= L[j * B + i] * v[j];
        }
      }
      v[i] = s;
    }
  }
  __syncthreads();

  if (tid < B) {
    z[row_start + block_row * B + tid] = v[tid];
  }
}

/** Generic-B apply for non-templated tile sizes; dynamic shared mem. */
__global__ void ApplyBlockJacobiGenericKernel(
    const float *__restrict__ factors, int factor_offset, int B,
    const float *__restrict__ r, int row_start, int num_blocks,
    float *__restrict__ z) {
  int block_row = blockIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  extern __shared__ float smem[];
  float *L = smem;
  float *v = L + B * B;

  int tid = threadIdx.x;
  if (tid < B) {
    v[tid] = r[row_start + block_row * B + tid];
  }
  for (int i = tid; i < B * B; i += blockDim.x) {
    L[i] = factors[factor_offset + block_row * B * B + i];
  }
  __syncthreads();
  if (tid == 0) {
    for (int i = 1; i < B; ++i) {
      float s = v[i];
      for (int j = 0; j < i; ++j) {
        s -= L[i * B + j] * v[j];
      }
      v[i] = s;
    }
    for (int i = 0; i < B; ++i) {
      v[i] /= L[i * B + i];
    }
    for (int i = B - 2; i >= 0; --i) {
      float s = v[i];
      for (int j = i + 1; j < B; ++j) {
        s -= L[j * B + i] * v[j];
      }
      v[i] = s;
    }
  }
  __syncthreads();
  if (tid < B) {
    z[row_start + block_row * B + tid] = v[tid];
  }
}

/** Scalar-Jacobi apply (B == 1). */
__global__ void ApplyScalarJacobi(const float *__restrict__ factors,
                                  int factor_offset,
                                  const float *__restrict__ r, int row_start,
                                  int n, float *__restrict__ z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  z[row_start + idx] = r[row_start + idx] / factors[factor_offset + idx];
}

/**
 * @brief One-thread-per-block apply, optimized for small B (≤ 8).
 *
 * The per-block work (3*B² FMAs ≈ 27 for B=3, 108 for B=6) fits in a
 * handful of registers — well below the per-thread state budget that
 * limits occupancy.  Holding `L` and `v` in thread-local registers
 * eliminates the per-CTA shared-memory store/load round-trip and lets a
 * full CTA process 256 different blocks instead of 256 threads ganging
 * up on one block.  For SBA with 1 M landmark tiles (B=3), this drops
 * the grid from 1 M CTAs to ~4 K — a >200× reduction in launch and
 * scheduling overhead.
 *
 * @tparam B Compile-time tile side length, must be small enough for the
 *           per-thread register file (we instantiate for {2, 3, 4, 6}).
 */
template <int B>
__global__ void
ApplyBlockJacobiPerThreadKernel(const float *__restrict__ factors,
                                int factor_offset,
                                const float *__restrict__ r, int row_start,
                                int num_blocks, float *__restrict__ z) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  const float *f = factors + factor_offset + block_row * B * B;
  const float *r_in = r + row_start + block_row * B;
  float *z_out = z + row_start + block_row * B;

  float L[B * B];
  float v[B];
#pragma unroll
  for (int i = 0; i < B * B; ++i) {
    L[i] = f[i];
  }
#pragma unroll
  for (int i = 0; i < B; ++i) {
    v[i] = r_in[i];
  }

  // Forward solve L y = r (unit-lower triangular).
#pragma unroll
  for (int i = 1; i < B; ++i) {
    float s = v[i];
#pragma unroll
    for (int j = 0; j < B; ++j) {
      if (j < i) {
        s -= L[i * B + j] * v[j];
      }
    }
    v[i] = s;
  }
  // Diagonal: D w = y.
#pragma unroll
  for (int i = 0; i < B; ++i) {
    v[i] /= L[i * B + i];
  }
  // Back: L^T z = w.
#pragma unroll
  for (int i = B - 2; i >= 0; --i) {
    float s = v[i];
#pragma unroll
    for (int j = 0; j < B; ++j) {
      if (j > i) {
        s -= L[j * B + i] * v[j];
      }
    }
    v[i] = s;
  }

#pragma unroll
  for (int i = 0; i < B; ++i) {
    z_out[i] = v[i];
  }
}

// =============================================================================
// Device kernels — PCG vector / scalar ops
// =============================================================================

/**
 * @brief Fused: compute `α = rz_old / <p, q>` and apply
 *        `x ← x + α p ; r ← r − α q` in a single kernel.
 *
 * Thread 0 reads the two scalar slots, computes α, stores it in
 * shared memory; all threads then perform the axpy on their slice.
 * This eliminates the separate `ComputeAlphaKernel` launch
 * (~1.2 µs / call × thousands of iters = several percent of total
 * runtime on the small-Minimize SBA fixture).
 */
__global__ void PcgUpdateKernel(const float *__restrict__ rz_old_ptr,
                                const float *__restrict__ pAp_ptr,
                                const float *__restrict__ p,
                                const float *__restrict__ Ap,
                                float *__restrict__ x, float *__restrict__ r,
                                int n) {
  __shared__ float a;
  if (threadIdx.x == 0) {
    float denom = pAp_ptr[0];
    a = (denom > 0.f) ? rz_old_ptr[0] / denom : 0.f;
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] += a * p[i];
  r[i] -= a * Ap[i];
}

/**
 * @brief Fused: compute `β = rz_new / rz_old`, store `rz_old ← rz_new`,
 *        and update the search direction `p ← z + β p` in a single
 *        kernel.  Eliminates the separate `ComputeBetaKernel` launch
 *        and writes the new `rz_old` from the first CTA's thread 0
 *        (subsequent CTAs don't read the slot in this kernel).
 */
__global__ void PcgDirectionKernel(const float *__restrict__ rz_new_ptr,
                                   float *__restrict__ rz_old_ptr,
                                   const float *__restrict__ z,
                                   float *__restrict__ p, int n) {
  __shared__ float b;
  if (threadIdx.x == 0) {
    float num = rz_new_ptr[0];
    float denom = rz_old_ptr[0];
    b = (fabsf(denom) > 0.f) ? num / denom : 0.f;
    // Only the first CTA's thread 0 writes rz_old; all other CTAs
    // skip the write (the rz_old slot is read by the *next*
    // iteration's PcgUpdate, which happens after this kernel
    // completes, so any one writer is enough).
    if (blockIdx.x == 0) {
      rz_old_ptr[0] = num;
    }
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  p[i] = z[i] + b * p[i];
}

/** Zero one or two scalar slots. */
__global__ void ZeroScalarKernel(float *out) { out[0] = 0.f; }
__global__ void Zero2ScalarKernel(float *out_a, float *out_b) {
  out_a[0] = 0.f;
  out_b[0] = 0.f;
}

/** Performs `dst += local` on a single device scalar via a per-block warp
 *  reduction + atomicAdd.  Helper for fused dot kernels below. */
__device__ inline void ReduceAndAddToScalar(float local, float *dst) {
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    local += __shfl_down_sync(0xffffffffu, local, off);
  }
  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[wid] = local;
  }
  __syncthreads();
  if (wid == 0) {
    int nw = (blockDim.x + 31) >> 5;
    float s = (lane < nw) ? warp_sums[lane] : 0.f;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if (lane == 0) {
      atomicAdd(dst, s);
    }
  }
}

/**
 * @brief Block-stride dot product `out[0] += <a, b>` (atomic add).
 *
 * Each CTA does a register-level reduction over its slice, then a
 * single-warp reduction inside shared memory, then one atomicAdd into
 * the scalar slot.  Caller must zero `out[0]` first (the companion
 * @ref ZeroScalarKernel kernel above does this).
 */
__global__ void DotKernel(const float *__restrict__ a,
                          const float *__restrict__ b, int n,
                          float *__restrict__ out) {
  float s = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    s += a[i] * b[i];
  }
  ReduceAndAddToScalar(s, out);
}

/**
 * @brief Two-output dot kernel: writes `<a, b>` and `<a, c>` in a
 *        single pass over `a`.
 *
 * Halves the number of vector reads and saves one kernel launch per
 * PCG iteration (only one `ZeroScalarKernel` pair instead of two).
 * Each thread accumulates two partial sums in registers and the
 * per-block reduction is identical to @ref DotKernel.
 */
__global__ void DualDotKernel(const float *__restrict__ a,
                              const float *__restrict__ b,
                              const float *__restrict__ c, int n,
                              float *__restrict__ out_ab,
                              float *__restrict__ out_ac) {
  float sab = 0.f;
  float sac = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float ai = a[i];
    sab += ai * b[i];
    sac += ai * c[i];
  }
  ReduceAndAddToScalar(sab, out_ab);
  ReduceAndAddToScalar(sac, out_ac);
}

// =============================================================================
// Host dispatch helpers
// =============================================================================

/** Picks the right ExtractAndFactor specialization for B. */
void DispatchExtractAndFactor(cudaStream_t stream, int B, int row_start,
                              int num_blocks, int factor_offset,
                              float pivot_floor,
                              const CSRSparseMatrix &matrix,
                              float *factors) {
  if (num_blocks == 0) {
    return;
  }
  const int *row_off = matrix.row_offsets.data();
  const int *col_idx = matrix.col_ids.data();
  const float *vals = matrix.values.data();

  if (B == 1) {
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    ExtractScalarJacobi<<<blocks, threads, 0, stream>>>(
        row_off, col_idx, vals, row_start, num_blocks, factor_offset,
        pivot_floor, factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }

#define LAUNCH_FACTOR_PER_THREAD(BVAL)                                         \
  case BVAL: {                                                                 \
    int threads = 256;                                                         \
    int blocks = (num_blocks + threads - 1) / threads;                         \
    ExtractAndFactorPerThreadKernel<BVAL>                                      \
        <<<blocks, threads, 0, stream>>>(row_off, col_idx, vals, row_start,    \
                                          num_blocks, factor_offset,           \
                                          pivot_floor, factors);               \
    break;                                                                     \
  }

  // Small-B path: one block per thread.
  switch (B) {
    LAUNCH_FACTOR_PER_THREAD(2);
    LAUNCH_FACTOR_PER_THREAD(3);
    LAUNCH_FACTOR_PER_THREAD(4);
    LAUNCH_FACTOR_PER_THREAD(5);
    LAUNCH_FACTOR_PER_THREAD(6);
  default:
    break;
  }
#undef LAUNCH_FACTOR_PER_THREAD
  if (B >= 2 && B <= 6) {
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }

  // Large-B path: one CTA per block.
  int threads = ((B + 31) / 32) * 32;
#define LAUNCH_FACTOR(BVAL)                                                    \
  case BVAL:                                                                   \
    ExtractAndFactorBlockDiagonalsKernel<BVAL>                                 \
        <<<num_blocks, threads, 0, stream>>>(row_off, col_idx, vals,           \
                                              row_start, num_blocks,           \
                                              factor_offset, pivot_floor,      \
                                              factors);                        \
    break

  switch (B) {
    LAUNCH_FACTOR(7);
    LAUNCH_FACTOR(8);
    LAUNCH_FACTOR(15);
    LAUNCH_FACTOR(16);
  default: {
    size_t shared_bytes = static_cast<size_t>(B) * B * sizeof(float);
    ExtractAndFactorGenericKernel<<<num_blocks, threads, shared_bytes, stream>>>(
        row_off, col_idx, vals, B, row_start, num_blocks, factor_offset,
        pivot_floor, factors);
    break;
  }
  }
#undef LAUNCH_FACTOR
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** Picks the right ApplyBlockJacobi specialization for B.
 *
 *  For small B (≤ 6) we use a "one block per thread" path which moves
 *  the per-tile factor + residual into thread-local registers and
 *  packs ~256 tiles per CTA.  This minimizes launch / scheduling
 *  overhead for problems with millions of small tiles (e.g. SBA
 *  landmarks).  For larger B we fall back to the per-CTA path that
 *  stages the factor through shared memory. */
void DispatchApplyPrecond(cudaStream_t stream, int B, int row_start,
                          int num_blocks, int factor_offset,
                          const float *factors, const float *r, float *z) {
  if (num_blocks == 0) {
    return;
  }
  if (B == 1) {
    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    ApplyScalarJacobi<<<blocks, threads, 0, stream>>>(factors, factor_offset, r,
                                                      row_start, num_blocks, z);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }

#define LAUNCH_APPLY_PER_THREAD(BVAL)                                          \
  case BVAL: {                                                                 \
    int threads = 256;                                                         \
    int blocks = (num_blocks + threads - 1) / threads;                         \
    ApplyBlockJacobiPerThreadKernel<BVAL>                                      \
        <<<blocks, threads, 0, stream>>>(factors, factor_offset, r,            \
                                          row_start, num_blocks, z);           \
    break;                                                                     \
  }

  // Small-B path: one block per thread, lots of blocks per CTA.
  switch (B) {
    LAUNCH_APPLY_PER_THREAD(2);
    LAUNCH_APPLY_PER_THREAD(3);
    LAUNCH_APPLY_PER_THREAD(4);
    LAUNCH_APPLY_PER_THREAD(5);
    LAUNCH_APPLY_PER_THREAD(6);
  default:
    break; // fall through to per-CTA path below
  }
#undef LAUNCH_APPLY_PER_THREAD
  if (B >= 2 && B <= 6) {
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }

  // Large-B path: one CTA per block, factor staged through shared mem.
  int threads = ((B + 31) / 32) * 32;
#define LAUNCH_APPLY(BVAL)                                                     \
  case BVAL:                                                                   \
    ApplyBlockJacobiKernel<BVAL><<<num_blocks, threads, 0, stream>>>(          \
        factors, factor_offset, r, row_start, num_blocks, z);                  \
    break

  switch (B) {
    LAUNCH_APPLY(7);
    LAUNCH_APPLY(8);
    LAUNCH_APPLY(15);
    LAUNCH_APPLY(16);
  default: {
    size_t shared_bytes = (static_cast<size_t>(B) * B + B) * sizeof(float);
    ApplyBlockJacobiGenericKernel<<<num_blocks, threads, shared_bytes,
                                    stream>>>(factors, factor_offset, B, r,
                                              row_start, num_blocks, z);
    break;
  }
  }
#undef LAUNCH_APPLY
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/** Enqueues `out[0] = <a, b>` (clear-then-reduce).
 *
 *  We use `cudaMemsetAsync` instead of a one-thread kernel because for
 *  a single-float clear the runtime can short-circuit it to a tiny
 *  memory operation, saving the per-kernel-launch dispatch overhead
 *  (~1 µs) that dominates on small-matrix workloads where PCG
 *  performs hundreds of dots per Solve. */
void DotAsync(cudaStream_t stream, const float *a, const float *b, int n,
              float *out) {
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(out, 0, sizeof(float), stream));
  int threads = 256;
  int blocks = std::min(1024, (n + threads - 1) / threads);
  DotKernel<<<blocks, threads, 0, stream>>>(a, b, n, out);
}

/** Enqueues `out_ab = <a, b>` and `out_ac = <a, c>` in a single pass.
 *  Equivalent to two @ref DotAsync calls but with half the global reads
 *  AND a single 8-byte memset (vs two 4-byte memsets) — the two output
 *  slots are required to be adjacent in memory, which the caller
 *  guarantees by laying them out next to each other in @c d_scratch_. */
void DualDotAsync(cudaStream_t stream, const float *a, const float *b,
                  const float *c, int n, float *out_ab, float *out_ac) {
  // Verified adjacent at the call site (kRzNew, kRnorm2 in d_scratch_).
  assert(out_ac == out_ab + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(out_ab, 0, 2 * sizeof(float), stream));
  int threads = 256;
  int blocks = std::min(1024, (n + threads - 1) / threads);
  DualDotKernel<<<blocks, threads, 0, stream>>>(a, b, c, n, out_ab, out_ac);
}

} // namespace

// =============================================================================
// BlockSparsePCGSolver
// =============================================================================

BlockSparsePCGSolver::BlockSparsePCGSolver(BlockSparsePCGOptions options)
    : options_(std::move(options)) {
  if (options_.block_size < 1 || options_.block_size > kMaxBlockSize) {
    throw std::invalid_argument(
        "BlockSparsePCGSolver: block_size must be in [1, 16]");
  }
  for (const auto &seg : options_.block_layout) {
    if (seg.second < 1 || seg.second > kMaxBlockSize) {
      throw std::invalid_argument(
          "BlockSparsePCGSolver: block_layout segment size must be in "
          "[1, 16]");
    }
  }
  if (options_.check_period < 1) {
    options_.check_period = 1;
  }
}

BlockSparsePCGSolver::~BlockSparsePCGSolver() = default;

bool BlockSparsePCGSolver::BuildSegmentTables(int matrix_dim) {
  segments_.clear();
  // Normalize the layout: use the user-supplied block_layout if it's
  // non-empty; otherwise treat the whole matrix as a single segment of
  // size `options_.block_size`.
  layout_.clear();
  if (!options_.block_layout.empty()) {
    layout_ = options_.block_layout;
  } else {
    if (matrix_dim % options_.block_size != 0) {
      LogError("BlockSparsePCGSolver: matrix dim {} not divisible by uniform "
               "block_size {}",
               matrix_dim, options_.block_size);
      return false;
    }
    layout_.push_back({matrix_dim / options_.block_size, options_.block_size});
  }

  int row_cursor = 0;
  int block_cursor = 0;
  size_t factor_cursor = 0;
  for (const auto &[count, size] : layout_) {
    if (count <= 0) {
      continue;
    }
    if (size < 1 || size > kMaxBlockSize) {
      LogError("BlockSparsePCGSolver: invalid segment size {}", size);
      return false;
    }
    Segment s;
    s.block_size = size;
    s.num_blocks = count;
    s.row_start = row_cursor;
    s.factor_offset = static_cast<int>(factor_cursor);
    s.block_row_start = block_cursor;
    segments_.push_back(s);

    row_cursor += count * size;
    block_cursor += count;
    factor_cursor += static_cast<size_t>(count) *
                     (size == 1 ? 1ull
                                : static_cast<size_t>(size) * size);
  }
  if (row_cursor != matrix_dim) {
    LogError("BlockSparsePCGSolver: layout covers {} rows but matrix has {}",
             row_cursor, matrix_dim);
    return false;
  }
  total_blocks_ = block_cursor;
  total_factor_floats_ = factor_cursor;
  return true;
}

bool BlockSparsePCGSolver::Initialize(cudaStream_t stream,
                                      const Problem &problem,
                                      const CSRSparseMatrix &spd_matrix,
                                      const dvector<float> &rhs,
                                      dvector<float> &result) {
  int n = static_cast<int>(spd_matrix.NumRows());
  if (n != static_cast<int>(rhs.size()) ||
      n != static_cast<int>(result.size())) {
    LogError(
        "BlockSparsePCGSolver: dim mismatch (matrix={}, rhs={}, result={})", n,
        rhs.size(), result.size());
    return false;
  }
  // When @p problem carries state batches, the block-Jacobi layout is
  // ALWAYS derived afresh from them — each non-empty batch contributes
  // a segment with `size = TangentSize()` and
  // `count = NumStateBlocks() - NumConstStateBlocks()`.  Consecutive
  // segments of equal size are merged so the dispatch loop has one
  // entry per distinct-size group (typical: 1 entry for PGO, 2 for
  // SBA).  Re-deriving on every Initialize is what makes the same
  // solver instance work across a stream of differently-dimensioned
  // problems (e.g. the binary SBA test fixture iterates over many
  // such problems with one minimizer).
  //
  // Falls back to the user-supplied @c options_.block_layout (or the
  // uniform @c options_.block_size) only when @p problem has no
  // registered state batches — the path tests exercise when they call
  // the solver directly on a raw matrix.
  std::vector<std::pair<int, int>> derived_layout;
  for (const auto *sb : problem.GetStateBatches()) {
    if (sb == nullptr) {
      continue;
    }
    int t = static_cast<int>(sb->TangentSize());
    int count =
        static_cast<int>(sb->NumStateBlocks() - sb->NumConstStateBlocks());
    if (count == 0) {
      continue;
    }
    if (!derived_layout.empty() && derived_layout.back().second == t) {
      derived_layout.back().first += count;
    } else {
      derived_layout.emplace_back(count, t);
    }
  }
  if (!derived_layout.empty()) {
    options_.block_layout = std::move(derived_layout);
  }
  if (!BuildSegmentTables(n)) {
    return false;
  }
  matrix_size_ = n;

  if (precond_factors_.size() < total_factor_floats_) {
    precond_factors_.resize(total_factor_floats_);
  }
  if (r_.size() < static_cast<size_t>(n)) {
    r_.resize(n);
    z_.resize(n);
    p_.resize(n);
    Ap_.resize(n);
  }
  if (d_scratch_.size() < 7) {
    d_scratch_.resize(7);
  }

  // cuSPARSE SpMV setup.  The descriptor is reused across all PCG steps
  // and across all Solve calls until the matrix structure changes.
  mat_desc_ = cuSPARSEMatrixDescription(
      n, n, static_cast<int>(spd_matrix.NumNonZeros()), spd_matrix);
  auto handle =
      static_cast<cusparseHandle_t>(cusparse_handle_.GetHandle(stream));

  auto matA = static_cast<cusparseSpMatDescr_t>(mat_desc_.GetDescription());
  cusparseDnVecDescr_t vecX = nullptr;
  cusparseDnVecDescr_t vecY = nullptr;
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecX, n, p_.data(), CUDA_R_32F));
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecY, n, Ap_.data(), CUDA_R_32F));

  float alpha = 1.f;
  float beta = 0.f;
  size_t buffer_size = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
  if (buffer_size > spmv_buffer_.size()) {
    spmv_buffer_.resize(buffer_size);
  }
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_.data()));
  WARN_ON_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
  WARN_ON_CUSPARSE_ERROR(cusparseDestroyDnVec(vecY));
  return true;
}

bool BlockSparsePCGSolver::Solve(cudaStream_t stream,
                                 const CSRSparseMatrix &spd_matrix,
                                 const dvector<float> &rhs,
                                 dvector<float> &result) {
  int n = static_cast<int>(spd_matrix.NumRows());
  if (n != static_cast<int>(rhs.size()) ||
      n != static_cast<int>(result.size())) {
    LogError(
        "BlockSparsePCGSolver: dim mismatch (matrix={}, rhs={}, result={})", n,
        rhs.size(), result.size());
    return false;
  }
  if (n == 0) {
    return true;
  }
  if (n != matrix_size_) {
    // Recovery path: matrix dim changed since the last Initialize.  Use
    // a default-constructed Problem; the cached options_.block_layout
    // (set by the prior Initialize) is reused if non-empty, otherwise
    // the uniform options_.block_size path takes over.
    Problem empty_problem;
    if (!Initialize(stream, empty_problem, spd_matrix, rhs, result)) {
      return false;
    }
  }

  // Refresh the cuSPARSE descriptor's value pointer; the matrix's
  // structure is unchanged so no re-preprocess is needed.
  mat_desc_.UpdatePointers(spd_matrix);

  // -----------------------------------------------------------------
  // 1. Rebuild the block-Jacobi preconditioner from current H values.
  // -----------------------------------------------------------------
  for (const auto &s : segments_) {
    DispatchExtractAndFactor(stream, s.block_size, s.row_start, s.num_blocks,
                             s.factor_offset, options_.pivot_floor, spd_matrix,
                             precond_factors_.data());
  }

  auto handle =
      static_cast<cusparseHandle_t>(cusparse_handle_.GetHandle(stream));
  auto matA = static_cast<cusparseSpMatDescr_t>(mat_desc_.GetDescription());

  // -----------------------------------------------------------------
  // 2. Initialize PCG with x_0 = 0 ⇒ r_0 = b.
  // -----------------------------------------------------------------
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(result.data(), 0, n * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(r_.data(), rhs.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // z_0 = M^{-1} r_0, p_0 = z_0.
  for (const auto &s : segments_) {
    DispatchApplyPrecond(stream, s.block_size, s.row_start, s.num_blocks,
                         s.factor_offset, precond_factors_.data(), r_.data(),
                         z_.data());
  }
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(p_.data(), z_.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // -----------------------------------------------------------------
  // 3. Compute initial inner products on device.
  // -----------------------------------------------------------------
  // Scratch slots: [0] alpha, [1] beta, [2] <p, Ap>, [3] rz_old (= <r, z>),
  // [4] rz_new, [5] ||r||^2, [6] ||b||^2.
  enum : int {
    kAlpha = 0,
    kBeta = 1,
    kPAp = 2,
    kRzOld = 3,
    kRzNew = 4,
    kRnorm2 = 5,
    kBnorm2 = 6
  };

  DotAsync(stream, r_.data(), z_.data(), n, d_scratch_.data() + kRzOld);
  DotAsync(stream, rhs.data(), rhs.data(), n, d_scratch_.data() + kBnorm2);

  // Pull ||b||^2 once for the host-side convergence threshold.  This is
  // the ONLY guaranteed host sync; the rest of the loop polls residual
  // norm only every `check_period` iterations.
  float b_norm2 = 0.f;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&b_norm2, d_scratch_.data() + kBnorm2,
                                      sizeof(float), cudaMemcpyDeviceToHost,
                                      stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  float abs_tol2 = options_.absolute_tolerance * options_.absolute_tolerance;
  float rel_tol2 = options_.relative_tolerance * options_.relative_tolerance;
  float stop_thresh = fmaxf(abs_tol2, rel_tol2 * fmaxf(b_norm2, 1e-30f));

  // -----------------------------------------------------------------
  // 4. Per-iteration loop.
  // -----------------------------------------------------------------
  // p_/Ap_ may have capacity > n across successive Solves on differently
  // sized matrices, so we construct fresh dense-vector descriptors of
  // explicit size n each Solve.  Cheap host calls.
  cusparseDnVecDescr_t vecX = nullptr;
  cusparseDnVecDescr_t vecY = nullptr;
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecX, n, p_.data(), CUDA_R_32F));
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecY, n, Ap_.data(), CUDA_R_32F));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const int check_period = options_.check_period;

  int it = 0;
  for (; it < options_.max_iterations; ++it) {
    // Ap = H * p.
    float spmv_alpha = 1.f;
    float spmv_beta = 0.f;
    THROW_ON_CUSPARSE_ERROR(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &spmv_alpha, matA, vecX,
        &spmv_beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        spmv_buffer_.data()));

    // <p, Ap>.
    DotAsync(stream, p_.data(), Ap_.data(), n, d_scratch_.data() + kPAp);

    // α = rz_old / <p, Ap>; x ← x + α p; r ← r − α Ap (fused kernel).
    PcgUpdateKernel<<<blocks, threads, 0, stream>>>(
        d_scratch_.data() + kRzOld, d_scratch_.data() + kPAp, p_.data(),
        Ap_.data(), result.data(), r_.data(), n);

    // z = M^{-1} r.
    for (const auto &s : segments_) {
      DispatchApplyPrecond(stream, s.block_size, s.row_start, s.num_blocks,
                           s.factor_offset, precond_factors_.data(), r_.data(),
                           z_.data());
    }
    // Fused: rz_new = <r, z>, ||r||^2 = <r, r>.  Single pass over r.
    DualDotAsync(stream, r_.data(), z_.data(), r_.data(), n,
                 d_scratch_.data() + kRzNew, d_scratch_.data() + kRnorm2);

    // β = rz_new / rz_old; rz_old ← rz_new; p ← z + β p (fused).
    PcgDirectionKernel<<<blocks, threads, 0, stream>>>(
        d_scratch_.data() + kRzNew, d_scratch_.data() + kRzOld, z_.data(),
        p_.data(), n);

    // Convergence poll every `check_period` iters: one D2H of one float
    // + one stream sync.  Far cheaper than a host sync per iteration.
    if (((it + 1) % check_period) == 0 ||
        (it + 1) == options_.max_iterations) {
      float r_norm2 = 0.f;
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&r_norm2, d_scratch_.data() + kRnorm2,
                                          sizeof(float),
                                          cudaMemcpyDeviceToHost, stream));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      if (r_norm2 <= stop_thresh) {
        ++it;
        break;
      }
    }
  }
  WARN_ON_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
  WARN_ON_CUSPARSE_ERROR(cusparseDestroyDnVec(vecY));
  last_iterations_ = it;
  return true;
}

} // namespace cunls
