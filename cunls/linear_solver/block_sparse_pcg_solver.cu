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

#include "cunls/linear_solver/block_sparse_pcg_solver.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"

namespace cunls {

namespace {

constexpr int kMaxBlockSize = 16;

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------

/**
 * @brief Extracts the dense B x B diagonal tiles from a symmetric CSR matrix
 *        and stores their LDLT factors.
 *
 * One thread block handles one diagonal block.  Threads cooperate to scan the
 * rows in [block_row * B, block_row * B + B), pull the entries with column id
 * in the same range, and assemble a dense B x B tile in shared memory.  A
 * single-thread LDLT then runs over the symmetric tile (sizes encountered here
 * are 3-6, so the serial cost is negligible compared to the global I/O).
 *
 * The factored block is written back to ``factors`` in row-major layout:
 * the strict lower-triangle holds L (1's on the diagonal implicitly), and the
 * diagonal of the stored tile holds D.  Upper triangle is undefined and not
 * read.
 */
template <int B>
__global__ void ExtractAndFactorBlockDiagonals(const int *__restrict__ row_off,
                                               const int *__restrict__ col_idx,
                                               const float *__restrict__ values,
                                               int num_blocks, float pivot_floor,
                                               float *__restrict__ factors) {
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

  // Each thread takes one of the B rows.
  if (tid < B) {
    int global_row = block_row * B + tid;
    int start = row_off[global_row];
    int end = row_off[global_row + 1];
    int col_lo = block_row * B;
    int col_hi = col_lo + B;
    for (int k = start; k < end; ++k) {
      int c = col_idx[k];
      if (c >= col_lo && c < col_hi) {
        tile[tid * B + (c - col_lo)] = values[k];
      }
    }
  }
  __syncthreads();

  // Symmetrize so the LDLT below can read either triangle.  CSR is
  // symmetric for J^T J and Levenberg-Marquardt damping is on the diagonal,
  // so off-diagonal entries should already match; symmetrizing keeps the
  // factorization stable when only the lower triangle is stored.
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

  // Serial LDLT in shared memory.  B is small (3..16); a single thread is the
  // simplest correct implementation.  Outer-product update with diagonal pivot
  // and a small floor to keep the preconditioner stable on near-singular tiles.
  if (tid == 0) {
    for (int k = 0; k < B; ++k) {
      float d = tile[k * B + k];
      if (fabsf(d) < pivot_floor) {
        d = pivot_floor;
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

  // Write factors back (row-major).
  float *out = factors + block_row * B * B;
  for (int i = tid; i < B * B; i += blockDim.x) {
    out[i] = tile[i];
  }
}

/**
 * @brief Applies the precomputed block-LDLT preconditioner: z = M^{-1} r.
 *
 * One block per diagonal tile, B threads per block.  Reads r into registers,
 * solves L y = r, D w = y, L^T z = w in three serial sweeps.  Sizes are tiny
 * (B in {3,6}) so the work fits in registers; the kernel is bandwidth-bound on
 * the factor read.
 */
template <int B>
__global__ void ApplyBlockJacobiPreconditioner(
    const float *__restrict__ factors, const float *__restrict__ r,
    float *__restrict__ z, int num_blocks) {
  int block_row = blockIdx.x;
  if (block_row >= num_blocks) {
    return;
  }
  __shared__ float L[B * B];
  __shared__ float v[B];

  int tid = threadIdx.x;
  if (tid < B) {
    v[tid] = r[block_row * B + tid];
  }
  for (int i = tid; i < B * B; i += blockDim.x) {
    L[i] = factors[block_row * B * B + i];
  }
  __syncthreads();

  // Forward solve L y = r (unit diagonal); D w = y; L^T z = w.  Serialized on
  // thread 0 — the per-block work is 3*B^2 FMAs which dominates over any
  // attempt to parallelize across B threads.
  if (tid == 0) {
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
    for (int i = 0; i < B; ++i) {
      v[i] /= L[i * B + i];
    }
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
    z[block_row * B + tid] = v[tid];
  }
}

/** Scalar Jacobi fallback (B == 1) — z = r / diag. */
__global__ void ApplyScalarJacobi(const float *__restrict__ factors,
                                  const float *__restrict__ r,
                                  float *__restrict__ z, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float d = factors[i];
  z[i] = r[i] / d;
}

/** Scalar Jacobi extractor (B == 1) — pulls diag(H) with a pivot floor. */
__global__ void ExtractScalarJacobi(const int *__restrict__ row_off,
                                    const int *__restrict__ col_idx,
                                    const float *__restrict__ values, int n,
                                    float pivot_floor,
                                    float *__restrict__ factors) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int start = row_off[i];
  int end = row_off[i + 1];
  float d = pivot_floor;
  for (int k = start; k < end; ++k) {
    if (col_idx[k] == i) {
      d = fmaxf(fabsf(values[k]), pivot_floor);
      break;
    }
  }
  factors[i] = d;
}

/** y = a*x + b*z, all length n.  Used for PCG vector updates. */
__global__ void AxpyKernel(float a, const float *__restrict__ x, float b,
                           const float *__restrict__ z, float *__restrict__ y,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  y[i] = a * x[i] + b * z[i];
}

/** x = a*p + x, r = r - a*Ap, plus partial-sum dot products of (new_r, z') —
 *  this kernel is the hot path of PCG and is intentionally minimal.
 *  ``alpha`` is read once from device memory so that the host can keep
 *  enqueueing kernels without waiting on the previous dot product. */
__global__ void PcgUpdateKernel(const float *__restrict__ alpha_ptr,
                                const float *__restrict__ p,
                                const float *__restrict__ Ap,
                                float *__restrict__ x, float *__restrict__ r,
                                int n) {
  __shared__ float a;
  if (threadIdx.x == 0) {
    a = alpha_ptr[0];
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] += a * p[i];
  r[i] -= a * Ap[i];
}

/** p = z + beta * p, length n.  ``beta`` is fetched from device memory. */
__global__ void PcgDirectionKernel(const float *__restrict__ beta_ptr,
                                   const float *__restrict__ z,
                                   float *__restrict__ p, int n) {
  __shared__ float b;
  if (threadIdx.x == 0) {
    b = beta_ptr[0];
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  p[i] = z[i] + b * p[i];
}

/** Single-thread device kernel: alpha = rz_old / pAp (with guard). */
__global__ void ComputeAlphaKernel(const float *__restrict__ rz_old,
                                   const float *__restrict__ pAp,
                                   float *__restrict__ alpha) {
  float denom = pAp[0];
  alpha[0] = (denom > 0.f) ? rz_old[0] / denom : 0.f;
}

/** Single-thread device kernel: beta = rz_new / rz_old; rz_old <- rz_new. */
__global__ void ComputeBetaKernel(const float *__restrict__ rz_new,
                                  float *__restrict__ rz_old,
                                  float *__restrict__ beta) {
  float num = rz_new[0];
  float denom = rz_old[0];
  beta[0] = (fabsf(denom) > 0.f) ? num / denom : 0.f;
  rz_old[0] = num;
}

/**
 * @brief Single-pass dot product writing the result to ``out`` on device.
 *
 * Uses one block-stride loop per CTA, a warp reduction, and an atomicAdd into
 * out[0].  Caller must zero out[0] before launch.  Fine for the sizes we hit
 * here (a few thousand to ~100k floats).
 */
__global__ void DotKernelZero(float *__restrict__ out) { out[0] = 0.f; }

__global__ void DotKernel(const float *__restrict__ a,
                          const float *__restrict__ b, int n,
                          float *__restrict__ out) {
  float s = 0.f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    s += a[i] * b[i];
  }
  // Warp reduction.
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffffu, s, off);
  }
  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[wid] = s;
  }
  __syncthreads();
  if (wid == 0) {
    int nw = (blockDim.x + 31) >> 5;
    s = (lane < nw) ? warp_sums[lane] : 0.f;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if (lane == 0) {
      atomicAdd(out, s);
    }
  }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

void LaunchExtractAndFactor(cudaStream_t stream, int B,
                            const CSRSparseMatrix &matrix, int num_blocks,
                            float pivot_floor, dvector<float> &factors) {
  if (B == 1) {
    int n = num_blocks;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ExtractScalarJacobi<<<blocks, threads, 0, stream>>>(
        matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
        n, pivot_floor, factors.data());
    return;
  }

  int threads = ((B + 31) / 32) * 32; // round up to a full warp
  switch (B) {
  case 2:
    ExtractAndFactorBlockDiagonals<2><<<num_blocks, threads, 0, stream>>>(
        matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
        num_blocks, pivot_floor, factors.data());
    break;
  case 3:
    ExtractAndFactorBlockDiagonals<3><<<num_blocks, threads, 0, stream>>>(
        matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
        num_blocks, pivot_floor, factors.data());
    break;
  case 6:
    ExtractAndFactorBlockDiagonals<6><<<num_blocks, threads, 0, stream>>>(
        matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
        num_blocks, pivot_floor, factors.data());
    break;
  case 7:
    ExtractAndFactorBlockDiagonals<7><<<num_blocks, threads, 0, stream>>>(
        matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
        num_blocks, pivot_floor, factors.data());
    break;
  default:
    throw std::invalid_argument(
        "BlockSparsePCGSolver: unsupported block_size (must be 1,2,3,6,7)");
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchApplyPrecond(cudaStream_t stream, int B, const float *factors,
                        const float *r, float *z, int num_blocks) {
  if (B == 1) {
    int n = num_blocks;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ApplyScalarJacobi<<<blocks, threads, 0, stream>>>(factors, r, z, n);
    return;
  }
  int threads = ((B + 31) / 32) * 32;
  switch (B) {
  case 2:
    ApplyBlockJacobiPreconditioner<2>
        <<<num_blocks, threads, 0, stream>>>(factors, r, z, num_blocks);
    break;
  case 3:
    ApplyBlockJacobiPreconditioner<3>
        <<<num_blocks, threads, 0, stream>>>(factors, r, z, num_blocks);
    break;
  case 6:
    ApplyBlockJacobiPreconditioner<6>
        <<<num_blocks, threads, 0, stream>>>(factors, r, z, num_blocks);
    break;
  case 7:
    ApplyBlockJacobiPreconditioner<7>
        <<<num_blocks, threads, 0, stream>>>(factors, r, z, num_blocks);
    break;
  default:
    throw std::invalid_argument("BlockSparsePCGSolver: unsupported block_size");
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void DotAsync(cudaStream_t stream, const float *a, const float *b, int n,
              float *d_out) {
  DotKernelZero<<<1, 1, 0, stream>>>(d_out);
  int threads = 256;
  int blocks = std::min(1024, (n + threads - 1) / threads);
  DotKernel<<<blocks, threads, 0, stream>>>(a, b, n, d_out);
}

} // namespace

// -----------------------------------------------------------------------------
// BlockSparsePCGSolver
// -----------------------------------------------------------------------------

BlockSparsePCGSolver::BlockSparsePCGSolver(BlockSparsePCGOptions options)
    : options_(options) {
  if (options_.block_size < 1 || options_.block_size > kMaxBlockSize) {
    throw std::invalid_argument(
        "BlockSparsePCGSolver: block_size must be in [1, 16]");
  }
}

BlockSparsePCGSolver::~BlockSparsePCGSolver() = default;

bool BlockSparsePCGSolver::Initialize(cudaStream_t stream,
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
  int B = options_.block_size;
  if (n % B != 0) {
    LogError("BlockSparsePCGSolver: matrix size {} not divisible by block "
             "size {}",
             n, B);
    return false;
  }
  matrix_size_ = n;
  num_blocks_ = n / B;

  size_t factors_size =
      static_cast<size_t>(num_blocks_) * static_cast<size_t>(B * B);
  if (B == 1) {
    factors_size = static_cast<size_t>(n);
  }
  if (precond_factors_.size() < factors_size) {
    precond_factors_.resize(factors_size);
  }
  if (r_.size() < static_cast<size_t>(n)) {
    r_.resize(n);
    z_.resize(n);
    p_.resize(n);
    Ap_.resize(n);
  }
  if (d_scratch_.size() < 2) {
    d_scratch_.resize(2);
  }

  // Build the cuSPARSE descriptor for SpMV.  Use raw cusparseDnVec with
  // explicit size = n so capacity-vs-size drift in p_/Ap_ across successive
  // Solves on differently-sized matrices doesn't break the SpMV preprocess.
  mat_desc_ =
      cuSPARSEMatrixDescription(n, n, static_cast<int>(spd_matrix.NumNonZeros()),
                                spd_matrix);
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
  spmv_buffer_ready_ = true;
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
    if (!Initialize(stream, spd_matrix, rhs, result)) {
      return false;
    }
  }
  int B = options_.block_size;

  // Refresh the cuSPARSE descriptor's value pointer (structure is fixed across
  // calls; the CSR matrix can move in memory between Solves).
  mat_desc_.UpdatePointers(spd_matrix);

  // Refresh preconditioner from the current matrix values.
  LaunchExtractAndFactor(stream, B, spd_matrix, num_blocks_,
                         options_.pivot_floor, precond_factors_);

  auto handle =
      static_cast<cusparseHandle_t>(cusparse_handle_.GetHandle(stream));
  auto matA = static_cast<cusparseSpMatDescr_t>(mat_desc_.GetDescription());

  // PCG with zero initial guess.  The Gauss-Newton step is reset every outer
  // iteration so warm-starting from the prior step would seed PCG far from the
  // new solution; zeroing is both faster and more robust here.
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(result.data(), 0, n * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(r_.data(), rhs.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // z = M^{-1} r
  LaunchApplyPrecond(stream, B, precond_factors_.data(), r_.data(), z_.data(),
                     num_blocks_);
  // p = z
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(p_.data(), z_.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  // Scratch layout (all device-resident, never copied to host during the inner
  // loop): [0] alpha, [1] beta, [2] pAp, [3] rz_old, [4] rz_new, [5] r_norm2,
  // [6] b_norm2.
  enum : int {
    kAlpha = 0,
    kBeta = 1,
    kPAp = 2,
    kRzOld = 3,
    kRzNew = 4,
    kRnorm2 = 5,
    kBnorm2 = 6,
    kScalarCount = 7
  };
  if (d_scratch_.size() < kScalarCount) {
    d_scratch_.resize(kScalarCount);
  }

  // rz_old = <r, z> ; b_norm2 = <b, b>
  DotAsync(stream, r_.data(), z_.data(), n, d_scratch_.data() + kRzOld);
  DotAsync(stream, rhs.data(), rhs.data(), n, d_scratch_.data() + kBnorm2);

  // Pull b_norm2 once for the host-side stopping threshold (the dot still
  // runs async; the sync below is paid only once, not per iteration).
  float b_norm2 = 0.f;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&b_norm2, d_scratch_.data() + kBnorm2,
                                      sizeof(float), cudaMemcpyDeviceToHost,
                                      stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  float abs_tol2 = options_.absolute_tolerance * options_.absolute_tolerance;
  float rel_tol2 = options_.relative_tolerance * options_.relative_tolerance;
  float stop_thresh = fmaxf(abs_tol2, rel_tol2 * fmaxf(b_norm2, 1e-30f));

  // Build dense-vector descriptors with explicit size = n.  dvector::resize
  // is grow-only, so p_/Ap_ may have capacity > n across successive Solves on
  // problems with different dimensions; using p_.size() in the descriptor
  // would mismatch matA's row count.
  cusparseDnVecDescr_t vecX = nullptr;
  cusparseDnVecDescr_t vecY = nullptr;
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecX, n, p_.data(), CUDA_R_32F));
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&vecY, n, Ap_.data(), CUDA_R_32F));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  // Convergence is polled every kCheckPeriod iterations.  Polling more often
  // turns the inner loop back into a sequence of host syncs; polling less
  // often risks doing extra work after the iteration has already converged.
  // 2 is a safe default — small PGO systems converge in a handful of iterates
  // and don't tolerate a long fixed period, while SBA still benefits from
  // the avoided alpha/beta-on-host syncs that the rest of the inner loop now
  // sidesteps.
  constexpr int kCheckPeriod = 2;
  int it = 0;
  for (; it < options_.max_iterations; ++it) {
    // Ap = A * p
    float spmv_alpha = 1.f;
    float spmv_beta = 0.f;
    THROW_ON_CUSPARSE_ERROR(cusparseSpMV(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &spmv_alpha, matA, vecX, &spmv_beta,
                                         vecY, CUDA_R_32F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         spmv_buffer_.data()));

    // pAp = <p, Ap>
    DotAsync(stream, p_.data(), Ap_.data(), n, d_scratch_.data() + kPAp);
    // alpha = rz_old / pAp on device.
    ComputeAlphaKernel<<<1, 1, 0, stream>>>(d_scratch_.data() + kRzOld,
                                            d_scratch_.data() + kPAp,
                                            d_scratch_.data() + kAlpha);
    PcgUpdateKernel<<<blocks, threads, 0, stream>>>(
        d_scratch_.data() + kAlpha, p_.data(), Ap_.data(), result.data(),
        r_.data(), n);

    // z = M^{-1} r ; rz_new = <r, z> ; r_norm2 = <r, r>
    LaunchApplyPrecond(stream, B, precond_factors_.data(), r_.data(), z_.data(),
                       num_blocks_);
    DotAsync(stream, r_.data(), r_.data(), n, d_scratch_.data() + kRnorm2);
    DotAsync(stream, r_.data(), z_.data(), n, d_scratch_.data() + kRzNew);

    // beta = rz_new / rz_old, then rz_old <- rz_new (single device kernel).
    ComputeBetaKernel<<<1, 1, 0, stream>>>(d_scratch_.data() + kRzNew,
                                           d_scratch_.data() + kRzOld,
                                           d_scratch_.data() + kBeta);
    PcgDirectionKernel<<<blocks, threads, 0, stream>>>(
        d_scratch_.data() + kBeta, z_.data(), p_.data(), n);

    // Convergence poll every kCheckPeriod iterations: one D2H of one float
    // plus a stream sync.  Cheap relative to a full inner-loop sync.
    if (((it + 1) % kCheckPeriod) == 0 ||
        (it + 1) == options_.max_iterations) {
      float r_norm2 = 0.f;
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&r_norm2,
                                          d_scratch_.data() + kRnorm2,
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
