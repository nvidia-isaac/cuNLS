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

#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/dense_linear_solver.h"

namespace cunls {
namespace {

/// Absolute threshold below which a pivot or diagonal element is considered
/// numerically zero, causing the factorization or solve to report failure.
constexpr float kDiagonalEpsilonAbs = 1e-7f;

constexpr int kWarpSize = 32;
constexpr int kMaxWarpsPerBlock = 8;

/// Number of int status flags exchanged between the GPU kernels and the host.
/// Index 0 holds the factorization result; index 1 holds the solve result.
constexpr int kNumStatuses = 2;

constexpr int kMaxBlockSize = kMaxWarpsPerBlock * kWarpSize;

/// @brief Selects the smallest block size (a power of two, at most 256) that
///        is >= n.  Keeping the block small when n is small reduces wasted
///        threads that would just idle in the synchronization barriers.
inline int SelectBlockSize(int n) {
  if (n <= 32) return 32;
  if (n <= 64) return 64;
  if (n <= 128) return 128;
  return kMaxBlockSize;
}

// ---------------------------------------------------------------------------
// Warp-level primitives
// ---------------------------------------------------------------------------

/// @brief Warp-level reduction that finds the lane holding the maximum
///        absolute value, propagating both the value and its index to lane 0.
__device__ __forceinline__ void WarpReduceMaxAbs(float& val, int& idx) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const float other_val = __shfl_down_sync(0xffffffff, val, offset);
    const int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

/// @brief Warp-level sum reduction.
__device__ __forceinline__ float WarpReduceSum(float val) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// CSR -> Dense conversion kernel
// ---------------------------------------------------------------------------

/// @brief Scatters CSR values into a dense row-major matrix.
///
/// Each warp processes one row: threads in the warp iterate over the row's
/// non-zero entries in parallel and write them to the corresponding column
/// position in the dense output.  The caller must zero-initialize
/// @p dense_matrix before launch.
__global__ void csr_to_dense_kernel(const int* __restrict__ row_offsets,
                                    const int* __restrict__ col_ids,
                                    const float* __restrict__ values,
                                    int num_rows,
                                    float* __restrict__ dense_matrix) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
  if (row >= num_rows) {
    return;
  }
  const int lane = threadIdx.x % kWarpSize;
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  float* const dense_row = dense_matrix + row * num_rows;
  for (int idx = row_start + lane; idx < row_end; idx += kWarpSize) {
    dense_row[col_ids[idx]] = values[idx];
  }
}

// ---------------------------------------------------------------------------
// Pivoted LDLT factorization kernel
// ---------------------------------------------------------------------------

/// @brief Single-block kernel that computes P^T A P = L D L^T in-place.
///
/// The algorithm at each step k = 0 .. n-1:
///   1. **Diagonal pivoting**: find the row/column with the largest absolute
///      diagonal element among indices [k, n).  When @p check_status is
///      non-zero and the best pivot is below kDiagonalEpsilonAbs, the kernel
///      sets *status = 0 and returns early.
///   2. **Symmetric swap**: swap rows k and pk, then columns k and pk, and
///      record the permutation.
///   3. **Schur complement update**: divide column k below the diagonal by the
///      pivot, then rank-1 update the trailing (n-k-1) x (n-k-1) sub-matrix.
///
/// On success *status is set to 1 (only written when @p check_status != 0).
///
/// @param A        Dense row-major input matrix (n x n), read-only.
/// @param n        Matrix dimension.
/// @param ldlt     Output: L and D packed in one n x n buffer.
/// @param permutation Output: permutation vector of length n.
/// @param status   Output: 1 on success, 0 on singular pivot (only when
///                 @p check_status != 0).
/// @param check_status  When non-zero, enable pivot-value checks and status
///                      reporting; when zero, skip them for lower latency.
__global__ void factorize_symmetric_pivoted_ldlt_kernel(
    const float* __restrict__ A, int n, float* __restrict__ ldlt,
    int* __restrict__ permutation, int* __restrict__ status,
    int check_status) {
  if (blockIdx.x != 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int num_warps = block_size / kWarpSize;

  __shared__ float warp_vals[kMaxWarpsPerBlock];
  __shared__ int warp_idxs[kMaxWarpsPerBlock];
  __shared__ int pivot_index_sh;
  __shared__ int factorization_ok;

  if (tid == 0) {
    factorization_ok = 1;
  }

  // Copy A -> ldlt and initialize the identity permutation.
  for (int idx = tid; idx < n * n; idx += block_size) {
    ldlt[idx] = A[idx];
  }
  for (int i = tid; i < n; i += block_size) {
    permutation[i] = i;
  }
  __syncthreads();

  for (int k = 0; k < n; ++k) {
    // --- Step 1: find the best diagonal pivot in [k, n) ------------------
    float local_best = -1.0f;
    int local_idx = k;
    for (int i = k + tid; i < n; i += block_size) {
      const float a = fabsf(ldlt[i * n + i]);
      if (a > local_best) {
        local_best = a;
        local_idx = i;
      }
    }

    // Two-level reduction: first within each warp, then across warps.
    WarpReduceMaxAbs(local_best, local_idx);
    if (lane == 0) {
      warp_vals[warp_id] = local_best;
      warp_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
      float v = (lane < num_warps) ? warp_vals[lane] : -1.0f;
      int ix = (lane < num_warps) ? warp_idxs[lane] : k;
      WarpReduceMaxAbs(v, ix);
      if (lane == 0) {
        pivot_index_sh = ix;
        if (check_status && v <= kDiagonalEpsilonAbs) {
          factorization_ok = 0;
        }
      }
    }
    __syncthreads();
    if (check_status && !factorization_ok) {
      if (tid == 0) {
        *status = 0;
      }
      return;
    }

    // --- Step 2: symmetric row/column swap if pivot != k -----------------
    const int pk = pivot_index_sh;
    if (pk != k) {
      for (int j = tid; j < n; j += block_size) {
        const float tmp = ldlt[k * n + j];
        ldlt[k * n + j] = ldlt[pk * n + j];
        ldlt[pk * n + j] = tmp;
      }
      __syncthreads();
      for (int i = tid; i < n; i += block_size) {
        const float tmp = ldlt[i * n + k];
        ldlt[i * n + k] = ldlt[i * n + pk];
        ldlt[i * n + pk] = tmp;
      }
      if (tid == 0) {
        const int tmp = permutation[k];
        permutation[k] = permutation[pk];
        permutation[pk] = tmp;
      }
      __syncthreads();
    }

    const float pivot = ldlt[k * n + k];
    if (check_status) {
      if (tid == 0 && fabsf(pivot) <= kDiagonalEpsilonAbs) {
        factorization_ok = 0;
      }
      __syncthreads();
      if (!factorization_ok) {
        if (tid == 0) *status = 0;
        return;
      }
    }

    // --- Step 3: Schur complement update ---------------------------------
    const float inv_pivot = 1.0f / pivot;
    // Scale sub-diagonal column k by 1/pivot to form L(:,k).
    for (int i = k + 1 + tid; i < n; i += block_size) {
      ldlt[i * n + k] *= inv_pivot;
    }
    __syncthreads();

    // Rank-1 update of the trailing sub-matrix.
    const int remaining = n - k - 1;
    const int total = remaining * remaining;
    for (int idx = tid; idx < total; idx += block_size) {
      const int i = k + 1 + idx / remaining;
      const int j = k + 1 + idx % remaining;
      ldlt[i * n + j] -= ldlt[i * n + k] * pivot * ldlt[j * n + k];
    }
    __syncthreads();
  }

  if (check_status && tid == 0) {
    *status = 1;
  }
}

// ---------------------------------------------------------------------------
// Triangular / diagonal solve kernel
// ---------------------------------------------------------------------------

/// @brief Single-block kernel that solves L D L^T x_p = P b, then un-permutes.
///
/// Given the factored form P^T A P = L D L^T from the factorization kernel,
/// this kernel performs:
///   1. Permute the RHS:  permuted_rhs = P * rhs.
///   2. Forward substitution:  L * y = permuted_rhs  (L has unit diagonal).
///   3. Diagonal solve:  D * z = y  (when @p check_status != 0, checks for
///      zero diagonals and sets *status = 0 / returns early).
///   4. Backward substitution:  L^T * x_p = z.
///   5. Un-permute:  solution[permutation[i]] = x_p[i].
///
/// @param ldlt        Factored matrix (L and D packed, n x n, row-major).
/// @param permutation Pivot permutation vector of length n.
/// @param rhs         Right-hand side vector of length n.
/// @param n           System dimension.
/// @param permuted_rhs     Scratch buffer of length n.
/// @param intermediate     Scratch buffer of length n (y, then z).
/// @param permuted_sol     Scratch buffer of length n (x_p).
/// @param solution         Output solution vector of length n.
/// @param status           Output: 1 on success, 0 on zero diagonal (only
///                         when @p check_status != 0).
/// @param check_status  When non-zero, enable diagonal checks and status
///                      reporting; when zero, skip them for lower latency.
__global__ void solve_from_pivoted_ldlt_kernel(
    const float* __restrict__ ldlt, const int* __restrict__ permutation,
    const float* __restrict__ rhs, int n, float* __restrict__ permuted_rhs,
    float* __restrict__ intermediate, float* __restrict__ permuted_sol,
    float* __restrict__ solution, int* __restrict__ status,
    int check_status) {
  if (blockIdx.x != 0) {
    return;
  }

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int num_warps = block_size / kWarpSize;

  __shared__ float warp_sums[kMaxWarpsPerBlock];
  __shared__ int valid_solve;

  if (tid == 0) {
    valid_solve = 1;
  }
  __syncthreads();

  // Step 1: permute the RHS.
  for (int i = tid; i < n; i += block_size) {
    permuted_rhs[i] = rhs[permutation[i]];
  }
  __syncthreads();

  // Step 2: forward substitution  L y = P b.
  for (int i = 0; i < n; ++i) {
    float partial = 0.0f;
    for (int j = tid; j < i; j += block_size) {
      partial += ldlt[i * n + j] * intermediate[j];
    }
    partial = WarpReduceSum(partial);
    if (lane == 0) {
      warp_sums[warp_id] = partial;
    }
    __syncthreads();
    if (warp_id == 0) {
      float s = (lane < num_warps) ? warp_sums[lane] : 0.0f;
      s = WarpReduceSum(s);
      if (lane == 0) {
        intermediate[i] = permuted_rhs[i] - s;
      }
    }
    __syncthreads();
  }

  // Step 3: diagonal solve  D z = y.
  if (check_status) {
    for (int i = tid; i < n; i += block_size) {
      if (fabsf(ldlt[i * n + i]) <= kDiagonalEpsilonAbs) {
        atomicExch(&valid_solve, 0);
      }
    }
    __syncthreads();
    if (valid_solve == 0) {
      if (tid == 0) {
        *status = 0;
      }
      return;
    }
  }

  for (int i = tid; i < n; i += block_size) {
    intermediate[i] /= ldlt[i * n + i];
  }
  __syncthreads();

  // Step 4: backward substitution  L^T x_p = z.
  for (int i = n - 1; i >= 0; --i) {
    float partial = 0.0f;
    for (int j = i + 1 + tid; j < n; j += block_size) {
      partial += ldlt[j * n + i] * permuted_sol[j];
    }
    partial = WarpReduceSum(partial);
    if (lane == 0) {
      warp_sums[warp_id] = partial;
    }
    __syncthreads();
    if (warp_id == 0) {
      float s = (lane < num_warps) ? warp_sums[lane] : 0.0f;
      s = WarpReduceSum(s);
      if (lane == 0) {
        permuted_sol[i] = intermediate[i] - s;
      }
    }
    __syncthreads();
  }

  // Step 5: un-permute to obtain the solution in the original ordering.
  for (int i = tid; i < n; i += block_size) {
    solution[permutation[i]] = permuted_sol[i];
  }

  if (check_status && tid == 0) {
    *status = 1;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host-side launcher functions
// ---------------------------------------------------------------------------

/// @brief Launches the pivoted LDLT factorization kernel on the given stream.
///
/// @param status  Device pointer to a single int; set to 1 on success, 0 on
///                failure (singular or near-singular pivot). Only written when
///                @p check_status is true.
/// @param check_status  When true, the kernel checks pivots for near-zero
///                      values and reports status; when false, skips checks.
void FactorizeSymmetricPivotedLDLT(cudaStream_t stream,
                                   const float* dense_symmetric_matrix, int n,
                                   float* ldlt_factor, int* permutation,
                                   int* status, bool check_status) {
  if (n == 0) {
    return;
  }

  const int threads = SelectBlockSize(n);
  factorize_symmetric_pivoted_ldlt_kernel<<<1, threads, 0, stream>>>(
      dense_symmetric_matrix, n, ldlt_factor, permutation, status,
      check_status ? 1 : 0);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/// @brief Launches the LDLT solve kernel on the given stream.
///
/// @param status  Device pointer to a single int; set to 1 on success, 0 on
///                failure (zero diagonal in D). Only written when
///                @p check_status is true.
/// @param check_status  When true, the kernel checks diagonal elements and
///                      reports status; when false, skips checks.
void SolveFromPivotedLDLT(cudaStream_t stream, const float* ldlt_factor,
                          const int* permutation, const float* rhs, int n,
                          float* permuted_rhs, float* intermediate_solution,
                          float* permuted_solution, float* solution,
                          int* status, bool check_status) {
  if (n == 0) {
    return;
  }
  const int threads = SelectBlockSize(n);
  solve_from_pivoted_ldlt_kernel<<<1, threads, 0, stream>>>(
      ldlt_factor, permutation, rhs, n, permuted_rhs, intermediate_solution,
      permuted_solution, solution, status, check_status ? 1 : 0);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// DenseLDLTSolver public API
// ---------------------------------------------------------------------------

bool DenseLDLTSolver::Initialize(cudaStream_t stream,
                                 const CSRSparseMatrix& spd_matrix,
                                 const dvector<float>& rhs,
                                 dvector<float>& result) {
  (void)stream;
  const size_t matrix_size = spd_matrix.NumRows();
  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size,
             rhs.size());
    return false;
  }
  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size,
             result.size());
    return false;
  }
  EnsureBuffersSize(matrix_size);
  return true;
}

bool DenseLDLTSolver::Solve(cudaStream_t stream,
                            const CSRSparseMatrix& spd_matrix,
                            const dvector<float>& rhs, dvector<float>& result) {
  const size_t matrix_size = spd_matrix.NumRows();
  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size,
             rhs.size());
    return false;
  }
  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size,
             result.size());
    return false;
  }
  if (matrix_size == 0) {
    return true;
  }

  EnsureBuffersSize(matrix_size);

  ConvertCSRToDense(stream, spd_matrix, dense_matrix_);

  const int n = static_cast<int>(matrix_size);

  FactorizeSymmetricPivotedLDLT(stream, dense_matrix_.data(), n,
                                ldlt_factor_.data(), permutation_.data(),
                                status_.data(), safety_checks_enabled_);

  SolveFromPivotedLDLT(stream, ldlt_factor_.data(), permutation_.data(),
                       rhs.data(), n, permuted_rhs_.data(),
                       intermediate_solution_.data(), permuted_solution_.data(),
                       result.data(), status_.data() + 1,
                       safety_checks_enabled_);

  if (safety_checks_enabled_) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
        status_pinned_.data(), status_.data(),
        kNumStatuses * sizeof(int), cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (status_pinned_[0] == 0) {
      LogError("LDLT factorization failed (singular or near-singular pivot)");
      return false;
    }
    if (status_pinned_[1] == 0) {
      LogError("LDLT solve failed (zero diagonal encountered)");
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// DenseLDLTSolver private helpers
// ---------------------------------------------------------------------------

void DenseLDLTSolver::EnsureBuffersSize(size_t n) {
  const size_t matrix_elements = n * n;
  if (dense_matrix_.size() != matrix_elements) {
    dense_matrix_.resize(matrix_elements);
  }
  if (ldlt_factor_.size() != matrix_elements) {
    ldlt_factor_.resize(matrix_elements);
  }
  if (permutation_.size() != n) {
    permutation_.resize(n);
  }
  if (permuted_rhs_.size() != n) {
    permuted_rhs_.resize(n);
  }
  if (permuted_solution_.size() != n) {
    permuted_solution_.resize(n);
  }
  if (intermediate_solution_.size() != n) {
    intermediate_solution_.resize(n);
  }
  if (status_.size() != kNumStatuses) {
    status_.resize(kNumStatuses);
    status_pinned_.resize(kNumStatuses);
  }
}

void DenseLDLTSolver::ConvertCSRToDense(cudaStream_t stream,
                                        const CSRSparseMatrix& matrix,
                                        dvector<float>& dense_matrix) {
  const int num_rows = static_cast<int>(matrix.NumRows());
  if (num_rows == 0) {
    return;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(
      dense_matrix.data(), 0,
      static_cast<size_t>(num_rows) * num_rows * sizeof(float), stream));

  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  csr_to_dense_kernel<<<blocks, kThreads, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
      num_rows, dense_matrix.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
