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

#include <cuda_pipeline.h>

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/block_tridiagonal_solver.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch.h"

namespace cunls {
namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/// Pivots below this are replaced (singular systems; no read-back reports them).
constexpr float kPivotFloor = 1e-30f;

__global__ void scatter_matrix_kernel(const float *values, const int *entry_slot, size_t nnz,
                                      float *blocks) {
  const size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (e >= nnz) return;
  const int slot = entry_slot[e];
  if (slot >= 0) blocks[slot] = values[e];
}

__global__ void gather_rhs_kernel(const float *rhs, const int *row_slot, size_t rows,
                                  float *vectors) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r < rows) vectors[row_slot[r]] = rhs[r];
}

__global__ void scatter_solution_kernel(const float *vectors, const int *row_slot, size_t rows,
                                        float *result) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r < rows) result[r] = vectors[row_slot[r]];
}

/** Asynchronous copy of an M x M block from global memory into shared rows of stride LD. */
template <int M, int LD>
__device__ void PrefetchBlock(float *dst, const float *__restrict__ src, int lane) {
#pragma unroll
  for (int e0 = 0; e0 < M * M; e0 += kWarp) {
    const int e = e0 + lane;
    if (e < M * M) __pipeline_memcpy_async(dst + (e / M) * LD + e % M, src + e, sizeof(float));
  }
}

/** Coalesced copy of an M x M block from shared rows of stride LD to global memory. */
template <int M, int LD>
__device__ void StoreBlock(float *__restrict__ dst, const float *src, int lane) {
#pragma unroll
  for (int e0 = 0; e0 < M * M; e0 += kWarp) {
    const int e = e0 + lane;
    if (e < M * M) dst[e] = src[(e / M) * LD + e % M];
  }
}

/**
 * One warp per subproblem: block Cholesky of the block-tridiagonal matrix (the
 * Riccati recursion) and the two substitutions. Lane i owns row i of the stage
 * blocks in registers; pivots and columns move by warp shuffles, rows of W by
 * broadcast reads from shared memory. The blocks of the next stage are
 * prefetched (cp.async) while the current one is processed.
 *
 * Forward, k = 0..K-1:
 *   S = D_k - W_kᵀ W_k  (W_0 = 0);  S = L_k L_kᵀ;  stored over D_k
 *   y_k = L_k⁻¹ (b_k - W_kᵀ y_{k-1});  stored over b_k
 *   W_{k+1} = L_k⁻¹ C_k;  stored over C_k
 * Backward, k = K-1..0:
 *   x_k = L_k⁻ᵀ (y_k - W_{k+1} x_{k+1});  stored over b_k
 *
 * Rows past a stage's size (padding up to M) are identity rows with zero
 * right-hand side; lanes past M only take part in the shuffles.
 */
template <int M>
__global__ void __launch_bounds__(kWarp)
    block_tridiagonal_kernel(int num_stages, const int *__restrict__ stage_sizes,
                             float *__restrict__ blocks, size_t c_offset,
                             float *__restrict__ vectors, int *__restrict__ num_singular) {
  constexpr int LD = M + 1;            // rows read one per lane: conflict free
  constexpr int LW = (M + 3) / 4 * 4;  // rows read by every lane (broadcast, float4)
  constexpr int MM = M * M;
  constexpr unsigned kFull = 0xffffffffu;
  __shared__ float Dbuf[2][M * LD];            // D_k, then L_k
  __shared__ float Cbuf[2][M * LD];            // C_k, then W_{k+1}
  __shared__ __align__(16) float Wsh[M * LW];  // rows of W (the current W_k, then W_{k+1})
  __shared__ float bbuf[2][kWarp];
  __shared__ int size_buf[2][2];  // sizes of stages k and k + 1

  const int p = blockIdx.x;
  const int lane = threadIdx.x;
  const int row = lane < M ? lane : 0;  // lanes past M read row 0 and never write
  const int K = num_stages;
  float *D = blocks + static_cast<size_t>(p) * K * MM;
  float *C = blocks + c_offset + static_cast<size_t>(p) * K * MM;
  float *b = vectors + static_cast<size_t>(p) * K * M;
  const int *sizes = stage_sizes + static_cast<size_t>(p) * K;

  auto prefetch = [&](int k) {
    const int buf = k & 1;
    PrefetchBlock<M, LD>(Dbuf[buf], D + static_cast<size_t>(k) * MM, lane);
    if (k + 1 < K) PrefetchBlock<M, LD>(Cbuf[buf], C + static_cast<size_t>(k) * MM, lane);
    if (lane < M) __pipeline_memcpy_async(&bbuf[buf][lane], b + k * M + lane, sizeof(float));
    if (lane < 2 && k + lane < K) {
      __pipeline_memcpy_async(&size_buf[buf][lane], sizes + k + lane, sizeof(int));
    }
    __pipeline_commit();
  };

  float wcol[M];  // column `lane` of W_k
  float y = 0.f;  // lane i: y_k[i] (forward), x_k[i] (backward)
  prefetch(0);
  for (int k = 0; k < K; ++k) {
    __pipeline_wait_prior(0);
    __syncwarp();
    const int buf = k & 1;
    const int size = size_buf[buf][0];
    const int next_size = k + 1 < K ? size_buf[buf][1] : 0;
    const float bk = lane < size ? bbuf[buf][lane] : 0.f;
    float s[M];
    float c_row[M];
    const float *Ds = Dbuf[buf];
    const float *Cs = Cbuf[buf];
#pragma unroll
    for (int c = 0; c < M; ++c) {
      s[c] = lane < size ? Ds[row * LD + c] : (c == lane ? 1.f : 0.f);
      c_row[c] = (k + 1 < K && lane < size && c < next_size) ? Cs[row * LD + c] : 0.f;
    }
    __syncwarp();
    if (k + 1 < K) prefetch(k + 1);  // the other buffer: free since stage k - 1

    // S -= W_kᵀ W_k: row i needs W_k[r][i] (own column) and the rows of W_k.
    float t = bk;
    if (k > 0) {
#pragma unroll
      for (int r = 0; r < M; ++r) {
        const float wr = wcol[r];
        t -= wr * __shfl_sync(kFull, y, r);  // y holds y_{k-1}
#pragma unroll
        for (int c4 = 0; c4 < LW; c4 += 4) {
          const float4 v = *reinterpret_cast<const float4 *>(&Wsh[r * LW + c4]);
          if (c4 + 0 < M) s[c4 + 0] -= wr * v.x;
          if (c4 + 1 < M) s[c4 + 1] -= wr * v.y;
          if (c4 + 2 < M) s[c4 + 2] -= wr * v.z;
          if (c4 + 3 < M) s[c4 + 3] -= wr * v.w;
        }
      }
    }

    // Cholesky, lane i keeps row i of L_k in s.
    float inv_diag = 1.f;
#pragma unroll
    for (int j = 0; j < M; ++j) {
      const float raw_pivot = __shfl_sync(kFull, s[j], j);
      // A non-positive (or NaN) pivot: the system is singular. Counted, then
      // replaced by a tiny positive one so the recursion completes.
      if (lane == j && !(raw_pivot > kPivotFloor)) atomicAdd(num_singular, 1);
      const float pivot = fmaxf(raw_pivot, kPivotFloor);
      const float inv = rsqrtf(pivot);
      const float d = pivot * inv;
      const float lij = lane > j ? s[j] * inv : (lane == j ? d : 0.f);
      s[j] = lij;
      if (lane == j) inv_diag = inv;
#pragma unroll
      for (int l = j + 1; l < M; ++l) {
        const float llj = __shfl_sync(kFull, lij, l);
        if (lane >= l) s[l] -= lij * llj;
      }
    }

    // y_k = L_k⁻¹ t, column by column.
#pragma unroll
    for (int j = 0; j < M; ++j) {
      const float yj = __shfl_sync(kFull, t * inv_diag, j);
      if (lane == j) y = yj;
      if (lane > j) t -= s[j] * yj;
    }

    // L_k and y_k back to global memory (L through shared, coalesced).
    float *Dw = Dbuf[buf];
    if (lane < M) {
#pragma unroll
      for (int c = 0; c < M; ++c) Dw[lane * LD + c] = s[c];
      b[k * M + lane] = y;
    }
    __syncwarp();
    StoreBlock<M, LD>(D + static_cast<size_t>(k) * MM, Dw, lane);

    // W_{k+1} = L_k⁻¹ C_k, row by row: row j is final once rows < j are
    // eliminated; it is published to Wsh and eliminated from the rows below.
    if (k + 1 < K) {
#pragma unroll
      for (int j = 0; j < M; ++j) {
        if (lane == j) {
#pragma unroll
          for (int c = 0; c < M; ++c) {
            c_row[c] *= inv_diag;
            Wsh[j * LW + c] = c_row[c];
          }
        }
        __syncwarp();
        if (lane > j && lane < M) {
          const float lij = s[j];
#pragma unroll
          for (int c4 = 0; c4 < LW; c4 += 4) {
            const float4 v = *reinterpret_cast<const float4 *>(&Wsh[j * LW + c4]);
            if (c4 + 0 < M) c_row[c4 + 0] -= lij * v.x;
            if (c4 + 1 < M) c_row[c4 + 1] -= lij * v.y;
            if (c4 + 2 < M) c_row[c4 + 2] -= lij * v.z;
            if (c4 + 3 < M) c_row[c4 + 3] -= lij * v.w;
          }
        }
      }
      __syncwarp();
#pragma unroll
      for (int r = 0; r < M; ++r) wcol[r] = Wsh[r * LW + row];
      StoreBlock<M, LW>(C + static_cast<size_t>(k) * MM, Wsh, lane);
    }
    __syncwarp();
  }

  // Back substitution. The forward pass's global writes must be visible to
  // the asynchronous copies below.
  __threadfence_block();
  __syncwarp();
  auto prefetch_back = [&](int k) {
    const int buf = k & 1;
    PrefetchBlock<M, LD>(Dbuf[buf], D + static_cast<size_t>(k) * MM, lane);
    if (k + 1 < K) PrefetchBlock<M, LD>(Cbuf[buf], C + static_cast<size_t>(k) * MM, lane);
    if (lane < M) __pipeline_memcpy_async(&bbuf[buf][lane], b + k * M + lane, sizeof(float));
    __pipeline_commit();
  };
  prefetch_back(K - 1);
  float x_next = 0.f;
  for (int k = K - 1; k >= 0; --k) {
    __pipeline_wait_prior(0);
    __syncwarp();
    const int buf = k & 1;
    const float *L = Dbuf[buf];
    const float *Wn = Cbuf[buf];  // W_{k+1}
    float t = lane < M ? bbuf[buf][lane] : 0.f;
    if (k + 1 < K) {
#pragma unroll
      for (int c = 0; c < M; ++c) t -= Wn[row * LD + c] * __shfl_sync(kFull, x_next, c);
    }
    const float inv_diag = 1.f / L[row * LD + row];
    float x = 0.f;
#pragma unroll
    for (int j = M - 1; j >= 0; --j) {
      const float xj = __shfl_sync(kFull, t * inv_diag, j);
      if (lane == j) x = xj;
      if (lane < j) t -= L[j * LD + row] * xj;
    }
    __syncwarp();
    if (k > 0) prefetch_back(k - 1);
    if (lane < M) b[k * M + lane] = x;
    x_next = x;
  }
}

using KernelPtr = void (*)(int, const int *, float *, size_t, float *, int *);

template <int... Ms>
constexpr std::array<KernelPtr, sizeof...(Ms)> MakeKernelTable(std::integer_sequence<int, Ms...>) {
  return {&block_tridiagonal_kernel<Ms + 1>...};
}

/// block_tridiagonal_kernel<M> for M = 1..32, indexed by M - 1.
constexpr auto kKernels = MakeKernelTable(std::make_integer_sequence<int, 32>{});

template <typename T>
std::vector<T> Download(const T *device, size_t n) {
  std::vector<T> host(n);
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
  return host;
}

bool Fail(const std::string &msg) {
  LogError("BlockTridiagonalSolver: {}", msg);
  return false;
}

}  // namespace

bool BlockTridiagonalSolver::Initialize(cudaStream_t stream, const Problem &problem,
                                        const CSRSparseMatrix &spd_matrix,
                                        const dvector<float> &rhs, dvector<float> &result) {
  num_rows_ = spd_matrix.NumRows();
  if (rhs.size() != num_rows_ || result.size() != num_rows_) {
    return Fail("rhs / result size does not match the matrix");
  }
  const auto &batches = problem.GetStateBatches();
  const auto &stage_ids = problem.StateStages();
  const auto &problem_ids = problem.StateProblemIds();
  if (stage_ids.size() != batches.size()) {
    return Fail("the problem has no state stages (Problem::SetStateStages)");
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Every unknown (reduced row) in the order of the normal equations: state
  // batches in order, their non-constant states in order, tangent components.
  struct Unknown {
    int problem, stage, offset;
  };
  std::vector<Unknown> unknowns;
  unknowns.reserve(num_rows_);
  num_problems_ = static_cast<int>(problem.NumProblems());
  std::vector<std::vector<int>> sizes(num_problems_);  // [p][k]
  for (size_t b = 0; b < batches.size(); ++b) {
    const StateBatch *batch = batches[b];
    const size_t n = batch->NumActiveStates();
    const int tangent = static_cast<int>(batch->TangentSize());
    std::vector<char> constant(n, 0);
    for (int id : Download(batch->ConstStateIds(), batch->NumConstStates())) {
      if (id >= 0 && static_cast<size_t>(id) < n) constant[id] = 1;
    }
    const std::vector<int> stages = Download(stage_ids[b], n);
    const std::vector<int> ids =
        problem_ids.size() == batches.size() ? Download(problem_ids[b], n) : std::vector<int>(n, 0);
    for (size_t s = 0; s < n; ++s) {
      if (constant[s]) continue;
      const int p = ids[s], k = stages[s];
      if (p < 0 || p >= num_problems_ || k < 0) {
        return Fail("invalid subproblem id or stage of state " + std::to_string(s) +
                    " of state batch " + std::to_string(b));
      }
      if (static_cast<int>(sizes[p].size()) <= k) sizes[p].resize(k + 1, 0);
      for (int c = 0; c < tangent; ++c) unknowns.push_back({p, k, sizes[p][k]++});
    }
  }
  if (unknowns.size() != num_rows_) {
    return Fail("the unknowns of the problem do not match the matrix rows");
  }
  num_stages_ = 1;
  stage_size_ = 1;
  for (const auto &s : sizes) {
    num_stages_ = std::max(num_stages_, static_cast<int>(s.size()));
    for (int v : s) stage_size_ = std::max(stage_size_, v);
  }
  if (stage_size_ > kMaxStageSize) {
    return Fail("a stage has " + std::to_string(stage_size_) + " unknowns; at most " +
                std::to_string(kMaxStageSize) + " are supported");
  }
  const int P = num_problems_, K = num_stages_, m = stage_size_;
  const size_t mm = static_cast<size_t>(m) * m;
  const size_t c_offset = static_cast<size_t>(P) * K * mm;

  std::vector<int> stage_sizes(static_cast<size_t>(P) * K, 0);
  for (int p = 0; p < P; ++p) {
    for (size_t k = 0; k < sizes[p].size(); ++k) stage_sizes[p * K + k] = sizes[p][k];
  }
  std::vector<int> row_slot(num_rows_);
  for (size_t r = 0; r < num_rows_; ++r) {
    const Unknown &u = unknowns[r];
    row_slot[r] = (u.problem * K + u.stage) * m + u.offset;
  }
  const std::vector<int> offsets = Download(spd_matrix.row_offsets.data(), num_rows_ + 1);
  const std::vector<int> cols = Download(spd_matrix.col_ids.data(), spd_matrix.NumNonZeros());
  std::vector<int> entry_slot(cols.size(), -1);
  for (size_t r = 0; r < num_rows_; ++r) {
    const Unknown &a = unknowns[r];
    for (int e = offsets[r]; e < offsets[r + 1]; ++e) {
      const Unknown &c = unknowns[cols[e]];
      if (a.problem != c.problem) {
        return Fail("the matrix couples two subproblems");
      }
      const size_t base = static_cast<size_t>(a.problem * K + a.stage) * mm;
      if (c.stage == a.stage) {
        entry_slot[e] = static_cast<int>(base + a.offset * m + c.offset);
      } else if (c.stage == a.stage + 1) {
        entry_slot[e] = static_cast<int>(c_offset + base + a.offset * m + c.offset);
      } else if (c.stage != a.stage - 1) {
        return Fail("the matrix couples stages " + std::to_string(a.stage) + " and " +
                    std::to_string(c.stage) + ": it is not block tridiagonal in stage order");
      }
    }
  }
  if (2 * c_offset > static_cast<size_t>(INT32_MAX)) {
    return Fail("the block storage exceeds 2^31 floats");
  }
  entry_slot_ = dvector<int>(entry_slot);
  row_slot_ = dvector<int>(row_slot);
  stage_sizes_ = dvector<int>(stage_sizes);
  blocks_.resize(2 * c_offset);
  vectors_.resize(static_cast<size_t>(P) * K * m);
  num_singular_.resize(1);  // here, not in Solve: Solve may run under CUDA-graph capture
  LogMessage("BlockTridiagonalSolver: {} subproblems, {} stages, stage blocks of {}", P, K, m);
  return true;
}

bool BlockTridiagonalSolver::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                                   const dvector<float> &rhs, dvector<float> &result) {
  if (spd_matrix.NumRows() != num_rows_ || rhs.size() != num_rows_ || result.size() != num_rows_ ||
      entry_slot_.size() != spd_matrix.NumNonZeros()) {
    return Fail("the system does not match the one the solver was initialized for");
  }
  if (num_rows_ == 0) return true;
  const size_t nnz = spd_matrix.NumNonZeros();
  const size_t c_offset = blocks_.size() / 2;
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(blocks_.data(), 0, blocks_.size() * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(vectors_.data(), 0, vectors_.size() * sizeof(float), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(num_singular_.data(), 0, sizeof(int), stream));
  scatter_matrix_kernel<<<Blocks(nnz), kThreads, 0, stream>>>(
      spd_matrix.values.data(), entry_slot_.data(), nnz, blocks_.data());
  gather_rhs_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(rhs.data(), row_slot_.data(),
                                                                num_rows_, vectors_.data());
  kKernels[stage_size_ - 1]<<<num_problems_, kWarp, 0, stream>>>(
      num_stages_, stage_sizes_.data(), blocks_.data(), c_offset, vectors_.data(),
      num_singular_.data());
  scatter_solution_kernel<<<Blocks(num_rows_), kThreads, 0, stream>>>(
      vectors_.data(), row_slot_.data(), num_rows_, result.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  if (safety_checks_enabled_) {
    int num_singular = 0;
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&num_singular, num_singular_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    if (num_singular > 0) {
      return Fail(std::to_string(num_singular) +
                  " non-positive pivot(s): the system is singular (add damping or priors)");
    }
  }
  return true;
}

}  // namespace cunls
