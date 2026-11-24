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

#include <stdexcept>
#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

/**
 * row_of[e] = the row of entry e of a CSR pattern, or the block row of tile e
 * of a BSR pattern; one thread per (block) row.
 */
__global__ void row_of_entry_kernel(const int *__restrict__ row_offsets, int num_rows,
                                    int *__restrict__ row_of) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_rows) return;
  for (int e = row_offsets[r]; e < row_offsets[r + 1]; ++e) row_of[e] = r;
}

/**
 * The working left-hand side written straight into the solver's storage, one
 * thread per value e of H's pattern: s_r s_c H_e (column scale), plus the
 * damping on the diagonal, then a masked row or column cleared to the
 * identity (a positive diagonal is kept) — stored at slot(e) (the identity, entry_slot[e] or
 * entry_slot64[e]; -1: not stored).
 *
 * Rows and columns: scalar CSR, entry_row[e] and col_ids[e]; BSR (kBlock),
 * tile t = e / b², with its block row in `rows` and block column in col_ids.
 */
template <bool kBlock, typename Slot>
__global__ void write_working_lhs_kernel(
    size_t num_values, int block_size, const int *__restrict__ rows,
    const int *__restrict__ col_ids, const float *__restrict__ hessian,
    const Slot *__restrict__ entry_slot, const float *__restrict__ scale,
    const float *__restrict__ damping, const float *__restrict__ mask,
    float *__restrict__ storage) {
  const size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (e >= num_values) return;
  int64_t slot = static_cast<int64_t>(e);
  if (entry_slot != nullptr) {
    slot = static_cast<int64_t>(entry_slot[e]);
    if (slot < 0) return;
  }
  int r, c;
  if (kBlock) {
    const int area = block_size * block_size;
    const size_t t = e / area;
    const int sub = static_cast<int>(e % area);
    r = rows[t] * block_size + sub / block_size;
    c = col_ids[t] * block_size + sub % block_size;
  } else {
    r = rows[e];
    c = col_ids[e];
  }
  float v = hessian[e];
  if (scale != nullptr) v *= scale[r] * scale[c];
  if (r == c && damping != nullptr) v += damping[r];
  if (mask != nullptr && (mask[r] == 0.f || mask[c] == 0.f)) {
    v = r == c ? (v > 0.f ? v : 1.f) : 0.f;
  }
  storage[slot] = v;
}

}  // namespace

NormalEquations::NormalEquations(SparseLinearSolverPtr solver) : solver_(std::move(solver)) {}

void NormalEquations::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                                 const std::vector<bool> &) {
  // Two independent conditions have to hold.  The tangent dimensions must share
  // a factor, or there are no tiles to form; and the solver must be able to read
  // tiles, since a CSR-only backend is better served by a natively assembled CSR
  // than by one expanded from blocks.
  const int problem_block_size = ChooseHessianBlockSize(problem);
  block_size_ = solver_->SupportsBlockStorage() ? problem_block_size : 1;

  if (UsesBlockStorage()) {
    assembler_.Initialize(stream, problem, num_cols, block_size_, bsr_hessian_);
    // The working left-hand side lives in the solver; only the tile rows are needed here.
    row_of_entry_.resize(bsr_hessian_.NumBlocks());
    if (bsr_hessian_.num_block_rows > 0) {
      row_of_entry_kernel<<<Blocks(bsr_hessian_.num_block_rows), kThreads, 0, stream>>>(
          bsr_hessian_.row_offsets.data(), bsr_hessian_.num_block_rows, row_of_entry_.data());
      THROW_ON_CUDA_ERROR(cudaGetLastError());
    }
    LogMessage("Hessian storage: BSR with block size {}", block_size_);
    hessian_step_.resize(num_cols);  // StepCurvature() allocates nothing in a solve
    return;
  }

  // Say which of the two conditions failed.  Falling back to scalar costs the
  // block-storage delta, and a silent fallback is the kind of thing that only
  // shows up as an unexplained regression in a profile.
  if (problem_block_size <= 1) {
    LogMessage("Hessian storage: CSR (tangent dimensions share no common factor)");
  } else {
    LogMessage(
        "Hessian storage: CSR (solver does not accept block storage; "
        "problem would have supported block size {})",
        problem_block_size);
  }

  assembler_.Initialize(stream, problem, num_cols, csr_hessian_);
  // The pattern is fixed from here on: locate the diagonal and the row of
  // every entry once (the working left-hand side lives in the solver).
  FindDiagonalPositions(stream, csr_hessian_, csr_diagonal_);
  row_of_entry_.resize(csr_hessian_.NumNonZeros());
  const int n = static_cast<int>(csr_hessian_.NumRows());
  if (n > 0) {
    row_of_entry_kernel<<<Blocks(n), kThreads, 0, stream>>>(csr_hessian_.row_offsets.data(), n,
                                                            row_of_entry_.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  // StepCurvature() allocates nothing in a solve: H δ, and cuSPARSE's handle and SpMV workspace
  // (one product with the pattern's current values; the result is not used).
  hessian_step_.resize(num_cols);
  if (n > 0) {
    const dvector<float> x(0.f, n);
    MultiplyHessian(stream, x, hessian_step_);
  }
}

void NormalEquations::Assemble(cudaStream_t stream, const Problem &, const float *jacobians,
                               const float *residuals, dvector<float> &rhs) {
  if (UsesBlockStorage()) {
    assembler_.Assemble(stream, jacobians, residuals, bsr_hessian_, rhs);
    return;
  }
  assembler_.Assemble(stream, jacobians, residuals, csr_hessian_, rhs);
}

void NormalEquations::ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  if (UsesBlockStorage()) {
    ExtractDiagonal(stream, bsr_hessian_, diagonal);
    return;
  }
  ExtractDiagonalAt(stream, csr_hessian_, csr_diagonal_, diagonal);
}

void NormalEquations::WriteWorkingLhsToSolver(cudaStream_t stream) {
  const SolverMatrixStorage storage = solver_->MatrixStorage();
  const size_t num_rows =
      UsesBlockStorage() ? static_cast<size_t>(bsr_hessian_.NumRows()) : csr_hessian_.NumRows();
  if (storage.values == nullptr && num_rows > 0) {
    throw std::logic_error(
        "NormalEquations::WriteWorkingLhsToSolver: the solver has no matrix storage (got null "
        "MatrixStorage().values, expected the storage of an initialized solver). Fix: call "
        "PrepareSolver() before Solve().");
  }
  solver_->ClearMatrixStorage(stream);
  const bool block = UsesBlockStorage();
  const size_t num_values = block ? bsr_hessian_.values.size() : csr_hessian_.values.size();
  if (num_values == 0) return;
  const int *rows = row_of_entry_.data();
  const int *col_ids = block ? bsr_hessian_.col_ids.data() : csr_hessian_.col_ids.data();
  const float *hessian = block ? bsr_hessian_.values.data() : csr_hessian_.values.data();
  const float *scale = ColumnScale() != nullptr ? ColumnScale()->data() : nullptr;
  const float *damping = Damping() != nullptr ? Damping()->data() : nullptr;
  const float *mask = Mask() != nullptr ? Mask()->data() : nullptr;
  auto launch = [&](auto kernel, const auto *slot) {
    kernel<<<Blocks(num_values), kThreads, 0, stream>>>(num_values, block_size_, rows, col_ids,
                                                        hessian, slot, scale, damping, mask,
                                                        storage.values);
  };
  if (storage.entry_slot64 != nullptr) {
    if (block) {
      launch(write_working_lhs_kernel<true, int64_t>, storage.entry_slot64);
    } else {
      launch(write_working_lhs_kernel<false, int64_t>, storage.entry_slot64);
    }
  } else if (block) {
    launch(write_working_lhs_kernel<true, int>, storage.entry_slot);
  } else {
    launch(write_working_lhs_kernel<false, int>, storage.entry_slot);
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void NormalEquations::MultiplyHessian(cudaStream_t stream, const dvector<float> &x,
                                      dvector<float> &y) const {
  if (UsesBlockStorage()) {
    MultiplyBSRByDenseVector(stream, bsr_hessian_, x, y);
    return;
  }
  const int n = static_cast<int>(csr_hessian_.NumRows());
  MultiplyCSRByDenseVector(stream, cusparse_handle_.GetHandle(stream), csr_hessian_, n, n,
                           static_cast<int>(csr_hessian_.NumNonZeros()), x, y, spmv_buffer_);
}

void NormalEquations::StepCurvature(cudaStream_t stream, const dvector<float> &step,
                                    ProblemPartition &partition, float *out) const {
  hessian_step_.resize(step.size());
  MultiplyHessian(stream, step, hessian_step_);
  partition.SumRows(stream, step.data(), hessian_step_.data(), nullptr, out);
}

bool NormalEquations::PrepareSolver(cudaStream_t stream, const Problem &problem,
                                    const dvector<float> &rhs, dvector<float> &step) {
  // The solver analyzes the pattern; each Solve() writes the matrix into its storage.
  if (UsesBlockStorage()) return solver_->Initialize(stream, problem, bsr_hessian_, rhs, step);
  return solver_->Initialize(stream, problem, csr_hessian_, rhs, step);
}

bool NormalEquations::Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) {
  solver_->SetRowActivity(RowActivity() != nullptr ? RowActivity()->data() : nullptr);
  WriteWorkingLhsToSolver(stream);
  return solver_->SolveStoredMatrix(stream, rhs, step);
}

}  // namespace cunls
