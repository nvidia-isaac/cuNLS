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

#include <utility>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/block_tridiagonal_solver.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

// Stage layout (BlockTridiagonalSolver): H as the solver's blocks. Slot s < c
// (c = P K m m) is entry (i, j) of D block pk = s / (m m): rows (pk, i) and
// (pk, j); slot c + s is entry (i, j) of C block pk: rows (pk, i), (pk + 1, j).
// vec_row maps a stage-vector index pk * m + i to its unknown, or -1 (padding).

/** vec_row[row_slot[r]] = r (vec_row preset to -1). */
__global__ void stage_vec_row_kernel(const int *row_slot, size_t num_rows, int *vec_row) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r < num_rows) vec_row[row_slot[r]] = static_cast<int>(r);
}

/**
 * The working left-hand side from H, one thread per slot: s_r s_c H (column
 * scale), plus the damping on the diagonal, then masked rows and columns
 * cleared to the identity (a positive diagonal is kept), as the CSR / BSR
 * layouts do in sequence. Padding slots are 0.
 */
__global__ void stage_working_lhs_kernel(const float *__restrict__ hessian, float *__restrict__ lhs,
                                         size_t num_slots, size_t c_offset, int m, int K,
                                         const int *__restrict__ vec_row,
                                         const float *__restrict__ scale,
                                         const float *__restrict__ damping,
                                         const float *__restrict__ mask) {
  const size_t s = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (s >= num_slots) return;
  const size_t mm = static_cast<size_t>(m) * m;
  const bool d_block = s < c_offset;
  const size_t t = d_block ? s : s - c_offset;
  const size_t pk = t / mm;
  const int i = static_cast<int>((t % mm) / m), j = static_cast<int>(t % m);
  const int r = vec_row[pk * m + i];
  int c = -1;
  if (d_block) {
    c = vec_row[pk * m + j];
  } else if (static_cast<int>(pk % K) + 1 < K) {
    c = vec_row[(pk + 1) * m + j];
  }
  if (r < 0 || c < 0) {
    lhs[s] = 0.f;
    return;
  }
  float v = hessian[s];
  if (scale != nullptr) v *= scale[r] * scale[c];
  const bool diagonal = d_block && i == j;
  if (diagonal && damping != nullptr) v += damping[r];
  if (mask != nullptr && (mask[r] == 0.f || mask[c] == 0.f)) {
    v = diagonal ? (v > 0.f ? v : 1.f) : 0.f;
  }
  lhs[s] = v;
}

/** diagonal[r] = H_rr from the D block of row r. */
__global__ void stage_diagonal_kernel(const float *__restrict__ hessian,
                                      const int *__restrict__ row_slot, size_t num_rows, int m,
                                      float *__restrict__ diagonal) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r >= num_rows) return;
  const int v = row_slot[r];
  const size_t pk = v / m;
  const int off = v % m;
  diagonal[r] = hessian[pk * m * m + static_cast<size_t>(off) * m + off];
}

/** xv[row_slot[r]] = x[r] (xv zeroed: padding stays 0). */
__global__ void stage_gather_kernel(const float *__restrict__ x, const int *__restrict__ row_slot,
                                    size_t num_rows, float *__restrict__ xv) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r < num_rows) xv[row_slot[r]] = x[r];
}

/**
 * y = H x, one thread per row r = (pk, off): its row of D_k, its row of C_k
 * (stages k, k + 1) and its column of C_{k-1} (H is symmetric; C stores the
 * upper blocks only). xv: x in stage-vector order (padding 0).
 */
__global__ void stage_multiply_kernel(const float *__restrict__ hessian,
                                      const float *__restrict__ xv,
                                      const int *__restrict__ row_slot, size_t num_rows,
                                      size_t c_offset, int m, int K, float *__restrict__ y) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (r >= num_rows) return;
  const int v = row_slot[r];
  const size_t pk = v / m;
  const int off = v % m, k = static_cast<int>(pk % K);
  const size_t mm = static_cast<size_t>(m) * m;
  const float *D = hessian + pk * mm + static_cast<size_t>(off) * m;
  const float *x = xv + pk * m;
  float sum = 0.f;
  for (int j = 0; j < m; ++j) sum += D[j] * x[j];
  if (k + 1 < K) {
    const float *C = hessian + c_offset + pk * mm + static_cast<size_t>(off) * m;
    for (int j = 0; j < m; ++j) sum += C[j] * x[m + j];
  }
  if (k > 0) {
    const float *C = hessian + c_offset + (pk - 1) * mm + off;  // column `off` of C_{k-1}
    for (int i = 0; i < m; ++i) sum += C[static_cast<size_t>(i) * m] * x[i - m];
  }
  y[r] = sum;
}

}  // namespace

NormalEquations::NormalEquations(SparseLinearSolverPtr solver) : solver_(std::move(solver)) {}

void NormalEquations::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                                 const std::vector<bool> &) {
  // Two independent conditions have to hold.  The tangent dimensions must share
  // a factor, or there are no tiles to form; and the solver must be able to read
  // tiles, since a CSR-only backend is better served by a natively assembled CSR
  // than by one expanded from blocks.
  stage_solver_ = nullptr;
  if (auto *stage_solver = dynamic_cast<BlockTridiagonalSolver *>(solver_.get())) {
    // Stage layout: the pattern only gives the solver its block map; H is
    // assembled straight into the stage blocks. The solver's symbolic setup
    // runs here (PrepareSolver() has nothing left to analyze). If it fails,
    // the CSR layout below is set up and PrepareSolver() reports the failure.
    block_size_ = 1;
    assembler_.Initialize(stream, problem, num_cols, csr_hessian_);
    dvector<float> rhs(static_cast<size_t>(num_cols)), result(static_cast<size_t>(num_cols));
    if (stage_solver->Initialize(stream, problem, csr_hessian_, rhs, result)) {
      stage_solver_ = stage_solver;
      stage_hessian_.resize(stage_solver->NumBlockValues());
      const size_t vector_size = static_cast<size_t>(stage_solver->NumProblems()) *
                                 stage_solver->NumStages() * stage_solver->StageSize();
      stage_vec_row_.resize(vector_size);
      stage_vector_.resize(vector_size);
      THROW_ON_CUDA_ERROR(
          cudaMemsetAsync(stage_vec_row_.data(), 0xff, vector_size * sizeof(int), stream));
      const size_t rows = static_cast<size_t>(num_cols);
      if (rows > 0) {
        stage_vec_row_kernel<<<Blocks(rows), kThreads, 0, stream>>>(stage_solver->RowSlots().data(),
                                                                    rows, stage_vec_row_.data());
        THROW_ON_CUDA_ERROR(cudaGetLastError());
      }
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));  // rhs / result go out of scope
      LogMessage("Hessian storage: stage blocks of the block-tridiagonal solver");
      return;
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  const int problem_block_size = ChooseHessianBlockSize(problem);
  block_size_ = solver_->SupportsBlockStorage() ? problem_block_size : 1;

  if (UsesBlockStorage()) {
    assembler_.Initialize(stream, problem, num_cols, block_size_, bsr_hessian_);
    CopyBSRSparseMatrix(stream, bsr_hessian_, bsr_lhs_);  // the pattern, once
    LogMessage("Hessian storage: BSR with block size {}", block_size_);
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
  // The pattern is fixed from here on: copy it and locate the diagonal once.
  CopyCSRSparseMatrix(stream, csr_hessian_, csr_lhs_);
  FindDiagonalPositions(stream, csr_hessian_, csr_diagonal_);
}

void NormalEquations::Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                               const float *residuals, dvector<float> &rhs) {
  if (UsesStageStorage()) {
    assembler_.AssembleIntoSlots(stream, problem, jacobians, residuals, csr_hessian_,
                                 stage_solver_->EntrySlots().data(), stage_hessian_.data(),
                                 stage_solver_->NumUsedBlockValues(), rhs);
    return;
  }
  if (UsesBlockStorage()) {
    assembler_.Assemble(stream, problem, jacobians, residuals, bsr_hessian_, rhs);
    return;
  }
  assembler_.Assemble(stream, problem, jacobians, residuals, csr_hessian_, rhs);
}

void NormalEquations::ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  if (UsesStageStorage()) {
    const size_t rows = stage_solver_->RowSlots().size();
    diagonal.resize(rows);
    if (rows == 0) return;
    stage_diagonal_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        stage_hessian_.data(), stage_solver_->RowSlots().data(), rows, stage_solver_->StageSize(),
        diagonal.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
  if (UsesBlockStorage()) {
    ExtractDiagonal(stream, bsr_hessian_, diagonal);
    return;
  }
  ExtractDiagonalAt(stream, csr_hessian_, csr_diagonal_, diagonal);
}

void NormalEquations::BuildWorkingLhs(cudaStream_t stream) {
  if (UsesStageStorage()) {
    const size_t slots = stage_solver_->NumUsedBlockValues();
    if (slots == 0) return;
    stage_working_lhs_kernel<<<Blocks(slots), kThreads, 0, stream>>>(
        stage_hessian_.data(), stage_solver_->WorkingBlocks(), slots,
        stage_solver_->NumBlockValues() / 2, stage_solver_->StageSize(), stage_solver_->NumStages(),
        stage_vec_row_.data(), ColumnScale() != nullptr ? ColumnScale()->data() : nullptr,
        Damping() != nullptr ? Damping()->data() : nullptr,
        Mask() != nullptr ? Mask()->data() : nullptr);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
  if (UsesBlockStorage()) {
    CopyBSRValues(stream, bsr_hessian_, bsr_lhs_);
    if (ColumnScale() != nullptr) {
      ScaleSymmetric(stream, bsr_lhs_, *ColumnScale(), tile_row_scratch_);
    }
    if (Damping() != nullptr) AddScaledDiagonal(stream, 1.f, *Damping(), bsr_lhs_, bsr_lhs_);
    if (Mask() != nullptr) ZeroMaskedRowsColumns(stream, bsr_lhs_, *Mask());
    return;
  }
  CopyCSRValues(stream, csr_hessian_, csr_lhs_);
  if (ColumnScale() != nullptr) ScaleSymmetricCSR(stream, csr_lhs_, *ColumnScale());
  if (Damping() != nullptr) AddScaledDiagonalAt(stream, 1.f, *Damping(), csr_diagonal_, csr_lhs_);
  if (Mask() != nullptr) ZeroMaskedRowsColumns(stream, csr_lhs_, *Mask());
}

void NormalEquations::MultiplyHessian(cudaStream_t stream, const dvector<float> &x,
                                      dvector<float> &y) const {
  if (UsesStageStorage()) {
    const size_t rows = stage_solver_->RowSlots().size();
    y.resize(rows);
    if (rows == 0) return;
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(stage_vector_.data(), 0, stage_vector_.size() * sizeof(float), stream));
    stage_gather_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        x.data(), stage_solver_->RowSlots().data(), rows, stage_vector_.data());
    stage_multiply_kernel<<<Blocks(rows), kThreads, 0, stream>>>(
        stage_hessian_.data(), stage_vector_.data(), stage_solver_->RowSlots().data(), rows,
        stage_solver_->NumBlockValues() / 2, stage_solver_->StageSize(), stage_solver_->NumStages(),
        y.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
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
  BuildWorkingLhs(stream);
  if (UsesStageStorage()) return true;  // the solver was set up by Initialize()
  if (UsesBlockStorage()) {
    return solver_->Initialize(stream, problem, bsr_lhs_, rhs, step);
  }
  return solver_->Initialize(stream, problem, csr_lhs_, rhs, step);
}

bool NormalEquations::Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) {
  BuildWorkingLhs(stream);
  if (UsesStageStorage()) return stage_solver_->SolveWorkingBlocks(stream, rhs, step);
  solver_->SetRowActivity(RowActivity() != nullptr ? RowActivity()->data() : nullptr);
  if (UsesBlockStorage()) {
    return solver_->Solve(stream, bsr_lhs_, rhs, step);
  }
  return solver_->Solve(stream, csr_lhs_, rhs, step);
}

}  // namespace cunls
