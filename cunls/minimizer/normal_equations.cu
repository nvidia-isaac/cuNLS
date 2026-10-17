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

#include "cunls/common/log.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

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
  if (UsesBlockStorage()) {
    assembler_.Assemble(stream, problem, jacobians, residuals, bsr_hessian_, rhs);
    return;
  }
  assembler_.Assemble(stream, problem, jacobians, residuals, csr_hessian_, rhs);
}

void NormalEquations::ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  if (UsesBlockStorage()) {
    ExtractDiagonal(stream, bsr_hessian_, diagonal);
    return;
  }
  ExtractDiagonalAt(stream, csr_hessian_, csr_diagonal_, diagonal);
}

void NormalEquations::BuildWorkingLhs(cudaStream_t stream) {
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
  if (UsesBlockStorage()) {
    return solver_->Initialize(stream, problem, bsr_lhs_, rhs, step);
  }
  return solver_->Initialize(stream, problem, csr_lhs_, rhs, step);
}

bool NormalEquations::Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) {
  BuildWorkingLhs(stream);
  if (UsesBlockStorage()) {
    return solver_->Solve(stream, bsr_lhs_, rhs, step);
  }
  return solver_->Solve(stream, csr_lhs_, rhs, step);
}

}  // namespace cunls
