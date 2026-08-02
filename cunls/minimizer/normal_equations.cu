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

#include "cunls/common/log.h"
#include "cunls/minimizer/normal_equations.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

void NormalEquations::Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                                 bool solver_supports_block_storage) {
  csr_dims_.Invalidate();

  // Block storage only pays off when the tangent dimensions share a factor and
  // the solver can read tiles; a CSR-only backend would just have to expand
  // them again.
  block_size_ = solver_supports_block_storage ? ChooseHessianBlockSize(problem) : 1;

  if (UsesBlockStorage()) {
    assembler_.Initialize(stream, problem, num_cols, block_size_, bsr_hessian_);
    LogMessage("Hessian storage: BSR with block size {}", block_size_);
    return;
  }

  assembler_.Initialize(stream, problem, num_cols, csr_hessian_);
  int num_rows = 0, num_matrix_cols = 0, num_nonzeros = 0;
  ExtractMatrixMetadata(stream, csr_hessian_, num_rows, num_matrix_cols, num_nonzeros);
  csr_dims_.Set(num_rows, num_matrix_cols, num_nonzeros);
}

void NormalEquations::Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                               const float *residuals, dvector<float> &rhs) {
  if (UsesBlockStorage()) {
    assembler_.Assemble(stream, problem, jacobians, residuals, bsr_hessian_, rhs);
    CopyBSRSparseMatrix(stream, bsr_hessian_, bsr_lhs_);
    return;
  }
  assembler_.Assemble(stream, problem, jacobians, residuals, csr_hessian_, rhs);
  CopyCSRSparseMatrix(stream, csr_hessian_, csr_lhs_);
}

void NormalEquations::ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  if (UsesBlockStorage()) {
    ExtractDiagonal(stream, bsr_hessian_, diagonal);
    return;
  }
  ExtractDiagonal(stream, csr_hessian_, diagonal);
}

void NormalEquations::ExtractLhsDiagonal(cudaStream_t stream, dvector<float> &diagonal) const {
  if (UsesBlockStorage()) {
    ExtractDiagonal(stream, bsr_lhs_, diagonal);
    return;
  }
  ExtractDiagonal(stream, csr_lhs_, diagonal);
}

void NormalEquations::AddScaledDiagonalToLhs(cudaStream_t stream, float scale,
                                             const dvector<float> &diagonal) {
  if (UsesBlockStorage()) {
    AddScaledDiagonal(stream, scale, diagonal, bsr_lhs_, bsr_lhs_);
    return;
  }
  AddScaledDiagonal(stream, scale, diagonal, csr_lhs_, csr_lhs_);
}

void NormalEquations::ScaleLhsSymmetric(cudaStream_t stream, const dvector<float> &scale) {
  if (UsesBlockStorage()) {
    ScaleSymmetric(stream, bsr_lhs_, scale, tile_row_scratch_);
    return;
  }
  ScaleSymmetricCSR(stream, csr_lhs_, scale);
}

void NormalEquations::WeightedSquaredStepAsync(cudaStream_t stream, void *cusparse_handle,
                                               const dvector<float> &step, float *d_out,
                                               float *d_partials, dvector<uint8_t> &buffer) {
  if (UsesBlockStorage()) {
    ComputeWeightedSquaredStepAsync(stream, bsr_hessian_, step, spmv_scratch_, d_out, d_partials);
    return;
  }
  ComputeWeightedSquaredStepAsync(stream, cusparse_handle, csr_hessian_, csr_dims_.num_rows,
                                  csr_dims_.num_cols, csr_dims_.num_nonzeros, step, spmv_scratch_,
                                  buffer, d_out, d_partials);
}

bool NormalEquations::InitializeSolver(cudaStream_t stream, CSRSparseLinearSolver &solver,
                                       const Problem &problem, const dvector<float> &rhs,
                                       dvector<float> &step) {
  if (UsesBlockStorage()) {
    return solver.Initialize(stream, problem, bsr_lhs_, rhs, step);
  }
  return solver.Initialize(stream, problem, csr_lhs_, rhs, step);
}

bool NormalEquations::Solve(cudaStream_t stream, CSRSparseLinearSolver &solver,
                            const dvector<float> &rhs, dvector<float> &step) {
  if (UsesBlockStorage()) {
    return solver.Solve(stream, bsr_lhs_, rhs, step);
  }
  return solver.Solve(stream, csr_lhs_, rhs, step);
}

}  // namespace cunls
