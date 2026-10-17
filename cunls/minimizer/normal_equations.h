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

#pragma once

#include <cuda_runtime.h>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/block_hessian_assembler.h"
#include "cunls/minimizer/bsr_matrix.h"
#include "cunls/minimizer/linear_system.h"

namespace cunls {

class Problem;
class ProblemPartition;

/**
 * @brief The assembled normal equations (LinearSystemType::Assembled).
 *
 * Holds two matrices: the Hessian `H` as assembled, and the working
 * left-hand side handed to the solver, rebuilt from `H` before every solver
 * call (PrepareSolver(), Solve()) with the declared column scale, damping and
 * mask. Both are stored in one of two
 * layouts, chosen once at Initialize():
 *
 * - **Block (BSR)** when the state tangent dimensions share a common factor and
 *   the solver can read tiles. One column index per dense tile instead of one
 *   per scalar entry, which is bandwidth the solver's SpMV no longer moves.
 * - **Scalar (CSR)** otherwise.
 *
 * Which layout is live is an implementation detail; every operation
 * dispatches internally.
 *
 * A BSR matrix with `block_size == 1` is byte-for-byte the same layout as CSR,
 * so the scalar case could in principle be expressed as block storage too. It
 * is kept separate deliberately: at `block_size == 1` every block operation
 * degenerates to its scalar counterpart but runs through less-optimized code —
 * notably cuSPARSE's `csrmv`, which reaches ~90% of peak bandwidth and which a
 * one-tile-per-entry BSR kernel cannot beat. Block storage is a win only when
 * the tiles are real.
 */
class NormalEquations : public LinearSystem {
 public:
  /** @brief Takes ownership of the solver of the working left-hand side (not null). */
  explicit NormalEquations(SparseLinearSolverPtr solver);

  /**
   * @brief Derives the sparsity pattern of H and picks the storage layout:
   * block storage when the tangent sizes share a factor and the solver
   * accepts it (SparseLinearSolver::SupportsBlockStorage()), scalar CSR
   * otherwise. Every batch is evaluated in its natural order (`reorderable`
   * is not used).
   */
  void Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                  const std::vector<bool> &reorderable) override;

  /** @copydoc LinearSystem::JacobianValuesSize */
  size_t JacobianValuesSize() const override { return assembler_.JacobianValuesSize(); }

  /** @brief Assembles `H` and `rhs` = -Jᵀr. */
  void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                const float *residuals, dvector<float> &rhs) override;

  /** @copydoc LinearSystem::ExtractHessianDiagonal */
  void ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const override;

  /** @brief Builds the working left-hand side and runs the solver's symbolic setup. */
  bool PrepareSolver(cudaStream_t stream, const Problem &problem, const dvector<float> &rhs,
                     dvector<float> &step) override;

  /** @brief Builds the working left-hand side and solves with it. */
  bool Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) override;

  /** @brief δᵀHδ per subproblem, from H δ (SpMV) in either layout. */
  void StepCurvature(cudaStream_t stream, const dvector<float> &step, ProblemPartition &partition,
                     float *out) const override;

  // Test hooks: the matrices and products the layouts produce.

  /** @brief y = H x with the assembled, undamped Hessian, in either layout. */
  void MultiplyHessian(cudaStream_t stream, const dvector<float> &x, dvector<float> &y) const;

  /** @brief True when the block layout is live for the current problem. */
  bool UsesBlockStorage() const { return block_size_ > 1; }

  /**
   * @brief Rebuilds the working left-hand side from H with the declared scale,
   * damping and mask (PrepareSolver() and Solve() do this first).
   */
  void BuildWorkingLhs(cudaStream_t stream);

  /** @brief Working left-hand side in scalar storage; empty under block storage. */
  const CSRSparseMatrix &LhsCSR() const { return csr_lhs_; }

  /** @brief Working left-hand side in block storage; empty under scalar storage. */
  const BSRSparseMatrix &LhsBSR() const { return bsr_lhs_; }

 private:
  SparseLinearSolverPtr solver_;     ///< Solver of the working left-hand side.
  BlockHessianAssembler assembler_;  ///< Pattern and assembly of H from the factors.
  int block_size_ = 1;               ///< Tile size of the block layout; 1: scalar layout.

  // Exactly one layout is populated, decided by block_size_.
  CSRSparseMatrix csr_hessian_;    ///< Scalar layout: H as assembled.
  CSRSparseMatrix csr_lhs_;        ///< Scalar layout: the working left-hand side (H's pattern).
  dvector<int> csr_diagonal_;      ///< Scalar layout: position of each diagonal entry.
  BSRSparseMatrix bsr_hessian_;    ///< Block layout: H as assembled.
  BSRSparseMatrix bsr_lhs_;        ///< Block layout: the working left-hand side.
  dvector<int> tile_row_scratch_;  ///< Block layout: tile -> block row (column scaling).

  // Scratch of MultiplyHessian() / StepCurvature().
  mutable cuSPARSEHandle cusparse_handle_;  ///< Scalar-layout SpMV.
  mutable dvector<uint8_t> spmv_buffer_;    ///< cuSPARSE workspace of the SpMV.
  mutable dvector<float> hessian_step_;     ///< H δ.
};

}  // namespace cunls
