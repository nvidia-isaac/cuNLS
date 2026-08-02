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

#include "cunls/common/types.h"
#include "cunls/linear_solver/csr_sparse_linear_solver.h"
#include "cunls/minimizer/block_hessian_assembler.h"
#include "cunls/minimizer/bsr_matrix.h"

namespace cunls {

class Problem;

/**
 * @brief The assembled normal equations, and the storage they live in.
 *
 * Holds two matrices: the Hessian `H` as assembled, and a working copy the
 * minimizer is free to damp and scale before handing it to the solver. Both
 * are stored in one of two layouts, chosen once at Initialize():
 *
 * - **Block (BSR)** when the state tangent dimensions share a common factor and
 *   the solver can read tiles. One column index per dense tile instead of one
 *   per scalar entry, which is bandwidth the solver's SpMV no longer moves.
 * - **Scalar (CSR)** otherwise.
 *
 * Which layout is live is an implementation detail. Callers work in terms of
 * "the Hessian" and "the working left-hand side"; every operation dispatches
 * internally, so no branch on storage leaks into the minimizers.
 *
 * A BSR matrix with `block_size == 1` is byte-for-byte the same layout as CSR,
 * so the scalar case could in principle be expressed as block storage too. It
 * is kept separate deliberately: at `block_size == 1` every block operation
 * degenerates to its scalar counterpart but runs through less-optimized code —
 * notably cuSPARSE's `csrmv`, which reaches ~90% of peak bandwidth and which a
 * one-tile-per-entry BSR kernel cannot beat. Block storage is a win only when
 * the tiles are real.
 */
class NormalEquations {
 public:
  /**
   * @brief Derives the sparsity pattern and picks the storage layout.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param num_cols Number of free tangent dimensions in the reduced system.
   * @param solver_supports_block_storage Whether the active solver can consume
   *        block storage; see CSRSparseLinearSolver::SupportsBlockStorage.
   */
  void Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                  bool solver_supports_block_storage);

  /** @brief Floats needed for the per-factor Jacobian buffer Assemble() reads. */
  size_t JacobianValuesSize() const { return assembler_.JacobianValuesSize(); }

  /**
   * @brief Assembles `H` and `rhs`, then refreshes the working left-hand side.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem (must match Initialize).
   * @param jacobians Per-factor dense Jacobian blocks.
   * @param residuals Residual vector.
   * @param[out] rhs Right-hand side `-J^T r`.
   */
  void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                const float *residuals, dvector<float> &rhs);

  /** @brief Diagonal of the assembled (undamped) Hessian. */
  void ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const;

  /** @brief Diagonal of the working left-hand side. */
  void ExtractLhsDiagonal(cudaStream_t stream, dvector<float> &diagonal) const;

  /** @brief Adds `scale * diag(diagonal)` to the working left-hand side. */
  void AddScaledDiagonalToLhs(cudaStream_t stream, float scale, const dvector<float> &diagonal);

  /** @brief Applies `A_ij *= scale[i] * scale[j]` to the working left-hand side. */
  void ScaleLhsSymmetric(cudaStream_t stream, const dvector<float> &scale);

  /**
   * @brief Async `step^T H step` against the assembled (undamped) Hessian.
   *
   * @param stream CUDA stream for GPU operations.
   * @param cusparse_handle Opaque cuSPARSE handle, used by the scalar path.
   * @param step Step vector.
   * @param[out] d_out Device destination for the scalar.
   * @param[out] d_partials Reduction scratch; see device_reduction.h.
   * @param[out] buffer Scratch for the scalar path's SpMV.
   */
  void WeightedSquaredStepAsync(cudaStream_t stream, void *cusparse_handle,
                                const dvector<float> &step, float *d_out, float *d_partials,
                                dvector<uint8_t> &buffer);

  /** @brief Hands the working left-hand side to the solver for symbolic setup. */
  bool InitializeSolver(cudaStream_t stream, CSRSparseLinearSolver &solver, const Problem &problem,
                        const dvector<float> &rhs, dvector<float> &step);

  /** @brief Solves with the working left-hand side. */
  bool Solve(cudaStream_t stream, CSRSparseLinearSolver &solver, const dvector<float> &rhs,
             dvector<float> &step);

  /** @brief True when the block layout is live for the current problem. */
  bool UsesBlockStorage() const { return block_size_ > 1; }

  /** @brief Working left-hand side in scalar storage; empty under block storage. */
  const CSRSparseMatrix &LhsCSR() const { return csr_lhs_; }

  /** @brief Working left-hand side in block storage; empty under scalar storage. */
  const BSRSparseMatrix &LhsBSR() const { return bsr_lhs_; }

 private:
  BlockHessianAssembler assembler_;

  // Exactly one pair is populated, decided by block_size_.
  CSRSparseMatrix csr_hessian_;
  CSRSparseMatrix csr_lhs_;
  BSRSparseMatrix bsr_hessian_;
  BSRSparseMatrix bsr_lhs_;

  CSRMatrixDimensions csr_dims_;   ///< Cached dims for the scalar SpMV.
  dvector<int> tile_row_scratch_;  ///< Tile-to-block-row map for scaling.
  dvector<float> spmv_scratch_;    ///< SpMV result, either layout.

  int block_size_ = 1;
};

}  // namespace cunls
