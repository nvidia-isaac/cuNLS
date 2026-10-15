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

#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

namespace internal {
struct SchurMatrixView;
}  // namespace internal

/**
 * @brief Automatic state-block Schur elimination for SPD CSR/BSR systems.
 *
 * Selects independent free tangent states across batches, solves the retained
 * system internally, and recovers the complete update. No factor-type hints or
 * additional options are needed. Does not delegate to another solver backend.
 * Initialize must be repeated when the state layout or matrix pattern changes.
 * Solve refreshes all numeric factors and checks the original-system residual.
 * See docs/sphinx/generalized_schur.rst for equations and numerical limits.
 */
class GeneralizedSchurSolver final : public SparseLinearSolver {
 public:
  /** @brief Accept BSR matrices without expanding them to scalar CSR storage. */
  bool SupportsBlockStorage() const override;

  /**
   * @brief Build the symbolic elimination plan for a scalar CSR matrix.
   *
   * Free tangent-state dimensions come from problem. An empty problem instead
   * treats each scalar row as one state. Rows in each logical dense state block
   * must share a sorted column pattern. No numeric factorization is cached here.
   *
   * @param stream Stream used for validation, selection, and workspace setup.
   * @param problem Source of the free-state layout; constant states are excluded.
   * @param matrix Symmetric working matrix with a fixed block sparsity pattern.
   * @param rhs Right-hand side; used to validate the required vector length.
   * @param solution Preallocated output with the same length as rhs.
   * @return False for incompatible dimensions or an invalid block pattern.
   *
   * Call again when the state layout or sparsity changes. Numeric matrix values
   * may change between Solve calls without rebuilding this plan.
   */
  bool Initialize(cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &matrix,
                  const dvector<float> &rhs, dvector<float> &solution) override;

  /**
   * @brief Build the same plan directly from BSR storage.
   *
   * Logical state blocks may span several BSR tiles. If problem has no state
   * metadata, each BSR tile is treated as one state block. The vector-size,
   * symmetry, and reinitialization requirements match the CSR overload.
   */
  bool Initialize(cudaStream_t stream, const Problem &problem, const BSRSparseMatrix &matrix,
                  const dvector<float> &rhs, dvector<float> &solution) override;

  /**
   * @brief Solve matrix * solution = rhs using the initialized CSR pattern.
   *
   * Refreshes pivot factors and the Schur preconditioner, solves the retained
   * system internally, and recovers eliminated updates in the original ordering.
   * The input solution is not used as a warm start. Local factors and Schur
   * accumulation use double precision; vectors use float. The backend checks
   * the original-system residual and may refine the update with the same factors.
   *
   * @return True when the full residual meets the internal tolerance; false for
   * dimension mismatches, invalid pivots, nonfinite values, or failed accuracy.
   * On failure, solution may contain an incomplete update. CUDA/cuBLAS errors
   * follow the library's exception-based error handling. Host convergence checks
   * synchronize stream; this method is not a fully asynchronous enqueue.
   */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &matrix, const dvector<float> &rhs,
             dvector<float> &solution) override;

  /** @brief BSR overload with the same numeric and failure contract as CSR Solve. */
  bool Solve(cudaStream_t stream, const BSRSparseMatrix &matrix, const dvector<float> &rhs,
             dvector<float> &solution) override;

 private:
  using MatrixView = internal::SchurMatrixView;

  /** @brief Select which diagonal factors a dimension group represents. */
  enum class BlockRole { Eliminated, Retained };

  /**
   * @brief State blocks sharing a factorization role and a kernel specialization.
   * dimension == 0 selects the generic kernel; actual sizes remain in the state
   * offsets. block_ids contains original logical-state indices on the GPU.
   */
  struct DimensionGroup {
    BlockRole role;
    int dimension;
    dvector<int> block_ids;
  };

  /** @brief Common symbolic setup for the CSR and BSR adapters. */
  bool Initialize(cudaStream_t stream, const Problem &problem, MatrixView matrix,
                  const dvector<float> &rhs, dvector<float> &solution);

  /** @brief Factor and invert all diagonal blocks of the requested role. */
  void InvertDimensionGroups(cudaStream_t stream, BlockRole role);

  /** @brief Synchronize stream and read the device factorization/validation flag. */
  bool CheckDeviceSuccess(cudaStream_t stream);

  /** @brief View packed D^-1 B values using the original matrix's column graph. */
  MatrixView MakeTransformedView(MatrixView matrix) const;

  /** @brief Compute output = S * input in retained order, explicitly or implicitly. */
  void ApplySchurOperator(cudaStream_t stream, MatrixView matrix, const float *input,
                          float *output);

  /** @brief Apply inverse Schur diagonal blocks to the current reduced residual. */
  void ApplyPreconditioner(cudaStream_t stream);

  /** @brief Common numeric factorization, solve, and full-residual verification. */
  bool Solve(cudaStream_t stream, MatrixView matrix, const dvector<float> &rhs,
             dvector<float> &solution);

  /**
   * @brief Reduce rhs, solve S, and recover the full solution with existing factors.
   *
   * Used for the initial solve and refinement corrections. rhs_norm scales the
   * reduced stopping criterion. The caller verifies the full-system residual;
   * this helper does not independently establish convergence.
   */
  void SolveWithCurrentFactors(cudaStream_t stream, MatrixView matrix, const float *rhs,
                               float *solution, float rhs_norm);

  /** @brief Factorization buckets built during symbolic initialization. */
  std::vector<DimensionGroup> dimension_groups_;

  /** @brief Total number of free scalar tangent coordinates. */
  int num_rows_ = 0;

  /** @brief Number of logical free-state blocks. */
  int num_state_blocks_ = 0;

  /** @brief Scalar coordinates selected for elimination. */
  int num_eliminated_rows_ = 0;

  /** @brief Scalar coordinates in the reduced system. */
  int num_retained_rows_ = 0;

  /** @brief Number of independent eliminated state blocks. */
  int num_eliminated_blocks_ = 0;

  /** @brief Number of retained state blocks. */
  int num_retained_blocks_ = 0;

  /** @brief Prefix offsets of each logical state in the original row order. */
  dvector<int> state_row_offsets_;

  /** @brief Original scalar row to logical state-block index. */
  dvector<int> row_state_ids_;

  /** @brief Selection state: 0 undecided, 1 eliminated, 2 excluded/retained. */
  dvector<int> selection_status_;

  /** @brief Temporary flags for a single independent-set selection round. */
  dvector<int> selected_in_round_;

  /** @brief Stored scalar columns per state row, including its diagonal block. */
  dvector<int> state_row_widths_;

  /** @brief Eliminated block indices in stable original order. */
  dvector<int> eliminated_block_ids_;

  /** @brief Retained block indices in stable original order. */
  dvector<int> retained_block_ids_;

  /** @brief Compact eliminated index to original scalar row. */
  dvector<int> eliminated_row_ids_;

  /** @brief Compact retained index to original scalar row. */
  dvector<int> retained_row_ids_;

  /** @brief Original row to compact retained index; -1 marks eliminated rows. */
  dvector<int> retained_row_indices_;

  /** @brief One device integer set by validation or failed Cholesky pivots. */
  dvector<int> device_failure_flag_;

  /** @brief Flags for exactly zero numerical rows in the working matrix. */
  dvector<int> unobservable_rows_;

  /** @brief State prefixes within the selected or retained scalar partition. */
  dvector<int> partition_row_offsets_;

  /** @brief Original rows assigned a full CTA for residual evaluation. */
  dvector<int> long_residual_row_ids_;

  /** @brief Offsets of row-major dense blocks in diagonal/inverse storage. */
  dvector<size_t> diagonal_block_offsets_;

  /** @brief Offsets of packed transformed rows for eliminated state blocks. */
  dvector<size_t> transformed_offsets_;

  /** @brief Whether the plan materializes the dense reduced matrix. */
  bool use_explicit_schur_ = false;

  /** @brief Whether the retained system uses the small Cholesky terminal path. */
  bool use_direct_solve_ = false;

  /** @brief Pivot blocks and Schur diagonal blocks; generic inversion scratch. */
  dvector<double> diagonal_blocks_;

  /** @brief Explicit row-major reduced matrix, when selected by the planner. */
  dvector<double> schur_matrix_;

  /** @brief Dense inverses of eliminated pivots and retained preconditioner blocks. */
  dvector<double> inverse_blocks_;

  /** @brief Packed D^-1 B values; retained rows have no allocation here. */
  dvector<double> transformed_cross_blocks_;

  /** @brief Small reduced-system Cholesky factor for the direct path. */
  dvector<double> terminal_factor_;

  /** @brief D^-1 b_e in original row order; only eliminated entries are defined. */
  dvector<float> eliminated_rhs_solution_;

  /** @brief D^-1 B times a retained vector, stored in original row order. */
  dvector<float> eliminated_product_;

  /** @brief Current reduced solution in compact retained order. */
  dvector<float> retained_solution_;

  /** @brief b_r - B^T D^-1 b_e in compact retained order. */
  dvector<float> reduced_rhs_;

  /** @brief CG residual g - S*x_r. */
  dvector<float> reduced_residual_;

  /** @brief Block-Jacobi preconditioned reduced residual. */
  dvector<float> preconditioned_residual_;

  /** @brief Current CG search direction. */
  dvector<float> search_direction_;

  /** @brief Schur operator applied to the CG search direction. */
  dvector<float> operator_product_;

  /** @brief Device dot products: current r^Tz, p^TSp, next r^Tz, spare. */
  dvector<float> cg_dot_products_;

  /** @brief Original-system residual used for validation and refinement. */
  dvector<float> full_residual_;

  /** @brief Full-order refinement update before adding it to the solution. */
  dvector<float> correction_;

  /** @brief Lazily initialized handle for vector norms and dot products. */
  cuBLASHandle cublas_handle_;
};

}  // namespace cunls
