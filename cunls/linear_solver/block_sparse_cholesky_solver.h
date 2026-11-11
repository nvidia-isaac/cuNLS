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
#include "cunls/linear_solver/block_sparse_cholesky_analysis.h"
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

/**
 * @brief Supernodal sparse Cholesky factorization on the GPU, without
 * external dependencies (no cuDSS).
 *
 * Initialize() finds the supervariables and their graph on the device, then
 * analyzes the graph on the host (internal::AnalyzeCholesky): a fill-reducing
 * ordering, supernodes cut into panels of at most 64 columns, and a level
 * schedule; the map of A's nonzeros into the factor is built on the device.
 * Every Solve() then factors A = L Lᵀ and solves with it, all on the device,
 * without host synchronization unless safety checks are on:
 *   - assembly: the lower triangle of A scattered into the panels;
 *   - per level, one kernel factors every panel of the level, one CTA per
 *     64 rows below its diagonal block (blocked by 16 columns in shared
 *     memory; the diagonal block redundantly per CTA), and takes the panel's
 *     step of the forward substitution along (the right-hand side as one
 *     more row); a second kernel applies each panel's update L_b L_bᵀ to the
 *     panels it targets, one CTA per 64 x 64 tile (atomics only where two
 *     panels of the level update the same target);
 *   - backward substitution, level by level in reverse, one CTA per 64 rows
 *     of a panel; the last CTA of a panel solves its diagonal block.
 *
 * Meant for the reduced systems of bundle adjustment (Schur complements),
 * where it replaces a general sparse direct solver: dense covisibility (a
 * few thousand columns, about 20% block fill), banded visual-inertial chains
 * (tens of thousands of columns) and small dense systems.
 *
 * A pivot within 1e-5 of zero relative to its original diagonal entry
 * (about 100 float epsilons) is a rank deficiency (a free gauge of Gauss-Newton): its
 * unknown is fixed at zero, which solves a consistent semidefinite system.
 * A clearly negative or non-finite pivot sets a device flag; with safety
 * checks (the default) Solve() reads it back and returns false. Without them
 * the failure shows up as non-finite values in the result.
 *
 * With SetRowActivity() (the minimizers declare the rows of the subproblems
 * still iterating in a partitioned problem), panels whose rows are all idle
 * are skipped, so a batch costs in proportion to its active subproblems.
 */
class BlockSparseCholeskySolver : public SparseLinearSolver {
 public:
  BlockSparseCholeskySolver();

  using SparseLinearSolver::Initialize;  // the BSR overloads: block storage is not supported
  using SparseLinearSolver::Solve;

  /** @brief Analyzes the pattern of @p spd_matrix (host) and allocates the factor. */
  bool Initialize(cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) override;

  /** @brief Factors @p spd_matrix (the pattern of Initialize()) and solves for @p rhs. */
  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
             dvector<float> &result) override;

  /** @brief Skips reading back the pivot-failure flag after each factorization. */
  void DisableSafetyChecks() override { safety_checks_ = false; }

  /**
   * @brief Rows the next Solve() calls need (see
   * SparseLinearSolver::SetRowActivity), or nullptr for all.
   *
   * A panel whose columns are all idle is skipped: not factored, no updates
   * from it, and a zero solution. Exact because a panel never mixes connected
   * components of the matrix (supernodes merge only along the elimination
   * tree), and the idle rows of a partitioned problem are whole subproblems,
   * i.e. whole components.
   */
  void SetRowActivity(const int *row_active) override { row_active_ = row_active; }

  /** @brief The analysis of the last Initialize() (host side). */
  const internal::CholeskyPlan &Plan() const { return plan_; }

 private:
  internal::CholeskyPlan plan_;      ///< Host analysis.
  bool safety_checks_ = true;        ///< Read back the failure flag after each Solve().
  const int *row_active_ = nullptr;  ///< SetRowActivity(): needed rows, or nullptr for all.
  dvector<int> panel_active_;        ///< Per panel: 1 if any of its rows is needed.

  // Device copies of the plan.
  dvector<int> perm_;                              ///< Permuted column -> original column.
  dvector<internal::CholeskyPlan::Panel> panels_;  ///< The panels.
  dvector<int> below_rows_;                        ///< Below rows of all panels.
  dvector<internal::CholeskyPlan::Run> runs_;      ///< Update runs of all panels.
  dvector<int> relmap_;  ///< Target row positions of the runs (built on the device).
  dvector<internal::CholeskyPlan::FactorItem> factor_items_;  ///< Factor CTAs by level.
  dvector<internal::CholeskyPlan::UpdateItem> update_items_;  ///< Update CTAs by level.
  dvector<int> assembly_dst_;        ///< Per nonzero of A: its factor offset, or -1 (upper).
  dvector<int64_t> assembly_dst64_;  ///< The same for a factor of more than 2^31 entries.
  bool wide_assembly_ = false;       ///< assembly_dst64_ is in use (not assembly_dst_).
  dvector<int> diagonal_index_;      ///< Per permuted column: its diagonal nonzero of A, or -1.

  // Numeric storage.
  dvector<float> factor_;    ///< Panels of L (column-major).
  dvector<float> diagonal_;  ///< Per panel: L_D, row-major.
  dvector<float> original_;  ///< Per permuted column: the diagonal entry of A (zero pivots).
  dvector<float> work_;      ///< Permuted right-hand side, reduced by the forward substitution.
  dvector<float> solution_;  ///< Permuted L^{-1} b, then the solution.
  /// Backward substitution: L_bᵀ x per column (zero between solves).
  dvector<float> accumulator_;
  dvector<int> counter_;       ///< Backward substitution: finished CTAs per panel (zero between).
  dvector<int> failure_;       ///< Set by a non-positive or non-finite pivot.
  pvector<int> failure_host_;  ///< Pinned read-back of failure_.
};

}  // namespace cunls
