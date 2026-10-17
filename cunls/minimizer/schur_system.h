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

#include <cstdint>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/hessian_structure.h"
#include "cunls/minimizer/linear_system.h"

namespace cunls {

namespace internal {
namespace schur {

/** @brief Per residual batch geometry, for kernels that walk all factors (internal). */
struct BatchGeometry {
  long long jacobian_offset;  ///< Start of the batch in the Jacobian buffer.
  long long residual_offset;  ///< Start of the batch in the residual buffer.
  int factor_begin;           ///< Global index of the batch's first factor.
  int num_residuals;          ///< m: residuals per factor.
  int tangent_dim;            ///< n: Jacobian columns per factor (sum of slot sizes).
  int num_slots;              ///< State slots per factor.
  int slot_begin;             ///< Start of the batch's slots in the slot tables.
  int col_offset;             ///< Start of the batch in HessianStructureBuilder::FactorCols().
};

/** @brief A state batch's range of free columns and its role in the elimination (internal). */
struct ColumnRange {
  int first_col;   ///< First free column of the batch.
  int end_col;     ///< One past its last free column.
  int tangent;     ///< Tangent size of its states.
  int first_id;    ///< Landmark id (eliminated) or retained block id of its first free state.
  int eliminated;  ///< Nonzero for a batch of Problem::EliminatedStateBatches().
};

}  // namespace schur
}  // namespace internal

/**
 * @brief The linear system with the landmark states eliminated
 * (LinearSystemType::SchurComplement).
 *
 * (Here S is the Schur complement; the column scale of LinearSystem is
 * applied to the full system before the elimination.)
 *
 * Partitions the free columns into the eliminated landmarks e (the batches of
 * Problem::SetEliminatedStateBatches) and the retained states r. No factor
 * touches two landmarks, so the landmark block D of the Hessian is block
 * diagonal and each Solve() computes, for the modified system (column scale,
 * damping and mask applied before elimination):
 *
 *     S = C - W D⁻¹ Wᵀ,   g = b_r - W D⁻¹ b_e,   S x_r = g,
 *     x_e,i = D_i⁻¹ (b_e,i - Σ_p W_piᵀ x_r,p),
 *
 * without ever forming the full Hessian. Per (retained state p, landmark i)
 * pair, with L_i the Cholesky factor of D_i, V_pi = W_pi L_i⁻ᵀ and
 * h_i = L_i⁻¹ b_e,i:
 *
 *     S_pp = C^R_pp + Σ_i (A_pi - V_pi V_piᵀ)   (A_pi: the pair's own J_pᵀJ_p;
 *                                               no large-minus-large cancellation)
 *     S_pq = C^R_pq - Σ_i V_pi V_qiᵀ,           g_p = b_r,p - Σ_i V_pi h_i.
 *
 * S (upper block triangle, variable block sizes) is expanded to the full
 * scalar CSR and solved with BlockSparseCholeskySolver (CreateLinearSystem;
 * S is sparse, and banded for visual-inertial chains, so a dense
 * factorization does not scale and an iterative one is not exact enough).
 *
 * Requirements (checked in Initialize()): every factor touches at most one
 * free landmark; landmarks have tangent size 1 to 3; the retained states that
 * share factors with landmarks all have one tangent size, 3, 6 or 9. Other
 * factors (between retained states only) may have any shape.
 *
 * Work per Solve() (P pairs, L landmarks, Σ k_i² clique products):
 *   - pose-major pass over the pairs: contract the pair's Jacobians, eliminate
 *     the landmark (3x3 Cholesky in registers), write V, and reduce the
 *     diagonal brackets and g with warp-segmented sums (one atomic per
 *     retained state and warp);
 *   - landmark-major pass: one warp per landmark forms -V_pi V_qiᵀ for every
 *     pair of its observations from shared memory, with coalesced atomics;
 *   - the backend solve of S, then one thread per landmark back-substitutes.
 */
class SchurSystem : public LinearSystem {
 public:
  /**
   * @brief Takes ownership of the solver of the reduced system.
   *
   * @param solver Solver of the reduced system S.
   * @param check_failures Read back, after every Solve(), whether a landmark
   *        block could not be factored even with a diagonal shift (non-finite
   *        values), and return false if so (one synchronization;
   *        MinimizerOptions::disable_safety_checks off). Without it, such a
   *        landmark makes the step NaN, which the minimizer rejects.
   */
  SchurSystem(SparseLinearSolverPtr solver, bool check_failures);

  /**
   * @brief Builds the elimination plan on the GPU: landmark and retained
   * numbering, the (state, landmark) pairs, the pattern of S and the maps
   * that write into it, and the expansion of S to scalar CSR.
   *
   * @throws std::invalid_argument if the problem declares no eliminated state
   *         batches, or violates the requirements of the class comment.
   */
  void Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                  const std::vector<bool> &reorderable) override;

  size_t JacobianValuesSize() const override { return structure_.JacobianValuesSize(); }

  /**
   * @brief For a reorderable batch with landmark factors: its landmark
   * factors in the pose-major order of their pairs (then its other factors),
   * so the pair passes stream the Jacobians.
   */
  const int *EvaluationOrder(size_t batch) const override;

  /** @brief rhs = -Jᵀr, diag(H) and the landmark blocks D, one pass over the factors. */
  void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                const float *residuals, dvector<float> &rhs) override;

  void ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const override;

  /** @brief Forms S and runs the backend's symbolic setup on it. */
  bool PrepareSolver(cudaStream_t stream, const Problem &problem, const dvector<float> &rhs,
                     dvector<float> &step) override;

  /** @brief Eliminates, solves S x_r = g and back-substitutes the landmarks. */
  bool Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) override;

  /** @brief δᵀHδ per subproblem as Σ over factors of ‖J_f δ_f‖². */
  void StepCurvature(cudaStream_t stream, const dvector<float> &step, ProblemPartition &partition,
                     float *out) const override;

  /** @brief The reduced system S of the last Solve() (full scalar CSR, for tests). */
  const CSRSparseMatrix &ReducedMatrix() const { return reduced_; }

  /** @brief Retained (reduced) dimension. */
  int NumRetainedColumns() const { return num_retained_cols_; }

 private:
  /// Temporaries shared by the setup phases (defined in the implementation).
  struct SetupScratch;

  // Setup phases of Initialize(), in order.
  /** @brief Batch geometry, slot tables, and the landmark and retained numbering. */
  void SetUpGeometry(cudaStream_t stream, const Problem &problem, SetupScratch &scratch);
  /** @brief Classifies the factors and checks the requirements of the class comment. */
  void ClassifyFactors(cudaStream_t stream, SetupScratch &scratch);
  /** @brief The (retained state, landmark) pairs and their pose-major order. */
  void BuildPairs(cudaStream_t stream, SetupScratch &scratch);
  /** @brief The pose-major entries and the Jacobian evaluation order. */
  void BuildEvaluationOrder(cudaStream_t stream, const std::vector<bool> &reorderable,
                            SetupScratch &scratch);
  /** @brief The block pattern of S and the maps that write into it. */
  void BuildReducedPattern(cudaStream_t stream, SetupScratch &scratch);
  /** @brief The scalar CSR of S handed to the solver. */
  void BuildReducedCsr(cudaStream_t stream, SetupScratch &scratch);
  /** @brief The per-solve buffers. */
  void AllocateNumericBuffers(cudaStream_t stream, const SetupScratch &scratch);

  /** @brief Builds S and g from the current Jacobians and declared modifications. */
  void FormReducedSystem(cudaStream_t stream, const dvector<float> &rhs);

  /** @brief Per residual batch and per factor: where its Jacobians, slots and landmark are. */
  struct FactorTables {
    dvector<internal::schur::BatchGeometry> batches;  ///< Per residual batch.
    std::vector<int> batch_offsets;  ///< Per residual batch: its first factor (global index).
    std::vector<bool> reordered;     ///< Per residual batch: evaluated in evaluation_order.
    dvector<int> slot_sizes;         ///< Per (batch, slot): tangent size.
    dvector<int> slot_offsets;       ///< Per (batch, slot): first Jacobian column of the slot.
    dvector<int> landmark;           ///< Per factor: its landmark, or -1.
    dvector<int> landmark_slot;      ///< Per factor: its landmark's slot, or -1.
    dvector<int> item;               ///< Per factor: its item (position in its batch's evaluation).
    dvector<int> evaluation_order;   ///< Per item, batches concatenated: its batch-local factor.
    dvector<int> item_cols;          ///< HessianStructureBuilder::FactorCols() in item order.
  };

  /**
   * @brief The (retained state, landmark) pairs that share factors, numbered
   * landmark-major, and their entries (one per landmark factor and retained
   * slot) in pose-major order, the order of V and of the Jacobians.
   */
  struct PairTables {
    dvector<int> landmark;              ///< Per pair: its landmark.
    dvector<int> state;                 ///< Per pair: its retained state (block row of S).
    dvector<int> landmark_offsets;      ///< Per landmark: its pairs [offset, next).
    dvector<int> pose_major;            ///< Pose-major position -> pair.
    dvector<int> pose_major_position;   ///< Pair -> pose-major position.
    dvector<int> entry_offsets;         ///< Per pose-major pair: its entries [offset, next).
    dvector<long long> entry_jacobian;  ///< Per entry: its factor's first Jacobian value.
    dvector<long long> entry_residual;  ///< Per entry: its factor's first residual.
    /// Per entry: retained slot column | landmark slot column << 8 | m << 16 | n << 24.
    dvector<int> entry_meta;
    /// Off-diagonal pass, one CTA each: retained state, pose-major pairs [begin, end).
    dvector<int3> chunks;
  };

  /** @brief The pattern of S and the maps that write into it and expand it. */
  struct ReducedPattern {
    dvector<int> state_col;             ///< Per retained state: first column of S.
    dvector<int> state_natural_col;     ///< Per retained state: first column of the full system.
    dvector<int> state_diagonal;        ///< Per retained state: its diagonal block of S.
    dvector<int> landmark_natural_col;  ///< Per landmark: first column of the full system.
    dvector<int> upper_row_offsets;  ///< Per retained state: its upper blocks of S [offset, next).
    dvector<int> upper_col;          ///< Per block of S (upper): its column block.
    dvector<int> value_offsets;      ///< Per block of S (upper): its first value (row-major).
    dvector<int> rr_factor;          ///< Per C^R item: its factor.
    dvector<int> rr_slots;           ///< Per C^R item: slot a | slot b << 8 | transposed << 16.
    dvector<int> rr_block;           ///< Per C^R item: its block of S.
    dvector<int> col_natural;        ///< Per column of S: column of the full system.
    dvector<int> col_diagonal;       ///< Per column of S: position of its diagonal value.
    dvector<int> csr_source;         ///< Per nonzero of the scalar CSR: position in the values.
  };

  int NumFactors() const { return static_cast<int>(factors_.landmark.size()); }
  int NumBatches() const { return static_cast<int>(factors_.batches.size()); }
  int NumLandmarks() const { return static_cast<int>(pattern_.landmark_natural_col.size()); }
  int NumRetainedStates() const { return static_cast<int>(pattern_.state_col.size()); }
  int NumPairs() const { return static_cast<int>(pairs_.landmark.size()); }

  SparseLinearSolverPtr solver_;       ///< Solver of the reduced system.
  bool check_failures_ = false;        ///< See the constructor.
  HessianStructureBuilder structure_;  ///< Layout and per-(factor, slot) columns.

  int num_cols_ = 0;           ///< Free columns of the full system.
  int num_retained_cols_ = 0;  ///< Columns of S.
  int landmark_dim_ = 0;       ///< DE: tangent size of the landmarks.
  int pair_block_dim_ = 0;     ///< DP: tangent size of the retained states paired with landmarks.
  FactorTables factors_;
  PairTables pairs_;
  ReducedPattern pattern_;

  // Numeric state.
  const float *jacobians_ = nullptr;  ///< Assemble(): per-factor Jacobians.
  dvector<float> hessian_diagonal_;   ///< diag(H), unscaled.
  dvector<float> landmark_hessian_;   ///< Per landmark: upper triangle of D_i, unscaled (SoA).
  dvector<float> s_values_;           ///< S, upper block triangle, row-major blocks.
  dvector<float> pair_v_;             ///< V per pair, pose-major SoA.
  CSRSparseMatrix reduced_;           ///< S, full scalar CSR, for the solver.
  dvector<float> reduced_rhs_;        ///< g.
  dvector<float> reduced_step_;       ///< x_r.
  mutable dvector<float> factor_curvature_;  ///< Per factor ‖J_f δ_f‖² (StepCurvature()).
  dvector<int> failure_;                     ///< Nonzero: a landmark block failed to factor.
};

}  // namespace cunls
