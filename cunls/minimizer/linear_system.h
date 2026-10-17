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

#include <cstddef>
#include <vector>

#include "cunls/common/types.h"

namespace cunls {

class Problem;
class ProblemPartition;

/**
 * @brief How each iteration's linear system is formed and solved
 * (MinimizerOptions::linear_system).
 */
enum class LinearSystemType {
  /**
   * Assemble the full normal equations H = JᵀJ and solve them with
   * MinimizerOptions::sparse_linear_solver_type. Works for every problem.
   */
  Assembled,
  /**
   * Eliminate the state batches of Problem::SetEliminatedStateBatches (the
   * landmarks of a bundle adjustment) with a Schur complement, solve the
   * reduced system of the remaining states with BlockSparseCholeskySolver
   * (sparse_linear_solver_type is not used), and recover the eliminated
   * states per landmark.
   */
  SchurComplement,
};

/**
 * @brief The linear system of one minimizer iteration.
 *
 * Created by the minimizer (MinimizerOptions::linear_system); subclasses of
 * Minimizer and InnerSolve::restrict_system receive it to declare their
 * modifications. Not meant to be implemented outside the library.
 *
 * The Gauss-Newton system H δ = b at the current states, with H = JᵀJ and
 * b = -Jᵀr from the per-factor Jacobians and residuals, as modified by the
 * minimizer: a column scaling s (solve S H S z = S b), a damping a added to
 * the diagonal of S H S, and a mask m whose zero entries hold their unknowns
 * at 0. The modifications are declared with SetColumnScale(), SetDamping()
 * and SetMask() (by pointer: the vectors stay owned by the caller and must
 * outlive the next Solve()) and applied by Solve(); H itself is never
 * modified, so the damping can be changed and the system solved again
 * without a new Assemble(). Each new linearization starts with
 * ClearModifications().
 *
 * Implementations: NormalEquations (assembles H and hands it to a
 * SparseLinearSolver) and SchurSystem (eliminates the landmark states and
 * solves the reduced system). The minimizer owns one, chosen by
 * MinimizerOptions::linear_system.
 */
class LinearSystem {
 public:
  virtual ~LinearSystem() = default;

  /**
   * @brief Symbolic setup for `problem`: sparsity patterns and maps. Called
   * whenever the problem's structure changes.
   *
   * @param stream CUDA stream.
   * @param problem The problem; its free tangent columns are laid out by state
   *        batch (see ComputeStateColumnOffsets).
   * @param num_cols Number of free tangent columns.
   * @param reorderable Per residual batch: whether its Jacobians can be
   *        evaluated in EvaluationOrder() (analytic Jacobians; numeric
   *        differentiation keeps the natural order).
   */
  virtual void Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                          const std::vector<bool> &reorderable) = 0;

  /** @brief Floats of the per-factor Jacobian buffer Assemble() reads. */
  virtual size_t JacobianValuesSize() const = 0;

  /**
   * @brief The order in which the factors of residual batch `batch` are
   * evaluated for Assemble() (device array of NumActiveFactors() factor
   * indices: item t evaluates factor order[t]; FactorBatch::Evaluate's
   * factor_ids), or nullptr for the natural order. Only for batches marked
   * reorderable in Initialize().
   */
  virtual const int *EvaluationOrder(size_t batch) const { return nullptr; }

  /**
   * @brief Builds H and b = -Jᵀr at the current states.
   *
   * @param stream CUDA stream.
   * @param problem The problem of Initialize().
   * @param jacobians Per-item row-major Jacobian blocks (items in
   *        EvaluationOrder()), residual batches concatenated in problem order.
   *        Kept by pointer: must stay unchanged until the next Assemble().
   * @param residuals Residuals, laid out the same way; kept like `jacobians`.
   * @param[out] rhs b, one entry per free tangent column (unscaled, unmasked).
   */
  virtual void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobians,
                        const float *residuals, dvector<float> &rhs) = 0;

  /** @brief diag(H), unscaled and undamped, one entry per column. */
  virtual void ExtractHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const = 0;

  /**
   * @brief Prepares the solver for this structure (symbolic factorization),
   * with the system as currently modified. Called once per structure, after
   * the first Assemble() and before the first Solve().
   *
   * @param stream CUDA stream.
   * @param problem The problem of Initialize().
   * @param rhs Right-hand side vector (sizes the solver's setup).
   * @param step Solution vector (sizes the solver's setup; not written).
   * @return false if the solver rejects the system.
   */
  virtual bool PrepareSolver(cudaStream_t stream, const Problem &problem, const dvector<float> &rhs,
                             dvector<float> &step) = 0;

  /**
   * @brief Solves the modified system S H S z + diag(a) z = rhs (masked).
   *
   * @param stream CUDA stream.
   * @param rhs Right-hand side, already scaled and masked by the caller.
   * @param[out] step z, in scaled coordinates (the caller maps it back).
   * @return false on a solver failure.
   */
  virtual bool Solve(cudaStream_t stream, const dvector<float> &rhs, dvector<float> &step) = 0;

  /**
   * @brief out[p] = δᵀ H δ over the rows of subproblem p, with the undamped,
   * unscaled H of the last Assemble() (the curvature term of the
   * Levenberg-Marquardt model).
   *
   * @param stream CUDA stream.
   * @param step δ, one entry per column, in physical coordinates.
   * @param partition The subproblems.
   * @param[out] out Device array of partition.NumProblems() floats.
   */
  virtual void StepCurvature(cudaStream_t stream, const dvector<float> &step,
                             ProblemPartition &partition, float *out) const = 0;

  /**
   * @brief Column scale s, or nullptr for none. Not copied: the vector must
   * stay alive, and is read by every later Solve().
   */
  void SetColumnScale(const dvector<float> *scale) { column_scale_ = scale; }

  /**
   * @brief Damping a added to the diagonal of the scaled system, or nullptr
   * for none. Not copied, as SetColumnScale().
   */
  void SetDamping(const dvector<float> *damping) { damping_ = damping; }

  /**
   * @brief Mask m, or nullptr for none: the unknowns with m = 0 get a zero row
   * and column and a unit diagonal, so their step is 0 (the caller zeroes
   * their right-hand side). Not copied, as SetColumnScale().
   */
  void SetMask(const dvector<float> *mask) { mask_ = mask; }

  /** @brief Clears the column scale, damping and mask (a new linearization). */
  void ClearModifications() {
    column_scale_ = nullptr;
    damping_ = nullptr;
    mask_ = nullptr;
  }

  /** @brief diag(S H S): the diagonal of the scaled, undamped system. */
  void ExtractScaledHessianDiagonal(cudaStream_t stream, dvector<float> &diagonal) const;

 protected:
  /** @brief The declared column scale; nullptr for none. */
  const dvector<float> *ColumnScale() const { return column_scale_; }
  /** @brief The declared damping; nullptr for none. */
  const dvector<float> *Damping() const { return damping_; }
  /** @brief The declared mask; nullptr for none. */
  const dvector<float> *Mask() const { return mask_; }

 private:
  const dvector<float> *column_scale_ = nullptr;  ///< See SetColumnScale().
  const dvector<float> *damping_ = nullptr;       ///< See SetDamping().
  const dvector<float> *mask_ = nullptr;          ///< See SetMask().
};

}  // namespace cunls
