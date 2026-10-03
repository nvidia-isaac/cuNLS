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
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

/**
 * @brief Direct solver for normal equations that are block tridiagonal in
 * time: trajectory optimization and MPC (one subproblem per trajectory).
 *
 * The ordering comes from the problem: Problem::SetStateStages gives every
 * state a stage (time index), Problem::SetProblemPartition the subproblem.
 * Within a subproblem, the unknowns of stage k form one block; factors may
 * only couple the same or adjacent stages, so the matrix of each subproblem
 * is block tridiagonal with blocks D_k (stage k) and C_k (stages k, k+1).
 * Initialize checks this and fails (false, with a logged error) otherwise.
 *
 * Solve: block Cholesky (the Riccati recursion of LQR) per subproblem,
 * S_0 = D_0, W_k = L_{k-1}^{-1} C_{k-1}, S_k = D_k - W_kᵀ W_k = L_k L_kᵀ,
 * then forward and back substitution. One warp per subproblem, blocks in
 * shared memory, O(K m³) work for K stages of at most m unknowns (m <= 32).
 * Stages smaller than m are padded with identity rows. No host
 * synchronization: the solve can be captured in a CUDA graph. A non-positive
 * pivot (a singular system) is replaced by a tiny positive one, as no
 * read-back reports it.
 *
 * Known gaps (measured on an RTX PRO 5000 Blackwell):
 *
 * - Large batches are bound by the scalar CSR Hessian, not by this kernel.
 *   For 4096 quadrotors (K = 41, M = 16, ~126M nonzeros) one real-time step
 *   takes 10.4 ms (target 10 ms): Hessian assembly into CSR 2.0 ms, the
 *   scatter of CSR into the stage blocks here 1.5 ms, this kernel 1.0 ms,
 *   Levenberg-Marquardt's SpMV 0.7 ms, bound masking and diagonal updates
 *   ~0.8 ms. Fix: let NormalEquations assemble directly into the stage
 *   blocks (a third storage layout next to CSR / BSR, with its own diagonal,
 *   masking and SpMV operations), which removes the scatter and most CSR
 *   passes.
 * - A single long horizon is latency bound: the recursion is serial in the
 *   stages, ~1.4 µs (M = 5) to ~5 µs (M = 16) per stage, i.e. 72 µs for
 *   K = 51, M = 5 and 205 µs for K = 41, M = 16. Fix: parallel (cyclic)
 *   reduction across stages for long horizons with few subproblems.
 * - Stage blocks are limited to M <= 32 unknowns (one warp, one row per
 *   lane); larger stages would need a multi-warp variant.
 */
class BlockTridiagonalSolver : public SparseLinearSolver {
 public:
  using SparseLinearSolver::Initialize;  // the BSR overloads: block storage is not supported
  using SparseLinearSolver::Solve;

  bool Initialize(cudaStream_t stream, const Problem &problem, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) override;

  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
             dvector<float> &result) override;

  /** @brief Largest stage block size supported (one warp per subproblem). */
  static constexpr int kMaxStageSize = 32;

 private:
  int num_problems_ = 0;  ///< P
  int num_stages_ = 0;    ///< K (the largest stage index + 1)
  int stage_size_ = 0;    ///< m (the largest stage block)
  size_t num_rows_ = 0;
  /// Per CSR entry: index into blocks_ (D blocks, then C blocks), or -1 (lower half).
  dvector<int> entry_slot_;
  /// Per row of the system: index into the stage vectors (p, k, offset).
  dvector<int> row_slot_;
  /// Per (p, k): number of unknowns of the stage (rows past it are padding).
  dvector<int> stage_sizes_;
  dvector<float> blocks_;   ///< P*K*m*m D blocks (then L), P*K*m*m C blocks (then W).
  dvector<float> vectors_;  ///< P*K*m right-hand side, then solution.
};

}  // namespace cunls
