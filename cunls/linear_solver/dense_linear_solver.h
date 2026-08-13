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
#include "cunls/linear_solver/dense_linear_solver_base.h"

namespace cunls {

/**
 * @brief Dense GPU linear solver based on pivoted LDLT factorization.
 *
 * Densifies the symmetric coefficient matrix (from either sparse layout; see
 * DenseLinearSolverBase) and solves A x = b via:
 *  1) Symmetric pivoted LDLT factorization: P^T A P = L D L^T
 *  2) Triangular/diagonal solves in the permuted system
 *  3) Permutation back to the original variable ordering
 *
 * Both the factorization and solve phases run as single-block CUDA kernels.
 * Each kernel writes a success/failure flag (1 or 0) into a device-side int
 * buffer.  After both kernels are enqueued, a single cudaMemcpyAsync copies
 * the two flags into a pinned host buffer, followed by a stream
 * synchronization.  This avoids per-kernel sync and gives the caller an
 * accurate bool return from Solve().
 */
class DenseLDLTSolver : public DenseLinearSolverBase {
 protected:
  /**
   * @copydoc DenseLinearSolverBase::EnsureBuffersSize
   *
   * Pre-allocates the LDLT factors, permutation, scratch vectors and status
   * buffers so that subsequent Solve() calls do not allocate.
   */
  void EnsureBuffersSize(cudaStream_t stream, size_t n) final;

  /**
   * @brief Factorizes the dense matrix via pivoted LDLT and solves.
   *
   * Handles symmetric matrices including indefinite ones (not limited to SPD).
   *
   * The pipeline is:
   *   1. Pivoted LDLT factorization kernel -> writes status_[0].
   *   2. Triangular/diagonal solve kernel  -> writes status_[1].
   *   3. Single async copy of status_[0..1] to status_pinned_[0..1].
   *   4. Stream synchronization.
   *   5. Host-side check of both status flags.
   *
   * If the factorization encounters a (near-)singular pivot, status_[0] is
   * set to 0 and the function returns false.  If the solve encounters a zero
   * diagonal element, status_[1] is set to 0 and the function returns false.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param n Matrix dimension.
   * @param rhs The right-hand side vector b.
   * @param result Output vector x.
   * @return true on success, false on a singular pivot or zero diagonal.
   */
  bool FactorizeAndSolve(cudaStream_t stream, int n, const dvector<float> &rhs,
                         dvector<float> &result) final;

 private:
  dvector<float> ldlt_factor_;            ///< In-place LDLT factor storage.
  dvector<int> permutation_;              ///< Pivot permutation vector.
  dvector<float> permuted_rhs_;           ///< P * b scratch vector.
  dvector<float> permuted_solution_;      ///< Permuted solution scratch.
  dvector<float> intermediate_solution_;  ///< Intermediate solve scratch.

  /// Device-side kernel status flags (index 0 = factorize, index 1 = solve).
  /// Each kernel writes 1 on success or 0 on failure.
  dvector<int> status_;

  /// Pinned host mirror of status_, used as the destination of a single async
  /// device-to-host copy so that the result can be read on the CPU after one
  /// stream synchronization.
  pvector<int> status_pinned_;
};

}  // namespace cunls
