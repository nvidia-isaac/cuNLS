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

namespace cunls {

/**
 * @brief Dense GPU linear solver based on pivoted LDLT factorization.
 *
 * Converts the input CSR symmetric matrix to a dense row-major matrix and
 * solves A x = b via:
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
class DenseLDLTSolver : public CSRSparseLinearSolver {
 public:
  /**
   * @brief Allocates internal dense buffers for the given matrix size.
   *
   * Validates that dimensions of @p spd_matrix, @p rhs, and @p result are
   * consistent, then pre-allocates all working buffers (dense matrix, LDLT
   * factors, permutation, scratch vectors, and status buffers) so that
   * subsequent Solve() calls do not allocate.
   *
   * @param stream CUDA stream (unused; buffers are allocated synchronously).
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  bool Initialize(cudaStream_t stream,
                  const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs,
                  dvector<float>& result) final;

  /**
   * @brief Converts CSR to dense and solves via pivoted LDLT factorization.
   *
   * Handles symmetric matrices including indefinite ones (not limited to SPD).
   *
   * The pipeline is:
   *   1. CSR -> dense conversion (one kernel).
   *   2. Pivoted LDLT factorization kernel -> writes status_[0].
   *   3. Triangular/diagonal solve kernel  -> writes status_[1].
   *   4. Single async copy of status_[0..1] to status_pinned_[0..1].
   *   5. Stream synchronization.
   *   6. Host-side check of both status flags.
   *
   * If the factorization encounters a (near-)singular pivot, status_[0] is
   * set to 0 and the function returns false.  If the solve encounters a zero
   * diagonal element, status_[1] is set to 0 and the function returns false.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false on dimension mismatch, singular pivot,
   *         or zero diagonal.
   */
  bool Solve(cudaStream_t stream,
             const CSRSparseMatrix& spd_matrix, const dvector<float>& rhs,
             dvector<float>& result) final;

 private:
  /**
   * @brief Ensures all internal buffers are (re-)allocated for an n x n system.
   *
   * Called from both Initialize() and Solve().  Each buffer is resized only
   * when its current size differs from the required size, so repeated calls
   * with the same n are essentially free.
   *
   * @param n Number of rows (and columns) of the dense system.
   */
  void EnsureBuffersSize(size_t n);

  /**
   * @brief Converts a CSR matrix to dense row-major format on the GPU.
   *
   * Zeroes the dense output, then scatters CSR values into the correct
   * positions using a warp-per-row kernel.
   *
   * @param stream CUDA stream for the memset and kernel launch.
   * @param matrix Input CSR matrix.
   * @param dense_matrix Output dense buffer (must be pre-allocated to n*n).
   */
  void ConvertCSRToDense(cudaStream_t stream, const CSRSparseMatrix& matrix,
                         dvector<float>& dense_matrix);

  dvector<float> dense_matrix_;           ///< Dense row-major copy of A.
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
