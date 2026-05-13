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
#include "cunls/linear_solver/csr_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Configuration options for the block-sparse PCG solver.
 *
 * The solver implements Preconditioned Conjugate Gradient with block-Jacobi
 * preconditioning on a symmetric-positive-definite CSR matrix. Block size is
 * inferred from the matrix structure: the user supplies a fixed block dimension
 * B and the diagonal B x B tiles of the matrix are factored once per outer
 * call to ``Solve`` and applied at every PCG iteration.
 */
struct BlockSparsePCGOptions {
  /** Block dimension used by the block-Jacobi preconditioner.
   *  Must divide the matrix size.  B = 1 falls back to scalar Jacobi. */
  int block_size = 6;

  /** Maximum number of PCG iterations. */
  int max_iterations = 200;

  /** Convergence threshold on the relative residual ||r_k|| / ||b||. */
  float relative_tolerance = 1e-3f;

  /** Absolute threshold on ||r_k|| for early exit. */
  float absolute_tolerance = 1e-30f;

  /** Floor added to LDLT diagonal pivots to keep the preconditioner
   *  numerically invertible when the diagonal block is near-singular. */
  float pivot_floor = 1e-12f;
};

/**
 * @brief Block-Jacobi preconditioned conjugate gradient solver.
 *
 * Solves H x = b for symmetric positive (semi-)definite H stored in CSR
 * format.  The preconditioner consists of the dense ``B x B`` diagonal tiles of
 * H, factored independently with one warp per block using a small in-register
 * LDLT.  SpMV is delegated to cuSPARSE (CSR Hermitian SpMV).
 *
 * The implementation is intended for normal equations from Gauss-Newton /
 * Levenberg-Marquardt where:
 *  - H has natural block structure aligned with state blocks (e.g. 6 for SE3,
 *    3 for Vector<3>, ...);
 *  - the matrix structure does not change between ``Solve`` calls inside one
 *    Minimize, so allocations and the cuSPARSE descriptor are reused;
 *  - the previous step is a good warm start for the next iteration.
 */
class BlockSparsePCGSolver : public CSRSparseLinearSolver {
public:
  explicit BlockSparsePCGSolver(BlockSparsePCGOptions options = {});
  ~BlockSparsePCGSolver() override;

  bool Initialize(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                  const dvector<float> &rhs, dvector<float> &result) final;

  bool Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
             const dvector<float> &rhs, dvector<float> &result) final;

  /** @brief Number of PCG iterations consumed by the most recent ``Solve``. */
  int LastIterations() const { return last_iterations_; }

private:
  BlockSparsePCGOptions options_;

  /** Number of rows in the system (cached on Initialize / refreshed on Solve).
   */
  int matrix_size_ = 0;
  /** Number of block rows = matrix_size_ / block_size. */
  int num_blocks_ = 0;

  /** Stored LDLT factors of every B x B diagonal tile, packed contiguously
   *  (block i begins at index i * B * B; lower-triangle is L, diagonal is D). */
  dvector<float> precond_factors_;

  /** Scratch device vectors used by PCG. */
  dvector<float> r_;
  dvector<float> z_;
  dvector<float> p_;
  dvector<float> Ap_;

  /** Two-slot device scalar buffer used for fused reductions. */
  dvector<float> d_scratch_;

  /** Persistent SpMV state. */
  cuSPARSEHandle cusparse_handle_;
  cuSPARSEMatrixDescription mat_desc_;
  dvector<uint8_t> spmv_buffer_;
  bool spmv_buffer_ready_ = false;

  int last_iterations_ = 0;
};

} // namespace cunls
