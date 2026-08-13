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

#include <vector>

#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/hessian_structure.h"

namespace cunls {

class Problem;

/**
 * @brief Assembles the normal equations directly from per-factor Jacobian
 * blocks.
 *
 * The classic path materializes the whole sparse Jacobian `J` in CSR, forms
 * `H = J^T J` with one warp per *residual row* (`m * n^2` scattered global
 * atomics per factor), and then runs a separate cuSPARSE SpMV for
 * `b = -J^T r`.
 *
 * This class instead contracts each factor locally.  A factor batch already
 * writes its Jacobian as `NumFactors()` dense row-major `m x n` blocks, so a
 * single kernel can read `J_f`, form `H_f = J_f^T J_f` and `b_f = -J_f^T r_f`
 * in shared memory, and scatter both into the global CSR.  That drops the
 * atomic count by exactly a factor of `m` and removes the triplet-to-CSR
 * conversion, the `J^T J` kernel and the RHS SpMV from every iteration.
 *
 * The saving above is layout-independent: it comes from contracting per factor
 * rather than per residual row, and it is the same whether the target is scalar
 * CSR or block BSR.  Both are supported, selected by NormalEquations and passed
 * in via the `block_size` overload of Initialize().  There is no intermediate
 * Hessian and no conversion between the two — the assembler scatters straight
 * into whichever layout it was initialized for.
 *
 * Scattering is direct in either case because the pattern produced by
 * HessianStructureBuilder lays out each block pair contiguously within a row
 * and at the same row-relative offset for every row of the block.  For scalar
 * CSR the address is `row_offsets[col_a + i] + write_offset(a,b) + j`; for BSR
 * the same offsets index tiles instead of entries.
 */
class BlockHessianAssembler {
 public:
  /**
   * @brief Builds the Hessian sparsity pattern and the per-factor scatter maps.
   *
   * Must be called whenever the problem structure changes.  The maps cost
   * `num_factors * (nb + nb^2)` integers per factor batch, where `nb` is the
   * number of state blocks a factor touches.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param num_cols Number of free tangent dimensions in the reduced system.
   * @param[out] hessian CSR Hessian; structure filled, values left unset.
   */
  void Initialize(cudaStream_t stream, const Problem &problem, int num_cols,
                  CSRSparseMatrix &hessian);

  /**
   * @brief Same, but targeting uniform block storage.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param num_cols Number of free tangent dimensions in the reduced system.
   * @param block_size Tile edge; see ChooseHessianBlockSize().
   * @param[out] hessian BSR Hessian; structure filled, values left unset.
   */
  void Initialize(cudaStream_t stream, const Problem &problem, int num_cols, int block_size,
                  BSRSparseMatrix &hessian);

  /**
   * @brief Scatter-accumulates `H = J^T J` and `rhs = -J^T r`.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem (must match Initialize).
   * @param jacobian_values Per-factor dense Jacobian blocks, batches
   *        concatenated in problem order; see JacobianValuesSize().
   * @param residuals Residual vector, batches concatenated in problem order.
   * @param[in,out] hessian CSR Hessian initialized by Initialize(); values are
   *        zeroed and then accumulated.
   * @param[out] rhs Right-hand side `-J^T r`; resized to `num_cols`.
   */
  void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobian_values,
                const float *residuals, CSRSparseMatrix &hessian, dvector<float> &rhs);

  /** @brief Same, into the BSR Hessian prepared by the block-size overload. */
  void Assemble(cudaStream_t stream, const Problem &problem, const float *jacobian_values,
                const float *residuals, BSRSparseMatrix &hessian, dvector<float> &rhs);

  /** @brief Total floats needed for the per-factor Jacobian value buffer. */
  size_t JacobianValuesSize() const { return structure_builder_.JacobianValuesSize(); }

 private:
  /** @brief Per-residual-batch constants uploaded once for the kernel. */
  struct BatchPlan {
    HessianBatchLayout layout;  ///< Geometry and flat-buffer offsets.
    /// n entries: block index owning each local column.
    dvector<int> block_of_col;
    /// n entries: offset of each local column inside its block.
    dvector<int> offset_in_block;
    /// n entries: index of the enclosing tile column, for block storage.
    dvector<int> tile_of_col;
    /// n entries: offset within that tile, for block storage.
    dvector<int> sub_of_col;
    /// n(n+1)/2 entries: the upper-triangle (p, q) pairs of H_f packed as
    /// `(p << 16) | q`.  H_f is symmetric, so enumerating the triangle halves
    /// the shared-memory traffic and the FLOPs; a precomputed table keeps
    /// every lane busy, which a `q < p` skip inside the square loop would not.
    dvector<int> triangular_pairs;
  };

  /** @brief Fills plans_ from the structure builder's layout. */
  void BuildPlans(const Problem &problem);

  /** @brief Zeroes the value array and the right-hand side. */
  void PrepareOutputs(cudaStream_t stream, dvector<float> &values, dvector<float> &rhs);

  /** @brief Launches one assembly kernel per residual batch. */
  void LaunchAssembly(cudaStream_t stream, const float *jacobian_values, const float *residuals,
                      const int *row_offsets, float *hessian_values, float *rhs,
                      bool block_storage);

  /// Owns the Hessian sparsity pattern and the per-factor scatter maps.
  HessianStructureBuilder structure_builder_;
  std::vector<BatchPlan> plans_;

  int num_cols_ = 0;
  /// Tile edge of the target storage; 1 means scalar CSR.
  int block_size_ = 1;

  profiler::Domain profiler_domain_{"BlockHessianAssembler"};
};

}  // namespace cunls
