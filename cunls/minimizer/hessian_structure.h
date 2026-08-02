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

namespace cunls {

class Problem;

/** @brief Per-residual-batch geometry and offsets into the flat scatter maps. */
struct HessianBatchLayout {
  int num_factors = 0;         ///< Factors in the batch.
  int residual_dim = 0;        ///< m: residual dimension of one factor.
  int tangent_dim = 0;         ///< n: sum of the factor's state block sizes.
  int num_blocks = 0;          ///< nb: state blocks a factor touches.
  size_t jacobian_offset = 0;  ///< Offset into the flat Jacobian value buffer.
  size_t residual_offset = 0;  ///< Offset into the flat residual vector.
  size_t col_offset = 0;       ///< Offset into FactorCols(), stride nb.
  size_t pair_offset = 0;      ///< Offset into WriteOffsets(), stride nb*nb.
};

/**
 * @brief Derives the J^T J sparsity pattern from factor-graph connectivity.
 *
 * Every factor connecting state blocks A,B contributes dense sub-blocks
 * (A,A), (A,B), (B,A), (B,B) to the Hessian.  The set of distinct block pairs
 * is the block-level sparsity pattern; expanding it gives the scalar CSR.
 *
 * The whole derivation runs on the GPU: resolve each factor's state pointers to
 * global columns, pack every candidate pair into a 64-bit key, then sort and
 * segment.  The previous host implementation spent most of a large SBA solve
 * inside an `unordered_set` of 8 M keys plus a `std::sort` of 5 M pairs.
 *
 * The resulting CSR has a property `BlockHessianAssembler` depends on: within
 * one block row, the columns of a given block pair are contiguous, and the
 * offset of that run relative to the row start is identical for every row of
 * the block.  That offset is exactly what `WriteOffsets()` returns, so the
 * assembler gets its scatter map as a by-product of the segmentation instead of
 * binary-searching the expanded column-index array.
 *
 * `output.values` is resized but left uninitialized.
 */
class HessianStructureBuilder {
 public:
  /**
   * @brief Builds the scalar CSR pattern, and optionally the scatter maps.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param num_cols Free tangent dimensions in the reduced system.
   * @param[out] output CSR matrix; row_offsets/col_ids filled, values sized.
   * @param want_scatter_maps When false, WriteOffsets() is left empty and the
   *        final scatter pass is skipped (the J^T J path does not need it).
   */
  void Build(cudaStream_t stream, const Problem &problem, int num_cols, CSRSparseMatrix &output,
             bool want_scatter_maps);

  /**
   * @brief Builds the same pattern in uniform block storage.
   *
   * Block-pair discovery is already block-level, so this is the *cheaper* of
   * the two expansions: each state-block pair emits
   * `(row_tangent / b) x (col_tangent / b)` tiles instead of
   * `row_tangent * col_tangent` scalar column indices.
   *
   * WriteOffsets() is then measured in tiles rather than scalar columns.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param num_cols Free tangent dimensions; must be a multiple of block_size.
   * @param block_size Tile edge; see ChooseHessianBlockSize().
   * @param[out] output BSR matrix; structure filled, values sized.
   * @param want_scatter_maps As for the CSR overload.
   */
  void Build(cudaStream_t stream, const Problem &problem, int num_cols, int block_size,
             BSRSparseMatrix &output, bool want_scatter_maps);

  /** @brief Per-residual-batch geometry, indexed as the problem's batches. */
  const std::vector<HessianBatchLayout> &Layout() const { return layout_; }

  /** @brief Global column of each (factor, block); -1 when constant. */
  const dvector<int> &FactorCols() const { return factor_cols_; }

  /**
   * @brief Row-relative offset of each (factor, block_a, block_b) run.
   *
   * -1 when either block is a constant state.  Empty unless Build() was called
   * with `want_scatter_maps`.
   */
  const dvector<int> &WriteOffsets() const { return write_offsets_; }

  /** @brief Total floats needed for the per-factor Jacobian value buffer. */
  size_t JacobianValuesSize() const { return jacobian_values_size_; }

 private:
  /**
   * @brief Shared front half: layout, column resolution, key sort, segmentation.
   *
   * Leaves the deduplicated block pairs in @p pair_fields and returns their
   * count; both expansions differ only in what they do with them.
   */
  int DiscoverBlockPairs(cudaStream_t stream, const Problem &problem, int num_cols,
                         dvector<int> &pair_fields, dvector<int> &pair_order,
                         dvector<int> &pair_group, size_t &num_valid);

  /** @brief Fills layout_ and the flat-buffer offsets from the problem. */
  void BuildLayout(const Problem &problem);

  /**
   * @brief Resolves every (factor, block) slot to a global column.
   *
   * @param[out] tangent_at_col Tangent size of the block owning each column.
   */
  void ResolveFactorColumns(cudaStream_t stream, const Problem &problem, int num_cols,
                            dvector<int> &tangent_at_col);

  std::vector<HessianBatchLayout> layout_;
  dvector<int> factor_cols_;
  dvector<int> write_offsets_;

  /// Reusable pinned staging buffer for D2H readbacks.
  pvector<int> pinned_buf_;

  size_t total_pairs_ = 0;
  size_t jacobian_values_size_ = 0;
};

}  // namespace cunls
