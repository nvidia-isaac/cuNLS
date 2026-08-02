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

#include <cuda/std/array>
#include <vector>

#include "device_vector.h"
#include "pinned_vector.h"

namespace cunls {

/**
 * @brief Fixed-size vector of floats stored as a cuda::std::array.
 *
 * Suitable for use in both host and device code. Elements are stored
 * contiguously in memory.
 *
 * @tparam Dim Number of elements in the vector.
 */
template <int Dim>
using Vector = cuda::std::array<float, Dim>;

/**
 * @brief Fixed-size square matrix of floats stored in row-major order.
 *
 * Stored as a flat cuda::std::array of Dim*Dim elements. Suitable for
 * use in both host and device code.
 *
 * @tparam Dim Number of rows (and columns) in the square matrix.
 */
template <int Dim>
using Matrix = cuda::std::array<float, Dim * Dim>;

/**
 * @brief SE(3) transformation matrix representation.
 *
 * Represents a 4x4 homogeneous transformation matrix in row-major order
 * stored as a contiguous array of 16 floats. The matrix format is:
 * [R00 R01 R02 tx]
 * [R10 R11 R12 ty]
 * [R20 R21 R22 tz]
 * [0   0   0   1 ]
 * where R is a 3x3 rotation matrix and [tx, ty, tz] is the translation vector.
 */
using SE3Transform = Matrix<4>;

/**
 * @brief SL(4) homogeneous matrix (4x4, row-major, det = 1 in exact
 * arithmetic).
 */
using SL4Transform = Matrix<4>;

/**
 * @brief Alias for a device (GPU) vector.
 * @tparam T Element type (must be trivially copyable).
 */
template <class T>
using dvector = DeviceVector<T>;

/**
 * @brief Alias for a host (CPU) vector (std::vector).
 * @tparam T Element type.
 */
template <class T>
using hvector = std::vector<T>;

/**
 * @brief Alias for a pinned (CPU) vector (PinnedVector).
 * @tparam T Element type.
 */
template <class T>
using pvector = PinnedVector<T>;

/**
 * @brief Compressed Sparse Row (CSR) matrix stored in GPU memory.
 *
 * Represents a sparse matrix using three arrays:
 * - row_offsets: indices into col_ids/values for the start of each row
 *   (size = num_rows + 1).
 * - col_ids: column indices of non-zero entries (size = num_nonzeros).
 * - values: non-zero values (size = num_nonzeros).
 */
/**
 * @brief Cached sparse matrix dimensions.
 *
 * Stores num_rows, num_cols, and num_nonzeros so that repeated GPU scans
 * (e.g. thrust::max_element + cudaStreamSynchronize) can be avoided in
 * hot loops. Populated once during initialization via ExtractMatrixMetadata
 * and reused thereafter.
 */
struct CSRMatrixDimensions {
  int num_rows = -1;
  int num_cols = -1;
  int num_nonzeros = -1;

  bool IsValid() const { return num_rows >= 0; }

  void Set(int rows, int cols, int nnz) {
    num_rows = rows;
    num_cols = cols;
    num_nonzeros = nnz;
  }

  void Invalidate() {
    num_rows = -1;
    num_cols = -1;
    num_nonzeros = -1;
  }
};

/**
 * @brief Compressed Sparse Row (CSR) matrix stored in GPU memory.
 *
 * Represents a sparse matrix using three arrays:
 * - row_offsets: indices into col_ids/values for the start of each row
 *   (size = num_rows + 1).
 * - col_ids: column indices of non-zero entries (size = num_nonzeros).
 * - values: non-zero values (size = num_nonzeros).
 */
struct CSRSparseMatrix {
  dvector<int> row_offsets; ///< Row offset array (num_rows + 1 entries).
  dvector<int> col_ids;      ///< Column index array (num_nonzeros entries).
  dvector<float> values;     ///< Non-zero value array (num_nonzeros entries).

  /**
   * @brief Returns the number of rows in the matrix.
   * @return Number of rows (row_offsets.size() - 1).
   */
  size_t NumRows() const { return row_offsets.empty() ? 0 : row_offsets.size() - 1; }

  /**
   * @brief Returns the number of non-zero entries in the matrix.
   * @return Number of non-zero values.
   */
  size_t NumNonZeros() const { return values.size(); }
};

/**
 * @brief Block Sparse Row (BSR) matrix stored in GPU memory.
 *
 * A square matrix partitioned into uniform `block_size x block_size` tiles.
 * Only tiles containing at least one structural non-zero are stored:
 * - row_offsets: index into col_ids/values-tiles per block row
 *   (size = num_block_rows + 1).
 * - col_ids: block-column index of each stored tile (size = NumBlocks()).
 * - values: tiles laid out row-major and contiguously, i.e. entry (k, l) of
 *   tile `t` lives at `values[t * block_size * block_size + k * block_size + l]`
 *   (CUSPARSE_DIRECTION_ROW).
 *
 * The Hessian of a factor graph is naturally block structured: every state
 * block contributes a dense tile per neighbour. Storing it this way keeps one
 * column index per tile instead of one per scalar entry, which is where the
 * SpMV bandwidth saving comes from — the values are identical either way.
 *
 * Requires every state block's tangent dimension to be a multiple of
 * `block_size`; see ChooseHessianBlockSize().
 */
struct BSRSparseMatrix {
  dvector<int> row_offsets; ///< Block-row offsets (num_block_rows + 1 entries).
  dvector<int> col_ids;      ///< Block-column index per tile.
  dvector<float> values;     ///< Tiles, row-major, block_size^2 floats each.

  int block_size = 1; ///< Tile edge length.
  /**
   * @brief Largest number of tiles in any block row.
   *
   * Selects the SpMV schedule.  Factor-graph Hessians come in two shapes: a
   * pose graph is near-uniform with a handful of tiles per row, while bundle
   * adjustment is extremely skewed (a pose row holds one tile per observation
   * of that camera, a landmark row a handful).  One schedule cannot serve both.
   */
  int max_tiles_per_row = 0;
  int num_block_rows = 0; ///< Number of block rows (= block columns).

  /** @brief Number of stored tiles. */
  size_t NumBlocks() const { return col_ids.size(); }

  /** @brief Scalar row/column count. */
  int NumRows() const { return num_block_rows * block_size; }

  /** @brief Number of stored scalar entries (tiles are dense). */
  size_t NumNonZeros() const { return values.size(); }
};

/**
 * @brief Per-factor dense Jacobian blocks, concatenated across residual
 * batches.
 *
 * Each factor batch writes `NumFactors()` dense row-major blocks of
 * `ResidualsSize() x sum(StateBlockSizes())` floats, and the batches are laid
 * out back to back.  There is no global sparse Jacobian: the Hessian is
 * assembled from these blocks directly (see BlockHessianAssembler).
 */
using PerFactorJacobians = dvector<float>;

} // namespace cunls
