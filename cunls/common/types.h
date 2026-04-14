/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @brief SL(4) homogeneous matrix (4x4, row-major, det = 1 in exact arithmetic).
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
  dvector<int> row_offsets;   ///< Row offset array (num_rows + 1 entries).
  dvector<int> col_ids;       ///< Column index array (num_nonzeros entries).
  dvector<float> values;      ///< Non-zero value array (num_nonzeros entries).

  /**
   * @brief Returns the number of rows in the matrix.
   * @return Number of rows (row_offsets.size() - 1).
   */
  size_t NumRows() const {
    return row_offsets.empty() ? 0 : row_offsets.size() - 1;
  }

  /**
   * @brief Returns the number of non-zero entries in the matrix.
   * @return Number of non-zero values.
   */
  size_t NumNonZeros() const { return values.size(); }
};

/**
 * @brief COO-like sparse structure storing row and column indices.
 *
 * Used as a triplet/coordinate format for building sparse matrices
 * before conversion to CSR.
 */
struct TripletSparseStructure {
  dvector<int> row_ids;   ///< Row indices of non-zero entries.
  dvector<int> col_ids;   ///< Column indices of non-zero entries.
};

/**
 * @brief Sparse Jacobian matrix in triplet format.
 *
 * Stores both the sparsity structure (row/column indices) and the
 * non-zero values of the Jacobian.
 */
struct SparseJacobian {
  TripletSparseStructure structure;  ///< Row and column indices.
  dvector<float> values;             ///< Non-zero Jacobian values.
};

}  // namespace cunls
