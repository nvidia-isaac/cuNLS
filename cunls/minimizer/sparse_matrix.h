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

namespace cunls {

/**
 * @brief Extracts dimensions and nonzero count from a CSR sparse matrix.
 *
 * Determines the number of rows from the row_offsets array size, the number
 * of columns from the maximum column index, and the number of nonzeros from
 * the values array size.
 *
 * @param stream CUDA stream for GPU operations.
 * @param matrix CSR sparse matrix to extract metadata from.
 * @param[out] num_rows Number of rows in the matrix.
 * @param[out] num_cols Number of columns in the matrix.
 * @param[out] num_nonzeros Number of nonzero elements.
 */
void ExtractMatrixMetadata(cudaStream_t stream, const CSRSparseMatrix &matrix, int &num_rows,
                           int &num_cols, int &num_nonzeros);

/**
 * @brief Extracts the diagonal elements from a CSR sparse matrix.
 *
 * Uses a warp-cooperative CUDA kernel where each warp processes one row
 * to find and extract the diagonal element (where col == row).
 *
 * @param stream CUDA stream for GPU operations.
 * @param matrix CSR sparse matrix to extract diagonal from.
 * @param[out] diagonal Output vector of diagonal elements.
 */
void ExtractDiagonal(cudaStream_t stream, const CSRSparseMatrix &matrix, dvector<float> &diagonal);

/**
 * @brief Adds a scaled diagonal to a sparse matrix.
 *
 * Computes result = matrix + scale * diag(diagonal). First copies the input
 * matrix, then adds scale * diagonal[i] to each diagonal entry.
 *
 * @param stream CUDA stream for GPU operations.
 * @param scale Scaling factor for the diagonal values.
 * @param diagonal Vector of diagonal values to add.
 * @param matrix Input CSR sparse matrix.
 * @param[out] result Output CSR sparse matrix (may alias matrix for in-place).
 */
void AddScaledDiagonal(cudaStream_t stream, float scale, const dvector<float> &diagonal,
                       const CSRSparseMatrix &matrix, CSRSparseMatrix &result);

/**
 * @brief Creates a deep copy of a CSR sparse matrix.
 *
 * Copies all arrays (values, column indices, row offsets) from the input
 * matrix to the output matrix. No-op if input and output are the same object.
 *
 * @param stream CUDA stream for GPU operations.
 * @param input Source CSR sparse matrix.
 * @param[out] output Destination CSR sparse matrix.
 */
void CopyCSRSparseMatrix(cudaStream_t stream, const CSRSparseMatrix &input,
                         CSRSparseMatrix &output);

/**
 * @brief Symmetric diagonal scaling of a CSR matrix: A_ij *= scale[i] *
 * scale[j].
 *
 * @param stream CUDA stream.
 * @param[in,out] matrix CSR matrix updated in-place.
 * @param scale Length must match matrix row/column dimension (square H).
 */
void ScaleSymmetricCSR(cudaStream_t stream, CSRSparseMatrix &matrix, const dvector<float> &scale);

/**
 * @brief Sets v[i] = 1 / sqrt(max(v[i], floor_value)) for all i (in-place).
 *
 * Used after ExtractDiagonal when building S from Hessian diagonal.
 */
void InvertSqrtWithFloorInPlace(cudaStream_t stream, dvector<float> &v, float floor_value = 1e-12f);

// ---- Async variants: write scalar result to device memory, no D2H or sync --

void ElementwiseMultiplyInPlace(cudaStream_t stream, float *a, const float *b, size_t n);

/**
 * @brief positions[i] = index of the diagonal entry (i, i) in the values of a
 * square CSR matrix, or -1 when the row has none. Once per sparsity pattern.
 */
void FindDiagonalPositions(cudaStream_t stream, const CSRSparseMatrix &matrix,
                           dvector<int> &positions);

/** @brief diagonal[i] = values[positions[i]] (0 where positions[i] < 0). */
void ExtractDiagonalAt(cudaStream_t stream, const CSRSparseMatrix &matrix,
                       const dvector<int> &positions, dvector<float> &diagonal);

/** @brief values[positions[i]] += scale * diagonal[i] (in place). */
void AddScaledDiagonalAt(cudaStream_t stream, float scale, const dvector<float> &diagonal,
                         const dvector<int> &positions, CSRSparseMatrix &matrix);

/** @brief Copies only the values of a CSR matrix with the same pattern. */
void CopyCSRValues(cudaStream_t stream, const CSRSparseMatrix &input, CSRSparseMatrix &output);

/**
 * @brief For every i with mask[i] == 0: zeroes row i and column i of a
 * symmetric matrix with a symmetric, column-sorted pattern, except the
 * diagonal entry, which becomes 1 if it is not positive. Work proportional to
 * the masked rows (the column entries are found by binary search in the rows
 * they live in), so a mostly free mask costs about one read of the mask.
 */
void ZeroMaskedRowsColumns(cudaStream_t stream, CSRSparseMatrix &matrix,
                           const dvector<float> &mask);

/** @brief Block-storage counterpart of the CSR overload (scalar mask). */
void ZeroMaskedRowsColumns(cudaStream_t stream, BSRSparseMatrix &matrix,
                           const dvector<float> &mask);

/**
 * @brief Active-set refinement of projected Gauss-Newton: components free in
 * `held_mask` (1) but marked by `step_mask` (0: at a bound, the step pushes
 * outward) become held. Writes extra[i] = 0 for them, 1 elsewhere, updates
 * held_mask, and adds their count to *d_count (device int).
 */
void HoldOutwardSteps(cudaStream_t stream, const dvector<float> &step_mask,
                      dvector<float> &held_mask, dvector<float> &extra, int *d_count);

/** @brief y = A * x for scalar CSR storage (cuSPARSE SpMV; y is resized). */
void MultiplyCSRByDenseVector(cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix,
                              int num_rows, int num_cols, int num_nonzeros, const dvector<float> &x,
                              dvector<float> &y, dvector<uint8_t> &buffer);

}  // namespace cunls
