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
 * @brief Sets v[i] = 1 / sqrt(max(v[i], floor_value)) for all i (in-place).
 *
 * Builds the column scale S of ColumnScaling::HessianDiagonal from diag(H).
 */
void InvertSqrtWithFloorInPlace(cudaStream_t stream, dvector<float> &v, float floor_value = 1e-12f);

/** @brief a[i] *= b[i] for i < n (in place, asynchronous). */
void ElementwiseMultiplyInPlace(cudaStream_t stream, float *a, const float *b, size_t n);

/**
 * @brief positions[i] = index of the diagonal entry (i, i) in the values of a
 * square CSR matrix, or -1 when the row has none. Once per sparsity pattern.
 * The column ids of each row must be ascending (a binary search; the Hessian
 * builders produce such rows).
 */
void FindDiagonalPositions(cudaStream_t stream, const CSRSparseMatrix &matrix,
                           dvector<int> &positions);

/** @brief diagonal[i] = values[positions[i]] (0 where positions[i] < 0). */
void ExtractDiagonalAt(cudaStream_t stream, const CSRSparseMatrix &matrix,
                       const dvector<int> &positions, dvector<float> &diagonal);

/** @brief y = A * x for scalar CSR storage (cuSPARSE SpMV; y is resized). */
void MultiplyCSRByDenseVector(cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix,
                              int num_rows, int num_cols, int num_nonzeros, const dvector<float> &x,
                              dvector<float> &y, dvector<uint8_t> &buffer);

}  // namespace cunls
