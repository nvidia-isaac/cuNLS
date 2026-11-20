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

class Problem;

/**
 * @brief Largest uniform tile size that divides every state's tangent
 * dimension.
 *
 * Uniform BSR needs one tile edge for the whole matrix, but cuNLS problems mix
 * tangent dimensions (SE3 poses at 6, landmarks at 3, arbitrary vectors).  The
 * gcd is the largest edge that tiles every block pair exactly, so no padding is
 * ever stored: PGO gives 6, bundle adjustment gives 3.  A gcd of 1 means block
 * storage would degenerate to scalar CSR, and the caller should stay on CSR.
 *
 * Block column offsets are cumulative sums of tangent dimensions, so they are
 * automatically multiples of the gcd.
 *
 * The edge is capped at 16, the largest tile the block SpMV accumulates
 * (MultiplyBSRByDenseVector).
 *
 * @param problem The optimization problem.
 * @return The tile edge, or 1 when block storage is not worthwhile.
 */
int ChooseHessianBlockSize(const Problem &problem);

/**
 * @brief Extracts the main diagonal of a BSR matrix.
 *
 * @param stream CUDA stream for GPU operations.
 * @param matrix BSR matrix.
 * @param[out] diagonal Output of length matrix.NumRows().
 */
void ExtractDiagonal(cudaStream_t stream, const BSRSparseMatrix &matrix, dvector<float> &diagonal);

/** @brief y = A * x for block storage (y is resized to the row count). */
void MultiplyBSRByDenseVector(cudaStream_t stream, const BSRSparseMatrix &matrix,
                              const dvector<float> &x, dvector<float> &y);

/**
 * @brief y = A * x for block storage with the pattern of @p structure (its row
 * offsets, column indices, block size and longest row; not its values) and the
 * values @p values, laid out as structure.values would be. @p x and @p y are
 * device arrays of structure.NumRows() floats.
 */
void MultiplyBSRValuesByDenseVector(cudaStream_t stream, const BSRSparseMatrix &structure,
                                    const float *values, const float *x, float *y);

}  // namespace cunls
