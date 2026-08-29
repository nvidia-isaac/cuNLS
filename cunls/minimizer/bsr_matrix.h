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

#include "cunls/common/types.h"

namespace cunls {

class Problem;

/**
 * @brief Largest uniform tile size that divides every state block's tangent
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
 * @param problem The optimization problem.
 * @param max_block_size Upper bound on the returned edge length.
 * @return The tile edge, or 1 when block storage is not worthwhile.
 */
int ChooseHessianBlockSize(const Problem &problem, int max_block_size = 16);

/**
 * @brief Extracts the main diagonal of a BSR matrix.
 *
 * @param stream CUDA stream for GPU operations.
 * @param matrix BSR matrix.
 * @param[out] diagonal Output of length matrix.NumRows().
 */
void ExtractDiagonal(cudaStream_t stream, const BSRSparseMatrix &matrix, dvector<float> &diagonal);

/**
 * @brief Computes result = matrix + scale * diag(diagonal) for BSR storage.
 *
 * @param stream CUDA stream for GPU operations.
 * @param scale Scaling factor applied to the diagonal values.
 * @param diagonal Diagonal values to add; length matrix.NumRows().
 * @param matrix Input BSR matrix.
 * @param[out] result Output BSR matrix; may alias @p matrix for in-place.
 */
void AddScaledDiagonal(cudaStream_t stream, float scale, const dvector<float> &diagonal,
                       const BSRSparseMatrix &matrix, BSRSparseMatrix &result);

/**
 * @brief Symmetric diagonal scaling: A_ij *= scale[i] * scale[j].
 *
 * @param stream CUDA stream for GPU operations.
 * @param[in,out] matrix BSR matrix updated in place.
 * @param scale Length must equal matrix.NumRows().
 * @param[out] row_of_tile Caller-owned scratch mapping tile index to block row;
 *        resized as needed and rebuilt on each call.
 */
void ScaleSymmetric(cudaStream_t stream, BSRSparseMatrix &matrix, const dvector<float> &scale,
                    dvector<int> &row_of_tile);

/**
 * @brief Deep-copies a BSR matrix (structure and values).
 *
 * The structure is always copied, not just the values: equal sizes do not imply
 * equal connectivity when one solver instance is reused across problems.
 *
 * @param stream CUDA stream for GPU operations.
 * @param input Source matrix.
 * @param[out] output Destination matrix.
 */
void CopyBSRSparseMatrix(cudaStream_t stream, const BSRSparseMatrix &input,
                         BSRSparseMatrix &output);

/**
 * @brief Async BSR-weighted squared step: d_out[0] = step^T A step.
 *
 * @param stream CUDA stream for GPU operations.
 * @param matrix BSR matrix A.
 * @param step Step vector.
 * @param[out] scratch Scratch for the SpMV result.
 * @param d_out Device destination for the scalar.
 * @param d_partials Reduction scratch; see device_reduction.h.
 */
void ComputeWeightedSquaredStepAsync(cudaStream_t stream, const BSRSparseMatrix &matrix,
                                     const dvector<float> &step, dvector<float> &scratch,
                                     float *d_out, float *d_partials);

}  // namespace cunls
