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

#include <cuda_runtime.h>
#include <cusparse.h>

#include <memory>
#include <vector>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/jacobian_ops.h"

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
void ExtractMatrixMetadata(cudaStream_t stream, const CSRSparseMatrix& matrix,
                           int& num_rows, int& num_cols, int& num_nonzeros);

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
void ExtractDiagonal(cudaStream_t stream, const CSRSparseMatrix& matrix,
                     dvector<float>& diagonal);

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
void AddScaledDiagonal(cudaStream_t stream, float scale,
                       const dvector<float>& diagonal,
                       const CSRSparseMatrix& matrix, CSRSparseMatrix& result);

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
void CopyCSRSparseMatrix(cudaStream_t stream, const CSRSparseMatrix& input,
                         CSRSparseMatrix& output);

/**
 * @brief Converts a triplet sparse structure to CSR format with index mapping.
 *
 * Filters out invalid entries (col_id == -1) from the triplet structure,
 * converts valid entries to CSR format, and builds a mapping from triplet
 * indices to CSR indices. The mapping enables efficient value-only updates
 * on subsequent iterations without re-converting the structure.
 *
 * @param stream CUDA stream for GPU operations.
 * @param handle cuSPARSE library handle.
 * @param structure Input triplet sparse structure (may contain -1 in col_ids).
 * @param[out] csr Output CSR sparse matrix (structure filled, values zeroed).
 * @param[out] mapping Output index mapping: mapping[triplet_idx] = csr_idx, or -1.
 * @param[out] buffer Temporary buffer for intermediate computations.
 */
void ConvertTripletStructureToCSR(cudaStream_t stream, cusparseHandle_t handle,
                                  const TripletSparseStructure& structure,
                                  CSRSparseMatrix& csr, dvector<int>& mapping,
                                  dvector<uint8_t>& buffer);

/**
 * @brief Scatters Jacobian values from triplet format into CSR format.
 *
 * Uses the precomputed mapping from ConvertTripletStructureToCSR to copy
 * updated Jacobian values from the triplet representation directly into
 * their corresponding CSR positions. Much faster than full re-conversion.
 *
 * @param stream CUDA stream for GPU operations.
 * @param jacobian Sparse Jacobian in triplet format with updated values.
 * @param mapping Precomputed triplet-to-CSR index mapping.
 * @param[out] csr CSR sparse matrix whose values are updated.
 */
void ConvertTripletToCSRValues(cudaStream_t stream,
                               const SparseJacobian& jacobian,
                               const dvector<int>& mapping,
                               CSRSparseMatrix& csr);

/**
 * @brief Computes the right-hand side of the normal equations: rhs = -J^T * r.
 *
 * Performs sparse matrix-transpose-vector multiplication followed by negation
 * to produce the negative gradient used in Gauss-Newton / LM solvers.
 *
 * @param stream CUDA stream for GPU operations.
 * @param handle cuSPARSE library handle.
 * @param jacobian CSR sparse Jacobian matrix (J).
 * @param residuals Dense residual vector (r).
 * @param[out] rhs Output right-hand side vector (-J^T * r).
 * @param[out] buffer Temporary buffer for cuSPARSE operations.
 */
void ComputeRHS(cudaStream_t stream, cusparseHandle_t handle,
                const CSRSparseMatrix& jacobian,
                const dvector<float>& residuals, dvector<float>& rhs,
                dvector<uint8_t>& buffer);

/**
 * @brief Computes the squared L2 norm of a step vector.
 *
 * Computes step^T * step using a GPU inner product.
 *
 * @param stream CUDA stream for GPU operations.
 * @param step Step vector.
 * @return The squared L2 norm (scalar value).
 */
float ComputeSquaredStep(cudaStream_t stream, const dvector<float>& step);

/**
 * @brief Computes a diagonally-weighted squared step norm.
 *
 * Computes step^T * diag(weights) * step, i.e., sum(weights[i] * step[i]^2).
 * Used in Levenberg-Marquardt to evaluate predicted cost reduction.
 *
 * @param stream CUDA stream for GPU operations.
 * @param weights Diagonal weight values.
 * @param step Step vector.
 * @param[out] buffer Temporary buffer for intermediate computations.
 * @return The weighted squared norm (scalar value).
 */
float ComputeWeightedSquaredStep(cudaStream_t stream,
                                 const dvector<float>& weights,
                                 const dvector<float>& step,
                                 dvector<uint8_t>& buffer);

/**
 * @brief Computes a sparse-matrix-weighted squared step norm.
 *
 * Computes step^T * A * step using sparse matrix-vector multiplication
 * followed by an inner product. Used when the weighting is a full sparse
 * matrix rather than just diagonal weights.
 *
 * @param stream CUDA stream for GPU operations.
 * @param handle cuSPARSE library handle.
 * @param matrix Sparse weight matrix (A).
 * @param step Step vector.
 * @param[out] buffer Temporary buffer for cuSPARSE operations.
 * @return The weighted squared norm (scalar value).
 */
float ComputeWeightedSquaredStep(cudaStream_t stream, cusparseHandle_t handle,
                                 const CSRSparseMatrix& matrix,
                                 const dvector<float>& step,
                                 dvector<uint8_t>& buffer);

/**
 * @brief Performs sparse matrix multiplication with structure caching.
 *
 * Efficiently computes the product of two sparse matrices using cuSPARSE's
 * reuse API. The class caches the sparsity-pattern analysis across calls,
 * detecting structure changes via hashing so that only the numeric phase
 * is repeated when the pattern is unchanged.
 *
 * Primary use case: computing the approximate Hessian H = J^T * J from
 * the Jacobian matrix J during nonlinear least-squares optimization.
 */
class SparseMatrixMultiplication {
 public:
  /** @brief Constructs and initializes the cuSPARSE GEMM descriptor. */
  SparseMatrixMultiplication();

  /** @brief Destroys the cuSPARSE GEMM descriptor and frees resources. */
  ~SparseMatrixMultiplication();

  /**
   * @brief Initializes the GEMM structural analysis for A^T * A.
   *
   * Transposes the input matrix to obtain its structure, then runs the
   * full three-phase cuSPARSE GEMM reuse setup (work estimation, nonzero
   * analysis, copy preparation) and allocates the output matrix.
   * Must be called once whenever the sparsity pattern changes (i.e., for
   * each new optimization problem).
   *
   * @param stream CUDA stream for GPU operations.
   * @param input  Input sparse matrix A (typically the Jacobian in CSR).
   * @param[out] output Output sparse matrix A^T * A (structure allocated).
   */
  void Initialize(cudaStream_t stream, const CSRSparseMatrix& input,
                  CSRSparseMatrix& output);

  /**
   * @brief Computes A^T * A for a sparse matrix A.
   *
   * Transposes the input matrix to update values, then performs the
   * numeric phase of the cuSPARSE GEMM reuse API.  Initialize() must
   * have been called first for the current sparsity pattern.
   *
   * @param stream CUDA stream for GPU operations.
   * @param input Input sparse matrix A (typically the Jacobian).
   * @param[out] output Output sparse matrix A^T * A.
   */
  void ComputeSquaredMatrix(cudaStream_t stream, const CSRSparseMatrix& input,
                            CSRSparseMatrix& output);

 private:
  /**
   * @brief Transposes a CSR matrix using cuSPARSE CSR-to-CSC conversion.
   *
   * @param stream CUDA stream for GPU operations.
   * @param matrix Input CSR sparse matrix.
   * @param[out] transposed Output transposed matrix.
   */
  void Transpose(cudaStream_t stream, const CSRSparseMatrix& matrix,
                 CSRSparseMatrix& transposed);

  /**
   * @brief Estimates work buffer size for the cuSPARSE GEMM reuse API.
   *
   * @param handle cuSPARSE library handle.
   */
  void EstimateWork(cusparseHandle_t handle);

  /**
   * @brief Analyzes the nonzero pattern of the result matrix.
   *
   * Second phase of the cuSPARSE GEMM reuse API setup.
   *
   * @param handle cuSPARSE library handle.
   */
  void ReuseNonzeros(cusparseHandle_t handle);

  /**
   * @brief Prepares work buffers for copying computed values to the result.
   *
   * Third phase of the cuSPARSE GEMM reuse API setup.
   *
   * @param handle cuSPARSE library handle.
   */
  void ReuseCopy(cusparseHandle_t handle);

  cuSPARSEHandle handle_;          ///< cuSPARSE handle for transpose operations.
  CSRSparseMatrix temp_matrix_;    ///< Temporary storage for the transposed matrix.

  cusparseSpGEMMDescr_t gemm_description_;  ///< cuSPARSE GEMM descriptor.

  cuSPARSEMatrixDescription descrA_, descrB_, descrC_;  ///< Matrix descriptors.

  dvector<uint8_t> buffer1;  ///< Work buffer for GEMM phase 1 (work estimation).
  dvector<uint8_t> buffer2;  ///< Work buffer for GEMM phase 2 (nonzero analysis).
  dvector<uint8_t> buffer3;  ///< Work buffer for GEMM phase 2 (nonzero analysis).
  dvector<uint8_t> buffer4;  ///< Work buffer for GEMM phase 2 (nonzero analysis).
  dvector<uint8_t> buffer5;  ///< Work buffer for GEMM phase 3 (copy preparation).
};

}  // namespace cunls
