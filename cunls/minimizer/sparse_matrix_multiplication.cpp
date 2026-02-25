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

#include <cusparse.h>

#include <cassert>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

constexpr cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;

/**
 * @brief Constructor for SparseMatrixMultiplication
 *
 * Initializes the cuSPARSE GEMM descriptor used for sparse matrix
 * multiplication operations. The descriptor is reused across multiple
 * multiplications for better performance.
 *
 * @throws cusparse_exception if descriptor creation fails
 */
SparseMatrixMultiplication::SparseMatrixMultiplication() {
  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&gemm_description_));
}

/**
 * @brief Destructor for SparseMatrixMultiplication
 *
 * Cleans up the cuSPARSE GEMM descriptor and releases associated resources.
 * Warnings are logged if descriptor destruction fails rather than throwing
 * exceptions from the destructor.
 */
SparseMatrixMultiplication::~SparseMatrixMultiplication() {
  WARN_ON_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(gemm_description_));
}

/**
 * @brief Transposes a sparse matrix from CSR to CSC format
 *
 * Converts a Compressed Sparse Row (CSR) matrix to Compressed Sparse
 * Column (CSC) format, effectively computing the matrix transpose.
 * Uses cuSPARSE's csr2csc conversion for efficient GPU-based operation.
 *
 * @param stream CUDA stream for asynchronous execution
 * @param matrix Input CSR sparse matrix to transpose
 * @param transposed[out] Output transposed matrix in CSC format
 *
 * @throws cusparse_exception if transpose operation fails
 * @throws cuda_exception if CUDA operations fail
 */
void SparseMatrixMultiplication::Transpose(cudaStream_t stream,
                                           const CSRSparseMatrix& matrix,
                                           CSRSparseMatrix& transposed) {
  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, matrix, num_rows, num_cols, num_nonzeros);

  if (transposed.values.size() != num_nonzeros) {
    transposed.values.resize(num_nonzeros);
  }

  if (transposed.col_ids.size() != num_nonzeros) {
    transposed.col_ids.resize(num_nonzeros);
  }

  if (transposed.row_offsets.size() != num_cols + 1) {
    transposed.row_offsets.resize(num_cols + 1);
  }

  size_t bufferSize = 0;

  THROW_ON_CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(
      handle_.GetHandle(stream), num_rows, num_cols, num_nonzeros,
      matrix.values.data(),
      matrix.row_offsets.data(),
      matrix.col_ids.data(),
      transposed.values.data(),
      transposed.row_offsets.data(),
      transposed.col_ids.data(), CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG_DEFAULT, &bufferSize));

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  if (buffer1.size() < bufferSize) {
    buffer1.resize(bufferSize);
  }

  THROW_ON_CUSPARSE_ERROR(cusparseCsr2cscEx2(
      handle_.GetHandle(stream), num_rows, num_cols, num_nonzeros,
      matrix.values.data(),
      matrix.row_offsets.data(),
      matrix.col_ids.data(),
      transposed.values.data(),
      transposed.row_offsets.data(),
      transposed.col_ids.data(), CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer1.data()));
}

/**
 * @brief Computes the squared matrix A^T * A
 *
 * Performs the computation of A transpose times A, which is commonly
 * used in optimization algorithms (e.g., computing Hessian approximations
 * from Jacobian matrices). The operation is done in two steps:
 * 1. Transpose the input matrix
 * 2. Multiply the transposed matrix with the original
 *
 * @param stream CUDA stream for asynchronous execution
 * @param input Input sparse matrix A
 * @param output[out] Result matrix A^T * A
 *
 * @throws cusparse_exception if matrix operations fail
 * @throws cuda_exception if CUDA operations fail
 */
void SparseMatrixMultiplication::ComputeSquaredMatrix(
    cudaStream_t stream, const CSRSparseMatrix& input,
    CSRSparseMatrix& output) {
  Transpose(stream, input, temp_matrix_);
  Multiply(stream, temp_matrix_, input, output);
}

/**
 * @brief Estimates work buffer size for sparse matrix multiplication
 *
 * Queries cuSPARSE for the required buffer size for the work estimation
 * phase of sparse matrix multiplication. This is the first phase of the
 * three-phase cuSPARSE GEMM reuse API that allows for efficient repeated
 * multiplications with the same sparsity pattern.
 *
 * @param handle cuSPARSE handle for the operation
 *
 * @throws cusparse_exception if work estimation fails
 */
void SparseMatrixMultiplication::EstimateWork(cusparseHandle_t handle) {
  size_t bufferSize1 = 0;

  auto dA = descrA_.GetDescription();
  auto dB = descrB_.GetDescription();
  auto dC = descrC_.GetDescription();

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_workEstimation(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize1, NULL));

  buffer1.resize(bufferSize1);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_workEstimation(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize1,
      buffer1.data()));
}

/**
 * @brief Determines the nonzero pattern of the result matrix
 *
 * Analyzes the sparsity structure of the input matrices to determine
 * the nonzero pattern of the result matrix. This is the second phase
 * of the cuSPARSE GEMM reuse API. Allocates and initializes multiple
 * work buffers needed for the pattern analysis.
 *
 * @param handle cuSPARSE handle for the operation
 *
 * @throws cusparse_exception if nonzero pattern analysis fails
 */
void SparseMatrixMultiplication::ReuseNonzeros(cusparseHandle_t handle) {
  size_t bufferSize2 = 0;
  size_t bufferSize3 = 0;
  size_t bufferSize4 = 0;

  auto dA = descrA_.GetDescription();
  auto dB = descrB_.GetDescription();
  auto dC = descrC_.GetDescription();

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize2, NULL, &bufferSize3, NULL, &bufferSize4,
      NULL));

  buffer2.resize(bufferSize2);
  buffer3.resize(bufferSize3);
  buffer4.resize(bufferSize4);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize2, buffer2.data(),
      &bufferSize3, buffer3.data(), &bufferSize4,
      buffer4.data()));
}

/**
 * @brief Prepares for copying computed values to the result matrix
 *
 * Sets up the final work buffer needed for copying the computed values
 * to the result matrix. This is the third phase of the cuSPARSE GEMM
 * reuse API setup, after which the actual multiplication can be performed
 * efficiently multiple times.
 *
 * @param handle cuSPARSE handle for the operation
 *
 * @throws cusparse_exception if copy preparation fails
 */
void SparseMatrixMultiplication::ReuseCopy(cusparseHandle_t handle) {
  size_t bufferSize5 = 0;

  auto dA = descrA_.GetDescription();
  auto dB = descrB_.GetDescription();
  auto dC = descrC_.GetDescription();

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize5, NULL));

  buffer5.resize(bufferSize5);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(
      handle, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT,
      gemm_description_, &bufferSize5,
      buffer5.data()));
}

/**
 * @brief Initializes sparse matrix multiplication for new matrix structures
 *
 * Performs the complete initialization sequence for sparse matrix
 * multiplication when the structure of input matrices changes. This includes:
 * 1. Creating matrix descriptors for A, B, and C
 * 2. Estimating work requirements
 * 3. Analyzing nonzero patterns
 * 4. Allocating result matrix storage
 * 5. Preparing for value computation
 *
 * The function is optimized for the case where:
 * - matrixA is J^T (Jacobian transpose)
 * - matrixB is J (Jacobian)
 * - matrixC is the Hessian approximation (J^T * J)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param handle cuSPARSE handle for the operation
 * @param matrixA Left operand matrix (typically J^T)
 * @param matrixB Right operand matrix (typically J)
 * @param matrixC[out] Result matrix (resized and prepared for computation)
 *
 * @throws cusparse_exception if GEMM initialization fails
 */
void SparseMatrixMultiplication::TryInitializeGEMM(
    cudaStream_t stream, cusparseHandle_t handle,
    const CSRSparseMatrix& matrixA, const CSRSparseMatrix& matrixB,
    CSRSparseMatrix& matrixC) {
  auto h1 = std::hash<CSRSparseMatrix>()(matrixA);
  auto h2 = std::hash<CSRSparseMatrix>()(matrixB);

  if (h1 == matrixA_hash_ && h2 == matrixB_hash_) {
    return;
  }

  matrixA_hash_ = h1;
  matrixB_hash_ = h2;

  // MatrixA corresponds to JT
  // MatrixB corresponds to J
  // MatrixC corresponds to Hessian
  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, matrixB, num_rows, num_cols, num_nonzeros);

  descrA_ = std::move(
      cuSPARSEMatrixDescription(num_cols, num_rows, num_nonzeros, matrixA));
  descrB_ = std::move(
      cuSPARSEMatrixDescription(num_rows, num_cols, num_nonzeros, matrixB));
  descrC_ = std::move(cuSPARSEMatrixDescription(num_cols, num_cols));

  EstimateWork(handle);
  ReuseNonzeros(handle);

  // Clear the unnecessary data;
  buffer1.clear();
  buffer2.clear();

  // Obtain the estimated size of the output matrix
  int64_t C_num_rows, C_num_cols, C_nnz;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMatGetSize(
      descrC_.GetDescription(), &C_num_rows, &C_num_cols, &C_nnz));

  // Resize the output matrix accordingly
  matrixC.row_offsets.resize(C_num_rows + 1);
  matrixC.col_ids.resize(C_nnz);
  matrixC.values.resize(C_nnz, 0);

  descrC_.UpdatePointers(matrixC);
  ReuseCopy(handle);

  // Clear the unnecessary data;
  buffer3.clear();
}

/**
 * @brief Performs sparse matrix multiplication C = A * B
 *
 * Efficiently multiplies two sparse matrices using cuSPARSE's reuse API.
 * The function uses hash-based caching to detect when matrix structures
 * have changed, only re-initializing expensive setup operations when necessary.
 * This makes repeated multiplications with the same sparsity pattern very fast.
 *
 * The function is optimized for the specific use case where:
 * - matrixA represents J^T (Jacobian transpose)
 * - matrixB represents J (Jacobian matrix)
 * - matrixC represents the Hessian approximation (J^T * J)
 *
 * @param stream CUDA stream for asynchronous execution
 * @param matrixA Left operand sparse matrix
 * @param matrixB Right operand sparse matrix
 * @param matrixC[out] Result matrix C = A * B
 *
 * @throws cusparse_exception if matrix multiplication fails
 * @throws cuda_exception if CUDA operations fail
 *
 * @note The result matrix C is automatically resized if the input matrix
 *       structures have changed since the last multiplication.
 */
void SparseMatrixMultiplication::Multiply(cudaStream_t stream,
                                          const CSRSparseMatrix& matrixA,
                                          const CSRSparseMatrix& matrixB,
                                          CSRSparseMatrix& matrixC) {
  // MatrixA corresponds to JT
  // MatrixB corresponds to J
  // MatrixC corresponds to Hessian
  auto handle = handle_.GetHandle(stream);

  // Check if the structure of inputs hasnt changed, s.t. we dont
  // do the result analisys in vain.
  // If the structure of the jacobian has changed, recalculate
  // result structure and prepare working buffers.
  TryInitializeGEMM(stream, handle, matrixA, matrixB, matrixC);

  int64_t C_num_rows, C_num_cols, C_nnz;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMatGetSize(
      descrC_.GetDescription(), &C_num_rows, &C_num_cols, &C_nnz));

  assert(matrixC.row_offsets.size() == C_num_rows + 1);
  assert(matrixC.col_ids.size() == C_nnz);
  assert(matrixC.values.size() == C_nnz);

  constexpr float alpha = 1;
  constexpr float beta = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_compute(
      handle, operation, operation, &alpha, descrA_.GetDescription(),
      descrB_.GetDescription(), &beta, descrC_.GetDescription(), CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, gemm_description_));
}
}  // namespace cunls
