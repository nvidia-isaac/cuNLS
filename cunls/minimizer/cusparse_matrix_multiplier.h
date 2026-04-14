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

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/sparse_matrix_multiplier.h"

namespace cunls {

/**
 * @brief Computes A^T * A using cuSPARSE's SpGEMM reuse API.
 *
 * Caches the sparsity-pattern analysis across calls so that only the numeric
 * phase is repeated when the pattern is unchanged. Internally transposes the
 * input via CSR-to-CSC conversion and then multiplies A^T * A.
 */
class cuSPARSESparseMatrixMultiplier : public SparseMatrixMultiplier {
public:
  /** @brief Constructs and initializes the cuSPARSE GEMM descriptor. */
  cuSPARSESparseMatrixMultiplier();

  /** @brief Destroys the cuSPARSE GEMM descriptor and frees resources. */
  ~cuSPARSESparseMatrixMultiplier() override;

  /**
   * @brief Initializes the GEMM structural analysis for A^T * A.
   *
   * Transposes the input matrix to obtain its structure, then runs the
   * full three-phase cuSPARSE GEMM reuse setup (work estimation, nonzero
   * analysis, copy preparation) and allocates the output matrix.
   * Must be called once whenever the sparsity pattern changes.
   *
   * @param stream  CUDA stream for GPU operations.
   * @param problem Optimization problem (unused by this implementation).
   * @param input   Input sparse matrix A (typically the Jacobian in CSR).
   * @param[out] output Output sparse matrix A^T * A (structure allocated).
   */
  void Initialize(cudaStream_t stream, const Problem &problem,
                  const CSRSparseMatrix &input,
                  CSRSparseMatrix &output) override;

  /**
   * @brief Computes A^T * A for a sparse matrix A.
   *
   * Transposes the input matrix to update values, then performs the
   * numeric phase of the cuSPARSE GEMM reuse API.
   *
   * @param stream  CUDA stream for GPU operations.
   * @param problem Optimization problem (unused by this implementation).
   * @param input   Input sparse matrix A (typically the Jacobian).
   * @param[out] output Output sparse matrix A^T * A.
   */
  void ComputeSquaredMatrix(cudaStream_t stream, const Problem &problem,
                            const CSRSparseMatrix &input,
                            CSRSparseMatrix &output) override;

private:
  void Transpose(cudaStream_t stream, const CSRSparseMatrix &matrix,
                 CSRSparseMatrix &transposed);
  void EstimateWork(void *handle);
  void ReuseNonzeros(void *handle);
  void ReuseCopy(void *handle);

  cuSPARSEHandle handle_;
  CSRSparseMatrix temp_matrix_;

  void *gemm_description_ = nullptr;

  cuSPARSEMatrixDescription descrA_, descrB_, descrC_;

  dvector<uint8_t> buffer1;
  dvector<uint8_t> buffer2;
  dvector<uint8_t> buffer3;
  dvector<uint8_t> buffer4;
  dvector<uint8_t> buffer5;
};

} // namespace cunls
