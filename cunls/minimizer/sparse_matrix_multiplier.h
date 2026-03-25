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

#include <memory>

#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

/**
 * @brief Abstract base class for computing A^T * A of a sparse CSR matrix.
 *
 * Provides a two-phase interface: Initialize() precomputes the output sparsity
 * pattern once per problem structure, and ComputeSquaredMatrix() fills in
 * numerical values and may be called repeatedly as the input values change.
 * Concrete implementations can use different GPU strategies (e.g. cuSPARSE GEMM
 * reuse or custom CUDA kernels).
 */
class SparseMatrixMultiplier {
 public:
  /** @brief Virtual destructor for proper cleanup of derived instances. */
  virtual ~SparseMatrixMultiplier() = default;

  /**
   * @brief Analyzes the sparsity pattern of A^T * A and allocates the output.
   *
   * Must be called once whenever the sparsity pattern of @p input changes.
   * Implementations may use @p problem to extract structural hints (e.g. max
   * nonzeros per row) that accelerate the analysis.
   *
   * @param stream  CUDA stream for GPU operations.
   * @param problem Optimization problem providing structural information.
   * @param input   Input sparse matrix A (typically the Jacobian in CSR).
   * @param[out] output Output sparse matrix A^T * A (structure allocated).
   */
  virtual void Initialize(cudaStream_t stream, const Problem& problem,
                          const CSRSparseMatrix& input,
                          CSRSparseMatrix& output) = 0;

  /**
   * @brief Computes A^T * A for a sparse matrix A.
   *
   * The output sparsity structure must already be set by a prior call to
   * Initialize(). Only the numerical values are recomputed.
   *
   * @param stream  CUDA stream for GPU operations.
   * @param problem Optimization problem (may be used for kernel tuning).
   * @param input   Input sparse matrix A (typically the Jacobian).
   * @param[out] output Output sparse matrix A^T * A.
   */
  virtual void ComputeSquaredMatrix(cudaStream_t stream,
                                    const Problem& problem,
                                    const CSRSparseMatrix& input,
                                    CSRSparseMatrix& output) = 0;
};

/**
 * @brief Strategy for computing the approximate Hessian J^T * J.
 */
enum class SparseMatrixMultiplierType {
  cuSPARSE,  ///< cuSPARSE SpGEMM reuse API (transpose + multiply).
  Fast,      ///< Fast warp-efficient CUDA kernels with bitmap pattern
             ///< discovery.
};

/**
 * @brief Smart pointer type for sparse square multipliers.
 */
using SparseMatrixMultiplierPtr = std::unique_ptr<SparseMatrixMultiplier>;

/**
 * @brief Factory function to create a sparse square multiplier.
 *
 * @param type The strategy to use for computing A^T * A.
 * @return A unique pointer to the created multiplier instance.
 */
SparseMatrixMultiplierPtr CreateSparseMatrixMultiplier(
    SparseMatrixMultiplierType type);

}  // namespace cunls
