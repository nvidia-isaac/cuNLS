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

#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Base class for sparse linear solvers operating on CSR matrices.
 *
 * Provides a common interface for solving sparse symmetric positive definite
 * (SPD) linear systems Ax = b where the matrix A is stored in CSR (Compressed
 * Sparse Row) format. Derived classes implement specific solver strategies.
 */
class CSRSparseLinearSolver {
 public:
  /**
   * @brief Performs setup work for the sparse linear system.
   *
   * Typically runs symbolic analysis of the sparsity pattern; some modes may
   * also perform an initial numerical factorization. Must be called once
   * whenever the matrix structure changes, before any call to Solve.
   *
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param handle Opaque solver handle (e.g. cuDSSHandle_t). Caller provides
   *               and owns the handle; it is used to store and pass solver
   *               context across Initialize and Solve.
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Initialize(void* handle, const CSRSparseMatrix& spd_matrix,
                          const dvector<float>& rhs,
                          dvector<float>& result) = 0;

  /**
   * @brief Solves a sparse SPD linear system Ax = b.
   *
   * Performs factorization and solve phases to compute the solution x.
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param handle Opaque solver handle (e.g. cuDSSHandle_t) from Initialize.
   * @param spd_matrix The coefficient matrix A in CSR format (must be symmetric
   *                   positive definite).
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Solve(void* handle, const CSRSparseMatrix& spd_matrix,
                     const dvector<float>& rhs, dvector<float>& result) = 0;

  /** @brief Virtual destructor for proper cleanup of derived solver instances. */
  virtual ~CSRSparseLinearSolver() = default;
};
}  // namespace cunls
