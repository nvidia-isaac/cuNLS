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
 * @brief Base class for linear solvers operating on CSR matrices.
 *
 * Provides a common interface for solving sparse symmetric linear systems
 * Ax = b where the matrix A is stored in CSR (Compressed Sparse Row) format.
 * Derived classes implement specific solver strategies (e.g. cuDSS direct
 * factorization, dense pivoted LDLT).
 */
class CSRSparseLinearSolver {
 public:
  /**
   * @brief Performs setup work for the linear system.
   *
   * Typically runs symbolic analysis of the sparsity pattern; some modes may
   * also perform an initial numerical factorization. Must be called once
   * whenever the matrix structure changes, before any call to Solve.
   *
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Initialize(cudaStream_t stream,
                          const CSRSparseMatrix& spd_matrix,
                          const dvector<float>& rhs,
                          dvector<float>& result) = 0;

  /**
   * @brief Solves a linear system Ax = b.
   *
   * Performs factorization and solve phases to compute the solution x.
   * Both @p rhs and @p result must be pre-allocated with the same number of
   * elements as the number of rows in @p spd_matrix; the solver does not
   * resize them.
   *
   * @param stream CUDA stream for asynchronous GPU operations.
   * @param spd_matrix The coefficient matrix A in CSR format.
   * @param rhs The right-hand side vector b (size must equal matrix rows).
   * @param result Output vector x (size must equal matrix rows).
   * @return true on success, false if a dimension mismatch is detected.
   */
  virtual bool Solve(cudaStream_t stream,
                     const CSRSparseMatrix& spd_matrix,
                     const dvector<float>& rhs, dvector<float>& result) = 0;

  /**
   * @brief Disables post-factorization safety checks.
   *
   * By default, dense solvers copy a device-side status flag back to the
   * host after factorization and synchronize the stream to detect singular
   * or non-positive-definite matrices.  Calling this method skips the extra
   * device-to-host memcpy, stream synchronization, and (for the LDLT solver)
   * in-kernel pivot/diagonal checks, which can be a significant fraction of
   * the total solve time for small systems.
   */
  void DisableSafetyChecks() { safety_checks_enabled_ = false; }

  /** @brief Returns whether post-factorization safety checks are enabled. */
  bool SafetyChecksEnabled() const { return safety_checks_enabled_; }

  /** @brief Virtual destructor for proper cleanup of derived solver instances. */
  virtual ~CSRSparseLinearSolver() = default;

 protected:
  bool safety_checks_enabled_ = true;
};
}  // namespace cunls
