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

#include "cunls/linear_solver/csr_sparse_linear_solver.h"
#include "cunls/linear_solver/cudss_sparse_linear_solver.h"

namespace cunls {

/**
 * @brief Type of sparse linear solver to use.
 */
enum class SparseLinearSolverType {
  cuDSS,  ///< cuDSS-based solver using NVIDIA's cuDSS library.
};

/**
 * @brief Configuration struct for sparse linear solvers.
 *
 * Contains solver-specific configuration options. The active member depends
 * on the solver type specified in SparseLinearSolverType.
 */
struct SparseLinearSolverConfig {
  cuDSSLinearSolverOptions cudss_solver_options;
};

/**
 * @brief Smart pointer type for sparse linear solvers.
 */
using SparseLinearSolverPtr = std::unique_ptr<CSRSparseLinearSolver>;

/**
 * @brief Factory function to create a sparse linear solver.
 *
 * Creates and returns a solver instance based on the specified type and
 * configuration.
 *
 * @param type The type of solver to create.
 * @param config Solver-specific configuration options.
 * @return A unique pointer to the created solver instance.
 */
SparseLinearSolverPtr CreateCSRSparseLinearSolver(
    SparseLinearSolverType type, const SparseLinearSolverConfig& config);

}  // namespace cunls
