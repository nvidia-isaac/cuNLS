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

#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/common/cudss_helper.h"

namespace cunls {

/** @copydoc CreateCSRSparseLinearSolver
 * @throws std::invalid_argument If an unsupported solver type is specified.
 */
SparseLinearSolverPtr
CreateCSRSparseLinearSolver(SparseLinearSolverType type,
                            const SparseLinearSolverConfig &config) {
  switch (type) {
  case SparseLinearSolverType::cuDSS:
    return std::make_unique<cuDSSLinearSolver>(config.cudss_solver_options);
  case SparseLinearSolverType::DenseLDLT:
    return std::make_unique<DenseLDLTSolver>();
  case SparseLinearSolverType::DenseCholesky:
    return std::make_unique<DenseCholeskySolver>();
  case SparseLinearSolverType::DenseQR:
    return std::make_unique<DenseQRSolver>();
  case SparseLinearSolverType::BlockSparsePCG:
    return std::make_unique<BlockSparsePCGSolver>(
        config.block_sparse_pcg_options);
  default:
    throw std::invalid_argument("Invalid sparse linear solver type");
  }
}

} // namespace cunls