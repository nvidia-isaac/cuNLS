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
#include "cunls/minimizer/problem.h"

namespace cunls {

/**
 * @brief Builds the triplet (COO) sparse structure for the Jacobian matrix.
 *
 * Constructs row and column index arrays that define the sparsity pattern of
 * the Jacobian for the given optimization problem. Column indices corresponding
 * to constant (fixed) states are set to -1 to exclude them from the
 * reduced linear system.
 *
 * @param stream CUDA stream for asynchronous GPU operations.
 * @param problem The optimization problem defining factors and state batches.
 * @param[out] structure Output triplet sparse structure with row_ids and col_ids.
 */
void BuildTripletSparseStructure(cudaStream_t stream, const Problem& problem,
                                 TripletSparseStructure& structure);

/**
 * @brief Builds the triplet (COO) sparse structure for the Jacobian on the CPU.
 *
 * CPU-side implementation of BuildTripletSparseStructure. Constructs the same
 * sparsity pattern but performs all computation on the host. Column indices for
 * constant states are set to -1.
 *
 * @param problem The optimization problem defining factors and state batches.
 * @param[out] structure Output triplet sparse structure with row_ids and col_ids
 *                       (device vectors populated via host-to-device copy).
 */
void BuildTripletSparseStructureCPU(const Problem& problem,
                                    TripletSparseStructure& structure);
}  // namespace cunls
