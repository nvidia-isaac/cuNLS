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

#include "cunls/common/types.h"

namespace cunls {
namespace test_utils {

/**
 * @brief Expands a block Hessian into scalar CSR so the two storage layouts can
 *        be compared entry for entry.
 *
 * Test-only; cuNLS itself never converts between layouts. Column indices come
 * out sorted within each row.
 *
 * @param stream CUDA stream for GPU operations.
 * @param input BSR matrix.
 * @param[out] output CSR matrix; resized as needed.
 * @param[out] row_of_tile Caller-owned scratch mapping tile index to block row.
 */
void ExpandBSRToCSR(cudaStream_t stream, const BSRSparseMatrix &input, CSRSparseMatrix &output,
                    dvector<int> &row_of_tile);

}  // namespace test_utils
}  // namespace cunls
