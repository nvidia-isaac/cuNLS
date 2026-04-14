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

#include <string>

#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Dumps a CSR sparse matrix to a binary file for debugging/analysis.
 *
 * Appends the matrix data to the end of the specified file in a compact
 * binary format:
 *   - num_rows     (uint32_t)
 *   - num_cols     (uint32_t)
 *   - num_nonzeros (uint32_t)
 *   - row_offsets  (uint32_t[num_rows + 1])
 *   - col_ids      (uint32_t[num_nonzeros])
 *   - values       (float[num_nonzeros])
 *
 * The matrix data is first copied from device to host before writing.
 *
 * @param filename Path to the output binary file (created or appended to).
 * @param matrix   The CSR sparse matrix to dump.
 */
void DumpCSRSparseMatrixToFile(const std::string &filename,
                               const CSRSparseMatrix &matrix);

/**
 * @brief Dumps a device vector to a binary file for debugging/analysis.
 *
 * Appends the vector data to the end of the specified file in a compact
 * binary format:
 *   - size   (uint32_t)
 *   - values (float[size])
 *
 * The vector data is first copied from device to host before writing.
 *
 * @param filename Path to the output binary file (created or appended to).
 * @param vector   The device vector to dump.
 */
void DumpVectorToFile(const std::string &filename,
                      const dvector<float> &vector);

} // namespace cunls
