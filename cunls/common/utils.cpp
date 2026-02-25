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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "cunls/common/utils.h"

namespace cunls {

/** @copydoc DumpCSRSparseMatrixToFile */
void DumpCSRSparseMatrixToFile(const std::string& filename,
                               const CSRSparseMatrix& matrix) {
  // Copy device vectors to host vectors
  hvector<int> host_row_offsets(matrix.row_offsets.size());
  hvector<int> host_col_ids(matrix.col_ids.size());
  hvector<float> host_values(matrix.values.size());

  matrix.row_offsets.CopyToHost(host_row_offsets.data(), host_row_offsets.size());
  matrix.col_ids.CopyToHost(host_col_ids.data(), host_col_ids.size());
  matrix.values.CopyToHost(host_values.data(), host_values.size());

  // Compute matrix dimensions
  size_t num_rows = matrix.NumRows();
  size_t num_nonzeros = matrix.NumNonZeros();
  size_t num_cols = 0;
  if (num_nonzeros > 0) {
    num_cols = *std::max_element(host_col_ids.begin(),
                                 host_col_ids.begin() + num_nonzeros) +
               1;
  }

  // Open file in append binary mode
  std::ofstream file(filename, std::ios::app | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Write binary data according to the format
  uint32_t num_rows_u32 = static_cast<uint32_t>(num_rows);
  uint32_t num_cols_u32 = static_cast<uint32_t>(num_cols);
  uint32_t num_nonzeros_u32 = static_cast<uint32_t>(num_nonzeros);

  file.write(reinterpret_cast<const char*>(&num_rows_u32), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&num_cols_u32), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&num_nonzeros_u32),
             sizeof(uint32_t));

  // Convert and write row_offsets (int -> uint32_t)
  for (size_t i = 0; i < host_row_offsets.size(); ++i) {
    uint32_t offset = static_cast<uint32_t>(host_row_offsets[i]);
    file.write(reinterpret_cast<const char*>(&offset), sizeof(uint32_t));
  }

  // Convert and write col_ids (int -> uint32_t)
  for (size_t i = 0; i < num_nonzeros; ++i) {
    uint32_t col_id = static_cast<uint32_t>(host_col_ids[i]);
    file.write(reinterpret_cast<const char*>(&col_id), sizeof(uint32_t));
  }

  // Write values (float)
  file.write(reinterpret_cast<const char*>(host_values.data()),
             num_nonzeros * sizeof(float));

  file.close();
}

/** @copydoc DumpVectorToFile */
void DumpVectorToFile(const std::string& filename,
                      const dvector<float>& vector) {
  // Copy device vector to host vector
  hvector<float> host_vector(vector.size());
  vector.CopyToHost(host_vector.data(), host_vector.size());

  // Open file in append binary mode
  std::ofstream file(filename, std::ios::app | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Write binary data according to the format
  uint32_t size = static_cast<uint32_t>(vector.size());
  file.write(reinterpret_cast<const char*>(&size), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(host_vector.data()),
             size * sizeof(float));

  file.close();
}

}  // namespace cunls
