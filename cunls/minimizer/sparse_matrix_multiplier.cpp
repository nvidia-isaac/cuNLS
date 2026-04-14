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

#include "cunls/minimizer/sparse_matrix_multiplier.h"

#include <stdexcept>

#include "cunls/minimizer/cusparse_matrix_multiplier.h"
#include "cunls/minimizer/fast_matrix_multiplier.h"

namespace cunls {

SparseMatrixMultiplierPtr
CreateSparseMatrixMultiplier(SparseMatrixMultiplierType type) {
  switch (type) {
  case SparseMatrixMultiplierType::cuSPARSE:
    return std::make_unique<cuSPARSESparseMatrixMultiplier>();
  case SparseMatrixMultiplierType::Fast:
    return std::make_unique<FastSparseMatrixMultiplier>();
  default:
    throw std::invalid_argument("Invalid sparse square multiplier type");
  }
}

} // namespace cunls
