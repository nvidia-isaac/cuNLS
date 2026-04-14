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

#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/sparse_matrix_multiplier.h"

namespace cunls {

/**
 * @brief Computes J^T * J using custom warp-efficient CUDA kernels.
 *
 * Uses the Problem's factor graph connectivity to derive the output sparsity
 * pattern during initialization, then warp-cooperative scatter-multiply
 * during compute. Requires a Problem with at least one residual batch.
 */
class FastSparseMatrixMultiplier : public SparseMatrixMultiplier {
public:
  void Initialize(cudaStream_t stream, const Problem &problem,
                  const CSRSparseMatrix &input,
                  CSRSparseMatrix &output) override;

  void ComputeSquaredMatrix(cudaStream_t stream, const Problem &problem,
                            const CSRSparseMatrix &input,
                            CSRSparseMatrix &output) override;

private:
  void ComputeOutputStructure(cudaStream_t stream, const Problem &problem,
                              CSRSparseMatrix &output, int num_cols);

  int max_nnz_per_row_ = 0;
  dvector<int> position_map_; ///< Precomputed output positions, indexed as
                              ///< [input_nnz_idx * max_nnz_per_row + b].
  dvector<int> buffer_;       ///< Reusable scratch buffer for initialization.
  pvector<int> pinned_buf_;   ///< Reusable pinned buffer for D2H readbacks.
  profiler::Domain profiler_domain_{
      "FastSparseMatrixMultiplier"}; ///< Profiling domain.
};

} // namespace cunls
