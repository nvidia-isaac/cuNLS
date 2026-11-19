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

#include <cstdint>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/sparse_linear_solver_base.h"

namespace cunls {

namespace {

constexpr int kThreads = 256;

/** storage[slot(e)] = values[e] for every value e (slot: the identity, or the map; -1 skipped). */
template <typename Slot>
__global__ void copy_into_storage_kernel(size_t n, const float *__restrict__ values,
                                         const Slot *__restrict__ entry_slot,
                                         float *__restrict__ storage) {
  const size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (e >= n) return;
  if (entry_slot == nullptr) {
    storage[e] = values[e];
    return;
  }
  const Slot slot = entry_slot[e];
  if (slot >= 0) storage[slot] = values[e];
}

/** Copies `n` pattern values into the solver's storage, then solves from it. */
bool SolveFromValues(SparseLinearSolver &solver, cudaStream_t stream, size_t num_rows,
                     const float *values, size_t n, const dvector<float> &rhs,
                     dvector<float> &result) {
  if (rhs.size() != num_rows || result.size() != num_rows) {
    LogError(
        "SparseLinearSolver::Solve: rhs / result size does not match the matrix (got {} / "
        "{}, expected {})",
        rhs.size(), result.size(), num_rows);
    return false;
  }
  if (num_rows == 0) return true;
  const SolverMatrixStorage storage = solver.MatrixStorage();
  if (storage.values == nullptr) {
    LogError(
        "SparseLinearSolver::Solve: the solver has no matrix storage. Fix: Initialize() it "
        "with this matrix's pattern first.");
    return false;
  }
  solver.ClearMatrixStorage(stream);
  if (n > 0) {
    const unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    if (storage.entry_slot64 != nullptr) {
      copy_into_storage_kernel<<<blocks, kThreads, 0, stream>>>(n, values, storage.entry_slot64,
                                                                storage.values);
    } else {
      copy_into_storage_kernel<<<blocks, kThreads, 0, stream>>>(n, values, storage.entry_slot,
                                                                storage.values);
    }
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  return solver.SolveStoredMatrix(stream, rhs, result);
}

}  // namespace

bool SparseLinearSolver::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                               const dvector<float> &rhs, dvector<float> &result) {
  return SolveFromValues(*this, stream, spd_matrix.NumRows(), spd_matrix.values.data(),
                         spd_matrix.values.size(), rhs, result);
}

bool SparseLinearSolver::Solve(cudaStream_t stream, const BSRSparseMatrix &spd_matrix,
                               const dvector<float> &rhs, dvector<float> &result) {
  return SolveFromValues(*this, stream, static_cast<size_t>(spd_matrix.NumRows()),
                         spd_matrix.values.data(), spd_matrix.values.size(), rhs, result);
}

}  // namespace cunls
