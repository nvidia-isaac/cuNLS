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

// Select through SparseLinearSolverType::GeneralizedSchur.
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/generalized_schur_solver.h"
#include "cunls/minimizer/problem.h"

namespace cunls {

namespace internal {
// Logical state rows have identical block sparsity. Initialize checks that
// contract before the fast relative-offset gathers used by the Schur kernels.
struct SchurMatrixView {
  int num_rows;
  int tile_size;
  const int *row_offsets;
  const int *column_ids;
  const float *values;
  const double *transformed_values = nullptr;
  const size_t *transformed_offsets = nullptr;
  const int *row_state_ids = nullptr;
  const int *state_offsets = nullptr;

  __device__ int RowBegin(int row) const { return row_offsets[row / tile_size] * tile_size; }

  __device__ int RowEnd(int row) const { return row_offsets[row / tile_size + 1] * tile_size; }

  __device__ int ColumnIndex(int entry) const {
    return column_ids[entry / tile_size] * tile_size + entry % tile_size;
  }

  __device__ size_t ValueIndex(int row, int entry) const {
    return size_t(entry / tile_size) * tile_size * tile_size + (row % tile_size) * tile_size +
           entry % tile_size;
  }

  __device__ int FindColumn(int row, int column) const {
    int lower = row_offsets[row / tile_size], upper = row_offsets[row / tile_size + 1];
    while (lower < upper) {
      int middle = lower + (upper - lower) / 2;
      if (column_ids[middle] < column / tile_size) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    return lower < row_offsets[row / tile_size + 1] && column_ids[lower] == column / tile_size
               ? lower * tile_size + column % tile_size
               : -1;
  }

  __device__ double ReadEntry(int row, int entry) const {
    if (!transformed_values) {
      return values[ValueIndex(row, entry)];
    }
    int state_id = row_state_ids[row];
    return transformed_values[transformed_offsets[state_id] +
                              size_t(row - state_offsets[state_id]) *
                                  (RowEnd(row) - RowBegin(row)) +
                              entry - RowBegin(row)];
  }

  __device__ double ReadCoefficient(int row, int column) const {
    int entry = FindColumn(row, column);
    return entry < 0 ? 0. : ReadEntry(row, entry);
  }
};

}  // namespace internal

namespace {
using MatrixView = internal::SchurMatrixView;
// Internal policy; selecting the enum is the complete public configuration.
constexpr float kRelativeTolerance = 1e-3f;
constexpr float kAbsoluteTolerance = 1e-30f;
constexpr int kMaxIterations = 200;
constexpr int kMaxRefinementPasses = 3;
constexpr int kMaxExplicitDimension = 4096;
constexpr size_t kMaxCliqueContributions = 512000000;

__global__ void ValidateStructure(MatrixView matrix, int num_blocks, const int *state_offsets,
                                  int *failure_flag) {
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks) {
    return;
  }
  int block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  for (int row = block_begin; row < block_begin + block_dimension; ++row) {
    int begin = matrix.RowBegin(row), end = matrix.RowEnd(row);
    if (end - begin != matrix.RowEnd(block_begin) - matrix.RowBegin(block_begin)) {
      atomicExch(failure_flag, 1);
      return;
    }
    for (int entry = begin; entry < end; ++entry) {
      int column = matrix.ColumnIndex(entry);
      if (column < 0 || column >= matrix.num_rows ||
          (entry > begin && matrix.ColumnIndex(entry - 1) >= column) ||
          column != matrix.ColumnIndex(matrix.RowBegin(block_begin) + entry - begin)) {
        atomicExch(failure_flag, 1);
        return;
      }
    }
  }
}

__global__ void ComputeStateRowWidths(MatrixView matrix, int num_blocks, const int *state_offsets,
                                      int *state_row_widths) {
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id < num_blocks) {
    state_row_widths[block_id] =
        matrix.RowEnd(state_offsets[block_id]) - matrix.RowBegin(state_offsets[block_id]);
  }
}

__device__ unsigned HashStateId(unsigned state_id) {
  state_id ^= state_id >> 16;
  state_id *= 0x7feb352du;
  state_id ^= state_id >> 15;
  state_id *= 0x846ca68bu;
  return state_id ^ (state_id >> 16);
}

// Parallel local minima form an independent set. Stable tie-breaking avoids a
// sequential ordering on regular graphs; no graph edges leave device memory.
__global__ void SelectIndependentStates(MatrixView matrix, int num_blocks, const int *state_offsets,
                                        const int *row_state_ids, const int *state_row_widths,
                                        const int *status, int *chosen) {
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks || status[block_id]) {
    return;
  }
  long long score = (long long)state_row_widths[block_id] *
                    (state_offsets[block_id + 1] - state_offsets[block_id]);
  for (int entry = matrix.RowBegin(state_offsets[block_id]);
       entry < matrix.RowEnd(state_offsets[block_id]); ++entry) {
    int index = row_state_ids[matrix.ColumnIndex(entry)];
    if (index == block_id || status[index]) {
      continue;
    }
    long long neighbor_score =
        (long long)state_row_widths[index] * (state_offsets[index + 1] - state_offsets[index]);
    if (neighbor_score < score ||
        (neighbor_score == score &&
         (HashStateId(index) < HashStateId(block_id) ||
          (HashStateId(index) == HashStateId(block_id) && index < block_id)))) {
      return;
    }
  }
  chosen[block_id] = 1;
}

__global__ void MarkSelectedNeighbors(MatrixView matrix, int num_blocks, const int *state_offsets,
                                      const int *row_state_ids, const int *chosen, int *status) {
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks || status[block_id]) {
    return;
  }
  if (chosen[block_id]) {
    status[block_id] = 1;
    return;
  }
  for (int entry = matrix.RowBegin(state_offsets[block_id]);
       entry < matrix.RowEnd(state_offsets[block_id]); ++entry) {
    if (chosen[row_state_ids[matrix.ColumnIndex(entry)]]) {
      status[block_id] = 2;
      return;
    }
  }
}

// Stable block prefixes are built with the dimension buckets on the host.
// One coalesced pass expands them into scalar row lists; no device selection
// library, scratch allocations, or four separate compaction synchronizations.
__global__ void ExpandPartitionRows(int num_rows, const int *row_state_ids,
                                    const int *state_offsets, const int *status, const int *base,
                                    int *erows, int *rrows, int *retained_indices) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }
  int row_index = row_state_ids[row], index = base[row_index] + row - state_offsets[row_index];
  bool eliminated = status[row_index] == 1;
  if (eliminated) {
    erows[index] = row;
  } else {
    rrows[index] = row;
  }
  retained_indices[row] = eliminated ? -1 : index;
}

__global__ void ExtractDiagonalBlocks(MatrixView matrix, int num_blocks, const int *state_offsets,
                                      const size_t *diagonal_offsets, double *diagonal,
                                      int *unobservable_rows) {
  int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks) {
    return;
  }
  int block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  for (int row = 0; row < block_dimension; ++row) {
    for (int column = 0; column < block_dimension; ++column) {
      diagonal[diagonal_offsets[block_id] + row * block_dimension + column] =
          matrix.ReadCoefficient(block_begin + row, block_begin + column);
    }
  }
  // An exactly empty numerical row is unobservable (e.g. a landmark behind
  // every camera). Choose zero in that direction without damping any observed
  // equation. A nonzero RHS there is still rejected by the full residual check.
  for (int row = 0; row < block_dimension; ++row) {
    bool empty = diagonal[diagonal_offsets[block_id] + row * block_dimension + row] == 0.;
    if (empty) {
      for (int entry = matrix.RowBegin(block_begin + row); entry < matrix.RowEnd(block_begin + row);
           ++entry) {
        if (matrix.ReadEntry(block_begin + row, entry) != 0.) {
          empty = false;
          break;
        }
      }
    }
    unobservable_rows[block_begin + row] = empty;
    if (empty) {
      diagonal[diagonal_offsets[block_id] + row * block_dimension + row] = 1.;
    }
  }
}

__global__ void RegularizeUnobservableRows(int num_blocks, const int *block_ids,
                                           const int *state_offsets, const size_t *diagonal_offsets,
                                           const int *unobservable_rows, double *diagonal) {
  int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_index >= num_blocks) {
    return;
  }
  int block_id = block_ids[work_index], block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  for (int row = 0; row < block_dimension; ++row) {
    if (unobservable_rows[block_begin + row]) {
      diagonal[diagonal_offsets[block_id] + row * block_dimension + row] = 1.;
    }
  }
}

// Tiny diagonal blocks are inverted once per numeric solve, not per CG step.
// Cholesky and accumulation use double; no pivot flooring changes the system.
template <int FixedDimension>
__global__ void InvertDiagonalBlocks(int num_blocks, const int *block_ids, const int *state_offsets,
                                     const size_t *diagonal_offsets, double *diagonal,
                                     double *inverse_blocks, int *failure_flag) {
  int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_index >= num_blocks) {
    return;
  }
  int block_id = block_ids[work_index],
      block_dimension =
          FixedDimension ? FixedDimension : state_offsets[block_id + 1] - state_offsets[block_id];
  double local_factor[FixedDimension ? FixedDimension * FixedDimension : 1];
  double *cholesky_factor = FixedDimension ? local_factor : diagonal + diagonal_offsets[block_id];
  if (FixedDimension) {
    for (int index = 0; index < block_dimension * block_dimension; ++index) {
      cholesky_factor[index] = diagonal[diagonal_offsets[block_id] + index];
    }
  }
  double *block_inverse = inverse_blocks + diagonal_offsets[block_id];
  for (int row = 0; row < block_dimension; ++row) {
    for (int column = 0; column <= row; ++column) {
      double pivot_value = cholesky_factor[row * block_dimension + column];
      for (int inner = 0; inner < column; ++inner) {
        pivot_value -= cholesky_factor[row * block_dimension + inner] *
                       cholesky_factor[column * block_dimension + inner];
      }
      if (column == row) {
        if (!(pivot_value > 0.) || !isfinite(pivot_value)) {
          atomicExch(failure_flag, 1);
          return;
        }
        cholesky_factor[row * block_dimension + column] = sqrt(pivot_value);
      } else {
        cholesky_factor[row * block_dimension + column] =
            pivot_value / cholesky_factor[column * block_dimension + column];
      }
    }
  }
  for (int column = 0; column < block_dimension; ++column) {
    for (int row = 0; row < block_dimension; ++row) {
      double pivot_value = row == column ? 1. : 0.;
      for (int inner = 0; inner < row; ++inner) {
        pivot_value -= cholesky_factor[row * block_dimension + inner] *
                       block_inverse[inner * block_dimension + column];
      }
      block_inverse[row * block_dimension + column] =
          pivot_value / cholesky_factor[row * block_dimension + row];
    }
    for (int row = block_dimension - 1; row >= 0; --row) {
      double pivot_value = block_inverse[row * block_dimension + column];
      for (int inner = row + 1; inner < block_dimension; ++inner) {
        pivot_value -= cholesky_factor[inner * block_dimension + row] *
                       block_inverse[inner * block_dimension + column];
      }
      block_inverse[row * block_dimension + column] =
          pivot_value / cholesky_factor[row * block_dimension + row];
    }
  }
  for (int row = 0; row < block_dimension; ++row) {
    for (int column = 0; column < row; ++column) {
      double symmetric_value = .5 * (block_inverse[row * block_dimension + column] +
                                     block_inverse[column * block_dimension + row]);
      block_inverse[row * block_dimension + column] =
          block_inverse[column * block_dimension + row] = symmetric_value;
    }
  }
}

__global__ void TransformCrossBlocks(MatrixView matrix, int num_eliminated_blocks,
                                     const int *block_ids, const int *state_offsets,
                                     const int *retained_indices, const size_t *diagonal_offsets,
                                     const size_t *transformed_offsets,
                                     const double *inverse_blocks, double *transformed_values) {
  int work_index = blockIdx.x * 4 + threadIdx.x / 32;
  if (work_index >= num_eliminated_blocks) {
    return;
  }
  int block_id = block_ids[work_index], block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  for (int entry = matrix.RowBegin(block_begin) + threadIdx.x % 32;
       entry < matrix.RowEnd(block_begin); entry += 32) {
    int column = matrix.ColumnIndex(entry);
    if (retained_indices[column] < 0) {
      continue;
    }
    for (int row = 0; row < block_dimension; ++row) {
      double transformed_value = 0.;
      for (int inner = 0; inner < block_dimension; ++inner) {
        transformed_value +=
            (double)inverse_blocks[diagonal_offsets[block_id] + row * block_dimension + inner] *
            matrix.values[matrix.ValueIndex(
                block_begin + inner,
                matrix.RowBegin(block_begin + inner) + entry - matrix.RowBegin(block_begin))];
      }
      transformed_values[transformed_offsets[block_id] +
                         size_t(row) * (matrix.RowEnd(block_begin) - matrix.RowBegin(block_begin)) +
                         entry - matrix.RowBegin(block_begin)] = transformed_value;
    }
  }
}

__global__ void AssembleImplicitSchurDiagonal(MatrixView matrix, MatrixView transformed_matrix,
                                              const int *block_ids, const int *state_offsets,
                                              const int *retained_indices,
                                              const size_t *diagonal_offsets,
                                              double *diagonal_blocks) {
  int block_id = block_ids[blockIdx.x], block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  // One CTA per retained block; each warp owns one diagonal-block entry.
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int entry_index = warp; entry_index < block_dimension * block_dimension;
       entry_index += blockDim.x / 32) {
    int row = block_begin + entry_index / block_dimension,
        column = block_begin + entry_index % block_dimension;
    double sum = 0.;
    for (int entry = matrix.RowBegin(row) + lane; entry < matrix.RowEnd(row); entry += 32) {
      int inner = matrix.ColumnIndex(entry);
      if (retained_indices[inner] < 0) {
        sum += (double)matrix.values[matrix.ValueIndex(row, entry)] *
               transformed_matrix.ReadCoefficient(inner, column);
      }
    }
    for (int delta = 16; delta; delta /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, delta);
    }
    if (!lane) {
      diagonal_blocks[diagonal_offsets[block_id] + entry_index] =
          (double)matrix.ReadCoefficient(row, column) - sum;
    }
  }
}

// For compact retained systems, pay for the Schur clique updates once and
// reuse a dense operator. Double atomic accumulation prevents high-degree
// camera rows losing small contributions; Krylov vectors remain float32.
__global__ void CopyRetainedMatrix(MatrixView matrix, int num_retained_rows, const int *row_ids,
                                   const int *retained_indices, double *schur_matrix) {
  int row = blockIdx.x, original_row = row_ids[row];
  for (int entry = matrix.RowBegin(original_row) + threadIdx.x; entry < matrix.RowEnd(original_row);
       entry += blockDim.x) {
    int column = retained_indices[matrix.ColumnIndex(entry)];
    if (column >= 0) {
      schur_matrix[size_t(row) * num_retained_rows + column] =
          matrix.values[matrix.ValueIndex(original_row, entry)];
    }
  }
}

// The common three-dimensional pivot path stages each cross block once in
// shared memory. Adjacent lanes read adjacent columns; a warp broadcasts the
// left operand while reading the right operand without repeated sparse gathers.
// Work is restricted to the upper triangle: symmetry halves both FP64 work and
// contended atomics. The generic path has no degree or state-size bound.
__global__ void AccumulateSchurUpdates(MatrixView matrix, MatrixView transformed_matrix,
                                       int num_retained_rows, const int *block_ids,
                                       const int *state_offsets, const int *retained_indices,
                                       double *schur_matrix) {
  int block_id = block_ids[blockIdx.x], block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  int begin = matrix.RowBegin(block_begin), row_width = matrix.RowEnd(block_begin) - begin;
  __shared__ double cross[3 * 64], transformed[3 * 64];
  __shared__ int retained[64];
  if (block_dimension == 3 && row_width <= 64) {
    for (int index = threadIdx.x; index < row_width; index += blockDim.x) {
      retained[index] = retained_indices[matrix.ColumnIndex(begin + index)];
    }
    for (int index = threadIdx.x; index < 3 * row_width; index += blockDim.x) {
      int inner = index / row_width, column = index % row_width;
      bool active = retained_indices[matrix.ColumnIndex(begin + column)] >= 0;
      cross[inner * 64 + column] =
          active
              ? matrix.ReadEntry(block_begin + inner, matrix.RowBegin(block_begin + inner) + column)
              : 0.;
      transformed[inner * 64 + column] =
          active ? transformed_matrix.ReadEntry(block_begin + inner,
                                                matrix.RowBegin(block_begin + inner) + column)
                 : 0.;
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < row_width * row_width; pair += blockDim.x) {
      int row = pair / row_width, column = pair % row_width;
      int retained_row = retained[row], retained_column = retained[column];
      if (retained_row < 0 || retained_column < retained_row) {
        continue;
      }
      double sum = 0.;
#pragma unroll
      for (int inner = 0; inner < 3; ++inner) {
        sum += cross[inner * 64 + row] * transformed[inner * 64 + column];
      }
      atomicAdd(schur_matrix + size_t(retained_row) * num_retained_rows + retained_column, -sum);
    }
  } else {
    for (size_t pair = threadIdx.x; pair < size_t(row_width) * row_width; pair += blockDim.x) {
      int row = pair / row_width, column = pair % row_width;
      int retained_row = retained_indices[matrix.ColumnIndex(begin + row)],
          retained_column = retained_indices[matrix.ColumnIndex(begin + column)];
      if (retained_row < 0 || retained_column < retained_row) {
        continue;
      }
      double sum = 0.;
      for (int inner = 0; inner < block_dimension; ++inner) {
        sum += matrix.ReadEntry(block_begin + inner, matrix.RowBegin(block_begin + inner) + row) *
               transformed_matrix.ReadEntry(block_begin + inner,
                                            matrix.RowBegin(block_begin + inner) + column);
      }
      atomicAdd(schur_matrix + size_t(retained_row) * num_retained_rows + retained_column, -sum);
    }
  }
}

__global__ void MirrorUpperTriangle(int num_rows, double *matrix) {
  size_t row_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_index < size_t(num_rows) * num_rows) {
    int row = row_index / num_rows, column = row_index % num_rows;
    if (row > column) {
      matrix[row_index] = matrix[size_t(column) * num_rows + row];
    }
  }
}

// Direct terminal case for tiny retained systems; no nested solver object.
__global__ void FactorSmallSchurMatrix(int num_rows, const double *matrix, const int *row_ids,
                                       const int *unobservable_rows, double *factor,
                                       int *failure_flag) {
  __shared__ double cholesky_factor[64 * 64];
  __shared__ int factorization_failed;
  if (!threadIdx.x) {
    factorization_failed = 0;
  }
  for (int block_id = threadIdx.x; block_id < num_rows * num_rows; block_id += blockDim.x) {
    cholesky_factor[block_id] = matrix[block_id];
    if (block_id / num_rows == block_id % num_rows &&
        unobservable_rows[row_ids[block_id / num_rows]]) {
      cholesky_factor[block_id] = 1.;
    }
  }
  __syncthreads();
  for (int inner = 0; inner < num_rows; ++inner) {
    if (!threadIdx.x) {
      if (!(cholesky_factor[inner * num_rows + inner] > 0.) ||
          !isfinite(cholesky_factor[inner * num_rows + inner])) {
        factorization_failed = 1;
        atomicExch(failure_flag, 1);
      } else {
        cholesky_factor[inner * num_rows + inner] = sqrt(cholesky_factor[inner * num_rows + inner]);
      }
    }
    __syncthreads();
    if (factorization_failed) {
      return;
    }
    for (int block_id = inner + 1 + threadIdx.x; block_id < num_rows; block_id += blockDim.x) {
      cholesky_factor[block_id * num_rows + inner] /= cholesky_factor[inner * num_rows + inner];
    }
    __syncthreads();
    for (int entry_index = threadIdx.x; entry_index < num_rows * num_rows;
         entry_index += blockDim.x) {
      int block_id = entry_index / num_rows, index = entry_index % num_rows;
      if (block_id >= index && index > inner) {
        cholesky_factor[entry_index] -= cholesky_factor[block_id * num_rows + inner] *
                                        cholesky_factor[index * num_rows + inner];
      }
    }
    __syncthreads();
  }
  for (int block_id = threadIdx.x; block_id < num_rows * num_rows; block_id += blockDim.x) {
    factor[block_id] = cholesky_factor[block_id];
  }
}

__global__ void SolveSmallSchurSystem(int num_rows, const double *cholesky_factor, const float *rhs,
                                      float *solution) {
  double intermediate_solution[64];
  for (int row = 0; row < num_rows; ++row) {
    double substitution_value = rhs[row];
    for (int column = 0; column < row; ++column) {
      substitution_value -=
          cholesky_factor[row * num_rows + column] * intermediate_solution[column];
    }
    intermediate_solution[row] = substitution_value / cholesky_factor[row * num_rows + row];
  }
  for (int row = num_rows - 1; row >= 0; --row) {
    double substitution_value = intermediate_solution[row];
    for (int column = row + 1; column < num_rows; ++column) {
      substitution_value -=
          cholesky_factor[column * num_rows + row] * intermediate_solution[column];
    }
    intermediate_solution[row] = substitution_value / cholesky_factor[row * num_rows + row];
    solution[row] = intermediate_solution[row];
  }
}

__global__ void MultiplyDenseSchurMatrix(int num_rows, const double *matrix, const float *input,
                                         float *output) {
  int row = blockIdx.x;
  double sum = 0;
  for (int column = threadIdx.x; column < num_rows; column += 128) {
    sum += matrix[size_t(row) * num_rows + column] * input[column];
  }
  for (int shuffle_offset = 16; shuffle_offset; shuffle_offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, shuffle_offset);
  }
  __shared__ double warp_sums[4];
  if (threadIdx.x % 32 == 0) {
    warp_sums[threadIdx.x / 32] = sum;
  }
  __syncthreads();
  if (!threadIdx.x) {
    output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
  }
}

__global__ void ExtractExplicitSchurDiagonal(int num_blocks, int num_retained_rows,
                                             const double *schur_matrix, const int *block_ids,
                                             const int *state_offsets, const int *retained_indices,
                                             const size_t *diagonal_offsets,
                                             double *diagonal_blocks) {
  int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_index >= num_blocks) {
    return;
  }
  int block_id = block_ids[work_index], block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  for (int row = 0; row < block_dimension; ++row) {
    for (int column = 0; column < block_dimension; ++column) {
      diagonal_blocks[diagonal_offsets[block_id] + row * block_dimension + column] =
          schur_matrix[size_t(retained_indices[block_begin + row]) * num_retained_rows +
                       retained_indices[block_begin + column]];
    }
  }
}

__global__ void ApplyBlockInverse(int num_rows, const int *row_ids, const int *row_state_ids,
                                  const int *state_offsets, const size_t *diagonal_offsets,
                                  const double *inverse_blocks, const int *retained_indices,
                                  const float *input, float *output, bool input_uses_full_order) {
  int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (work_index >= num_rows) {
    return;
  }
  int row = row_ids[work_index], block_id = row_state_ids[row],
      block_begin = state_offsets[block_id],
      block_dimension = state_offsets[block_id + 1] - block_begin;
  double sum = 0.;
  for (int index = 0; index < block_dimension; ++index) {
    sum +=
        (double)inverse_blocks[diagonal_offsets[block_id] + (row - block_begin) * block_dimension +
                               index] *
        input[input_uses_full_order ? block_begin + index : retained_indices[block_begin + index]];
  }
  output[input_uses_full_order ? row : work_index] = sum;
}

// Retained vectors are compact. EliminatedRows work stays in original row order.
// CTA-per-row for long retained rows, warp-per-row for short pivot rows.
template <bool EliminatedRows, int ThreadsPerRow>
__global__ void MultiplyPartitionedMatrix(MatrixView matrix, int num_rows, const int *row_ids,
                                          const int *retained_indices, const float *input,
                                          const float *eliminated_values, float *output) {
  int work_index = ThreadsPerRow == 32 ? blockIdx.x * 4 + threadIdx.x / 32 : blockIdx.x;
  if (work_index >= num_rows) {
    return;
  }
  int row = row_ids[work_index];
  double sum = 0.;
  int lane = ThreadsPerRow == 32 ? threadIdx.x % 32 : threadIdx.x;
  for (int entry = matrix.RowBegin(row) + lane; entry < matrix.RowEnd(row);
       entry += ThreadsPerRow) {
    int column = matrix.ColumnIndex(entry), index = retained_indices[column];
    if constexpr (EliminatedRows) {
      if (index < 0) {
        continue;
      }
    }
    float input_value =
        index >= 0 ? input[index] : (EliminatedRows ? 0.f : -eliminated_values[column]);
    sum += (double)matrix.ReadEntry(row, entry) * input_value;
  }
  for (int shuffle_offset = 16; shuffle_offset; shuffle_offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, shuffle_offset);
  }
  if constexpr (ThreadsPerRow == 32) {
    if (threadIdx.x % 32 == 0) {
      output[EliminatedRows ? row : work_index] = sum;
    }
    return;
  }
  __shared__ double warp_sums[ThreadsPerRow / 32];
  if (threadIdx.x % 32 == 0) {
    warp_sums[threadIdx.x / 32] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum = 0.;
    for (int index = 0; index < ThreadsPerRow / 32; ++index) {
      sum += warp_sums[index];
    }
    output[EliminatedRows ? row : work_index] = sum;
  }
}

__global__ void AddRetainedRhs(int num_rows, const int *row_ids, const float *rhs,
                               float *reduced_rhs) {
  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index < num_rows) {
    reduced_rhs[row_index] += rhs[row_ids[row_index]];
  }
}

__global__ void RecoverFullSolution(int num_rows, const int *retained_indices,
                                    const float *retained_solution,
                                    const float *eliminated_rhs_solution,
                                    const float *eliminated_product, float *solution) {
  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index < num_rows) {
    solution[row_index] = retained_indices[row_index] >= 0
                              ? retained_solution[retained_indices[row_index]]
                              : eliminated_rhs_solution[row_index] - eliminated_product[row_index];
  }
}

__global__ void UpdateCgSolutionAndResidual(int num_rows, const float *residual_preconditioned_dot,
                                            const float *search_direction_operator_dot,
                                            const float *search_direction,
                                            const float *operator_product, float *solution,
                                            float *residual) {
  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index >= num_rows) {
    return;
  }
  float alpha = (*search_direction_operator_dot > 0.f && *residual_preconditioned_dot > 0.f)
                    ? *residual_preconditioned_dot / *search_direction_operator_dot
                    : 0.f;
  solution[row_index] += alpha * search_direction[row_index];
  residual[row_index] -= alpha * operator_product[row_index];
}

__global__ void UpdateCgDirection(int num_rows, const float *current_dot, const float *previous_dot,
                                  const float *preconditioned_residual, float *search_direction) {
  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index < num_rows) {
    search_direction[row_index] =
        preconditioned_residual[row_index] +
        (*previous_dot > 0.f ? *current_dot / *previous_dot : 0.f) * search_direction[row_index];
  }
}

__global__ void AddCorrection(int num_rows, const float *correction, float *solution) {
  int row_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_index < num_rows) {
    solution[row_index] += correction[row_index];
  }
}

template <int ThreadsPerRow>
__global__ void ComputeFullResidual(MatrixView matrix, const float *rhs, const float *solution,
                                    float *residual, const int *long_rows) {
  int row = ThreadsPerRow == 32 ? blockIdx.x * 4 + threadIdx.x / 32 : long_rows[blockIdx.x];
  int lane = threadIdx.x % 32;
  if (row >= matrix.num_rows) {
    return;
  }
  if constexpr (ThreadsPerRow == 32) {
    if (matrix.RowEnd(row) - matrix.RowBegin(row) > 128) {
      return;
    }
  }
  double sum = 0.;
  for (int entry = matrix.RowBegin(row) + (ThreadsPerRow == 32 ? lane : threadIdx.x);
       entry < matrix.RowEnd(row); entry += ThreadsPerRow) {
    sum +=
        (double)matrix.values[matrix.ValueIndex(row, entry)] * solution[matrix.ColumnIndex(entry)];
  }
  for (int shuffle_offset = 16; shuffle_offset; shuffle_offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, shuffle_offset);
  }
  if constexpr (ThreadsPerRow == 32) {
    if (!lane) {
      residual[row] = rhs[row] - sum;
    }
  } else {
    __shared__ double warp_sums[ThreadsPerRow / 32];
    if (!lane) {
      warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (!threadIdx.x) {
      sum = 0.;
      for (int index = 0; index < ThreadsPerRow / 32; ++index) {
        sum += warp_sums[index];
      }
      residual[row] = rhs[row] - sum;
    }
  }
}

}  // namespace

bool GeneralizedSchurSolver::SupportsBlockStorage() const { return true; }

bool GeneralizedSchurSolver::Initialize(cudaStream_t stream, const Problem &problem,
                                        const CSRSparseMatrix &matrix, const dvector<float> &rhs,
                                        dvector<float> &solution) {
  return Initialize(stream, problem,
                    {int(matrix.NumRows()), 1, matrix.row_offsets.data(), matrix.col_ids.data(),
                     matrix.values.data()},
                    rhs, solution);
}

bool GeneralizedSchurSolver::Initialize(cudaStream_t stream, const Problem &problem,
                                        const BSRSparseMatrix &matrix, const dvector<float> &rhs,
                                        dvector<float> &solution) {
  return Initialize(stream, problem,
                    {matrix.NumRows(), matrix.block_size, matrix.row_offsets.data(),
                     matrix.col_ids.data(), matrix.values.data()},
                    rhs, solution);
}

bool GeneralizedSchurSolver::Solve(cudaStream_t stream, const CSRSparseMatrix &matrix,
                                   const dvector<float> &rhs, dvector<float> &solution) {
  return Solve(stream,
               {int(matrix.NumRows()), 1, matrix.row_offsets.data(), matrix.col_ids.data(),
                matrix.values.data()},
               rhs, solution);
}

bool GeneralizedSchurSolver::Solve(cudaStream_t stream, const BSRSparseMatrix &matrix,
                                   const dvector<float> &rhs, dvector<float> &solution) {
  return Solve(stream,
               {matrix.NumRows(), matrix.block_size, matrix.row_offsets.data(),
                matrix.col_ids.data(), matrix.values.data()},
               rhs, solution);
}

bool GeneralizedSchurSolver::Initialize(cudaStream_t stream, const Problem &problem,
                                        MatrixView matrix, const dvector<float> &rhs,
                                        dvector<float> &solution) {
  if (matrix.num_rows != int(rhs.size()) || matrix.num_rows != int(solution.size()) ||
      matrix.num_rows <= 0) {
    return false;
  }
  num_rows_ = matrix.num_rows;
  std::vector<int> state_offsets{0}, row_state_ids;
  std::vector<size_t> diagonal_offsets{0};
  for (auto *batch : problem.GetStateBatches()) {
    int block_dimension = batch->TangentSize();
    size_t num_free_states = batch->NumActiveStates() - batch->NumConstStates();
    for (size_t block_id = 0; block_id < num_free_states; ++block_id) {
      state_offsets.push_back(state_offsets.back() + block_dimension);
      diagonal_offsets.push_back(diagonal_offsets.back() +
                                 size_t(block_dimension) * block_dimension);
    }
  }
  if (state_offsets.size() == 1) {
    for (int row = 0; row < num_rows_; row += matrix.tile_size) {
      state_offsets.push_back(row + matrix.tile_size);
      diagonal_offsets.push_back(diagonal_offsets.back() +
                                 size_t(matrix.tile_size) * matrix.tile_size);
    }
  }
  if (state_offsets.back() != num_rows_) {
    return false;
  }
  num_state_blocks_ = state_offsets.size() - 1;
  row_state_ids.resize(num_rows_);
  for (int block_id = 0; block_id < num_state_blocks_; ++block_id) {
    std::fill(row_state_ids.begin() + state_offsets[block_id],
              row_state_ids.begin() + state_offsets[block_id + 1], block_id);
  }
  state_row_offsets_ = dvector<int>(state_offsets);
  row_state_ids_ = dvector<int>(row_state_ids);
  diagonal_block_offsets_ = dvector<size_t>(diagonal_offsets);
  unobservable_rows_.resize(num_rows_);
  device_failure_flag_.resize(1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(device_failure_flag_.data(), 0, sizeof(int), stream));
  ValidateStructure<<<(num_state_blocks_ + 127) / 128, 128, 0, stream>>>(
      matrix, num_state_blocks_, state_row_offsets_.data(), device_failure_flag_.data());
  if (!CheckDeviceSuccess(stream)) {
    LogError("GeneralizedSchur: inconsistent logical block structure");
    return false;
  }
  selection_status_.resize(num_state_blocks_);
  selected_in_round_.resize(num_state_blocks_);
  state_row_widths_.resize(num_state_blocks_);
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(selection_status_.data(), 0, num_state_blocks_ * sizeof(int), stream));
  ComputeStateRowWidths<<<(num_state_blocks_ + 127) / 128, 128, 0, stream>>>(
      matrix, num_state_blocks_, state_row_offsets_.data(), state_row_widths_.data());
  for (int round = 0; round < 4; ++round) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(selected_in_round_.data(), 0, num_state_blocks_ * sizeof(int), stream));
    SelectIndependentStates<<<(num_state_blocks_ + 127) / 128, 128, 0, stream>>>(
        matrix, num_state_blocks_, state_row_offsets_.data(), row_state_ids_.data(),
        state_row_widths_.data(), selection_status_.data(), selected_in_round_.data());
    MarkSelectedNeighbors<<<(num_state_blocks_ + 127) / 128, 128, 0, stream>>>(
        matrix, num_state_blocks_, state_row_offsets_.data(), row_state_ids_.data(),
        selected_in_round_.data(), selection_status_.data());
  }
  // Selection metadata is also needed for dimension buckets and the fill
  // estimate. Transfer it once, and build all stable block prefixes in one pass.
  std::vector<int> statuses(num_state_blocks_), degrees(num_state_blocks_),
      partition_offsets(num_state_blocks_), eliminated_blocks, retained_blocks, long_rows;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(statuses.data(), selection_status_.data(),
                                      num_state_blocks_ * sizeof(int), cudaMemcpyDeviceToHost,
                                      stream));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(degrees.data(), state_row_widths_.data(),
                                      num_state_blocks_ * sizeof(int), cudaMemcpyDeviceToHost,
                                      stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  std::map<std::pair<BlockRole, int>, std::vector<int>> buckets;
  std::vector<size_t> transformed_offsets(num_state_blocks_);
  size_t transformed_size = 0;
  num_eliminated_rows_ = num_retained_rows_ = 0;
  size_t estimated_clique_pairs = 0;
  for (int block_id = 0; block_id < num_state_blocks_; ++block_id) {
    int block_dimension = state_offsets[block_id + 1] - state_offsets[block_id];
    bool is_eliminated = statuses[block_id] == 1;
    partition_offsets[block_id] = is_eliminated ? num_eliminated_rows_ : num_retained_rows_;
    (is_eliminated ? num_eliminated_rows_ : num_retained_rows_) += block_dimension;
    (is_eliminated ? eliminated_blocks : retained_blocks).push_back(block_id);
    transformed_offsets[block_id] = transformed_size;
    if (is_eliminated) {
      estimated_clique_pairs += size_t(degrees[block_id]) * degrees[block_id];
      transformed_size += size_t(block_dimension) * degrees[block_id];
    }
    if (degrees[block_id] > 128) {
      for (int row = state_offsets[block_id]; row < state_offsets[block_id + 1]; ++row) {
        long_rows.push_back(row);
      }
    }
    bool specialized = block_dimension == 1 || block_dimension == 2 || block_dimension == 3 ||
                       block_dimension == 4 || block_dimension == 6 || block_dimension == 7 ||
                       block_dimension == 9 || block_dimension == 15 || block_dimension == 16;
    buckets[{is_eliminated ? BlockRole::Eliminated : BlockRole::Retained,
             specialized ? block_dimension : 0}]
        .push_back(block_id);
  }
  num_eliminated_blocks_ = eliminated_blocks.size();
  num_retained_blocks_ = retained_blocks.size();
  eliminated_block_ids_ = dvector<int>(eliminated_blocks);
  retained_block_ids_ = dvector<int>(retained_blocks);
  long_residual_row_ids_ = dvector<int>(long_rows);
  partition_row_offsets_ = dvector<int>(partition_offsets);
  eliminated_row_ids_.resize(num_eliminated_rows_);
  retained_row_ids_.resize(num_retained_rows_);
  retained_row_indices_.resize(num_rows_);
  ExpandPartitionRows<<<(num_rows_ + 255) / 256, 256, 0, stream>>>(
      num_rows_, row_state_ids_.data(), state_row_offsets_.data(), selection_status_.data(),
      partition_row_offsets_.data(), eliminated_row_ids_.data(), retained_row_ids_.data(),
      retained_row_indices_.data());
  dimension_groups_.clear();
  for (auto &bucket : buckets) {
    dimension_groups_.push_back(
        {bucket.first.first, bucket.first.second, dvector<int>(bucket.second)});
  }
  diagonal_blocks_.resize(diagonal_offsets.back());
  inverse_blocks_.resize(diagonal_offsets.back());
  transformed_offsets_ = dvector<size_t>(transformed_offsets);
  transformed_cross_blocks_.resize(transformed_size);
  device_failure_flag_.resize(1);
  eliminated_rhs_solution_.resize(num_rows_);
  eliminated_product_.resize(num_rows_);
  full_residual_.resize(num_rows_);
  correction_.resize(num_rows_);
  retained_solution_.resize(num_retained_rows_);
  reduced_rhs_.resize(num_retained_rows_);
  reduced_residual_.resize(num_retained_rows_);
  preconditioned_residual_.resize(num_retained_rows_);
  search_direction_.resize(num_retained_rows_);
  operator_product_.resize(num_retained_rows_);
  cg_dot_products_.resize(4);
  size_t free_bytes, total_bytes;
  THROW_ON_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
  use_explicit_schur_ = num_retained_rows_ > 0 && num_retained_rows_ <= kMaxExplicitDimension &&
                        estimated_clique_pairs <= kMaxCliqueContributions &&
                        size_t(num_retained_rows_) * num_retained_rows_ * 12 < free_bytes / 4;
  if (use_explicit_schur_) {
    schur_matrix_.resize(size_t(num_retained_rows_) * num_retained_rows_);
  }
  use_direct_solve_ = use_explicit_schur_ && num_retained_rows_ <= 64;
  if (use_direct_solve_) {
    terminal_factor_.resize(size_t(num_retained_rows_) * num_retained_rows_);
  }
  LogMessage(
      "GeneralizedSchur: {} eliminated / {} retained dimensions, {} / {} state blocks, "
      "explicit={}",
      num_eliminated_rows_, num_retained_rows_, num_eliminated_blocks_, num_retained_blocks_,
      use_explicit_schur_);
  return true;
}

void GeneralizedSchurSolver::InvertDimensionGroups(cudaStream_t stream, BlockRole role) {
  for (auto &group : dimension_groups_) {
    if (group.role == role) {
      int num_group_blocks = group.block_ids.size();
#define LAUNCH_BLOCK_INVERSE(D)                                                        \
  InvertDiagonalBlocks<D><<<(num_group_blocks + 127) / 128, 128, 0, stream>>>(         \
      num_group_blocks, group.block_ids.data(), state_row_offsets_.data(),             \
      diagonal_block_offsets_.data(), diagonal_blocks_.data(), inverse_blocks_.data(), \
      device_failure_flag_.data())
      switch (group.dimension) {
        case 1:
          LAUNCH_BLOCK_INVERSE(1);
          break;
        case 2:
          LAUNCH_BLOCK_INVERSE(2);
          break;
        case 3:
          LAUNCH_BLOCK_INVERSE(3);
          break;
        case 4:
          LAUNCH_BLOCK_INVERSE(4);
          break;
        case 6:
          LAUNCH_BLOCK_INVERSE(6);
          break;
        case 7:
          LAUNCH_BLOCK_INVERSE(7);
          break;
        case 9:
          LAUNCH_BLOCK_INVERSE(9);
          break;
        case 15:
          LAUNCH_BLOCK_INVERSE(15);
          break;
        case 16:
          LAUNCH_BLOCK_INVERSE(16);
          break;
        default:
          LAUNCH_BLOCK_INVERSE(0);
      }
#undef LAUNCH_BLOCK_INVERSE
    }
  }
}

bool GeneralizedSchurSolver::CheckDeviceSuccess(cudaStream_t stream) {
  int failure_flag = 0;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&failure_flag, device_failure_flag_.data(), sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return failure_flag == 0;
}

GeneralizedSchurSolver::MatrixView GeneralizedSchurSolver::MakeTransformedView(
    MatrixView matrix) const {
  matrix.transformed_values = transformed_cross_blocks_.data();
  matrix.transformed_offsets = transformed_offsets_.data();
  matrix.row_state_ids = row_state_ids_.data();
  matrix.state_offsets = state_row_offsets_.data();
  return matrix;
}

void GeneralizedSchurSolver::ApplySchurOperator(cudaStream_t stream, MatrixView matrix,
                                                const float *input, float *output) {
  if (use_explicit_schur_) {
    MultiplyDenseSchurMatrix<<<num_retained_rows_, 128, 0, stream>>>(
        num_retained_rows_, schur_matrix_.data(), input, output);
    return;
  }
  MatrixView transformed_matrix = MakeTransformedView(matrix);
  if (num_eliminated_rows_) {
    MultiplyPartitionedMatrix<true, 32><<<(num_eliminated_rows_ + 3) / 4, 128, 0, stream>>>(
        transformed_matrix, num_eliminated_rows_, eliminated_row_ids_.data(),
        retained_row_indices_.data(), input, nullptr, eliminated_product_.data());
  }
  if (num_retained_rows_) {
    MultiplyPartitionedMatrix<false, 128><<<num_retained_rows_, 128, 0, stream>>>(
        matrix, num_retained_rows_, retained_row_ids_.data(), retained_row_indices_.data(), input,
        eliminated_product_.data(), output);
  }
}

void GeneralizedSchurSolver::ApplyPreconditioner(cudaStream_t stream) {
  ApplyBlockInverse<<<(num_retained_rows_ + 127) / 128, 128, 0, stream>>>(
      num_retained_rows_, retained_row_ids_.data(), row_state_ids_.data(),
      state_row_offsets_.data(), diagonal_block_offsets_.data(), inverse_blocks_.data(),
      retained_row_indices_.data(), reduced_residual_.data(), preconditioned_residual_.data(),
      false);
}

bool GeneralizedSchurSolver::Solve(cudaStream_t stream, MatrixView matrix,
                                   const dvector<float> &rhs, dvector<float> &solution) {
  if (matrix.num_rows != num_rows_ || int(rhs.size()) != num_rows_ ||
      int(solution.size()) != num_rows_) {
    return false;
  }
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  float rhs_norm = 0;
  THROW_ON_CUBLAS_ERROR(cublasSnrm2(handle, num_rows_, rhs.data(), 1, &rhs_norm));
  if (!std::isfinite(rhs_norm)) {
    return false;
  }
  if (rhs_norm == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(solution.data(), 0, num_rows_ * sizeof(float), stream));
    return true;
  }
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(device_failure_flag_.data(), 0, sizeof(int), stream));
  ExtractDiagonalBlocks<<<(num_state_blocks_ + 127) / 128, 128, 0, stream>>>(
      matrix, num_state_blocks_, state_row_offsets_.data(), diagonal_block_offsets_.data(),
      diagonal_blocks_.data(), unobservable_rows_.data());
  InvertDimensionGroups(stream, BlockRole::Eliminated);
  if (!CheckDeviceSuccess(stream)) {
    LogError("GeneralizedSchur: non-positive eliminated pivot");
    return false;
  }
  if (num_eliminated_blocks_) {
    TransformCrossBlocks<<<(num_eliminated_blocks_ + 3) / 4, 128, 0, stream>>>(
        matrix, num_eliminated_blocks_, eliminated_block_ids_.data(), state_row_offsets_.data(),
        retained_row_indices_.data(), diagonal_block_offsets_.data(), transformed_offsets_.data(),
        inverse_blocks_.data(), transformed_cross_blocks_.data());
  }
  MatrixView transformed_matrix = MakeTransformedView(matrix);
  if (use_explicit_schur_) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(schur_matrix_.data(), 0,
                        size_t(num_retained_rows_) * num_retained_rows_ * sizeof(double), stream));
    CopyRetainedMatrix<<<num_retained_rows_, 128, 0, stream>>>(
        matrix, num_retained_rows_, retained_row_ids_.data(), retained_row_indices_.data(),
        schur_matrix_.data());
    if (num_eliminated_blocks_) {
      AccumulateSchurUpdates<<<num_eliminated_blocks_, 128, 0, stream>>>(
          matrix, transformed_matrix, num_retained_rows_, eliminated_block_ids_.data(),
          state_row_offsets_.data(), retained_row_indices_.data(), schur_matrix_.data());
    }
    MirrorUpperTriangle<<<(size_t(num_retained_rows_) * num_retained_rows_ + 255) / 256, 256, 0,
                          stream>>>(num_retained_rows_, schur_matrix_.data());
    ExtractExplicitSchurDiagonal<<<(num_retained_blocks_ + 127) / 128, 128, 0, stream>>>(
        num_retained_blocks_, num_retained_rows_, schur_matrix_.data(), retained_block_ids_.data(),
        state_row_offsets_.data(), retained_row_indices_.data(), diagonal_block_offsets_.data(),
        diagonal_blocks_.data());
  } else if (num_retained_blocks_) {
    AssembleImplicitSchurDiagonal<<<num_retained_blocks_, 128, 0, stream>>>(
        matrix, transformed_matrix, retained_block_ids_.data(), state_row_offsets_.data(),
        retained_row_indices_.data(), diagonal_block_offsets_.data(), diagonal_blocks_.data());
  }
  if (num_retained_blocks_) {
    RegularizeUnobservableRows<<<(num_retained_blocks_ + 127) / 128, 128, 0, stream>>>(
        num_retained_blocks_, retained_block_ids_.data(), state_row_offsets_.data(),
        diagonal_block_offsets_.data(), unobservable_rows_.data(), diagonal_blocks_.data());
  }
  if (use_direct_solve_) {
    FactorSmallSchurMatrix<<<1, 128, 0, stream>>>(
        num_retained_rows_, schur_matrix_.data(), retained_row_ids_.data(),
        unobservable_rows_.data(), terminal_factor_.data(), device_failure_flag_.data());
  } else {
    InvertDimensionGroups(stream, BlockRole::Retained);
  }
  if (!CheckDeviceSuccess(stream)) {
    LogError("GeneralizedSchur: non-positive reduced preconditioner");
    return false;
  }
  SolveWithCurrentFactors(stream, matrix, rhs.data(), solution.data(), rhs_norm);
  for (int refinement = 0; refinement <= kMaxRefinementPasses; ++refinement) {
    ComputeFullResidual<32><<<(num_rows_ + 3) / 4, 128, 0, stream>>>(
        matrix, rhs.data(), solution.data(), full_residual_.data(), nullptr);
    if (!long_residual_row_ids_.empty()) {
      ComputeFullResidual<128><<<long_residual_row_ids_.size(), 128, 0, stream>>>(
          matrix, rhs.data(), solution.data(), full_residual_.data(),
          long_residual_row_ids_.data());
    }
    float residual_norm;
    THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    THROW_ON_CUBLAS_ERROR(cublasSnrm2(handle, num_rows_, full_residual_.data(), 1, &residual_norm));
    if (residual_norm <= std::max(kAbsoluteTolerance, kRelativeTolerance * rhs_norm)) {
      return true;
    }
    if (!std::isfinite(residual_norm)) {
      return false;
    }
    if (refinement == kMaxRefinementPasses) {
      LogError("GeneralizedSchur: residual {} exceeds tolerance (rhs {})", residual_norm, rhs_norm);
      return false;
    }
    SolveWithCurrentFactors(stream, matrix, full_residual_.data(), correction_.data(),
                            residual_norm);
    AddCorrection<<<(num_rows_ + 127) / 128, 128, 0, stream>>>(num_rows_, correction_.data(),
                                                               solution.data());
  }
  return false;
}

void GeneralizedSchurSolver::SolveWithCurrentFactors(cudaStream_t stream, MatrixView matrix,
                                                     const float *rhs, float *solution,
                                                     float rhs_norm) {
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  MatrixView transformed_matrix = MakeTransformedView(matrix);
  if (num_eliminated_rows_) {
    ApplyBlockInverse<<<(num_eliminated_rows_ + 127) / 128, 128, 0, stream>>>(
        num_eliminated_rows_, eliminated_row_ids_.data(), row_state_ids_.data(),
        state_row_offsets_.data(), diagonal_block_offsets_.data(), inverse_blocks_.data(),
        retained_row_indices_.data(), rhs, eliminated_rhs_solution_.data(), true);
  }
  if (num_retained_rows_) {
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(retained_solution_.data(), 0, num_retained_rows_ * sizeof(float), stream));
  }
  if (num_retained_rows_) {
    MultiplyPartitionedMatrix<false, 128><<<num_retained_rows_, 128, 0, stream>>>(
        matrix, num_retained_rows_, retained_row_ids_.data(), retained_row_indices_.data(),
        retained_solution_.data(), eliminated_rhs_solution_.data(), reduced_rhs_.data());
    AddRetainedRhs<<<(num_retained_rows_ + 127) / 128, 128, 0, stream>>>(
        num_retained_rows_, retained_row_ids_.data(), rhs, reduced_rhs_.data());
    if (use_direct_solve_) {
      SolveSmallSchurSystem<<<1, 1, 0, stream>>>(num_retained_rows_, terminal_factor_.data(),
                                                 reduced_rhs_.data(), retained_solution_.data());
    } else {
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(reduced_residual_.data(), reduced_rhs_.data(),
                                          num_retained_rows_ * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream));
      ApplyPreconditioner(stream);
      THROW_ON_CUDA_ERROR(cudaMemcpyAsync(search_direction_.data(), preconditioned_residual_.data(),
                                          num_retained_rows_ * sizeof(float),
                                          cudaMemcpyDeviceToDevice, stream));
      THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
      THROW_ON_CUBLAS_ERROR(cublasSdot(handle, num_retained_rows_, reduced_residual_.data(), 1,
                                       preconditioned_residual_.data(), 1,
                                       cg_dot_products_.data()));
      for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
        ApplySchurOperator(stream, matrix, search_direction_.data(), operator_product_.data());
        THROW_ON_CUBLAS_ERROR(cublasSdot(handle, num_retained_rows_, search_direction_.data(), 1,
                                         operator_product_.data(), 1, cg_dot_products_.data() + 1));
        UpdateCgSolutionAndResidual<<<(num_retained_rows_ + 127) / 128, 128, 0, stream>>>(
            num_retained_rows_, cg_dot_products_.data(), cg_dot_products_.data() + 1,
            search_direction_.data(), operator_product_.data(), retained_solution_.data(),
            reduced_residual_.data());
        ApplyPreconditioner(stream);
        THROW_ON_CUBLAS_ERROR(cublasSdot(handle, num_retained_rows_, reduced_residual_.data(), 1,
                                         preconditioned_residual_.data(), 1,
                                         cg_dot_products_.data() + 2));
        UpdateCgDirection<<<(num_retained_rows_ + 127) / 128, 128, 0, stream>>>(
            num_retained_rows_, cg_dot_products_.data() + 2, cg_dot_products_.data(),
            preconditioned_residual_.data(), search_direction_.data());
        THROW_ON_CUDA_ERROR(cudaMemcpyAsync(cg_dot_products_.data(), cg_dot_products_.data() + 2,
                                            sizeof(float), cudaMemcpyDeviceToDevice, stream));
        if ((iteration + 1) % 4 == 0) {
          float residual_norm;
          THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
          THROW_ON_CUBLAS_ERROR(
              cublasSnrm2(handle, num_retained_rows_, reduced_residual_.data(), 1, &residual_norm));
          THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
          if (!std::isfinite(residual_norm)) {
            break;
          }
          if (residual_norm <= std::max(kAbsoluteTolerance, .2f * kRelativeTolerance * rhs_norm)) {
            break;
          }
        }
      }
    }
  }
  if (num_eliminated_rows_) {
    MultiplyPartitionedMatrix<true, 32><<<(num_eliminated_rows_ + 3) / 4, 128, 0, stream>>>(
        transformed_matrix, num_eliminated_rows_, eliminated_row_ids_.data(),
        retained_row_indices_.data(), retained_solution_.data(), nullptr,
        eliminated_product_.data());
  }
  RecoverFullSolution<<<(num_rows_ + 127) / 128, 128, 0, stream>>>(
      num_rows_, retained_row_indices_.data(), retained_solution_.data(),
      eliminated_rhs_solution_.data(), eliminated_product_.data(), solution);
}
}  // namespace cunls
