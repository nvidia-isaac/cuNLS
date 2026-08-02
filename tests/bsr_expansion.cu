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

/**
 * @file bsr_expansion.cu
 * @brief Expands a block Hessian into scalar CSR, for tests only.
 *
 * cuNLS never needs this: a solver that cannot read tiles reports
 * SupportsBlockStorage() == false and is handed scalar CSR directly, so no
 * conversion happens on any path. It lives here rather than in the library
 * because its only purpose is to make the two storage layouts directly
 * comparable in the equivalence tests.
 */

#include "cunls/common/helper.h"
#include "tests/bsr_expansion.h"

namespace cunls {
namespace test_utils {
namespace {

constexpr int kBlockSize = 256;

int GridFor(size_t count) { return static_cast<int>((count + kBlockSize - 1) / kBlockSize); }

/** @brief Records, for each stored tile, the block row it belongs to. */
__global__ void FillRowOfTileKernel(int num_block_rows, const int *__restrict__ row_offsets,
                                    int *__restrict__ row_of_tile) {
  int block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= num_block_rows) {
    return;
  }
  for (int tile = row_offsets[block_row]; tile < row_offsets[block_row + 1]; tile++) {
    row_of_tile[tile] = block_row;
  }
}

/**
 * @brief Fills the CSR row offsets of the expanded matrix.
 *
 * Every tile in a block row contributes `block_size` columns to each of that
 * block row's `block_size` scalar rows, so a row's length follows from the
 * block row's tile count alone.
 */
__global__ void FillExpandedRowOffsetsKernel(int num_rows, int block_size,
                                             const int *__restrict__ block_row_offsets,
                                             int *__restrict__ row_offsets) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > num_rows) {
    return;
  }
  const int block_row = row / block_size;
  const int sub_row = row - block_row * block_size;
  const int whole = block_row_offsets[block_row] * block_size;
  const int partial = sub_row * (block_row_offsets[block_row + 1] - block_row_offsets[block_row]);
  row_offsets[row] = (whole + partial) * block_size;
}

/** @brief Scatters each tile entry to its scalar CSR position. */
__global__ void ExpandTilesToCSRKernel(size_t num_values, int block_size,
                                       const int *__restrict__ row_of_tile,
                                       const int *__restrict__ block_row_offsets,
                                       const int *__restrict__ block_col_ids,
                                       const float *__restrict__ values,
                                       const int *__restrict__ row_offsets,
                                       int *__restrict__ col_ids, float *__restrict__ out_values) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_values) {
    return;
  }
  const int tile_area = block_size * block_size;
  const size_t tile = idx / tile_area;
  const int within = static_cast<int>(idx - tile * tile_area);
  const int row_in_tile = within / block_size;
  const int col_in_tile = within - row_in_tile * block_size;

  const int block_row = row_of_tile[tile];
  const int tile_in_row = static_cast<int>(tile) - block_row_offsets[block_row];
  const int row = block_row * block_size + row_in_tile;
  const int slot = row_offsets[row] + tile_in_row * block_size + col_in_tile;
  col_ids[slot] = block_col_ids[tile] * block_size + col_in_tile;
  out_values[slot] = values[idx];
}

}  // namespace

void ExpandBSRToCSR(cudaStream_t stream, const BSRSparseMatrix &input, CSRSparseMatrix &output,
                    dvector<int> &row_of_tile) {
  const int num_rows = input.NumRows();
  output.row_offsets.resize(static_cast<size_t>(num_rows) + 1);
  output.col_ids.resize(input.NumNonZeros());
  output.values.resize(input.NumNonZeros());
  if (num_rows == 0) {
    THROW_ON_CUDA_ERROR(cudaMemsetAsync(output.row_offsets.data(), 0, sizeof(int), stream));
    return;
  }

  row_of_tile.resize(input.NumBlocks());
  FillRowOfTileKernel<<<GridFor(input.num_block_rows), kBlockSize, 0, stream>>>(
      input.num_block_rows, input.row_offsets.data(), row_of_tile.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  FillExpandedRowOffsetsKernel<<<GridFor(static_cast<size_t>(num_rows) + 1), kBlockSize, 0,
                                 stream>>>(num_rows, input.block_size, input.row_offsets.data(),
                                           output.row_offsets.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  ExpandTilesToCSRKernel<<<GridFor(input.values.size()), kBlockSize, 0, stream>>>(
      input.values.size(), input.block_size, row_of_tile.data(), input.row_offsets.data(),
      input.col_ids.data(), input.values.data(), output.row_offsets.data(), output.col_ids.data(),
      output.values.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace test_utils
}  // namespace cunls
