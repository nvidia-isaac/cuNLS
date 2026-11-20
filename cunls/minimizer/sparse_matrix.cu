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

#include <cusparse.h>

#include <cassert>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {
namespace {

constexpr int kThreads = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

__global__ void inv_sqrt_floor_kernel(float *__restrict__ v, int n, float floor_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = v[i];
  v[i] = rsqrtf(x < floor_value ? floor_value : x);  // a NaN diagonal stays NaN
}

__global__ void elementwise_multiply_kernel(float *__restrict__ a, const float *__restrict__ b,
                                            int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= b[i];
}

/** Index of the diagonal entry of `row` in the sorted col_ids of the row, or -1. */
__global__ void find_diagonal_kernel(const int *__restrict__ row_offsets,
                                     const int *__restrict__ col_ids, int num_rows,
                                     int *__restrict__ positions) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  int lo = row_offsets[row], hi = row_offsets[row + 1] - 1;
  int found = -1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const int c = col_ids[mid];
    if (c == row) {
      found = mid;
      break;
    }
    if (c < row) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  positions[row] = found;
}

__global__ void extract_diagonal_at_kernel(const float *__restrict__ values,
                                           const int *__restrict__ positions, int n,
                                           float *__restrict__ diagonal) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int e = positions[i];
  diagonal[i] = e >= 0 ? values[e] : 0.f;
}

}  // namespace

void InvertSqrtWithFloorInPlace(cudaStream_t stream, dvector<float> &v, float floor_value) {
  const int n = static_cast<int>(v.size());
  if (n == 0) return;
  inv_sqrt_floor_kernel<<<Blocks(n), kThreads, 0, stream>>>(v.data(), n, floor_value);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ElementwiseMultiplyInPlace(cudaStream_t stream, float *a, const float *b, size_t n) {
  if (n == 0) return;
  elementwise_multiply_kernel<<<Blocks(n), kThreads, 0, stream>>>(a, b, static_cast<int>(n));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void FindDiagonalPositions(cudaStream_t stream, const CSRSparseMatrix &matrix,
                           dvector<int> &positions) {
  const int n = static_cast<int>(matrix.NumRows());
  positions.resize(n);
  if (n == 0) return;
  find_diagonal_kernel<<<Blocks(n), kThreads, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), n, positions.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ExtractDiagonalAt(cudaStream_t stream, const CSRSparseMatrix &matrix,
                       const dvector<int> &positions, dvector<float> &diagonal) {
  const int n = static_cast<int>(positions.size());
  diagonal.resize(n);
  if (n == 0) return;
  extract_diagonal_at_kernel<<<Blocks(n), kThreads, 0, stream>>>(
      matrix.values.data(), positions.data(), n, diagonal.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void MultiplyCSRByDenseVector(cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix,
                              int num_rows, int num_cols, int num_nonzeros, const dvector<float> &x,
                              dvector<float> &y, dvector<uint8_t> &buffer) {
  auto cusparse_handle = static_cast<cusparseHandle_t>(handle);
  y.resize(num_rows);
  assert(x.size() == static_cast<size_t>(num_cols));

  cuSPARSEMatrixDescription matrix_description(num_rows, num_cols, num_nonzeros, matrix);
  cuSPARSEVectorDescription vec_x_description(x);
  cuSPARSEVectorDescription vec_y_description(y);
  auto mat_a = static_cast<cusparseSpMatDescr_t>(matrix_description.GetDescription());
  auto vec_x = static_cast<cusparseDnVecDescr_t>(vec_x_description.GetDescription());
  auto vec_y = static_cast<cusparseDnVecDescr_t>(vec_y_description.GetDescription());

  constexpr float alpha = 1;
  constexpr float beta = 0;
  constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t buffer_size = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_bufferSize(cusparse_handle, op, &alpha, mat_a, vec_x, &beta,
                                                  vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                  &buffer_size));
  buffer.resize(buffer_size);
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(cusparse_handle, op, &alpha, mat_a, vec_x, &beta,
                                                  vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                                  buffer.data()));
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV(cusparse_handle, op, &alpha, mat_a, vec_x, &beta, vec_y,
                                       CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer.data()));
}

}  // namespace cunls
