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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cassert>

#include "cunls/common/cusparse_helper.h"
#include "cunls/minimizer/device_reduction.h"
#include "cunls/minimizer/sparse_matrix.h"

#define WARP_SIZE 32

namespace cunls {

/**
 * Extracts metadata from a CSR sparse matrix including dimensions and number of
 * non-zero elements.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param matrix CSR sparse matrix to extract metadata from
 * @param num_rows Output argument for the number of rows in the matrix
 * @param num_cols Output argument for the number of columns in the matrix
 * @param num_nonzeros Output argument for the number of non-zero elements
 *
 * The number of columns is determined by finding the maximum column index + 1.
 * Requires matrix to have at least one row and one non-zero element.
 */
void ExtractMatrixMetadata(cudaStream_t stream, const CSRSparseMatrix &matrix, int &num_rows,
                           int &num_cols, int &num_nonzeros) {
  num_rows = matrix.row_offsets.empty() ? 0 : static_cast<int>(matrix.row_offsets.size() - 1);
  num_nonzeros = static_cast<int>(matrix.values.size());

  // A fully-constrained problem (every state block constant) yields an empty
  // system; max_element over an empty range would dereference end().
  if (num_nonzeros == 0) {
    num_cols = 0;
    return;
  }

  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::device_ptr<const int> col_ids_ptr(matrix.col_ids.data());
  auto max_col_idx_it = thrust::max_element(stream_policy, col_ids_ptr, col_ids_ptr + num_nonzeros);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  num_cols = *max_col_idx_it + 1;
  assert(num_cols > 0);
}

/**
 * Functor to check if a value is not equal to -1.
 * Used to filter out invalid entries in sparse matrix structures where -1
 * indicates missing or invalid elements (e.g., due to constant states in
 * Jacobians).
 */
template <int Value>
struct NotEqualOperator {
  __host__ __device__ bool operator()(const int &x) { return x != Value; }
};

/**
 * CUDA kernel to extract diagonal elements from a CSR sparse matrix.
 * Uses warp-cooperative processing for efficient memory access patterns.
 *
 * @param row_ptr CSR row pointers array
 * @param col_ind CSR column indices array
 * @param values CSR values array
 * @param diag Output array to store diagonal elements
 * @param num_rows Number of rows in the matrix
 *
 * Thread organization: Each warp processes one row, with threads cooperating
 * to search for the diagonal element. Uses warp shuffles for reduction.
 */
__global__ void extract_diagonal_kernel(const int *__restrict__ row_ptr,
                                        const int *__restrict__ col_ind,
                                        const float *__restrict__ values, float *__restrict__ diag,
                                        int num_rows) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;  // warp-level row assignment
  if (row >= num_rows) {
    return;
  }

  int lane = threadIdx.x;  // thread within warp
  int start = row_ptr[row];
  int end = row_ptr[row + 1];

  float diag_val = 0.0f;

  // Warp-cooperative search
  for (int idx = start + lane; idx < end; idx += WARP_SIZE) {
    int col = col_ind[idx];
    if (col == row) {
      diag_val = values[idx];
    }
  }
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    diag_val += __shfl_down_sync(0xFFFFFFFF, diag_val, offset);
  }
  if (lane == 0) {
    diag[row] = diag_val;
  }
}

/**
 * CUDA kernel to add a scaled diagonal to the matrix diagonal elements.
 * Performs the operation: A[i,i] = A[i,i] + scale * diagonal[i] for each row i.
 *
 * @param row_offsets CSR row pointers array
 * @param col_indices CSR column indices array
 * @param values CSR values array (modified in-place)
 * @param scale Scaling factor for the diagonal values
 * @param diagonal Diagonal values to add
 * @param num_rows Number of rows in the matrix
 *
 * Uses warp-cooperative parallel search: one warp processes one row, with all
 * 32 lanes simultaneously checking different positions. For rows with <= 32
 * non-zeros, no iteration is needed. Uses ballot_sync for uniform warp exit.
 */
__global__ void add_scaled_diagonal_kernel(const int *__restrict__ row_offsets,
                                           const int *__restrict__ col_indices,
                                           float *__restrict__ values, float scale,
                                           const float *__restrict__ diagonal, int num_rows) {
  // One warp per row: warp_id identifies the row, lane_id is the thread within
  // warp
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);

  if (warp_id >= num_rows) {
    return;
  }

  const int row = warp_id;
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  const int row_len = row_end - row_start;

  // Warp-cooperative parallel search: each lane checks a strided position
  // For rows <= 32 elements, this executes without iteration
  for (int base = 0; base < row_len; base += WARP_SIZE) {
    const int offset = base + lane_id;
    const int idx = row_start + offset;

    // Each lane checks if its position is the diagonal element
    const bool is_diag = (offset < row_len) && (col_indices[idx] == row);

    // ballot_sync collects results from all lanes - uniform warp operation
    const unsigned mask = __ballot_sync(0xFFFFFFFF, is_diag);

    if (mask != 0) {
      // Diagonal found - only the lane that found it performs the update
      if (is_diag) {
        values[idx] += scale * diagonal[row];
      }
      return;  // All lanes exit together (uniform control flow)
    }
  }
}

/**
 * Creates a deep copy of a CSR sparse matrix.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param input Source CSR sparse matrix to copy from
 * @param output Destination CSR sparse matrix to copy to
 *
 * Resizes output arrays to match input dimensions and copies all data
 * (values, column indices, row offsets). Uses thrust::copy for efficient GPU
 * memory transfer.
 */
void CopyCSRSparseMatrix(cudaStream_t stream, const CSRSparseMatrix &input,
                         CSRSparseMatrix &output) {
  if (&input == &output) {
    return;
  }

  size_t num_nonzeros = input.values.size();
  size_t num_rows = input.row_offsets.size() - 1;
  output.col_ids.resize(num_nonzeros);
  output.values.resize(num_nonzeros);
  output.row_offsets.resize(num_rows + 1);

  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  thrust::device_ptr<const float> in_values_ptr(input.values.data());
  thrust::device_ptr<float> out_values_ptr(output.values.data());
  thrust::copy(stream_policy, in_values_ptr, in_values_ptr + input.values.size(), out_values_ptr);

  thrust::device_ptr<const int> in_col_ids_ptr(input.col_ids.data());
  thrust::device_ptr<int> out_col_ids_ptr(output.col_ids.data());
  thrust::copy(stream_policy, in_col_ids_ptr, in_col_ids_ptr + input.col_ids.size(),
               out_col_ids_ptr);

  thrust::device_ptr<const int> in_row_offsets_ptr(input.row_offsets.data());
  thrust::device_ptr<int> out_row_offsets_ptr(output.row_offsets.data());
  thrust::copy(stream_policy, in_row_offsets_ptr, in_row_offsets_ptr + input.row_offsets.size(),
               out_row_offsets_ptr);
}

/**
 * Symmetric diagonal scaling of CSR values: A_ij *= scale[i]*scale[j].
 * One CUDA warp per row; lanes stride over that row's nnz for coalesced access.
 */
__global__ void scale_symmetric_csr_rows_kernel(const int *__restrict__ row_offsets,
                                                const int *__restrict__ col_ids,
                                                float *__restrict__ values,
                                                const float *__restrict__ scale, int num_rows) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= num_rows) {
    return;
  }
  int lane = threadIdx.x;
  float sr = scale[row];
  int start = row_offsets[row];
  int end = row_offsets[row + 1];
  for (int idx = start + lane; idx < end; idx += WARP_SIZE) {
    int col = col_ids[idx];
    values[idx] *= sr * scale[col];
  }
}

void ScaleSymmetricCSR(cudaStream_t stream, CSRSparseMatrix &matrix, const dvector<float> &scale) {
  int num_rows = static_cast<int>(matrix.row_offsets.size() - 1);
  assert(static_cast<int>(scale.size()) == num_rows);
  dim3 block(WARP_SIZE, 8);
  dim3 grid((num_rows + block.y - 1) / block.y);
  scale_symmetric_csr_rows_kernel<<<grid, block, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(), scale.data(),
      num_rows);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

__global__ void inv_sqrt_floor_kernel(float *__restrict__ v, int n, float floor_value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float x = v[i];
  if (x < floor_value) {
    x = floor_value;
  }
  v[i] = rsqrtf(x);
}

void InvertSqrtWithFloorInPlace(cudaStream_t stream, dvector<float> &v, float floor_value) {
  int n = static_cast<int>(v.size());
  if (n == 0) {
    return;
  }
  constexpr int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  inv_sqrt_floor_kernel<<<grid, block_size, 0, stream>>>(v.data(), n, floor_value);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * Extracts the diagonal elements from a CSR sparse matrix.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param matrix CSR sparse matrix to extract diagonal from
 * @param diagonal Output vector to store diagonal elements
 *
 * Uses a warp-cooperative CUDA kernel for efficient diagonal extraction.
 * Each warp processes one row, searching for the diagonal element (where row ==
 * col).
 */
void ExtractDiagonal(cudaStream_t stream, const CSRSparseMatrix &matrix, dvector<float> &diagonal) {
  size_t num_rows = matrix.row_offsets.size() - 1;

  diagonal.resize(num_rows);
  dim3 block(32, 4);
  dim3 grid((num_rows + block.y - 1) / block.y);
  extract_diagonal_kernel<<<grid, block, 0, stream>>>(matrix.row_offsets.data(),
                                                      matrix.col_ids.data(), matrix.values.data(),
                                                      diagonal.data(), num_rows);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * Multiplies a sparse matrix by a dense vector using cuSPARSE library.
 * Performs either A*x or A^T*x depending on transpose_matrix flag.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param handle cuSPARSE library handle
 * @param matrix CSR sparse matrix (A)
 * @param transpose_matrix If true, computes A^T*x; if false, computes A*x
 * @param x Dense input vector
 * @param result Dense output vector to store the result
 * @param buffer Temporary buffer for cuSPARSE operations
 *
 * Uses cuSPARSE SpMV (Sparse Matrix-Vector multiplication) with preprocessing
 * for optimal performance. Buffer is automatically resized as needed.
 */
static void SpMVImpl(cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix, int num_rows,
                     int num_cols, int num_nonzeros, bool transpose_matrix, const dvector<float> &x,
                     dvector<float> &result, dvector<uint8_t> &buffer) {
  auto cusparse_handle = static_cast<cusparseHandle_t>(handle);

  result.resize(transpose_matrix ? num_cols : num_rows);
  assert(x.size() == static_cast<size_t>(transpose_matrix ? num_rows : num_cols));

  cuSPARSEMatrixDescription matrix_description(num_rows, num_cols, num_nonzeros, matrix);
  cuSPARSEVectorDescription vec_x_description(x);
  cuSPARSEVectorDescription vec_result_description(result);

  auto matA = static_cast<cusparseSpMatDescr_t>(matrix_description.GetDescription());
  auto vecX = static_cast<cusparseDnVecDescr_t>(vec_x_description.GetDescription());
  auto vecY = static_cast<cusparseDnVecDescr_t>(vec_result_description.GetDescription());

  constexpr float alpha = 1;
  constexpr float beta = 0;

  cusparseOperation_t operation =
      transpose_matrix ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

  size_t bufferSize = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_bufferSize(cusparse_handle, operation, &alpha, matA, vecX,
                                                  &beta, vecY, CUDA_R_32F,
                                                  CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  buffer.resize(bufferSize);

  auto buffer_ptr = buffer.data();

  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(cusparse_handle, operation, &alpha, matA, vecX,
                                                  &beta, vecY, CUDA_R_32F,
                                                  CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));

  THROW_ON_CUSPARSE_ERROR(cusparseSpMV(cusparse_handle, operation, &alpha, matA, vecX, &beta, vecY,
                                       CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));
}

/**
 * Adds a scaled diagonal to a sparse matrix: result = matrix + scale *
 * diag(diagonal).
 *
 * @param stream CUDA stream for asynchronous operations
 * @param scale Scaling factor for the diagonal values
 * @param diagonal Vector containing diagonal values to add
 * @param matrix Input CSR sparse matrix
 * @param result Output CSR sparse matrix (can be the same as input for in-place
 * operation)
 *
 * First copies the input matrix, then uses a CUDA kernel to add the scaled
 * diagonal elements to the existing diagonal entries of the matrix.
 */
void AddScaledDiagonal(cudaStream_t stream, float scale, const dvector<float> &diagonal,
                       const CSRSparseMatrix &matrix, CSRSparseMatrix &result) {
  int num_rows = diagonal.size();
  assert(num_rows + 1 == matrix.row_offsets.size());

  CopyCSRSparseMatrix(stream, matrix, result);

  // Launch one warp (32 threads) per row for warp-cooperative diagonal search
  constexpr int block_size = 256;  // Must be multiple of WARP_SIZE
  const int total_threads = num_rows * WARP_SIZE;
  const int blocks = (total_threads + block_size - 1) / block_size;
  add_scaled_diagonal_kernel<<<blocks, block_size, 0, stream>>>(
      result.row_offsets.data(), result.col_ids.data(), result.values.data(), scale,
      diagonal.data(), num_rows);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

__global__ void elementwise_multiply_kernel(float *__restrict__ a, const float *__restrict__ b,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= b[i];
}

void ElementwiseMultiplyInPlace(cudaStream_t stream, float *a, const float *b, size_t n) {
  if (n == 0) return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  elementwise_multiply_kernel<<<grid, kBlock, 0, stream>>>(a, b, static_cast<int>(n));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Async diagonally-weighted squared step: d_out[0] = step^T diag(w) step.
 */
void ComputeWeightedSquaredStepAsync(cudaStream_t stream, const dvector<float> &weights,
                                     const dvector<float> &step, float *d_out, float *d_partials) {
  assert(step.size() == weights.size());
  WeightedDotProductToDevice(stream, step.data(), weights.data(), step.data(), step.size(), d_out,
                             d_partials);
}

/**
 * @brief Async sparse-weighted squared step: d_out[0] = step^T A step.
 *
 * Runs the SpMV into a slice of `buffer` and reduces against `step`, so the
 * whole thing stays on the stream with no host synchronization.
 */
void ComputeWeightedSquaredStepAsync(cudaStream_t stream, void *handle,
                                     const CSRSparseMatrix &matrix, int num_rows, int num_cols,
                                     int num_nonzeros, const dvector<float> &step,
                                     dvector<uint8_t> &buffer, float *d_out, float *d_partials) {
  static thread_local dvector<float> spmv_result;
  SpMVImpl(stream, handle, matrix, num_rows, num_cols, num_nonzeros, /*transpose_matrix=*/false,
           step, spmv_result, buffer);
  DotProductToDevice(stream, step.data(), spmv_result.data(), step.size(), d_out, d_partials);
}

/**
 * @brief Computes the squared L2 norm of a step vector: step^T * step.
 *
 * @param stream CUDA stream for asynchronous operations.
 * @param step Step vector.
 * @return The squared L2 norm (scalar value).
 */
void ComputeSquaredStepAsync(cudaStream_t stream, const dvector<float> &step, float *d_out,
                             float *d_partials) {
  DotProductToDevice(stream, step.data(), step.data(), step.size(), d_out, d_partials);
}

}  // namespace cunls
