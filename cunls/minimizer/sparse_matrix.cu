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
void ExtractMatrixMetadata(cudaStream_t stream, const CSRSparseMatrix &matrix,
                           int &num_rows, int &num_cols, int &num_nonzeros) {
  num_rows = matrix.row_offsets.size() - 1;
  assert(num_rows > 0);

  num_nonzeros = matrix.values.size();
  assert(num_nonzeros > 0);

  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::device_ptr<const int> col_ids_ptr(matrix.col_ids.data());
  auto max_col_idx_it = thrust::max_element(stream_policy, col_ids_ptr,
                                            col_ids_ptr + num_nonzeros);
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
template <int Value> struct NotEqualOperator {
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
                                        const float *__restrict__ values,
                                        float *__restrict__ diag,
                                        int num_rows) {
  int row = blockIdx.x * blockDim.y + threadIdx.y; // warp-level row assignment
  if (row >= num_rows) {
    return;
  }

  int lane = threadIdx.x; // thread within warp
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
                                           float *__restrict__ values,
                                           float scale,
                                           const float *__restrict__ diagonal,
                                           int num_rows) {
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
      return; // All lanes exit together (uniform control flow)
    }
  }
}

/**
 * CUDA kernel to build a mapping from triplet indices to CSR indices.
 * For each entry in the original triplet structure, finds its position
 * in the CSR format using linear search within the row.
 *
 * @param triplet_row_ids Original triplet row indices
 * @param triplet_col_ids Original triplet column indices
 * @param num_triplets Total number of triplet entries
 * @param csr_row_offsets CSR row offsets array
 * @param csr_col_ids CSR column indices array
 * @param mapping Output mapping: mapping[triplet_idx] = csr_idx, or -1 if
 * invalid
 */
__global__ void build_triplet_to_csr_mapping_kernel(
    const int *__restrict__ triplet_row_ids,
    const int *__restrict__ triplet_col_ids, int num_triplets,
    const int *__restrict__ csr_row_offsets,
    const int *__restrict__ csr_col_ids, int *__restrict__ mapping) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_triplets) {
    return;
  }

  int col = triplet_col_ids[tid];
  if (col < 0) {
    mapping[tid] = -1;
    return;
  }

  int row = triplet_row_ids[tid];
  int start = csr_row_offsets[row];
  int end = csr_row_offsets[row + 1];

  // Linear search for col in csr_col_ids[start:end]
  for (int i = start; i < end; i++) {
    if (csr_col_ids[i] == col) {
      mapping[tid] = i;
      return;
    }
  }

  // Should not be reached for valid entries
  mapping[tid] = -1;
}

/**
 * CUDA kernel to scatter triplet values into CSR format using a precomputed
 * mapping. Each thread processes one triplet entry and writes its value to
 * the corresponding CSR position.
 *
 * @param triplet_values Source triplet values
 * @param mapping Precomputed triplet-to-CSR index mapping
 * @param num_triplets Total number of triplet entries
 * @param csr_values Destination CSR values array
 */
__global__ void
scatter_triplet_values_kernel(const float *__restrict__ triplet_values,
                              const int *__restrict__ mapping, int num_triplets,
                              float *__restrict__ csr_values) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_triplets) {
    return;
  }

  int csr_idx = mapping[tid];
  if (csr_idx >= 0) {
    csr_values[csr_idx] = triplet_values[tid];
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
  thrust::copy(stream_policy, in_values_ptr,
               in_values_ptr + input.values.size(), out_values_ptr);

  thrust::device_ptr<const int> in_col_ids_ptr(input.col_ids.data());
  thrust::device_ptr<int> out_col_ids_ptr(output.col_ids.data());
  thrust::copy(stream_policy, in_col_ids_ptr,
               in_col_ids_ptr + input.col_ids.size(), out_col_ids_ptr);

  thrust::device_ptr<const int> in_row_offsets_ptr(input.row_offsets.data());
  thrust::device_ptr<int> out_row_offsets_ptr(output.row_offsets.data());
  thrust::copy(stream_policy, in_row_offsets_ptr,
               in_row_offsets_ptr + input.row_offsets.size(),
               out_row_offsets_ptr);
}

/**
 * Symmetric diagonal scaling of CSR values: A_ij *= scale[i]*scale[j].
 * One CUDA warp per row; lanes stride over that row's nnz for coalesced access.
 */
__global__ void scale_symmetric_csr_rows_kernel(
    const int *__restrict__ row_offsets, const int *__restrict__ col_ids,
    float *__restrict__ values, const float *__restrict__ scale, int num_rows) {
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

void ScaleSymmetricCSR(cudaStream_t stream, CSRSparseMatrix &matrix,
                       const dvector<float> &scale) {
  int num_rows = static_cast<int>(matrix.row_offsets.size() - 1);
  assert(static_cast<int>(scale.size()) == num_rows);
  dim3 block(WARP_SIZE, 8);
  dim3 grid((num_rows + block.y - 1) / block.y);
  scale_symmetric_csr_rows_kernel<<<grid, block, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
      scale.data(), num_rows);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

__global__ void inv_sqrt_floor_kernel(float *__restrict__ v, int n,
                                      float floor_value) {
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

void InvertSqrtWithFloorInPlace(cudaStream_t stream, dvector<float> &v,
                                float floor_value) {
  int n = static_cast<int>(v.size());
  if (n == 0) {
    return;
  }
  constexpr int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  inv_sqrt_floor_kernel<<<grid, block_size, 0, stream>>>(v.data(), n,
                                                         floor_value);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * Per-column sum of squares ||J_{:,j}||_2^2 from CSR Jacobian entries.
 * One CUDA thread per nonzero; uses atomicAdd (O(nnz), no sort / no Thrust).
 */
__global__ void
jacobian_accum_col_sq_atomic_kernel(const int *__restrict__ col_ids,
                                    const float *__restrict__ values, int nnz,
                                    float *__restrict__ col_sums) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz) {
    return;
  }
  float t = values[i];
  atomicAdd(col_sums + col_ids[i], t * t);
}

__global__ void col_sq_to_inv_norm_kernel(float *__restrict__ col_sums,
                                          int num_cols, float eps) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= num_cols) {
    return;
  }
  float s = col_sums[j];
  if (s < eps) {
    s = eps;
  }
  col_sums[j] = rsqrtf(s);
}

void ComputeJacobianColumnScaling(cudaStream_t stream,
                                  const CSRSparseMatrix &jacobian, int num_cols,
                                  int num_nonzeros,
                                  dvector<float> &column_scale) {
  column_scale.resize(static_cast<size_t>(num_cols));

  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(column_scale.data(), 0,
                      static_cast<size_t>(num_cols) * sizeof(float), stream));

  constexpr int block_size = 256;
  int grid_nnz = (num_nonzeros + block_size - 1) / block_size;
  jacobian_accum_col_sq_atomic_kernel<<<grid_nnz, block_size, 0, stream>>>(
      jacobian.col_ids.data(), jacobian.values.data(), num_nonzeros,
      column_scale.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  int grid_cols = (num_cols + block_size - 1) / block_size;
  const float floor_value = 1e-12f;
  col_sq_to_inv_norm_kernel<<<grid_cols, block_size, 0, stream>>>(
      column_scale.data(), num_cols, floor_value);
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
void ExtractDiagonal(cudaStream_t stream, const CSRSparseMatrix &matrix,
                     dvector<float> &diagonal) {
  size_t num_rows = matrix.row_offsets.size() - 1;

  diagonal.resize(num_rows);
  dim3 block(32, 4);
  dim3 grid((num_rows + block.y - 1) / block.y);
  extract_diagonal_kernel<<<grid, block, 0, stream>>>(
      matrix.row_offsets.data(), matrix.col_ids.data(), matrix.values.data(),
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
static void SpMVImpl(cudaStream_t stream, void *handle,
                     const CSRSparseMatrix &matrix, int num_rows, int num_cols,
                     int num_nonzeros, bool transpose_matrix,
                     const dvector<float> &x, dvector<float> &result,
                     dvector<uint8_t> &buffer) {
  auto cusparse_handle = static_cast<cusparseHandle_t>(handle);

  result.resize(transpose_matrix ? num_cols : num_rows);
  assert(x.size() ==
         static_cast<size_t>(transpose_matrix ? num_rows : num_cols));

  cuSPARSEMatrixDescription matrix_description(num_rows, num_cols, num_nonzeros,
                                               matrix);
  cuSPARSEVectorDescription vec_x_description(x);
  cuSPARSEVectorDescription vec_result_description(result);

  auto matA =
      static_cast<cusparseSpMatDescr_t>(matrix_description.GetDescription());
  auto vecX =
      static_cast<cusparseDnVecDescr_t>(vec_x_description.GetDescription());
  auto vecY = static_cast<cusparseDnVecDescr_t>(
      vec_result_description.GetDescription());

  constexpr float alpha = 1;
  constexpr float beta = 0;

  cusparseOperation_t operation = transpose_matrix
                                      ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  size_t bufferSize = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_bufferSize(
      cusparse_handle, operation, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  buffer.resize(bufferSize);

  auto buffer_ptr = buffer.data();

  THROW_ON_CUSPARSE_ERROR(cusparseSpMV_preprocess(
      cusparse_handle, operation, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
      CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));

  THROW_ON_CUSPARSE_ERROR(cusparseSpMV(cusparse_handle, operation, &alpha, matA,
                                       vecX, &beta, vecY, CUDA_R_32F,
                                       CUSPARSE_SPMV_ALG_DEFAULT, buffer_ptr));
}

void MultiplySparseMatrixByDenseVector(cudaStream_t stream, void *handle,
                                       const CSRSparseMatrix &matrix,
                                       bool transpose_matrix,
                                       const dvector<float> &x,
                                       dvector<float> &result,
                                       dvector<uint8_t> &buffer) {
  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, matrix, num_rows, num_cols, num_nonzeros);
  SpMVImpl(stream, handle, matrix, num_rows, num_cols, num_nonzeros,
           transpose_matrix, x, result, buffer);
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
void AddScaledDiagonal(cudaStream_t stream, float scale,
                       const dvector<float> &diagonal,
                       const CSRSparseMatrix &matrix, CSRSparseMatrix &result) {
  int num_rows = diagonal.size();
  assert(num_rows + 1 == matrix.row_offsets.size());

  CopyCSRSparseMatrix(stream, matrix, result);

  // Launch one warp (32 threads) per row for warp-cooperative diagonal search
  constexpr int block_size = 256; // Must be multiple of WARP_SIZE
  const int total_threads = num_rows * WARP_SIZE;
  const int blocks = (total_threads + block_size - 1) / block_size;
  add_scaled_diagonal_kernel<<<blocks, block_size, 0, stream>>>(
      result.row_offsets.data(), result.col_ids.data(), result.values.data(),
      scale, diagonal.data(), num_rows);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * Converts a TripletSparseStructure to CSR format and builds a mapping from
 * triplet indices to CSR indices. The mapping enables efficient value updates
 * without re-computing the structure on each iteration.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param handle cuSPARSE library handle
 * @param structure Triplet sparse structure (may contain -1 for invalid
 * entries)
 * @param csr Output CSR sparse matrix (structure filled, values zeroed)
 * @param mapping Output mapping: mapping[triplet_idx] = csr_idx, or -1
 * @param buffer Temporary buffer for intermediate computations
 */
void ConvertTripletStructureToCSR(cudaStream_t stream, void *handle,
                                  const TripletSparseStructure &structure,
                                  CSRSparseMatrix &csr, dvector<int> &mapping,
                                  dvector<uint8_t> &buffer) {
  auto cusparse_handle = static_cast<cusparseHandle_t>(handle);
  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  const auto &col_ids = structure.col_ids;
  const auto &row_ids = structure.row_ids;
  size_t num_triplets = col_ids.size();

  if (num_triplets == 0) {
    csr.row_offsets.resize(1);
    csr.col_ids.resize(0);
    csr.values.resize(0);
    mapping.resize(0);
    THROW_ON_CUDA_ERROR(
        cudaMemsetAsync(csr.row_offsets.data(), 0, sizeof(int), stream));
    return;
  }

  // Count valid entries (col_id != -1) and find dimensions
  thrust::device_ptr<const int> col_ids_ptr(col_ids.data());
  int number_of_nonzeros =
      thrust::count_if(stream_policy, col_ids_ptr, col_ids_ptr + num_triplets,
                       NotEqualOperator<-1>());

  thrust::device_ptr<const int> row_ids_ptr(row_ids.data());
  auto max_row_idx_it = thrust::max_element(stream_policy, row_ids_ptr,
                                            row_ids_ptr + num_triplets);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  int num_rows = *max_row_idx_it + 1;
  assert(num_rows > 0);

  // Allocate CSR arrays
  csr.values.resize(number_of_nonzeros);
  csr.col_ids.resize(number_of_nonzeros);
  csr.row_offsets.resize(num_rows + 1);

  // Phase 1: Stream compaction to filter out invalid entries (col_id == -1),
  // producing COO format directly without intermediate triplet representation.
  size_t buffer_size_in_bytes = number_of_nonzeros * sizeof(int);

  if (buffer.size() < buffer_size_in_bytes) {
    buffer.resize(buffer_size_in_bytes);
  }

  auto coo_row_ids_ptr = reinterpret_cast<int *>(buffer.data());

  {
    auto input_begin = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(row_ids.data()),
                           thrust::device_pointer_cast(col_ids.data())));
    auto output_begin = thrust::make_zip_iterator(
        thrust::make_tuple(thrust::device_pointer_cast(coo_row_ids_ptr),
                           thrust::device_pointer_cast(csr.col_ids.data())));
    thrust::device_ptr<const int> stencil(col_ids.data());

    thrust::copy_if(stream_policy, input_begin, input_begin + num_triplets,
                    stencil, output_begin, NotEqualOperator<-1>());

    THROW_ON_CUDA_ERROR(cudaMemsetAsync(
        csr.values.data(), 0, number_of_nonzeros * sizeof(float), stream));
  }

  // Convert COO row indices to CSR row offsets
  THROW_ON_CUSPARSE_ERROR(cusparseXcoo2csr(
      cusparse_handle, coo_row_ids_ptr, number_of_nonzeros, num_rows,
      csr.row_offsets.data(), cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO));

  // Phase 2: Build mapping from triplet indices to CSR indices.
  // For each original triplet entry, linear search in the CSR
  // structure to find the corresponding CSR index.
  mapping.resize(num_triplets);
  {
    constexpr size_t block_size = 256;
    size_t num_blocks = (num_triplets + block_size - 1) / block_size;
    build_triplet_to_csr_mapping_kernel<<<num_blocks, block_size, 0, stream>>>(
        structure.row_ids.data(), structure.col_ids.data(), num_triplets,
        csr.row_offsets.data(), csr.col_ids.data(), mapping.data());
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

/**
 * Copies values from a SparseJacobian in triplet form to a CSRSparseMatrix
 * using a precomputed mapping from ConvertTripletStructureToCSR.
 * This is much faster than recomputing the full structure+values conversion
 * since it only scatters values to their precomputed positions.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param jacobian Sparse Jacobian in triplet format with updated values
 * @param mapping Precomputed mapping from triplet indices to CSR indices
 * @param csr CSR sparse matrix with precomputed structure (values updated)
 */
void ConvertTripletToCSRValues(cudaStream_t stream,
                               const SparseJacobian &jacobian,
                               const dvector<int> &mapping,
                               CSRSparseMatrix &csr) {
  assert(jacobian.values.size() == jacobian.structure.col_ids.size());
  assert(jacobian.values.size() == jacobian.structure.row_ids.size());
  assert(jacobian.values.size() == mapping.size());

  size_t num_triplets = jacobian.values.size();
  if (num_triplets == 0) {
    return;
  }

  constexpr size_t block_size = 256;
  size_t num_blocks = (num_triplets + block_size - 1) / block_size;
  scatter_triplet_values_kernel<<<num_blocks, block_size, 0, stream>>>(
      jacobian.values.data(), mapping.data(), num_triplets, csr.values.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * Computes the right-hand side vector for the normal equations: rhs = -J^T * r.
 * This is a key step in Gauss-Newton and Levenberg-Marquardt optimization
 * algorithms.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param jacobian CSR sparse Jacobian matrix (J)
 * @param residuals Dense residual vector (r)
 * @param rhs Output right-hand side vector (-J^T * r)
 * @param buffer Temporary buffer for sparse matrix operations
 *
 * First computes J^T * r using sparse matrix-vector multiplication,
 * then negates the result to get -J^T * r.
 */
__global__ void negate_kernel(float *__restrict__ data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] = -data[i];
}

void NegateVector(cudaStream_t stream, float *data, size_t n) {
  if (n == 0)
    return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  negate_kernel<<<grid, kBlock, 0, stream>>>(data, static_cast<int>(n));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

__global__ void elementwise_multiply_kernel(float *__restrict__ a,
                                            const float *__restrict__ b,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] *= b[i];
}

void ElementwiseMultiplyInPlace(cudaStream_t stream, float *a, const float *b,
                                size_t n) {
  if (n == 0)
    return;
  constexpr int kBlock = 256;
  int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  elementwise_multiply_kernel<<<grid, kBlock, 0, stream>>>(a, b,
                                                           static_cast<int>(n));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ComputeRHS(cudaStream_t stream, void *handle,
                const CSRSparseMatrix &jacobian,
                const dvector<float> &residuals, dvector<float> &rhs,
                dvector<uint8_t> &buffer) {
  constexpr bool transpose_matrix = true;
  MultiplySparseMatrixByDenseVector(stream, handle, jacobian, transpose_matrix,
                                    residuals, rhs, buffer);
  NegateVector(stream, rhs.data(), rhs.size());
}

void ComputeRHS(cudaStream_t stream, void *handle,
                const CSRSparseMatrix &jacobian, int num_rows, int num_cols,
                int num_nonzeros, const dvector<float> &residuals,
                dvector<float> &rhs, dvector<uint8_t> &buffer) {
  constexpr bool transpose_matrix = true;
  SpMVImpl(stream, handle, jacobian, num_rows, num_cols, num_nonzeros,
           transpose_matrix, residuals, rhs, buffer);
  NegateVector(stream, rhs.data(), rhs.size());
}

/**
 * Computes the weighted squared norm of a step vector: step^T * W * step,
 * where W is a diagonal weight matrix.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param weights Diagonal weight values (W)
 * @param step Step vector
 * @param buffer Temporary buffer for intermediate computations
 * @return The weighted squared norm (scalar value)
 *
 * Computes the inner product of the step vector with its element-wise
 * product with the weights: sum(step[i] * weights[i] * step[i]).
 * Used in trust region methods and optimization algorithms.
 */
void ComputeWeightedSquaredStepAsync(cudaStream_t stream,
                                     const dvector<float> &weights,
                                     const dvector<float> &step, float *d_out,
                                     float *d_partials) {
  assert(step.size() == weights.size());
  WeightedDotProductToDevice(stream, step.data(), weights.data(), step.data(),
                             step.size(), d_out, d_partials);
}

float ComputeWeightedSquaredStep(cudaStream_t stream,
                                 const dvector<float> &weights,
                                 const dvector<float> &step,
                                 dvector<uint8_t> &buffer) {
  assert(step.size() == weights.size());
  size_t partials_count = ReducePartialCount(step.size());
  buffer.resize((partials_count + 1) * sizeof(float));
  float *d_out = reinterpret_cast<float *>(buffer.data());
  float *d_partials = d_out + 1;

  WeightedDotProductToDevice(stream, step.data(), weights.data(), step.data(),
                             step.size(), d_out, d_partials);

  float result;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&result, d_out, sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  assert(result >= 0);
  return result;
}

/**
 * Computes the weighted squared norm using a sparse matrix: step^T * A * step.
 * This overload uses a sparse matrix A instead of diagonal weights.
 *
 * @param stream CUDA stream for asynchronous operations
 * @param matrix Sparse weight matrix (A)
 * @param step Step vector
 * @param buffer Temporary buffer for sparse matrix operations
 * @return The weighted squared norm (scalar value)
 *
 * First computes A * step using sparse matrix-vector multiplication,
 * then computes the inner product with the original step vector.
 * Used when the weighting is represented as a full sparse matrix rather than
 * just diagonal weights.
 */
static thread_local dvector<float> g_spmv_result;

static void WeightedSquaredStepSparseAsyncImpl(
    cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix,
    int num_rows, int num_cols, int num_nonzeros, const dvector<float> &step,
    dvector<uint8_t> &buffer, float *d_out, float *d_partials) {
  constexpr bool transpose_matrix = false;
  SpMVImpl(stream, handle, matrix, num_rows, num_cols, num_nonzeros,
           transpose_matrix, step, g_spmv_result, buffer);
  DotProductToDevice(stream, g_spmv_result.data(), step.data(),
                     g_spmv_result.size(), d_out, d_partials);
}

void ComputeWeightedSquaredStepAsync(
    cudaStream_t stream, void *handle, const CSRSparseMatrix &matrix,
    int num_rows, int num_cols, int num_nonzeros, const dvector<float> &step,
    dvector<uint8_t> &buffer, float *d_out, float *d_partials) {
  WeightedSquaredStepSparseAsyncImpl(stream, handle, matrix, num_rows, num_cols,
                                     num_nonzeros, step, buffer, d_out,
                                     d_partials);
}

float ComputeWeightedSquaredStep(cudaStream_t stream, void *handle,
                                 const CSRSparseMatrix &matrix,
                                 const dvector<float> &step,
                                 dvector<uint8_t> &buffer) {
  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, matrix, num_rows, num_cols, num_nonzeros);

  size_t partials_count = ReducePartialCount(step.size());
  dvector<float> d_scratch(partials_count + 1);
  float *d_out = d_scratch.data();
  float *d_partials = d_out + 1;

  WeightedSquaredStepSparseAsyncImpl(stream, handle, matrix, num_rows, num_cols,
                                     num_nonzeros, step, buffer, d_out,
                                     d_partials);

  float result;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&result, d_out, sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  assert(result >= 0);
  return result;
}

float ComputeWeightedSquaredStep(cudaStream_t stream, void *handle,
                                 const CSRSparseMatrix &matrix, int num_rows,
                                 int num_cols, int num_nonzeros,
                                 const dvector<float> &step,
                                 dvector<uint8_t> &buffer) {
  size_t partials_count = ReducePartialCount(step.size());
  dvector<float> d_scratch(partials_count + 1);
  float *d_out = d_scratch.data();
  float *d_partials = d_out + 1;

  WeightedSquaredStepSparseAsyncImpl(stream, handle, matrix, num_rows, num_cols,
                                     num_nonzeros, step, buffer, d_out,
                                     d_partials);

  float result;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&result, d_out, sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  assert(result >= 0);
  return result;
}

/**
 * @brief Computes the squared L2 norm of a step vector: step^T * step.
 *
 * @param stream CUDA stream for asynchronous operations.
 * @param step Step vector.
 * @return The squared L2 norm (scalar value).
 */
void ComputeSquaredStepAsync(cudaStream_t stream, const dvector<float> &step,
                             float *d_out, float *d_partials) {
  DotProductToDevice(stream, step.data(), step.data(), step.size(), d_out,
                     d_partials);
}

float ComputeSquaredStep(cudaStream_t stream, const dvector<float> &step) {
  size_t partials_count = ReducePartialCount(step.size());
  dvector<float> d_scratch(partials_count + 1);
  float *d_out = d_scratch.data();
  float *d_partials = d_out + 1;

  DotProductToDevice(stream, step.data(), step.data(), step.size(), d_out,
                     d_partials);

  float result;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&result, d_out, sizeof(float),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  assert(result >= 0);
  return result;
}
} // namespace cunls
