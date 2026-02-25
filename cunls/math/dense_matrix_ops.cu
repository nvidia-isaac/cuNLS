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

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <cmath>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cusolver_helper.h"
#include "cunls/common/types.h"
#include "cunls/math/dense_matrix_ops.h"

namespace cunls {

constexpr size_t block_size = 256;  ///< Thread block size for CUDA kernels

/**
 * @brief CUDA kernel to scale matrix rows by square root of eigenvalues.
 *
 * For each matrix in the batch, scales each row of the eigenvector matrix
 * by the square root of the corresponding eigenvalue. This is part of the
 * matrix square root computation: A^{1/2} = Q * D^{1/2} * Q^T.
 *
 * @param matrices Input/output matrices (device pointer, modified in-place)
 * @param matrix_size Size of each matrix (number of rows/columns)
 * @param pitch Pitch (leading dimension) of each matrix
 * @param eigenvalues Eigenvalues for each matrix (device pointer)
 * @param num_matrices Number of matrices in the batch
 */
__global__ void scale_row_kernel(float *matrices, size_t matrix_size,
                                 size_t pitch, float *eigenvalues,
                                 size_t num_matrices) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_matrices) {
    return;
  }

  float *matrix = matrices + tid * pitch * matrix_size;
  float *eigs = eigenvalues + tid * matrix_size;

#pragma unroll
  for (int i = 0; i < matrix_size; ++i) {
    float eig = eigs[i];
    eig = eig > 0.0f ? sqrtf(eig) : 0.0f;
#pragma unroll
    for (int j = 0; j < matrix_size; ++j) {
      matrix[i * pitch + j] *= eig;
    }
  }
}

/**
 * @brief Computes the square root of a batch of symmetric positive definite
 * matrices.
 *
 * Uses eigenvalue decomposition: A = Q * D * Q^T, then A^{1/2} = Q * D^{1/2} *
 * Q^T. The computation proceeds as follows:
 * 1. Compute eigenvalue decomposition using cuSolver
 * 2. Scale eigenvectors by square root of eigenvalues
 * 3. Reconstruct matrix square root using matrix multiplication
 *
 * @param cublas_handle Reference to an externally-owned cuBLAS handle.
 * @param stream CUDA stream for asynchronous operations.
 * @param spd_matrix Input/output matrices (device pointer, modified in-place).
 * @param matrix_size Size of each matrix (number of rows/columns).
 * @param pitch Pitch (leading dimension) of each matrix.
 * @param num_matrices Number of matrices in the batch.
 */
void ComputeSqrtMatrix(cuBLASHandle& cublas_handle, cudaStream_t stream,
                       float *spd_matrix, size_t matrix_size, size_t pitch,
                       size_t num_matrices) {
  if (spd_matrix == nullptr) {
    const std::string msg = "spd_matrix cannot be null";
    LogError(msg);
    throw std::invalid_argument(msg);
  }
  if (matrix_size == 0 || matrix_size > pitch) {
    const std::string msg = "Invalid matrix_size or pitch";
    LogError(msg);
    throw std::invalid_argument(msg);
  }
  if (num_matrices == 0) {
    return;  // Nothing to do
  }

  cuSolverHandle solver_handle;
  cuSolverInfo solver_info;

  auto handle = solver_handle.GetHandle(stream);
  auto params = solver_info.GetInfo();

  /// Temporary storage for eigenvector matrices (before scaling)
  dvector<float> temp_matrix(num_matrices * matrix_size * pitch);

  /// Storage for eigenvalues (one vector per matrix)
  dvector<float> eigenvalues(num_matrices * matrix_size);

  /// Storage for cuSolver info codes (one per matrix)
  dvector<int> info(num_matrices);

  auto eigenvalues_ptr =
      reinterpret_cast<float *>(eigenvalues.data());
  auto info_ptr =
      reinterpret_cast<int *>(info.data());
  auto temp_matrix_ptr =
      reinterpret_cast<float *>(temp_matrix.data());

  auto stream_policy = thrust::cuda::par_nosync.on(stream);

  // Step 1: Compute eigenvalue decomposition A = Q * D * Q^T
  // Query workspace size
  int lwork = 0;
  THROW_ON_CUSOLVER_ERROR(cusolverDnSsyevjBatched_bufferSize(
      handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, matrix_size,
      spd_matrix, pitch, eigenvalues_ptr, &lwork, params, num_matrices));

  /// Workspace buffer for cuSolver
  dvector<uint8_t> buffer(lwork);
  auto buffer_ptr =
      reinterpret_cast<float *>(buffer.data());

  // Perform eigenvalue decomposition (spd_matrix contains eigenvectors on
  // output)
  THROW_ON_CUSOLVER_ERROR(cusolverDnSsyevjBatched(
      handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, matrix_size,
      spd_matrix, pitch, eigenvalues_ptr, buffer_ptr, lwork, info_ptr, params,
      num_matrices));

  // Copy eigenvectors to temporary storage before scaling
  thrust::device_ptr<float> ptr = thrust::device_pointer_cast(spd_matrix);
  thrust::device_ptr<float> temp_dst_ptr(temp_matrix.data());
  thrust::copy(stream_policy, ptr, ptr + num_matrices * matrix_size * pitch,
               temp_dst_ptr);

  // Step 2: Scale eigenvectors by square root of eigenvalues
  // temp_matrix = Q * D^{1/2}
  {
    size_t num_blocks = (num_matrices + block_size - 1) / block_size;

    scale_row_kernel<<<num_blocks, block_size, 0, stream>>>(
        temp_matrix_ptr, matrix_size, pitch, eigenvalues_ptr, num_matrices);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  // Step 3: Reconstruct matrix square root: A^{1/2} = (Q * D^{1/2}) * Q^T
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  auto cublas_handle_ = cublas_handle.GetHandle(stream);
  size_t stride = matrix_size * pitch;

  // Compute spd_matrix = temp_matrix * Q^T = (Q * D^{1/2}) * Q^T
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T, matrix_size, matrix_size,
      matrix_size, &alpha, temp_matrix_ptr, pitch, stride, spd_matrix, pitch,
      stride, &beta, spd_matrix, pitch, stride, num_matrices));
}

}  // namespace cunls