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

#include <cublas_v2.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

/// Number of threads per CUDA block.
constexpr size_t kSO3BlockSize = 256;

/**
 * @brief CUDA kernel to collect SO(3) rotations from state pointers.
 *
 * @param state_pointers Array of state block pointers
 * @param num_factors Number of factors
 * @param rotations Output array for collected rotations
 */
__global__ void collect_so3_rotations_kernel(float const* const* state_pointers,
                                             size_t num_factors,
                                             Matrix<3>* rotations) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) {
    return;
  }

  auto rot_ptr = reinterpret_cast<const Matrix<3>*>(state_pointers[tid]);
  rotations[tid] = *rot_ptr;
}

SO3PriorFactorBatch::SO3PriorFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<3>* observations_ptr,
    size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      rotations_current_(num_factors),
      rotations_error_(num_factors) {}

bool SO3PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  size_t num_factors = NumFactors();

  // Step 1: Collect current rotations from state pointers into contiguous memory
  size_t num_blocks = (num_factors + kSO3BlockSize - 1) / kSO3BlockSize;
  collect_so3_rotations_kernel<<<num_blocks, kSO3BlockSize, 0, stream>>>(
      state_pointers, num_factors, rotations_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute R_error = R_target^T * R_current using cuBLAS
  // cuBLAS operates column-major. For row-major matrices:
  //   CUBLAS_OP_N on A reads A_col = A_row^T
  //   CUBLAS_OP_T on B reads B_col^T = (B_row^T)^T = B_row
  // Result_col = A_col * B_col^T = A_row^T * B_row
  // Result_row = Result_col^T = (A_row^T * B_row)^T = B_row^T * A_row
  // With A = R_current, B = R_target:
  //   Result_row = R_target^T * R_current  (exactly what we want)
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(rotations_current_.data()), mat_size,
      stride, reinterpret_cast<const float*>(observations_ptr_), mat_size,
      stride, &beta, reinterpret_cast<float*>(rotations_error_.data()),
      mat_size, stride, num_factors));

  // Step 3: Compute residual = Log(R_error) using SO(3) logarithm map
  constexpr size_t rotation_pitch = 3;
  constexpr size_t rotation_stride = 9;
  constexpr size_t twist_stride = 3;
  ComputeLogSO3(stream, reinterpret_cast<const float*>(rotations_error_.data()),
                rotation_pitch, rotation_stride, twist_stride, num_factors,
                residuals);

  // Step 4: Compute Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    constexpr size_t jacobian_pitch = 3;
    constexpr size_t jacobian_stride = 9;
    ComputeJacobianRightInverseSO3(stream, residuals, twist_stride,
                                   jacobian_pitch, jacobian_stride, num_factors,
                                   jacobians);
  }

  return true;
}

}  // namespace cunls
