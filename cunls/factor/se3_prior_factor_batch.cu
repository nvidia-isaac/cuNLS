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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

/// Number of threads per CUDA block.
constexpr size_t kSE3PriorBlockSize = 256;

/**
 * @brief CUDA kernel to collect SE(3) transforms from state pointers.
 *
 * @param state_pointers Array of state block pointers
 * @param num_factors Number of factors
 * @param transforms Output array for collected transforms
 */
__global__ void collect_se3_transforms_kernel(float const* const* state_pointers,
                                              size_t num_factors,
                                              SE3Transform* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) {
    return;
  }

  auto transform_ptr = reinterpret_cast<const SE3Transform*>(state_pointers[tid]);
  transforms[tid] = *transform_ptr;
}

SE3PriorFactorBatch::SE3PriorFactorBatch(
    cuBLASHandle& cublas_handle, const SE3Transform* observations_ptr,
    size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      cublas_handle_(cublas_handle),
      transforms_current_(num_factors),
      transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all targets
  CudaStream stream;
  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  ComputeInverseSE3(stream.GetStream(),
                    reinterpret_cast<const float*>(observations_ptr_), pitch,
                    stride, pitch, stride, num_factors_,
                    reinterpret_cast<float*>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SE3PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  size_t num_factors = NumFactors();

  // Step 1: Collect current transforms from state pointers into contiguous memory
  size_t num_blocks =
      (num_factors + kSE3PriorBlockSize - 1) / kSE3PriorBlockSize;
  collect_se3_transforms_kernel<<<num_blocks, kSE3PriorBlockSize, 0, stream>>>(
      state_pointers, num_factors, transforms_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute T_error = T_target^{-1} * T_current using cuBLAS
  // cuBLAS column-major convention with row-major data:
  //   Result_row = (A_col * B_col)^T where A_col = T_current^T, B_col = T_inv^T
  //   cublas computes: T_current^T * T_inv^T = (T_inv * T_current)^T
  //   Result_row = T_inv * T_current  (exactly what we want)
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;
  constexpr int stride = 16;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(transforms_current_.data()), mat_size,
      stride,
      reinterpret_cast<const float*>(observations_inverse_.data()), mat_size,
      stride, &beta,
      reinterpret_cast<float*>(transforms_error_.data()), mat_size, stride,
      num_factors));

  // Step 3: Compute residual = Log(T_error) using SE(3) logarithm map
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  constexpr size_t twist_stride = 6;
  ComputeLogSE3(
      stream, reinterpret_cast<const float*>(transforms_error_.data()),
      transform_pitch, transform_stride, twist_stride, num_factors, residuals);

  // Step 4: Compute Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    constexpr size_t jacobian_pitch = 6;
    constexpr size_t jacobian_stride = 36;
    ComputeJacobianRightInverseSE3(stream, residuals, twist_stride,
                                   jacobian_pitch, jacobian_stride, num_factors,
                                   jacobians);
  }

  return true;
}

}  // namespace cunls
