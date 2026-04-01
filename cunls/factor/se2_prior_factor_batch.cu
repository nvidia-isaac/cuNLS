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
#include "cunls/factor/se2_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kSE2PriorBlockSize = 256;

// SE(2) prior factor layout constants
constexpr size_t kSE2TransformStride = 9;
constexpr size_t kSE2TangentStride = 3;
constexpr size_t kSE2JacobianStride = 9;

/**
 * @brief Collect SE(2) transforms from state pointers into contiguous memory.
 */
__global__ void collect_se2_transforms_kernel(float const* const* state_pointers,
                                              size_t num_factors,
                                              Matrix<3>* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) {
    return;
  }

  auto transform_ptr = reinterpret_cast<const Matrix<3>*>(state_pointers[tid]);
  transforms[tid] = *transform_ptr;
}

SE2PriorFactorBatch::SE2PriorFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<3>* observations_ptr,
    size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      cublas_handle_(cublas_handle),
      transforms_current_(num_factors),
      transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all observations
  CudaStream stream;
  ComputeInverseSE2(stream.GetStream(),
                    reinterpret_cast<const float*>(observations_ptr_),
                    kSE2TransformStride, kSE2TransformStride, num_factors_,
                    reinterpret_cast<float*>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SE2PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSE2PriorBlockSize - 1) / kSE2PriorBlockSize;

  // Step 1: Gather current transforms from state pointers
  collect_se2_transforms_kernel<<<num_blocks, kSE2PriorBlockSize, 0, stream>>>(
      state_pointers, num_factors, transforms_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: T_error = T_target^{-1} * T_current via batched cuBLAS GEMM
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(transforms_current_.data()), mat_size,
      stride,
      reinterpret_cast<const float*>(observations_inverse_.data()), mat_size,
      stride, &beta,
      reinterpret_cast<float*>(transforms_error_.data()), mat_size, stride,
      num_factors));

  // Step 3: residual = Log(T_error)
  ComputeLogSE2(stream,
                reinterpret_cast<const float*>(transforms_error_.data()),
                kSE2TransformStride, kSE2TangentStride, num_factors,
                residuals);

  // Step 4: Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSE2(stream, residuals, kSE2TangentStride,
                                   kSE2JacobianStride, num_factors, jacobians);
  }

  return true;
}

}  // namespace cunls
