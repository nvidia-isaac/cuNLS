/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sl4_prior_factor_batch.h"
#include "cunls/math/sl_lie_math.h"

namespace cunls {

constexpr size_t kSL4PriorBlockSize = 256;

__global__ void collect_sl4_transforms_kernel(float const* const* state_pointers,
                                              size_t num_factors,
                                              SL4Transform* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  auto p = reinterpret_cast<const SL4Transform*>(state_pointers[tid]);
  transforms[tid] = *p;
}

SL4PriorFactorBatch::SL4PriorFactorBatch(cuBLASHandle& cublas_handle,
                                         const SL4Transform* observations_ptr,
                                         size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      cublas_handle_(cublas_handle),
      transforms_current_(num_factors),
      transforms_error_(num_factors) {
  CudaStream stream;
  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  ComputeInverseSL4(stream.GetStream(),
                    reinterpret_cast<const float*>(observations_ptr_), pitch,
                    stride, pitch, stride, num_factors_,
                    reinterpret_cast<float*>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SL4PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                   float const* const* state_pointers,
                                   cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSL4PriorBlockSize - 1) / kSL4PriorBlockSize;
  collect_sl4_transforms_kernel<<<num_blocks, kSL4PriorBlockSize, 0, stream>>>(
      state_pointers, num_factors, transforms_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

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

  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  constexpr size_t twist_stride = 15;
  ComputeLogSL4(stream,
                reinterpret_cast<const float*>(transforms_error_.data()),
                transform_pitch, transform_stride, twist_stride, num_factors,
                residuals);

  if (jacobians != nullptr) {
    constexpr size_t jacobian_pitch = 15;
    constexpr size_t jacobian_stride = 225;
    FillIdentity15x15(stream, num_factors, jacobians, jacobian_pitch,
                      jacobian_stride);
  }

  return true;
}

}  // namespace cunls
