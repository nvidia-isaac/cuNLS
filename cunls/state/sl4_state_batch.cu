/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/helper.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/state/sl4_state_batch.h"

namespace cunls {

void SL4StateBatch::Plus(const float* x, const float* delta, float* x_plus_delta,
                         cudaStream_t stream) {
  size_t n = NumStateBlocks();
  delta_transforms_.resize(n);
  twists_.resize(n * 15);

  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(twists_.data(), delta, n * 15 * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  constexpr size_t twist_stride = 15;
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  ComputeExpSL4(stream, reinterpret_cast<const float*>(twists_.data()),
                twist_stride, transform_pitch, transform_stride, n,
                reinterpret_cast<float*>(delta_transforms_.data()));

  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;
  constexpr int stride = 16;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(delta_transforms_.data()), mat_size, stride,
      x, mat_size, stride, &beta, x_plus_delta, mat_size, stride, n));
}

}  // namespace cunls
