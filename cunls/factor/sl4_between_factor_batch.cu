/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/sl4_between_factor_batch.h"
#include "cunls/math/sl_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

__global__ void collect_sl4_poses_kernel(float const* const* state_pointers,
                                       size_t num_factors, SL4Transform* pose_left,
                                       SL4Transform* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  pose_left[tid] = *reinterpret_cast<const SL4Transform*>(state_pointers[2 * tid]);
  pose_right[tid] =
      *reinterpret_cast<const SL4Transform*>(state_pointers[2 * tid + 1]);
}

__global__ void sl4_between_left_jacobian_kernel(const float* neg_adjoint,
                                                 float* jacobians,
                                                 size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  const float* A = neg_adjoint + tid * 225;
  float* J = jacobians + tid * 450;
#pragma unroll
  for (int r = 0; r < 15; ++r) {
#pragma unroll
    for (int c = 0; c < 15; ++c) {
      J[r * 30 + c] = A[r * 15 + c];
    }
  }
}

__global__ void sl4_between_right_jacobian_identity_kernel(float* jacobians,
                                                           size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  float* J = jacobians + tid * 450;
#pragma unroll
  for (int r = 0; r < 15; ++r) {
#pragma unroll
    for (int c = 0; c < 15; ++c) {
      J[r * 30 + 15 + c] = (r == c) ? 1.f : 0.f;
    }
  }
}

SL4BetweenFactorBatch::SL4BetweenFactorBatch(cuBLASHandle& cublas_handle,
                                            const SL4Transform* pose_deltas_ptr,
                                            size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      delta_adjoints_(num_factors * 225),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void SL4BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  constexpr size_t delta_pitch = 4;
  constexpr size_t delta_stride = 16;
  constexpr size_t adjoint_pitch = 15;
  constexpr size_t adjoint_stride = 225;
  ComputeAdjointSL4(stream, reinterpret_cast<const float*>(pose_deltas_ptr_),
                    delta_pitch, delta_stride, adjoint_pitch, adjoint_stride,
                    num_factors_, delta_adjoints_.data());
  ComputeNegateMatrix15x15(stream, delta_adjoints_.data(), adjoint_pitch,
                           adjoint_stride, num_factors_, delta_adjoints_.data());
}

bool SL4BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;
  collect_sl4_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  ComputeInverseSL4(stream, reinterpret_cast<const float*>(poses_left_.data()),
                    pitch, stride, pitch, stride, num_factors,
                    reinterpret_cast<float*>(poses_left_inverse_.data()));

  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_.data()), mat_size, stride,
      reinterpret_cast<const float*>(poses_left_inverse_.data()), mat_size,
      stride, &beta, reinterpret_cast<float*>(poses_left_.data()), mat_size,
      stride, num_factors));

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride,
      &beta, reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size,
      stride, num_factors));

  constexpr size_t twist_stride = 15;
  ComputeLogSL4(stream, reinterpret_cast<const float*>(poses_left_inverse_.data()),
                pitch, stride, twist_stride, num_factors, residuals);

  if (jacobians != nullptr) {
    sl4_between_left_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        delta_adjoints_.data(), jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    sl4_between_right_jacobian_identity_kernel<<<num_blocks, kBlockSize, 0,
                                                 stream>>>(jacobians,
                                                           num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
