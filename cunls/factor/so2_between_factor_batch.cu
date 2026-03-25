/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/so2_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// SO(2) between factor layout constants
constexpr size_t kSO2RotationStride = 4;
constexpr size_t kSO2AngleStride = 1;

/**
 * @brief Gather left and right SO(2) rotations from state pointers.
 */
__global__ void collect_so2_poses_kernel(float const* const* state_pointers,
                                       size_t num_factors, Matrix<2>* pose_left,
                                       Matrix<2>* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  pose_left[tid] = *reinterpret_cast<const Matrix<2>*>(state_pointers[2 * tid]);
  pose_right[tid] =
      *reinterpret_cast<const Matrix<2>*>(state_pointers[2 * tid + 1]);
}

/**
 * @brief Write exact Jacobian for SO(2) between factor.
 *
 * The between factor has residual dim 1 and two state blocks of dim 1 each.
 * Since SO(2) is abelian, J_r^{-1} = 1 and Ad(Delta) = 1, so:
 *   H_left  = -J_l^{-1} * Ad(Delta) = -1
 *   H_right =  J_r^{-1}             =  1
 */
__global__ void so2_between_jacobian_kernel(float* jacobians,
                                            size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  float* J = jacobians + tid * 2;
  J[0] = -1.f;
  J[1] = 1.f;
}

SO2BetweenFactorBatch::SO2BetweenFactorBatch(cuBLASHandle& cublas_handle,
                                             const Matrix<2>* pose_deltas_ptr,
                                             size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool SO2BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Step 1: Gather left and right rotations
  collect_so2_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute R_left^T (= R_left^{-1} for orthogonal matrices)
  ComputeTransposeSO2(stream,
                      reinterpret_cast<const float*>(poses_left_.data()),
                      kSO2RotationStride, kSO2RotationStride, num_factors,
                      reinterpret_cast<float*>(poses_left_inverse_.data()));

  // Step 3: Compute R_left^T * R_right via batched GEMM
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 2;
  constexpr int stride = 4;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_.data()), mat_size, stride,
      reinterpret_cast<const float*>(poses_left_inverse_.data()), mat_size,
      stride, &beta, reinterpret_cast<float*>(poses_left_.data()), mat_size,
      stride, num_factors));

  // Step 4: Multiply by delta^{-1}
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride,
      &beta, reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size,
      stride, num_factors));

  // Step 5: residual = Log(R_error)
  ComputeLogSO2(stream,
                reinterpret_cast<const float*>(poses_left_inverse_.data()),
                kSO2RotationStride, kSO2AngleStride, num_factors, residuals);

  // Step 6: Jacobian = [-1, 1]
  if (jacobians != nullptr) {
    so2_between_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
