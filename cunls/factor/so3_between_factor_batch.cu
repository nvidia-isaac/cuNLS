/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/so3_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// SO(3) between factor layout constants
constexpr size_t kRotationStride = 9;
constexpr size_t kTwistStride = 3;

/**
 * @brief Gather left and right SO(3) rotations from state pointers.
 */
__global__ void collect_so3_poses_kernel(float const* const* state_pointers,
                                       size_t num_factors, Matrix<3>* pose_left,
                                       Matrix<3>* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }
  pose_left[tid] = *reinterpret_cast<const Matrix<3>*>(state_pointers[2 * tid]);
  pose_right[tid] =
      *reinterpret_cast<const Matrix<3>*>(state_pointers[2 * tid + 1]);
}

SO3BetweenFactorBatch::SO3BetweenFactorBatch(cuBLASHandle& cublas_handle,
                                               const Matrix<3>* pose_deltas_ptr,
                                               size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      delta_adjoints_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void SO3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(
      delta_adjoints_.data(), pose_deltas_ptr_,
      num_factors_ * sizeof(Matrix<3>), cudaMemcpyDeviceToDevice, stream));
}

bool SO3BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Step 1: Gather left and right rotations
  collect_so3_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data(), poses_right_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: Compute R_left^T (= R_left^{-1})
  ComputeTransposeSO3(stream,
                      reinterpret_cast<const float*>(poses_left_.data()),
                      kRotationStride, kRotationStride, num_factors,
                      reinterpret_cast<float*>(poses_left_inverse_.data()));

  // Step 3: Compute (R_left^T * R_right) and then multiply by delta^T
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_.data()), mat_size, stride,
      reinterpret_cast<const float*>(poses_left_inverse_.data()), mat_size,
      stride, &beta, reinterpret_cast<float*>(poses_left_.data()), mat_size,
      stride, num_factors));

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_.data()), mat_size, stride,
      reinterpret_cast<const float*>(pose_deltas_ptr_), mat_size, stride, &beta,
      reinterpret_cast<float*>(poses_left_inverse_.data()), mat_size, stride,
      num_factors));

  // Step 4: residual = Log(R_error)
  constexpr size_t rotation_pitch = 3;
  ComputeLogSO3(stream, reinterpret_cast<const float*>(poses_left_inverse_.data()),
                rotation_pitch, kRotationStride, kTwistStride, num_factors,
                residuals);

  // Step 5: Compute left and right Jacobian blocks
  if (jacobians != nullptr) {
    ComputeLeftPoseJacobian(stream, residuals, jacobians);
    ComputeRightPoseJacobian(stream, residuals, &jacobians[3]);
  }

  return true;
}

void SO3BetweenFactorBatch::ComputeLeftPoseJacobian(cudaStream_t stream,
                                                    const float* residuals,
                                                    float* jacobians) const {
  constexpr size_t jacobian_block_size = 3;
  constexpr size_t jacobian_pitch = 6;
  constexpr size_t jacobian_stride = 18;
  size_t num_factors = NumFactors();

  // J_left = -delta^T * J_l^{-1}(residual)
  ComputeJacobianLeftInverseSO3(stream, residuals, kTwistStride, jacobian_pitch,
                                jacobian_stride, num_factors, jacobians);
  ComputeNegateMatrix(stream, jacobians, jacobian_block_size, jacobian_block_size,
                      jacobian_pitch, jacobian_stride, num_factors, jacobians);

  auto delta_adjoints_ptr = reinterpret_cast<const float*>(delta_adjoints_.data());
  constexpr size_t delta_adjoint_pitch = 3;
  constexpr size_t delta_adjoint_stride = 9;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr size_t mat_size = 3;

  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_adjoints_ptr, delta_adjoint_pitch, delta_adjoint_stride,
      reinterpret_cast<const float*>(jacobians), jacobian_pitch, jacobian_stride,
      &beta, reinterpret_cast<float*>(jacobians), jacobian_pitch,
      jacobian_stride, num_factors));
}

void SO3BetweenFactorBatch::ComputeRightPoseJacobian(cudaStream_t stream,
                                                     const float* residuals,
                                                     float* jacobians) const {
  size_t num_factors = NumFactors();
  constexpr size_t jacobian_pitch = 6;
  constexpr size_t jacobian_stride = 18;

  // J_right = J_r^{-1}(residual)
  ComputeJacobianRightInverseSO3(stream, residuals, kTwistStride, jacobian_pitch,
                                 jacobian_stride, num_factors, jacobians);
}

}  // namespace cunls
