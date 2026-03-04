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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/math/lie_math.h"

namespace cunls {

/// Number of threads per CUDA block for SE3 between cost kernel launches.
constexpr size_t block_size = 256;

/**
 * @brief CUDA kernel to collect poses from state pointers.
 *
 * Extracts SE(3) transformation matrices from the state pointer array
 * and stores them in contiguous memory for batch processing.
 *
 * @param state_pointers Array of state block pointers (device pointer to device pointers)
 * @param num_factors Number of factors in the batch
 * @param pose_left Output array for left poses (device pointer)
 * @param pose_right Output array for right poses (device pointer)
 */
__global__ void collect_poses_kernel(float const* const* state_pointers,
                                     size_t num_factors,
                                     SE3Transform* pose_left,
                                     SE3Transform* pose_right) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) {
    return;
  }

  auto pose_left_ptr =
      reinterpret_cast<const SE3Transform*>(state_pointers[2 * tid]);
  auto pose_right_ptr =
      reinterpret_cast<const SE3Transform*>(state_pointers[2 * tid + 1]);

  pose_left[tid] = *pose_left_ptr;
  pose_right[tid] = *pose_right_ptr;
}

/**
 * @brief Constructs the SE3 between factor batch.
 *
 * Allocates device memory for intermediate results and precomputes the
 * adjoint matrices of the pose deltas, which are reused during every
 * Evaluate() call for Jacobian computation.
 *
 * @param cublas_handle Reference to an externally-owned cuBLAS handle.
 * @param pose_deltas_ptr    Device pointer to SE3 pose delta constraints.
 * @param num_factors Number of factors in the batch.
 */
SE3BetweenFactorBatch::SE3BetweenFactorBatch(
    cuBLASHandle& cublas_handle, const SE3Transform* pose_deltas_ptr,
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

/**
 * @brief Evaluates residuals and optionally Jacobians for the SE3 between factor.
 *
 * Computes residual = Log(Delta * T_left^{-1} * T_right) for each pair of
 * poses. If jacobians is non-null, also computes the left and right pose
 * Jacobians using the SE(3) left/right inverse Jacobians and the precomputed
 * delta adjoint matrices.
 *
 * @param residuals   Output device pointer for residuals (6 floats per factor).
 * @param jacobians   Output device pointer for Jacobians (6x12 floats per
 *                    factor), or nullptr to skip Jacobian computation.
 * @param state_pointers  Device pointer to state block pointers.
 * @param stream      CUDA stream for asynchronous execution.
 * @return true on success.
 */
bool SE3BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                           float const* const* state_pointers,
                                           cudaStream_t stream) const {
  auto poses_left_ptr = poses_left_.data();
  auto poses_right_ptr = poses_right_.data();
  auto poses_left_inv_ptr = reinterpret_cast<float*>(
      poses_left_inverse_.data());

  size_t num_factors = NumFactors();

  size_t num_blocks = (num_factors + block_size - 1) / block_size;
  collect_poses_kernel<<<num_blocks, block_size, 0, stream>>>(
      state_pointers, num_factors, poses_left_ptr, poses_right_ptr);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  // Compute inverse of left poses: T_left_inv = T_left^{-1}
  ComputeInverseSE3(stream, reinterpret_cast<const float*>(poses_left_ptr),
                    pitch, stride, pitch, stride, num_factors,
                    poses_left_inv_ptr);

  auto handle = cublas_handle_.GetHandle(stream);

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr size_t mat_size = 4;

  // Compute relative transform: T_left := T_left_inv @ T_right
  // This overwrites poses_left_ with the relative transformation
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(poses_right_ptr), mat_size, stride,
      poses_left_inv_ptr, mat_size, stride, &beta,
      reinterpret_cast<float*>(poses_left_ptr), mat_size, stride,
      num_factors));

  // Compute relative transform: T_left_inv := Delta @ T_left =
  // = Delta @ T_left_inv @ Delta
  // This overwrites poses_left_inv
  auto delta_ptr = reinterpret_cast<const float*>(pose_deltas_ptr_);
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<float*>(poses_left_ptr), mat_size, stride, delta_ptr,
      mat_size, stride, &beta, poses_left_inv_ptr, mat_size, stride,
      num_factors));

  constexpr size_t twist_stride = 6;
  // Compute residual: residual = Log(Delta @ T_left_inv @ T_right)
  // Note: poses_left_ now contains the relative transform after the matrix
  // multiply
  ComputeLogSE3(stream, poses_left_inv_ptr, pitch, stride, twist_stride,
                num_factors, residuals);

  if (jacobians != nullptr) {
    ComputeLeftPoseJacobian(stream, residuals, jacobians);
    ComputeRightPoseJacobian(stream, residuals, &jacobians[6]);
  }

  return true;
}

/**
 * @brief Precomputes SE(3) adjoint matrices for all pose deltas.
 *
 * The adjoint of each delta transform is used during Jacobian computation
 * (specifically for the left pose Jacobian). This is called once during
 * construction and the results are cached in delta_adjoints_.
 *
 * @param stream CUDA stream for asynchronous execution.
 */
void SE3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  auto delta_ptr = reinterpret_cast<const float*>(pose_deltas_ptr_);
  auto delta_adjoints_ptr = reinterpret_cast<float*>(delta_adjoints_.data());

  constexpr size_t delta_pitch = 4;
  constexpr size_t delta_stride = 16;
  constexpr size_t delta_adjoint_pitch = 6;
  constexpr size_t delta_adjoint_stride = 36;

  ComputeAdjointSE3(stream, delta_ptr, delta_pitch, delta_stride,
                    delta_adjoint_pitch, delta_adjoint_stride,
                    num_factors_, delta_adjoints_ptr);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Computes the Jacobian with respect to the left SE(3) pose.
 *
 * The left Jacobian is: J_left = -Ad(Delta) * J_l^{-1}(residual)
 * where J_l^{-1} is the inverse left Jacobian of SE(3) evaluated at the
 * current residual, and Ad(Delta) is the precomputed adjoint of the delta.
 *
 * @param stream    CUDA stream for asynchronous execution.
 * @param residuals Current residual values (6 floats per factor).
 * @param jacobians Output Jacobian matrix (device pointer, modified in-place).
 *                  Writes to columns 0-5 of the full Jacobian block.
 */
void SE3BetweenFactorBatch::ComputeLeftPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  constexpr size_t jacobian_block_size = 6;
  constexpr size_t twist_stride = 6;
  constexpr size_t jacobian_pitch = 12;
  constexpr size_t jacobian_stride = 6 * 12;

  size_t num_factors = NumFactors();
  // Compute Jacobian with respect to first state block (left pose)
  // J_left = -J_l^{-1}(residual) where J_l is the left Jacobian of SE(3)
  ComputeJacobianLeftInverseSE3(stream, residuals, twist_stride, jacobian_pitch,
                                jacobian_stride, num_factors, jacobians);
  // Negate the Jacobian (chain rule: d/dT_left Log(T_left^{-1} @ T_right))
  ComputeNegateMatrix(stream, jacobians, jacobian_block_size,
                      jacobian_block_size, jacobian_pitch, jacobian_stride,
                      num_factors, jacobians);

  auto delta_adjoints_ptr = delta_adjoints_.data();
  constexpr size_t delta_adjoint_pitch = 6;
  constexpr size_t delta_adjoint_stride = 36;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr size_t mat_size = 6;

  auto handle = cublas_handle_.GetHandle(stream);

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(delta_adjoints_ptr), delta_adjoint_pitch,
      delta_adjoint_stride, reinterpret_cast<const float*>(jacobians),
      jacobian_pitch, jacobian_stride, &beta,
      reinterpret_cast<float*>(jacobians), jacobian_pitch, jacobian_stride,
      num_factors));
}

/**
 * @brief Computes the Jacobian with respect to the right SE(3) pose.
 *
 * The right Jacobian is: J_right = J_r^{-1}(residual) where J_r^{-1} is
 * the inverse right Jacobian of SE(3) evaluated at the current residual.
 *
 * @param stream    CUDA stream for asynchronous execution.
 * @param residuals Current residual values (6 floats per factor).
 * @param jacobians Output Jacobian matrix (device pointer, modified in-place).
 *                  Writes to columns 6-11 of the full Jacobian block.
 */
void SE3BetweenFactorBatch::ComputeRightPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  size_t num_factors = NumFactors();
  constexpr size_t twist_stride = 6;
  constexpr size_t jacobian_pitch = 12;
  constexpr size_t jacobian_stride = 6 * 12;
  // Compute Jacobian with respect to second state block (right pose)
  // J_right = J_r^{-1}(residual) where J_r is the right Jacobian of SE(3)
  ComputeJacobianRightInverseSE3(stream, residuals, twist_stride,
                                 jacobian_pitch, jacobian_stride,
                                 num_factors, jacobians);
}

}  // namespace cunls
