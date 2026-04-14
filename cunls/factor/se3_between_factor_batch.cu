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
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

/// Number of threads per CUDA block for SE3 between cost kernel launches.
constexpr size_t block_size = 256;

/**
 * @brief Fused kernel: collect L/R SE3 transforms, compute T_left^{-1},
 *        then compute error = Delta * T_left^{-1} * T_right in one pass.
 *
 * Replaces: collect_poses_kernel + ComputeInverseSE3 + 2x cuBLAS SGEMM.
 * Fully unrolled with SE3 structure exploitation: last row is always [0 0 0 1],
 * reducing 128 FMAs to ~78 FMAs + additions.  Zero local-memory arrays.
 */
__global__ void collect_and_compute_se3_between_error_kernel(
    float const* const* state_pointers, const SE3Transform* deltas,
    size_t num_factors, SE3Transform* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  // Load L rotation (3x3) and translation (3x1)
  const float l00 = L[0], l01 = L[1], l02 = L[2], l03 = L[3];
  const float l10 = L[4], l11 = L[5], l12 = L[6], l13 = L[7];
  const float l20 = L[8], l21 = L[9], l22 = L[10], l23 = L[11];

  // L_inv rotation = L_rot^T (transpose), L_inv translation = -L_rot^T * L_t
  // i00 = l00, i01 = l10, i02 = l20  (transposed)
  // i10 = l01, i11 = l11, i12 = l21
  // i20 = l02, i21 = l12, i22 = l22
  const float i03 = -(l00*l03 + l10*l13 + l20*l23);
  const float i13 = -(l01*l03 + l11*l13 + l21*l23);
  const float i23 = -(l02*l03 + l12*l13 + l22*l23);

  // Load R (3x4 active part)
  const float r00 = R[0], r01 = R[1], r02 = R[2], r03 = R[3];
  const float r10 = R[4], r11 = R[5], r12 = R[6], r13 = R[7];
  const float r20 = R[8], r21 = R[9], r22 = R[10], r23 = R[11];

  // temp = L_inv * R  (only compute 3x4 active part; row 3 = [0 0 0 1])
  // Using transposed L indices directly
  const float t00 = l00*r00 + l10*r10 + l20*r20;
  const float t01 = l00*r01 + l10*r11 + l20*r21;
  const float t02 = l00*r02 + l10*r12 + l20*r22;
  const float t03 = l00*r03 + l10*r13 + l20*r23 + i03;

  const float t10 = l01*r00 + l11*r10 + l21*r20;
  const float t11 = l01*r01 + l11*r11 + l21*r21;
  const float t12 = l01*r02 + l11*r12 + l21*r22;
  const float t13 = l01*r03 + l11*r13 + l21*r23 + i13;

  const float t20 = l02*r00 + l12*r10 + l22*r20;
  const float t21 = l02*r01 + l12*r11 + l22*r21;
  const float t22 = l02*r02 + l12*r12 + l22*r22;
  const float t23 = l02*r03 + l12*r13 + l22*r23 + i23;

  // Load D (3x4 active part)
  const float d00 = D[0], d01 = D[1], d02 = D[2], d03 = D[3];
  const float d10 = D[4], d11 = D[5], d12 = D[6], d13 = D[7];
  const float d20 = D[8], d21 = D[9], d22 = D[10], d23 = D[11];

  // error = D * temp  (only compute 3x4 active part; row 3 = [0 0 0 1])
  out[0]  = d00*t00 + d01*t10 + d02*t20;
  out[1]  = d00*t01 + d01*t11 + d02*t21;
  out[2]  = d00*t02 + d01*t12 + d02*t22;
  out[3]  = d00*t03 + d01*t13 + d02*t23 + d03;

  out[4]  = d10*t00 + d11*t10 + d12*t20;
  out[5]  = d10*t01 + d11*t11 + d12*t21;
  out[6]  = d10*t02 + d11*t12 + d12*t22;
  out[7]  = d10*t03 + d11*t13 + d12*t23 + d13;

  out[8]  = d20*t00 + d21*t10 + d22*t20;
  out[9]  = d20*t01 + d21*t11 + d22*t21;
  out[10] = d20*t02 + d21*t12 + d22*t22;
  out[11] = d20*t03 + d21*t13 + d22*t23 + d23;

  out[12] = 0.0f;
  out[13] = 0.0f;
  out[14] = 0.0f;
  out[15] = 1.0f;
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
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + block_size - 1) / block_size;

  // Fused: collect L/R + compute Delta * L^{-1} * R in one kernel
  collect_and_compute_se3_between_error_kernel<<<num_blocks, block_size, 0,
                                                 stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  constexpr size_t twist_stride = 6;
  ComputeLogSE3(stream,
                reinterpret_cast<const float*>(poses_left_inverse_.data()),
                pitch, stride, twist_stride, num_factors, residuals);

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

  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));

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
