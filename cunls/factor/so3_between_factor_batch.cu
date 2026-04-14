/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
 * @brief Fused kernel: collect L/R rotations from state pointers, compute
 *        R_error = Delta^T * (R_left^T * R_right) in one pass.
 *
 * Replaces collect + TransposeSO3 + two cuBLAS GEMM calls.
 * Fully unrolled: two 3x3 multiplies use 54 FMAs with zero local memory.
 */
__global__ void collect_and_compute_so3_between_error_kernel(
    float const* const* state_pointers, const Matrix<3>* deltas,
    size_t num_factors, Matrix<3>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  const float l0 = L[0], l1 = L[1], l2 = L[2];
  const float l3 = L[3], l4 = L[4], l5 = L[5];
  const float l6 = L[6], l7 = L[7], l8 = L[8];

  const float r0 = R[0], r1 = R[1], r2 = R[2];
  const float r3 = R[3], r4 = R[4], r5 = R[5];
  const float r6 = R[6], r7 = R[7], r8 = R[8];

  // temp = L^T * R  (27 FMAs, all in registers)
  const float m0 = l0*r0 + l3*r3 + l6*r6;
  const float m1 = l0*r1 + l3*r4 + l6*r7;
  const float m2 = l0*r2 + l3*r5 + l6*r8;
  const float m3 = l1*r0 + l4*r3 + l7*r6;
  const float m4 = l1*r1 + l4*r4 + l7*r7;
  const float m5 = l1*r2 + l4*r5 + l7*r8;
  const float m6 = l2*r0 + l5*r3 + l8*r6;
  const float m7 = l2*r1 + l5*r4 + l8*r7;
  const float m8 = l2*r2 + l5*r5 + l8*r8;

  const float d0 = D[0], d1 = D[1], d2 = D[2];
  const float d3 = D[3], d4 = D[4], d5 = D[5];
  const float d6 = D[6], d7 = D[7], d8 = D[8];

  // error = temp * D^T  (27 FMAs)
  out[0] = m0*d0 + m1*d1 + m2*d2;
  out[1] = m0*d3 + m1*d4 + m2*d5;
  out[2] = m0*d6 + m1*d7 + m2*d8;
  out[3] = m3*d0 + m4*d1 + m5*d2;
  out[4] = m3*d3 + m4*d4 + m5*d5;
  out[5] = m3*d6 + m4*d7 + m5*d8;
  out[6] = m6*d0 + m7*d1 + m8*d2;
  out[7] = m6*d3 + m7*d4 + m8*d5;
  out[8] = m6*d6 + m7*d7 + m8*d8;
}

/**
 * @brief Fused kernel: compute left Jacobian block with delta adjoint multiply.
 *
 * Reads -J_l^{-1}(residual) from the left 3x3 block of jacobians (pitch 6),
 * computes D * J in-place. Fully unrolled.
 */
__global__ void apply_delta_adjoint_to_left_jacobian_kernel(
    const Matrix<3>* delta_adjoints, float* jacobians,
    size_t num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ D = delta_adjoints[tid].data();
  float* __restrict__ J = &jacobians[tid * 18];

  const float d0 = D[0], d1 = D[1], d2 = D[2];
  const float d3 = D[3], d4 = D[4], d5 = D[5];
  const float d6 = D[6], d7 = D[7], d8 = D[8];

  // Load the left 3x3 block (columns 0-2 at pitch 6)
  const float j00 = J[0],  j01 = J[1],  j02 = J[2];
  const float j10 = J[6],  j11 = J[7],  j12 = J[8];
  const float j20 = J[12], j21 = J[13], j22 = J[14];

  // result = D * src  (27 FMAs)
  J[0]  = d0*j00 + d1*j10 + d2*j20;
  J[1]  = d0*j01 + d1*j11 + d2*j21;
  J[2]  = d0*j02 + d1*j12 + d2*j22;
  J[6]  = d3*j00 + d4*j10 + d5*j20;
  J[7]  = d3*j01 + d4*j11 + d5*j21;
  J[8]  = d3*j02 + d4*j12 + d5*j22;
  J[12] = d6*j00 + d7*j10 + d8*j20;
  J[13] = d6*j01 + d7*j11 + d8*j21;
  J[14] = d6*j02 + d7*j12 + d8*j22;
}

SO3BetweenFactorBatch::SO3BetweenFactorBatch(const Matrix<3>* pose_deltas_ptr,
                                               size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      delta_adjoints_(num_factors),
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

  // Fused: collect L/R + compute R_error = Delta^T * (L^T * R) in one kernel
  collect_and_compute_so3_between_error_kernel<<<num_blocks, kBlockSize, 0,
                                                 stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // residual = Log(R_error)
  constexpr size_t rotation_pitch = 3;
  ComputeLogSO3(stream, reinterpret_cast<const float*>(poses_left_inverse_.data()),
                rotation_pitch, kRotationStride, kTwistStride, num_factors,
                residuals);

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

  // Fused: multiply delta_adjoint * J_block in-place (replaces cuBLAS GEMM)
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;
  apply_delta_adjoint_to_left_jacobian_kernel<<<num_blocks, kBlockSize, 0,
                                                stream>>>(
      delta_adjoints_.data(), jacobians, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
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
