/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
 * @brief Fused kernel: collect L/R SO(2) rotations, compute
 *        R_error = (R_L^T * R_R) * Delta^{-1} in one pass.
 *
 * SO(2) is 2x2 so everything is done with scalar ops -- zero loops.
 * Replaces: collect + TransposeSO2 + 2x cuBLAS 2x2 GEMM.
 */
__global__ void collect_and_compute_so2_between_error_kernel(
    float const* const* state_pointers, const Matrix<2>* deltas,
    size_t num_factors, Matrix<2>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  // L^T * R  (L is orthogonal 2x2)
  const float l0=L[0], l1=L[1], l2=L[2], l3=L[3];
  const float r0=R[0], r1=R[1], r2=R[2], r3=R[3];

  const float m0 = l0*r0 + l2*r2;
  const float m1 = l0*r1 + l2*r3;
  const float m2 = l1*r0 + l3*r2;
  const float m3 = l1*r1 + l3*r3;

  // (L^T*R) * D  (D stores delta^{-1} in the cuBLAS convention)
  const float d0=D[0], d1=D[1], d2=D[2], d3=D[3];
  out[0] = m0*d0 + m1*d2;
  out[1] = m0*d1 + m1*d3;
  out[2] = m2*d0 + m3*d2;
  out[3] = m2*d1 + m3*d3;
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

SO2BetweenFactorBatch::SO2BetweenFactorBatch(const Matrix<2>* pose_deltas_ptr,
                                             size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool SO2BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Fused: collect L/R + compute (L^T * R) * D in one kernel
  collect_and_compute_so2_between_error_kernel<<<num_blocks, kBlockSize, 0,
                                                 stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

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
