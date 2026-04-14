/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// SE(2) between factor layout constants
constexpr size_t kSE2TransformStride = 9;
constexpr size_t kSE2TangentStride = 3;

/**
 * @brief Fused kernel: collect L/R SE(2), compute T_left^{-1} * T_right * Delta
 *        in one pass. SE(2) structure: last row [0 0 1].
 *
 * Replaces: collect + InverseSE2 + 2x cuBLAS 3x3 GEMM.
 */
__global__ void collect_and_compute_se2_between_error_kernel(
    float const* const* state_pointers, const Matrix<3>* deltas,
    size_t num_factors, Matrix<3>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  // L: [[c,-s,tx],[s,c,ty],[0,0,1]]
  const float lc = L[0], ls = L[3], ltx = L[2], lty = L[5];

  // L_inv: [[c,s,-c*tx-s*ty],[-s,c,s*tx-c*ty],[0,0,1]]
  const float i02 = -(lc*ltx + ls*lty);
  const float i12 = ls*ltx - lc*lty;

  // R active part
  const float r00=R[0], r01=R[1], r02=R[2];
  const float r10=R[3], r11=R[4], r12=R[5];

  // temp = L_inv * R (3x3, last row [0 0 1])
  const float t00 = lc*r00 + ls*r10;
  const float t01 = lc*r01 + ls*r11;
  const float t02 = lc*r02 + ls*r12 + i02;
  const float t10 = -ls*r00 + lc*r10;
  const float t11 = -ls*r01 + lc*r11;
  const float t12 = -ls*r02 + lc*r12 + i12;

  // error = D * temp (row-major: D_row * temp_row)
  // cuBLAS col-major: C_col = A_col * B_col => C_row = B_row * A_row = D * temp
  const float d00=D[0], d01=D[1], d02=D[2];
  const float d10=D[3], d11=D[4], d12=D[5];
  out[0] = d00*t00 + d01*t10;
  out[1] = d00*t01 + d01*t11;
  out[2] = d00*t02 + d01*t12 + d02;
  out[3] = d10*t00 + d11*t10;
  out[4] = d10*t01 + d11*t11;
  out[5] = d10*t02 + d11*t12 + d12;
  out[6] = 0.0f; out[7] = 0.0f; out[8] = 1.0f;
}

/**
 * @brief Compute exact Jacobian blocks for the SE(2) between factor.
 *
 * The residual is r = Log(Delta * T_left^{-1} * T_right).
 *
 * Right-perturbation Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)    (3x3, columns 0-2)
 *   H_right =  J_r^{-1}(r)                 (3x3, columns 3-5)
 *
 * where J_r^{-1} is the SE(2) LogmapDerivative, J_l^{-1}(xi) = J_r^{-1}(-xi),
 * and Ad(Delta) is the SE(2) adjoint of the measurement-inverse transform:
 *   Ad([[c,-s,tx],[s,c,ty],[0,0,1]]) = [[c,-s,ty],[s,c,-tx],[0,0,1]].
 *
 * @param residuals  Residual vectors (3 floats per factor)
 * @param deltas     Delta transforms (9 floats per factor, row-major 3x3)
 * @param delta_stride Stride between consecutive delta transforms
 * @param jacobians  Output 3x6 Jacobian matrices (18 floats per factor)
 * @param num_factors Number of factors in the batch
 */
__global__ void se2_between_jacobian_kernel(const float* residuals,
                                            const float* deltas,
                                            size_t delta_stride,
                                            float* jacobians,
                                            size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }

  const float* r = residuals + tid * 3;
  float v1 = r[0], v2 = r[1], alpha = r[2];

  // Read Delta transform: [[c,-s,tx],[s,c,ty],[0,0,1]] (row-major)
  const float* D = deltas + tid * delta_stride;
  float dc = D[0], ds = D[3], dtx = D[2], dty = D[5];

  // J_r^{-1}(r) and J_l^{-1}(r) = J_r^{-1}(-r)
  float Jl[9], Jr[9];

  if (fabsf(alpha) > 1e-3f) {
    float ai = 1.0f / alpha;
    float hch = 0.5f * sinf(alpha) / (1.0f - cosf(alpha));
    float ach = alpha * hch;
    float ha = 0.5f * alpha;

    Jr[0] = ach;
    Jr[1] = -ha;
    Jr[2] = v1 * ai - v1 * hch + 0.5f * v2;
    Jr[3] = ha;
    Jr[4] = ach;
    Jr[5] = v2 * ai - 0.5f * v1 - v2 * hch;
    Jr[6] = 0.0f;
    Jr[7] = 0.0f;
    Jr[8] = 1.0f;

    Jl[0] = ach;
    Jl[1] = ha;
    Jl[2] = v1 * ai - v1 * hch - 0.5f * v2;
    Jl[3] = -ha;
    Jl[4] = ach;
    Jl[5] = v2 * ai + 0.5f * v1 - v2 * hch;
    Jl[6] = 0.0f;
    Jl[7] = 0.0f;
    Jl[8] = 1.0f;
  } else {
    Jr[0] = 1.0f;
    Jr[1] = 0.0f;
    Jr[2] = 0.5f * v2;
    Jr[3] = 0.0f;
    Jr[4] = 1.0f;
    Jr[5] = -0.5f * v1;
    Jr[6] = 0.0f;
    Jr[7] = 0.0f;
    Jr[8] = 1.0f;

    Jl[0] = 1.0f;
    Jl[1] = 0.0f;
    Jl[2] = -0.5f * v2;
    Jl[3] = 0.0f;
    Jl[4] = 1.0f;
    Jl[5] = 0.5f * v1;
    Jl[6] = 0.0f;
    Jl[7] = 0.0f;
    Jl[8] = 1.0f;
  }

  // H_left = -J_l^{-1} * Ad(Delta)
  // Ad(Delta) = [[dc, -ds, dty], [ds, dc, -dtx], [0, 0, 1]]
  float* J = jacobians + tid * 18;

#pragma unroll
  for (int i = 0; i < 3; ++i) {
    float a = Jl[i * 3 + 0];
    float b = Jl[i * 3 + 1];
    float c = Jl[i * 3 + 2];
    J[i * 6 + 0] = -(a * dc + b * ds);
    J[i * 6 + 1] = -(a * (-ds) + b * dc);
    J[i * 6 + 2] = -(a * dty + b * (-dtx) + c);
  }

  // H_right = J_r^{-1}(r)
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    J[i * 6 + 3] = Jr[i * 3 + 0];
    J[i * 6 + 4] = Jr[i * 3 + 1];
    J[i * 6 + 5] = Jr[i * 3 + 2];
  }
}

SE2BetweenFactorBatch::SE2BetweenFactorBatch(const Matrix<3>* pose_deltas_ptr,
                                             size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool SE2BetweenFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Fused: collect L/R + compute (L^{-1} * R) * Delta in one kernel
  collect_and_compute_se2_between_error_kernel<<<num_blocks, kBlockSize, 0,
                                                 stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 5: residual = Log(error)
  ComputeLogSE2(stream,
                reinterpret_cast<const float*>(poses_left_inverse_.data()),
                kSE2TransformStride, kSE2TangentStride, num_factors,
                residuals);

  // Step 6: jacobians
  //   H_left  = -J_l^{-1}(r) * Ad(Delta)
  //   H_right =  J_r^{-1}(r)
  if (jacobians != nullptr) {
    se2_between_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        residuals, reinterpret_cast<const float*>(pose_deltas_ptr_),
        kSE2TransformStride, jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
