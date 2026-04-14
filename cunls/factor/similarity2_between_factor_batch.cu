/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity2_between_factor_batch.h"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

// Sim(2) between factor layout constants
constexpr size_t kSim2TransformStride = 9;
constexpr size_t kSim2TangentStride = 4;

/**
 * @brief Fused kernel: given pre-computed L^{-1}, read R from state_pointers,
 *        compute error = D * (L_inv * R). Fully unrolled 3x3.
 *
 * Replaces 2x cuBLAS 3x3 GEMM.
 */
__global__ void compute_sim2_between_error_kernel(
    float const* const* state_pointers, const Matrix<3>* left_inverse,
    const Matrix<3>* deltas, size_t num_factors, Matrix<3>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ I = left_inverse[tid].data();
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  const float i0=I[0], i1=I[1], i2=I[2];
  const float i3=I[3], i4=I[4], i5=I[5];
  const float i6=I[6], i7=I[7], i8=I[8];

  const float r0=R[0], r1=R[1], r2=R[2];
  const float r3=R[3], r4=R[4], r5=R[5];
  const float r6=R[6], r7=R[7], r8=R[8];

  // temp = L_inv * R  (full 3x3, 27 FMAs)
  const float t00 = i0*r0+i1*r3+i2*r6;
  const float t01 = i0*r1+i1*r4+i2*r7;
  const float t02 = i0*r2+i1*r5+i2*r8;
  const float t10 = i3*r0+i4*r3+i5*r6;
  const float t11 = i3*r1+i4*r4+i5*r7;
  const float t12 = i3*r2+i4*r5+i5*r8;
  const float t20 = i6*r0+i7*r3+i8*r6;
  const float t21 = i6*r1+i7*r4+i8*r7;
  const float t22 = i6*r2+i7*r5+i8*r8;

  // error = D * temp  (full 3x3, 27 FMAs)
  const float d0=D[0], d1=D[1], d2=D[2];
  const float d3=D[3], d4=D[4], d5=D[5];
  const float d6=D[6], d7=D[7], d8=D[8];

  out[0] = d0*t00+d1*t10+d2*t20;
  out[1] = d0*t01+d1*t11+d2*t21;
  out[2] = d0*t02+d1*t12+d2*t22;
  out[3] = d3*t00+d4*t10+d5*t20;
  out[4] = d3*t01+d4*t11+d5*t21;
  out[5] = d3*t02+d4*t12+d5*t22;
  out[6] = d6*t00+d7*t10+d8*t20;
  out[7] = d6*t01+d7*t11+d8*t21;
  out[8] = d6*t02+d7*t12+d8*t22;
}

/**
 * @brief Gather left Sim(2) poses from state pointers.
 */
__global__ void collect_sim2_left_poses_kernel(
    float const* const* state_pointers, size_t num_factors,
    Matrix<3>* pose_left) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;
  pose_left[tid] = *reinterpret_cast<const Matrix<3>*>(state_pointers[2 * tid]);
}

/**
 * @brief Compute exact Jacobian blocks for the Sim(2) between factor.
 *
 * Residual r = Log(Delta * T_left^{-1} * T_right), tangent = [u1,u2,w,lambda].
 *
 * Right-perturbation Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)    (4x4, columns 0-3)
 *   H_right =  J_r^{-1}(r)                 (4x4, columns 4-7)
 *
 * where J_l^{-1}(xi) = J_r^{-1}(-xi), and Ad(Delta) is the Sim(2) adjoint:
 *   Ad = [[s*R, [s*ty,-s*tx]^T, [-s*tx,-s*ty]^T],
 *         [0 0, 1, 0],
 *         [0 0, 0, 1]]
 */
__global__ void __launch_bounds__(256, 6)
    sim2_between_jacobian_kernel(const float* __restrict__ residuals,
                                 const float* __restrict__ deltas,
                                 size_t delta_stride,
                                 float* __restrict__ jacobians,
                                 size_t num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) {
    return;
  }

  const float* r = residuals + tid * 4;
  float u1 = r[0], u2 = r[1], w = r[2], lam = r[3];

  const float* D = deltas + tid * delta_stride;
  float dc = D[0], ds = D[3], dtx = D[2], dty = D[5];
  float scale = 1.0f / D[8];

  // Ad(Delta) -- sparse: rows 2,3 are [0,0,1,0] and [0,0,0,1]
  float a00 = scale * dc,  a01 = -scale * ds;
  float a02 = scale * dty, a03 = -scale * dtx;
  float a10 = scale * ds,  a11 = scale * dc;
  float a12 = -scale * dtx, a13 = -scale * dty;

  // Compute Jr and Jl in one pass: Jr = J_r^{-1}(r), Jl = J_r^{-1}(-r)
  float Jr[16];
  Sim2JrInv(u1, u2, w, lam, Jr);

  float Jl[16];
  Sim2JrInv(-u1, -u2, -w, -lam, Jl);

  float* J = jacobians + tid * 32;

  // H_left = -(Jl * Ad), exploit Ad sparsity: Ad rows 2-3 = identity rows
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float l0 = Jl[i*4], l1 = Jl[i*4+1], l2 = Jl[i*4+2], l3 = Jl[i*4+3];
    J[i*8+0] = -(l0*a00 + l1*a10);
    J[i*8+1] = -(l0*a01 + l1*a11);
    J[i*8+2] = -(l0*a02 + l1*a12 + l2);
    J[i*8+3] = -(l0*a03 + l1*a13 + l3);
  }

  // H_right = Jr
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    J[i*8+4] = Jr[i*4];   J[i*8+5] = Jr[i*4+1];
    J[i*8+6] = Jr[i*4+2]; J[i*8+7] = Jr[i*4+3];
  }
}

Similarity2BetweenFactorBatch::Similarity2BetweenFactorBatch(
    const Matrix<3>* pose_deltas_ptr, size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors) {}

bool Similarity2BetweenFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Collect left poses for inverse computation
  collect_sim2_left_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, num_factors, poses_left_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Compute T_left^{-1} using the library's Sim2 inverse
  ComputeInverseSim2(stream,
                     reinterpret_cast<const float*>(poses_left_.data()),
                     kSim2TransformStride, kSim2TransformStride, num_factors,
                     reinterpret_cast<float*>(poses_left_inverse_.data()));

  // Fused: (L^{-1} * R) * D in one kernel (replaces 2x cuBLAS GEMM)
  compute_sim2_between_error_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, poses_left_inverse_.data(), pose_deltas_ptr_,
      num_factors, poses_left_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 5: residual = Log(error)
  ComputeLogSim2(stream,
                 reinterpret_cast<const float*>(poses_left_.data()),
                 kSim2TransformStride, kSim2TangentStride, num_factors,
                 residuals);

  // Step 6: Jacobian = [-J_l^{-1}(r)*Ad(Delta) | J_r^{-1}(r)]
  if (jacobians != nullptr) {
    sim2_between_jacobian_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        residuals, reinterpret_cast<const float*>(pose_deltas_ptr_),
        kSim2TransformStride, jacobians, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
