/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
    float const *const *state_pointers, const Matrix<3> *deltas,
    size_t num_factors, Matrix<3> *errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors)
    return;

  const float *__restrict__ L = state_pointers[2 * tid];
  const float *__restrict__ R = state_pointers[2 * tid + 1];
  const float *__restrict__ D = deltas[tid].data();
  float *__restrict__ out = errors[tid].data();

  const float l0 = L[0], l1 = L[1], l2 = L[2];
  const float l3 = L[3], l4 = L[4], l5 = L[5];
  const float l6 = L[6], l7 = L[7], l8 = L[8];

  const float r0 = R[0], r1 = R[1], r2 = R[2];
  const float r3 = R[3], r4 = R[4], r5 = R[5];
  const float r6 = R[6], r7 = R[7], r8 = R[8];

  // temp = L^T * R  (27 FMAs, all in registers)
  const float m0 = l0 * r0 + l3 * r3 + l6 * r6;
  const float m1 = l0 * r1 + l3 * r4 + l6 * r7;
  const float m2 = l0 * r2 + l3 * r5 + l6 * r8;
  const float m3 = l1 * r0 + l4 * r3 + l7 * r6;
  const float m4 = l1 * r1 + l4 * r4 + l7 * r7;
  const float m5 = l1 * r2 + l4 * r5 + l7 * r8;
  const float m6 = l2 * r0 + l5 * r3 + l8 * r6;
  const float m7 = l2 * r1 + l5 * r4 + l8 * r7;
  const float m8 = l2 * r2 + l5 * r5 + l8 * r8;

  const float d0 = D[0], d1 = D[1], d2 = D[2];
  const float d3 = D[3], d4 = D[4], d5 = D[5];
  const float d6 = D[6], d7 = D[7], d8 = D[8];

  // error = temp * D^T  (27 FMAs)
  out[0] = m0 * d0 + m1 * d1 + m2 * d2;
  out[1] = m0 * d3 + m1 * d4 + m2 * d5;
  out[2] = m0 * d6 + m1 * d7 + m2 * d8;
  out[3] = m3 * d0 + m4 * d1 + m5 * d2;
  out[4] = m3 * d3 + m4 * d4 + m5 * d5;
  out[5] = m3 * d6 + m4 * d7 + m5 * d8;
  out[6] = m6 * d0 + m7 * d1 + m8 * d2;
  out[7] = m6 * d3 + m7 * d4 + m8 * d5;
  out[8] = m6 * d6 + m7 * d7 + m8 * d8;
}

// Compute one row of J_l^{-1}(phi) via the Rodrigues formula:
//   k1*I + k2*[phi]_x + k3*(phi * phi^T)
// with k1 = half_theta/tan(half_theta), k2 = -half_theta/theta,
//      k3 = (1 - k1)/theta^2, k4 = -0.5 for small-angle fallback.
__device__ __forceinline__ void so3_jl_inv_row(const float *phi, int r,
                                               float *row) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  float th2 = theta * theta;
  if (theta < 1e-5f) {
    float sx, sy, sz;
    if (r == 0) {
      sx = 0.f;
      sy = -phi[2];
      sz = phi[1];
    } else if (r == 1) {
      sx = phi[2];
      sy = 0.f;
      sz = -phi[0];
    } else {
      sx = -phi[1];
      sy = phi[0];
      sz = 0.f;
    }
    row[0] = (r == 0 ? 1.f : 0.f) - 0.5f * sx;
    row[1] = (r == 1 ? 1.f : 0.f) - 0.5f * sy;
    row[2] = (r == 2 ? 1.f : 0.f) - 0.5f * sz;
    return;
  }
  float half = 0.5f * theta;
  float k1 = half / tanf(half);
  float k2 = -half / theta;
  float k3 = (1.0f - k1) / th2;
  float pr = phi[r];
  float sx, sy, sz;
  if (r == 0) {
    sx = 0.f;
    sy = -phi[2];
    sz = phi[1];
  } else if (r == 1) {
    sx = phi[2];
    sy = 0.f;
    sz = -phi[0];
  } else {
    sx = -phi[1];
    sy = phi[0];
    sz = 0.f;
  }
  row[0] = (r == 0 ? k1 : 0.f) + k2 * sx + k3 * pr * phi[0];
  row[1] = (r == 1 ? k1 : 0.f) + k2 * sy + k3 * pr * phi[1];
  row[2] = (r == 2 ? k1 : 0.f) + k2 * sz + k3 * pr * phi[2];
}

// Fused kernel: computes BOTH left and right SO(3) Jacobians in one pass.
// Left  Jacobian (cols 0..2): -D * J_l^{-1}(r)  where D = Ad(Delta)
// Right Jacobian (cols 3..5):  J_r^{-1}(r) = J_l^{-1}(-r)
// 1 thread per factor, ~25 regs. Replaces 4 separate kernel launches.
__global__ void __launch_bounds__(256, 8) so3_between_fused_jacobians_kernel(
    const float *__restrict__ residuals,
    const Matrix<3> *__restrict__ delta_adjoints, int num_factors,
    float *__restrict__ jacobians) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors)
    return;

  const float *r = residuals + tid * kTwistStride;
  float phi[3] = {r[0], r[1], r[2]};

  const float *D = delta_adjoints[tid].data();
  float *J = jacobians + tid * 18;

  // Compute J_l^{-1}(phi) rows, multiply by -D, write left block (pitch 6)
  float jl[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    so3_jl_inv_row(phi, row, &jl[row * 3]);
  }

  // Left block: -D * J_l_inv
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    float d0 = D[row * 3], d1 = D[row * 3 + 1], d2 = D[row * 3 + 2];
    J[row * 6 + 0] = -(d0 * jl[0] + d1 * jl[3] + d2 * jl[6]);
    J[row * 6 + 1] = -(d0 * jl[1] + d1 * jl[4] + d2 * jl[7]);
    J[row * 6 + 2] = -(d0 * jl[2] + d1 * jl[5] + d2 * jl[8]);
  }

  // Right block: J_r^{-1}(phi) = J_l^{-1}(-phi)
  float neg_phi[3] = {-phi[0], -phi[1], -phi[2]};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    float jr[3];
    so3_jl_inv_row(neg_phi, row, jr);
    J[row * 6 + 3] = jr[0];
    J[row * 6 + 4] = jr[1];
    J[row * 6 + 5] = jr[2];
  }
}

SO3BetweenFactorBatch::SO3BetweenFactorBatch(const Matrix<3> *pose_deltas_ptr,
                                             size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr), num_factors_(num_factors),
      delta_adjoints_(num_factors), poses_left_(num_factors),
      poses_right_(num_factors), poses_left_inverse_(num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void SO3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(delta_adjoints_.data(), pose_deltas_ptr_,
                                      num_factors_ * sizeof(Matrix<3>),
                                      cudaMemcpyDeviceToDevice, stream));
}

bool SO3BetweenFactorBatch::Evaluate(float *residuals, float *jacobians,
                                     float const *const *state_pointers,
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
  ComputeLogSO3(
      stream, reinterpret_cast<const float *>(poses_left_inverse_.data()),
      rotation_pitch, kRotationStride, kTwistStride, num_factors, residuals);

  if (jacobians != nullptr) {
    so3_between_fused_jacobians_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        residuals, delta_adjoints_.data(), num_factors, jacobians);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

} // namespace cunls
