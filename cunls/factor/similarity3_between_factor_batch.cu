/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cublas_v2.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/math/dense_matrix_ops.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

constexpr size_t kSim3TransformStride = 16;
constexpr size_t kSim3TangentStride = 7;
constexpr size_t kSim3TangentDim = 7;
constexpr size_t kSim3AdjointStride = 49;
constexpr size_t kSim3JacobianPitch = 14;
constexpr size_t kSim3JacobianStride = 98;

/**
 * @brief Fused kernel: collect L/R Sim(3), compute (L^{-1} * R) * Delta.
 *
 * Full 4x4 row-major multiply (128 FMAs for 2 matmuls). Inline Sim(3) inverse.
 * Replaces: collect + InverseSim3 + 2x cuBLAS 4x4 GEMM.
 */
__global__ void collect_and_compute_sim3_between_error_kernel(
    float const* const* state_pointers, const Matrix<4>* deltas,
    size_t num_factors, Matrix<4>* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float* __restrict__ L = state_pointers[2 * tid];
  const float* __restrict__ R = state_pointers[2 * tid + 1];
  const float* __restrict__ D = deltas[tid].data();
  float* __restrict__ out = errors[tid].data();

  // Load L: 4x4, bottom-right = 1/s
  const float l00=L[0],l01=L[1],l02=L[2],l03=L[3];
  const float l10=L[4],l11=L[5],l12=L[6],l13=L[7];
  const float l20=L[8],l21=L[9],l22=L[10],l23=L[11];
  const float l_inv_s=L[15];
  const float s = 1.0f / l_inv_s;

  // L^{-1}: rotation part = R^T/s^2 (where sR = upper-left 3x3 of L)
  // det(sR) = s^3*det(R) = s^3, so (sR)^{-1} = adj(sR)/s^3
  // For orthogonal R: (sR)^{-1} = R^T/s
  const float inv_s = l_inv_s;
  const float i00=l00*inv_s, i01=l10*inv_s, i02=l20*inv_s;
  const float i10=l01*inv_s, i11=l11*inv_s, i12=l21*inv_s;
  const float i20=l02*inv_s, i21=l12*inv_s, i22=l22*inv_s;
  const float i03 = -(i00*l03 + i01*l13 + i02*l23);
  const float i13 = -(i10*l03 + i11*l13 + i12*l23);
  const float i23 = -(i20*l03 + i21*l13 + i22*l23);
  const float i33 = s;

  // Load R
  const float r0=R[0],r1=R[1],r2=R[2],r3=R[3];
  const float r4=R[4],r5=R[5],r6=R[6],r7=R[7];
  const float r8=R[8],r9=R[9],r10=R[10],r11=R[11];
  const float r12=R[12],r13=R[13],r14=R[14],r15=R[15];

  // temp = L^{-1} * R  (full 4x4)
  const float t00 = i00*r0+i01*r4+i02*r8+i03*r12;
  const float t01 = i00*r1+i01*r5+i02*r9+i03*r13;
  const float t02 = i00*r2+i01*r6+i02*r10+i03*r14;
  const float t03 = i00*r3+i01*r7+i02*r11+i03*r15;
  const float t10 = i10*r0+i11*r4+i12*r8+i13*r12;
  const float t11 = i10*r1+i11*r5+i12*r9+i13*r13;
  const float t12 = i10*r2+i11*r6+i12*r10+i13*r14;
  const float t13 = i10*r3+i11*r7+i12*r11+i13*r15;
  const float t20 = i20*r0+i21*r4+i22*r8+i23*r12;
  const float t21 = i20*r1+i21*r5+i22*r9+i23*r13;
  const float t22 = i20*r2+i21*r6+i22*r10+i23*r14;
  const float t23 = i20*r3+i21*r7+i22*r11+i23*r15;
  const float t30 = i33*r12;
  const float t31 = i33*r13;
  const float t32 = i33*r14;
  const float t33 = i33*r15;

  // error = D * temp  (row-major: D_row * temp_row)
  // cuBLAS col-major convention: C_col = A_col * B_col => C_row = B_row * A_row
  const float d0=D[0],d1=D[1],d2=D[2],d3=D[3];
  const float d4=D[4],d5=D[5],d6=D[6],d7=D[7];
  const float d8=D[8],d9=D[9],d10=D[10],d11=D[11];
  const float d12=D[12],d13=D[13],d14=D[14],d15=D[15];

  out[0]  = d0*t00+d1*t10+d2*t20+d3*t30;
  out[1]  = d0*t01+d1*t11+d2*t21+d3*t31;
  out[2]  = d0*t02+d1*t12+d2*t22+d3*t32;
  out[3]  = d0*t03+d1*t13+d2*t23+d3*t33;
  out[4]  = d4*t00+d5*t10+d6*t20+d7*t30;
  out[5]  = d4*t01+d5*t11+d6*t21+d7*t31;
  out[6]  = d4*t02+d5*t12+d6*t22+d7*t32;
  out[7]  = d4*t03+d5*t13+d6*t23+d7*t33;
  out[8]  = d8*t00+d9*t10+d10*t20+d11*t30;
  out[9]  = d8*t01+d9*t11+d10*t21+d11*t31;
  out[10] = d8*t02+d9*t12+d10*t22+d11*t32;
  out[11] = d8*t03+d9*t13+d10*t23+d11*t33;
  out[12] = d12*t00+d13*t10+d14*t20+d15*t30;
  out[13] = d12*t01+d13*t11+d14*t21+d15*t31;
  out[14] = d12*t02+d13*t12+d14*t22+d15*t32;
  out[15] = d12*t03+d13*t13+d14*t23+d15*t33;
}

Similarity3BetweenFactorBatch::Similarity3BetweenFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<4>* pose_deltas_ptr,
    size_t num_factors)
    : pose_deltas_ptr_(pose_deltas_ptr),
      num_factors_(num_factors),
      cublas_handle_(cublas_handle),
      poses_left_(num_factors),
      poses_right_(num_factors),
      poses_left_inverse_(num_factors),
      delta_adjoints_(kSim3AdjointStride * num_factors),
      jacobian_temp_(kSim3AdjointStride * num_factors) {
  CudaStream stream;
  ComputeDeltaAdjoints(stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

void Similarity3BetweenFactorBatch::ComputeDeltaAdjoints(cudaStream_t stream) {
  size_t num_factors = NumFactors();
  ComputeAdjointSim3(stream,
                      reinterpret_cast<const float*>(pose_deltas_ptr_),
                      kSim3TransformStride, delta_adjoints_.data(),
                      kSim3AdjointStride, num_factors);
}

void Similarity3BetweenFactorBatch::ComputeLeftPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  size_t num_factors = NumFactors();

  // Negate residuals into temp storage (reuse poses_left_ which is free after
  // residual computation; 16 floats/factor > 7 floats needed).
  float* neg_res = reinterpret_cast<float*>(poses_left_.data());
  ComputeNegateMatrix(stream, residuals, 1, kSim3TangentDim,
                      kSim3TangentDim, kSim3TangentStride,
                      num_factors, neg_res);

  // J_r^{-1}(-r) = J_l^{-1}(r) into jacobian_temp_ (dense 7x7, stride 49)
  ComputeJacobianRightInverseSim3(stream, neg_res, kSim3TangentStride,
                                  kSim3AdjointStride, num_factors,
                                  jacobian_temp_.data());

  // Negate to get -J_l^{-1}(r)
  ComputeNegateMatrix(stream, jacobian_temp_.data(), kSim3TangentDim,
                      kSim3TangentDim, kSim3TangentDim, kSim3AdjointStride,
                      num_factors, jacobian_temp_.data());

  // cuBLAS GEMM: H_left = (-J_l^{-1}) * Ad(Delta)
  // Row-major C = B * A via column-major cuBLAS: C_col = A_col * B_col.
  // Output writes directly into the left 7x7 block of the 7x14 Jacobian
  // using ldc=14 (full row width).
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 7;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_adjoints_.data(), mat_size, kSim3AdjointStride,
      jacobian_temp_.data(), mat_size, kSim3AdjointStride,
      &beta,
      jacobians, kSim3JacobianPitch, kSim3JacobianStride,
      num_factors));
}

void Similarity3BetweenFactorBatch::ComputeRightPoseJacobian(
    cudaStream_t stream, const float* residuals, float* jacobians) const {
  size_t num_factors = NumFactors();

  // J_r^{-1}(r) into jacobian_temp_ (dense 7x7, stride 49)
  ComputeJacobianRightInverseSim3(stream, residuals, kSim3TangentStride,
                                  kSim3AdjointStride, num_factors,
                                  jacobian_temp_.data());

  // Scatter into the right 7x7 block of each 7x14 Jacobian row
  ScatterToRightBlock(stream, jacobian_temp_.data(), kSim3TangentDim,
                      kSim3AdjointStride, jacobians + kSim3TangentDim,
                      kSim3JacobianPitch, kSim3JacobianStride, num_factors);
}

bool Similarity3BetweenFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks = (num_factors + kBlockSize - 1) / kBlockSize;

  // Fused: collect L/R + compute (L^{-1} * R) * Delta in one kernel
  collect_and_compute_sim3_between_error_kernel<<<num_blocks, kBlockSize, 0,
                                                  stream>>>(
      state_pointers, pose_deltas_ptr_, num_factors,
      poses_left_inverse_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 5: residual = Log(error)
  ComputeLogSim3(stream,
                 reinterpret_cast<const float*>(poses_left_inverse_.data()),
                 kSim3TransformStride, kSim3TangentStride, num_factors,
                 residuals);

  // Step 6: Jacobian = [-J_l^{-1}(r)*Ad(Delta) | J_r^{-1}(r)]
  if (jacobians != nullptr) {
    ComputeLeftPoseJacobian(stream, residuals, jacobians);
    ComputeRightPoseJacobian(stream, residuals, jacobians);
  }

  return true;
}

}  // namespace cunls
