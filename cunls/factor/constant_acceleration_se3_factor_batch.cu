/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "cunls/factor/constant_acceleration_se3_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSizeSE3CA = 256;

/** @brief pose_rel = pose_k^{-1} * pose_{k+1} (same as the CV factor). */
__global__ void ca_se3_relative_pose_kernel(float const *const *state_pointers, size_t num_factors,
                                            SE3Transform *pose_rel) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float *__restrict__ L = state_pointers[6 * tid + 0];
  const float *__restrict__ R = state_pointers[6 * tid + 1];
  float *__restrict__ out = pose_rel[tid].data();

  const float l00 = L[0], l01 = L[1], l02 = L[2], l03 = L[3];
  const float l10 = L[4], l11 = L[5], l12 = L[6], l13 = L[7];
  const float l20 = L[8], l21 = L[9], l22 = L[10], l23 = L[11];

  const float i03 = -(l00 * l03 + l10 * l13 + l20 * l23);
  const float i13 = -(l01 * l03 + l11 * l13 + l21 * l23);
  const float i23 = -(l02 * l03 + l12 * l13 + l22 * l23);

  const float r00 = R[0], r01 = R[1], r02 = R[2], r03 = R[3];
  const float r10 = R[4], r11 = R[5], r12 = R[6], r13 = R[7];
  const float r20 = R[8], r21 = R[9], r22 = R[10], r23 = R[11];

  out[0] = l00 * r00 + l10 * r10 + l20 * r20;
  out[1] = l00 * r01 + l10 * r11 + l20 * r21;
  out[2] = l00 * r02 + l10 * r12 + l20 * r22;
  out[3] = l00 * r03 + l10 * r13 + l20 * r23 + i03;

  out[4] = l01 * r00 + l11 * r10 + l21 * r20;
  out[5] = l01 * r01 + l11 * r11 + l21 * r21;
  out[6] = l01 * r02 + l11 * r12 + l21 * r22;
  out[7] = l01 * r03 + l11 * r13 + l21 * r23 + i13;

  out[8] = l02 * r00 + l12 * r10 + l22 * r20;
  out[9] = l02 * r01 + l12 * r11 + l22 * r21;
  out[10] = l02 * r02 + l12 * r12 + l22 * r22;
  out[11] = l02 * r03 + l12 * r13 + l22 * r23 + i23;

  out[12] = 0.0f;
  out[13] = 0.0f;
  out[14] = 0.0f;
  out[15] = 1.0f;
}

/**
 * @brief Assembles residuals and (optionally) the 18x36 dense Jacobian
 * block. Column order: [pose_k, pose_k+1, vel_k, vel_k+1, accel_k,
 * accel_k+1]. Pose-block Jacobians of r_vel and r_accel are treated as zero
 * (documented simplification).
 */
__global__ void ca_se3_assemble_kernel(float const *const *state_pointers, const Vector<6> *twist,
                                       const Matrix<6> *jl_inv, const Matrix<6> *jr_inv,
                                       const float *dt, size_t num_factors, float *residuals,
                                       float *jacobians) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  constexpr int kDim = 6;
  constexpr int kCols = 36;

  const float *__restrict__ vel_k = state_pointers[6 * tid + 2];
  const float *__restrict__ vel_k1 = state_pointers[6 * tid + 3];
  const float *__restrict__ accel_k = state_pointers[6 * tid + 4];
  const float *__restrict__ accel_k1 = state_pointers[6 * tid + 5];
  const float *__restrict__ tw = twist[tid].data();
  const float *__restrict__ jli = jl_inv[tid].data();
  const float dt_i = dt[tid];
  const float half_dt2 = 0.5f * dt_i * dt_i;

  float *__restrict__ res = residuals + tid * 3 * kDim;
#pragma unroll
  for (int i = 0; i < kDim; i++) {
    res[i] = tw[i] - dt_i * vel_k[i] - half_dt2 * accel_k[i];
    float acc_vel = -vel_k[i] - dt_i * accel_k[i];
    float acc_accel = -accel_k[i];
#pragma unroll
    for (int j = 0; j < kDim; j++) {
      acc_vel += jli[i * kDim + j] * vel_k1[j];
      acc_accel += jli[i * kDim + j] * accel_k1[j];
    }
    res[kDim + i] = acc_vel;
    res[2 * kDim + i] = acc_accel;
  }

  if (jacobians == nullptr) return;

  const float *__restrict__ jri = jr_inv[tid].data();
  float *__restrict__ jac = jacobians + tid * 3 * kDim * kCols;

#pragma unroll
  for (int i = 0; i < kDim; i++) {
    float *row_pose = jac + i * kCols;
    float *row_vel = jac + (kDim + i) * kCols;
    float *row_accel = jac + (2 * kDim + i) * kCols;
#pragma unroll
    for (int j = 0; j < kDim; j++) {
      const float delta_ij = (i == j) ? 1.0f : 0.0f;

      row_pose[j] = -jli[i * kDim + j];               // d(r_pose)/d(pose_k)
      row_pose[kDim + j] = jri[i * kDim + j];         // d(r_pose)/d(pose_k+1)
      row_pose[2 * kDim + j] = -dt_i * delta_ij;      // d(r_pose)/d(vel_k)
      row_pose[3 * kDim + j] = 0.0f;                  // d(r_pose)/d(vel_k+1)
      row_pose[4 * kDim + j] = -half_dt2 * delta_ij;  // d(r_pose)/d(accel_k)
      row_pose[5 * kDim + j] = 0.0f;                  // d(r_pose)/d(accel_k+1)

      row_vel[j] = 0.0f;                          // (simplified)
      row_vel[kDim + j] = 0.0f;                   // (simplified)
      row_vel[2 * kDim + j] = -delta_ij;          // d(r_vel)/d(vel_k)
      row_vel[3 * kDim + j] = jli[i * kDim + j];  // d(r_vel)/d(vel_k+1)
      row_vel[4 * kDim + j] = -dt_i * delta_ij;   // d(r_vel)/d(accel_k)
      row_vel[5 * kDim + j] = 0.0f;               // d(r_vel)/d(accel_k+1)

      row_accel[j] = 0.0f;                          // (simplified)
      row_accel[kDim + j] = 0.0f;                   // (simplified)
      row_accel[2 * kDim + j] = 0.0f;               // d(r_accel)/d(vel_k)
      row_accel[3 * kDim + j] = 0.0f;               // d(r_accel)/d(vel_k+1)
      row_accel[4 * kDim + j] = -delta_ij;          // d(r_accel)/d(accel_k)
      row_accel[5 * kDim + j] = jli[i * kDim + j];  // d(r_accel)/d(accel_k+1)
    }
  }
}

ConstantAccelerationSE3FactorBatch::ConstantAccelerationSE3FactorBatch(const float *dt_ptr,
                                                                       size_t num_factors)
    : dt_ptr_(dt_ptr),
      num_factors_(num_factors),
      pose_rel_(num_factors),
      twist_(num_factors),
      jl_inv_(num_factors),
      jr_inv_(num_factors) {}

bool ConstantAccelerationSE3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream) const {
  const size_t num_blocks = (num_factors_ + kBlockSizeSE3CA - 1) / kBlockSizeSE3CA;

  ca_se3_relative_pose_kernel<<<num_blocks, kBlockSizeSE3CA, 0, stream>>>(
      state_pointers, num_factors_, pose_rel_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t pose_pitch = 4;
  constexpr size_t pose_stride = 16;
  constexpr size_t twist_stride = 6;
  ComputeLogSE3(stream, reinterpret_cast<const float *>(pose_rel_.data()), pose_pitch, pose_stride,
                twist_stride, num_factors_, reinterpret_cast<float *>(twist_.data()));

  constexpr size_t jac_pitch = 6;
  constexpr size_t jac_stride = 36;
  ComputeJacobianLeftInverseSE3(stream, reinterpret_cast<const float *>(twist_.data()),
                                twist_stride, jac_pitch, jac_stride, num_factors_,
                                reinterpret_cast<float *>(jl_inv_.data()));

  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSE3(stream, reinterpret_cast<const float *>(twist_.data()),
                                   twist_stride, jac_pitch, jac_stride, num_factors_,
                                   reinterpret_cast<float *>(jr_inv_.data()));
  }

  ca_se3_assemble_kernel<<<num_blocks, kBlockSizeSE3CA, 0, stream>>>(
      state_pointers, twist_.data(), jl_inv_.data(), jr_inv_.data(), dt_ptr_, num_factors_,
      residuals, jacobians);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  return true;
}

}  // namespace cunls
