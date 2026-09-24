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
#include "cunls/factor/constant_acceleration_so3_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSizeSO3CA = 256;

__global__ void ca_so3_relative_pose_kernel(float const *const *state_pointers, size_t num_factors,
                                            Matrix<3> *pose_rel) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float *__restrict__ L = state_pointers[6 * tid + 0];
  const float *__restrict__ R = state_pointers[6 * tid + 1];
  float *__restrict__ out = pose_rel[tid].data();

#pragma unroll
  for (int i = 0; i < 3; i++)
#pragma unroll
    for (int j = 0; j < 3; j++) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < 3; k++) acc += L[k * 3 + i] * R[k * 3 + j];
      out[i * 3 + j] = acc;
    }
}

__global__ void ca_so3_assemble_kernel(float const *const *state_pointers, const Vector<3> *twist,
                                       const Matrix<3> *jl_inv, const Matrix<3> *jr_inv,
                                       const float *dt, size_t num_factors, float *residuals,
                                       float *jacobians) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  constexpr int kDim = 3;
  constexpr int kCols = 18;

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

      row_pose[j] = -jli[i * kDim + j];
      row_pose[kDim + j] = jri[i * kDim + j];
      row_pose[2 * kDim + j] = -dt_i * delta_ij;
      row_pose[3 * kDim + j] = 0.0f;
      row_pose[4 * kDim + j] = -half_dt2 * delta_ij;
      row_pose[5 * kDim + j] = 0.0f;

      row_vel[j] = 0.0f;
      row_vel[kDim + j] = 0.0f;
      row_vel[2 * kDim + j] = -delta_ij;
      row_vel[3 * kDim + j] = jli[i * kDim + j];
      row_vel[4 * kDim + j] = -dt_i * delta_ij;
      row_vel[5 * kDim + j] = 0.0f;

      row_accel[j] = 0.0f;
      row_accel[kDim + j] = 0.0f;
      row_accel[2 * kDim + j] = 0.0f;
      row_accel[3 * kDim + j] = 0.0f;
      row_accel[4 * kDim + j] = -delta_ij;
      row_accel[5 * kDim + j] = jli[i * kDim + j];
    }
  }
}

ConstantAccelerationSO3FactorBatch::ConstantAccelerationSO3FactorBatch(const float *dt_ptr,
                                                                       size_t num_factors)
    : dt_ptr_(dt_ptr),
      num_factors_(num_factors),
      pose_rel_(num_factors),
      twist_(num_factors),
      jl_inv_(num_factors),
      jr_inv_(num_factors) {}

bool ConstantAccelerationSO3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream) const {
  const size_t num_blocks = (num_factors_ + kBlockSizeSO3CA - 1) / kBlockSizeSO3CA;

  ca_so3_relative_pose_kernel<<<num_blocks, kBlockSizeSO3CA, 0, stream>>>(
      state_pointers, num_factors_, pose_rel_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t rot_pitch = 3;
  constexpr size_t rot_stride = 9;
  constexpr size_t twist_stride = 3;
  ComputeLogSO3(stream, reinterpret_cast<const float *>(pose_rel_.data()), rot_pitch, rot_stride,
                twist_stride, num_factors_, reinterpret_cast<float *>(twist_.data()));

  constexpr size_t jac_pitch = 3;
  constexpr size_t jac_stride = 9;
  ComputeJacobianLeftInverseSO3(stream, reinterpret_cast<const float *>(twist_.data()),
                                twist_stride, jac_pitch, jac_stride, num_factors_,
                                reinterpret_cast<float *>(jl_inv_.data()));

  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSO3(stream, reinterpret_cast<const float *>(twist_.data()),
                                   twist_stride, jac_pitch, jac_stride, num_factors_,
                                   reinterpret_cast<float *>(jr_inv_.data()));
  }

  ca_so3_assemble_kernel<<<num_blocks, kBlockSizeSO3CA, 0, stream>>>(
      state_pointers, twist_.data(), jl_inv_.data(), jr_inv_.data(), dt_ptr_, num_factors_,
      residuals, jacobians);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  return true;
}

}  // namespace cunls
