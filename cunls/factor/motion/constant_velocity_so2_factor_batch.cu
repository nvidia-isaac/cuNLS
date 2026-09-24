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
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kBlockSizeSO2CV = 256;

/** @brief Computes pose_rel = R_k^T * R_{k+1} for each factor (2x2, SO(2)). */
__global__ void cv_so2_relative_pose_kernel(float const *const *state_pointers, size_t num_factors,
                                            Matrix<2> *pose_rel) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float *__restrict__ L = state_pointers[4 * tid + 0];
  const float *__restrict__ R = state_pointers[4 * tid + 1];
  float *__restrict__ out = pose_rel[tid].data();

  const float l00 = L[0], l01 = L[1], l10 = L[2], l11 = L[3];
  const float r00 = R[0], r01 = R[1], r10 = R[2], r11 = R[3];

  // L^T * R
  out[0] = l00 * r00 + l10 * r10;
  out[1] = l00 * r01 + l10 * r11;
  out[2] = l01 * r00 + l11 * r10;
  out[3] = l01 * r01 + l11 * r11;
}

/**
 * @brief Assembles residuals and (optionally) the 2x4 Jacobian block. SO(2)
 * is abelian: J_l^{-1} = J_r^{-1} = 1, so both are hardcoded scalars.
 */
__global__ void cv_so2_assemble_kernel(float const *const *state_pointers, const float *twist,
                                       const float *dt, size_t num_factors, float *residuals,
                                       float *jacobians) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  const float vel_k = *state_pointers[4 * tid + 2];
  const float vel_k1 = *state_pointers[4 * tid + 3];
  const float tw = twist[tid];
  const float dt_i = dt[tid];

  float *__restrict__ res = residuals + tid * 2;
  res[0] = tw - dt_i * vel_k;
  res[1] = vel_k1 - vel_k;

  if (jacobians == nullptr) return;

  float *__restrict__ jac = jacobians + tid * 2 * 4;
  // r_pose row: [pose_k, pose_k+1, vel_k, vel_k+1]
  jac[0] = -1.0f;
  jac[1] = 1.0f;
  jac[2] = -dt_i;
  jac[3] = 0.0f;
  // r_vel row
  jac[4] = 0.0f;
  jac[5] = 0.0f;
  jac[6] = -1.0f;
  jac[7] = 1.0f;
}

ConstantVelocitySO2FactorBatch::ConstantVelocitySO2FactorBatch(const float *dt_ptr,
                                                               size_t num_factors)
    : dt_ptr_(dt_ptr), num_factors_(num_factors), pose_rel_(num_factors), twist_(num_factors) {}

bool ConstantVelocitySO2FactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream) const {
  const size_t num_blocks = (num_factors_ + kBlockSizeSO2CV - 1) / kBlockSizeSO2CV;

  cv_so2_relative_pose_kernel<<<num_blocks, kBlockSizeSO2CV, 0, stream>>>(
      state_pointers, num_factors_, pose_rel_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t rot_stride = 4;
  constexpr size_t angle_stride = 1;
  ComputeLogSO2(stream, reinterpret_cast<const float *>(pose_rel_.data()), rot_stride, angle_stride,
                num_factors_, twist_.data());

  cv_so2_assemble_kernel<<<num_blocks, kBlockSizeSO2CV, 0, stream>>>(
      state_pointers, twist_.data(), dt_ptr_, num_factors_, residuals, jacobians);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  return true;
}

}  // namespace cunls
