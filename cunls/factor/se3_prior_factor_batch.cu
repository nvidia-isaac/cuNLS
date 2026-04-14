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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kSE3PriorBlockSize = 256;

/**
 * @brief Fused kernel: collect SE(3) transform from state pointer and compute
 *        T_error = T_inv * T_current in one pass.
 *
 * Fully unrolled with SE3 structure exploitation: last row is [0 0 0 1],
 * so only 3x4 active part is computed (36 FMAs instead of 64).
 */
__global__ void collect_and_multiply_se3_prior_kernel(
    float const* const* state_pointers, const SE3Transform* obs_inverse,
    size_t num_factors, SE3Transform* errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;

  const float* __restrict__ C = state_pointers[tid];
  const float* __restrict__ I = obs_inverse[tid].data();
  float* __restrict__ out = errors[tid].data();

  const float c00=C[0], c01=C[1], c02=C[2], c03=C[3];
  const float c10=C[4], c11=C[5], c12=C[6], c13=C[7];
  const float c20=C[8], c21=C[9], c22=C[10], c23=C[11];

  const float i00=I[0], i01=I[1], i02=I[2], i03=I[3];
  const float i10=I[4], i11=I[5], i12=I[6], i13=I[7];
  const float i20=I[8], i21=I[9], i22=I[10], i23=I[11];

  // cuBLAS col-major with row-major data: result_row = I_row * C_row
  // (CUBLAS_OP_N,CUBLAS_OP_N with C_col=C_row^T, I_col=I_row^T gives
  //  C_row^T * I_row^T = (I_row * C_row)^T, so result_row = I * C)
  out[0]  = i00*c00 + i01*c10 + i02*c20;
  out[1]  = i00*c01 + i01*c11 + i02*c21;
  out[2]  = i00*c02 + i01*c12 + i02*c22;
  out[3]  = i00*c03 + i01*c13 + i02*c23 + i03;

  out[4]  = i10*c00 + i11*c10 + i12*c20;
  out[5]  = i10*c01 + i11*c11 + i12*c21;
  out[6]  = i10*c02 + i11*c12 + i12*c22;
  out[7]  = i10*c03 + i11*c13 + i12*c23 + i13;

  out[8]  = i20*c00 + i21*c10 + i22*c20;
  out[9]  = i20*c01 + i21*c11 + i22*c21;
  out[10] = i20*c02 + i21*c12 + i22*c22;
  out[11] = i20*c03 + i21*c13 + i22*c23 + i23;

  out[12] = 0.0f; out[13] = 0.0f; out[14] = 0.0f; out[15] = 1.0f;
}

SE3PriorFactorBatch::SE3PriorFactorBatch(const SE3Transform* observations_ptr,
                                         size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all targets
  CudaStream stream;
  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;
  ComputeInverseSE3(stream.GetStream(),
                    reinterpret_cast<const float*>(observations_ptr_), pitch,
                    stride, pitch, stride, num_factors_,
                    reinterpret_cast<float*>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SE3PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSE3PriorBlockSize - 1) / kSE3PriorBlockSize;

  // Fused: collect T_current from state pointers + compute T_inv * T_current
  collect_and_multiply_se3_prior_kernel<<<num_blocks, kSE3PriorBlockSize, 0,
                                          stream>>>(
      state_pointers, observations_inverse_.data(), num_factors,
      transforms_error_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 3: Compute residual = Log(T_error) using SE(3) logarithm map
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  constexpr size_t twist_stride = 6;
  ComputeLogSE3(
      stream, reinterpret_cast<const float*>(transforms_error_.data()),
      transform_pitch, transform_stride, twist_stride, num_factors, residuals);

  // Step 4: Compute Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    constexpr size_t jacobian_pitch = 6;
    constexpr size_t jacobian_stride = 36;
    ComputeJacobianRightInverseSE3(stream, residuals, twist_stride,
                                   jacobian_pitch, jacobian_stride, num_factors,
                                   jacobians);
  }

  return true;
}

}  // namespace cunls
