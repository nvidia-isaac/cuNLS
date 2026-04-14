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
#include "cunls/factor/similarity2_prior_factor_batch.h"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kSim2PriorBlockSize = 256;

// Sim(2) layout constants
constexpr size_t kSim2TransformStride = 9;
constexpr size_t kSim2TangentStride = 4;
constexpr size_t kSim2JacobianStride = 16;

/**
 * @brief Fused kernel: collect Sim(2) transform and compute T_inv * T_current.
 *
 * 3x3 row-major multiply, fully unrolled. Last row structure not assumed
 * since Sim(2) bottom-right is 1/s not 1.
 */
__global__ void
collect_and_multiply_sim2_prior_kernel(float const *const *state_pointers,
                                       const Matrix<3> *obs_inverse,
                                       size_t num_factors, Matrix<3> *errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors)
    return;

  const float *__restrict__ C = state_pointers[tid];
  const float *__restrict__ I = obs_inverse[tid].data();
  float *__restrict__ out = errors[tid].data();

  const float c0 = C[0], c1 = C[1], c2 = C[2];
  const float c3 = C[3], c4 = C[4], c5 = C[5];
  const float c6 = C[6], c7 = C[7], c8 = C[8];

  const float i0 = I[0], i1 = I[1], i2 = I[2];
  const float i3 = I[3], i4 = I[4], i5 = I[5];
  const float i6 = I[6], i7 = I[7], i8 = I[8];

  out[0] = i0 * c0 + i1 * c3 + i2 * c6;
  out[1] = i0 * c1 + i1 * c4 + i2 * c7;
  out[2] = i0 * c2 + i1 * c5 + i2 * c8;
  out[3] = i3 * c0 + i4 * c3 + i5 * c6;
  out[4] = i3 * c1 + i4 * c4 + i5 * c7;
  out[5] = i3 * c2 + i4 * c5 + i5 * c8;
  out[6] = i6 * c0 + i7 * c3 + i8 * c6;
  out[7] = i6 * c1 + i7 * c4 + i8 * c7;
  out[8] = i6 * c2 + i7 * c5 + i8 * c8;
}

Similarity2PriorFactorBatch::Similarity2PriorFactorBatch(
    const Matrix<3> *observations_ptr, size_t num_factors)
    : observations_ptr_(observations_ptr), num_factors_(num_factors),
      observations_inverse_(num_factors), transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all observations
  CudaStream stream;
  ComputeInverseSim2(stream.GetStream(),
                     reinterpret_cast<const float *>(observations_ptr_),
                     kSim2TransformStride, kSim2TransformStride, num_factors_,
                     reinterpret_cast<float *>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool Similarity2PriorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                           float const *const *state_pointers,
                                           cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSim2PriorBlockSize - 1) / kSim2PriorBlockSize;

  // Fused: collect T_current + compute T_inv * T_current
  collect_and_multiply_sim2_prior_kernel<<<num_blocks, kSim2PriorBlockSize, 0,
                                           stream>>>(
      state_pointers, observations_inverse_.data(), num_factors,
      transforms_error_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 3: residual = Log(T_error)
  ComputeLogSim2(
      stream, reinterpret_cast<const float *>(transforms_error_.data()),
      kSim2TransformStride, kSim2TangentStride, num_factors, residuals);

  // Step 4: Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSim2(stream, residuals, kSim2TangentStride,
                                    kSim2JacobianStride, num_factors,
                                    jacobians);
  }

  return true;
}

} // namespace cunls
