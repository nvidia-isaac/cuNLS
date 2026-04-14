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
#include "cunls/factor/similarity3_prior_factor_batch.h"
#include "cunls/math/sim_lie_math.h"

namespace cunls {

constexpr size_t kSim3PriorBlockSize = 256;

// Sim(3) layout constants
constexpr size_t kSim3TransformStride = 16;
constexpr size_t kSim3TangentStride = 7;
constexpr size_t kSim3JacobianStride = 49;

/**
 * @brief Fused kernel: collect Sim(3) transform and compute T_inv * T_current.
 *
 * Full 4x4 row-major multiply, fully unrolled (64 FMAs).
 * Cannot exploit last-row structure since Sim(3) bottom-right = 1/s.
 */
__global__ void
collect_and_multiply_sim3_prior_kernel(float const *const *state_pointers,
                                       const Matrix<4> *obs_inverse,
                                       size_t num_factors, Matrix<4> *errors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors)
    return;

  const float *__restrict__ C = state_pointers[tid];
  const float *__restrict__ I = obs_inverse[tid].data();
  float *__restrict__ out = errors[tid].data();

  const float c0 = C[0], c1 = C[1], c2 = C[2], c3 = C[3];
  const float c4 = C[4], c5 = C[5], c6 = C[6], c7 = C[7];
  const float c8 = C[8], c9 = C[9], c10 = C[10], c11 = C[11];
  const float c12 = C[12], c13 = C[13], c14 = C[14], c15 = C[15];

  const float i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3];
  const float i4 = I[4], i5 = I[5], i6 = I[6], i7 = I[7];
  const float i8 = I[8], i9 = I[9], i10 = I[10], i11 = I[11];
  const float i12 = I[12], i13 = I[13], i14 = I[14], i15 = I[15];

  out[0] = i0 * c0 + i1 * c4 + i2 * c8 + i3 * c12;
  out[1] = i0 * c1 + i1 * c5 + i2 * c9 + i3 * c13;
  out[2] = i0 * c2 + i1 * c6 + i2 * c10 + i3 * c14;
  out[3] = i0 * c3 + i1 * c7 + i2 * c11 + i3 * c15;
  out[4] = i4 * c0 + i5 * c4 + i6 * c8 + i7 * c12;
  out[5] = i4 * c1 + i5 * c5 + i6 * c9 + i7 * c13;
  out[6] = i4 * c2 + i5 * c6 + i6 * c10 + i7 * c14;
  out[7] = i4 * c3 + i5 * c7 + i6 * c11 + i7 * c15;
  out[8] = i8 * c0 + i9 * c4 + i10 * c8 + i11 * c12;
  out[9] = i8 * c1 + i9 * c5 + i10 * c9 + i11 * c13;
  out[10] = i8 * c2 + i9 * c6 + i10 * c10 + i11 * c14;
  out[11] = i8 * c3 + i9 * c7 + i10 * c11 + i11 * c15;
  out[12] = i12 * c0 + i13 * c4 + i14 * c8 + i15 * c12;
  out[13] = i12 * c1 + i13 * c5 + i14 * c9 + i15 * c13;
  out[14] = i12 * c2 + i13 * c6 + i14 * c10 + i15 * c14;
  out[15] = i12 * c3 + i13 * c7 + i14 * c11 + i15 * c15;
}

Similarity3PriorFactorBatch::Similarity3PriorFactorBatch(
    const Matrix<4> *observations_ptr, size_t num_factors)
    : observations_ptr_(observations_ptr), num_factors_(num_factors),
      observations_inverse_(num_factors), transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all observations
  CudaStream stream;
  ComputeInverseSim3(stream.GetStream(),
                     reinterpret_cast<const float *>(observations_ptr_),
                     kSim3TransformStride, kSim3TransformStride, num_factors_,
                     reinterpret_cast<float *>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool Similarity3PriorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                           float const *const *state_pointers,
                                           cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSim3PriorBlockSize - 1) / kSim3PriorBlockSize;

  // Fused: collect T_current + compute T_inv * T_current
  collect_and_multiply_sim3_prior_kernel<<<num_blocks, kSim3PriorBlockSize, 0,
                                           stream>>>(
      state_pointers, observations_inverse_.data(), num_factors,
      transforms_error_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 3: residual = Log(T_error)
  ComputeLogSim3(
      stream, reinterpret_cast<const float *>(transforms_error_.data()),
      kSim3TransformStride, kSim3TangentStride, num_factors, residuals);

  // Step 4: Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSim3(stream, residuals, kSim3TangentStride,
                                    kSim3JacobianStride, num_factors,
                                    jacobians);
  }

  return true;
}

} // namespace cunls
