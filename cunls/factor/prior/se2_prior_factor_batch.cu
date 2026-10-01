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
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {

constexpr size_t kSE2PriorBlockSize = 256;

// SE(2) prior factor layout constants
constexpr size_t kSE2TransformStride = 9;
constexpr size_t kSE2TangentStride = 3;
constexpr size_t kSE2JacobianStride = 9;

/**
 * @brief Fused kernel: collect SE(2) transform and compute T_inv * T_current.
 *
 * SE(2) 3x3 row-major multiply with structure exploitation: last row is [0 0
 * 1]. Only 3x3 active part computed (6 FMAs for rotation, 3 for translation).
 */
__global__ void collect_and_multiply_se2_prior_kernel(float const *const *state_pointers,
                                                      const Matrix<3> *obs_inverse,
                                                      size_t num_items, Matrix<3> *errors,
                                                      const int *factor_ids, int num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_items) return;

  const float *__restrict__ C = state_pointers[tid];
  const float *__restrict__ I =
      obs_inverse[FactorMeasurementIndex(tid, factor_ids, num_factors)].data();
  float *__restrict__ out = errors[tid].data();

  const float c00 = C[0], c01 = C[1], c02 = C[2];
  const float c10 = C[3], c11 = C[4], c12 = C[5];

  const float i00 = I[0], i01 = I[1], i02 = I[2];
  const float i10 = I[3], i11 = I[4], i12 = I[5];

  out[0] = i00 * c00 + i01 * c10;
  out[1] = i00 * c01 + i01 * c11;
  out[2] = i00 * c02 + i01 * c12 + i02;
  out[3] = i10 * c00 + i11 * c10;
  out[4] = i10 * c01 + i11 * c11;
  out[5] = i10 * c02 + i11 * c12 + i12;
  out[6] = 0.0f;
  out[7] = 0.0f;
  out[8] = 1.0f;
}

SE2PriorFactorBatch::SE2PriorFactorBatch(const SE2Transform *observations_ptr, size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      transforms_error_(num_factors) {
  // Pre-compute T_target^{-1} for all observations
  CudaStream stream;
  ComputeInverseSE2(stream.GetStream(), reinterpret_cast<const float *>(observations_ptr_),
                    kSE2TransformStride, kSE2TransformStride, num_factors_,
                    reinterpret_cast<float *>(observations_inverse_.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool SE2PriorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                   float const *const *state_pointers, cudaStream_t stream,
                                   const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumFactors() : num_factor_ids;
  if (num_items == 0 || NumFactors() == 0) {
    return true;
  }
  transforms_error_.resize(num_items);  // keeps capacity: allocates at most once per size
  size_t num_blocks = (num_items + kSE2PriorBlockSize - 1) / kSE2PriorBlockSize;

  // Fused: collect T_current + compute T_inv * T_current
  collect_and_multiply_se2_prior_kernel<<<num_blocks, kSE2PriorBlockSize, 0, stream>>>(
      state_pointers, observations_inverse_.data(), num_items, transforms_error_.data(),
      factor_ids, static_cast<int>(NumFactors()));
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 3: residual = Log(T_error)
  ComputeLogSE2(stream, reinterpret_cast<const float *>(transforms_error_.data()),
                kSE2TransformStride, kSE2TangentStride, num_items, residuals);

  // Step 4: Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSE2(stream, residuals, kSE2TangentStride, kSE2JacobianStride,
                                   num_items, jacobians);
  }

  return true;
}

}  // namespace cunls
