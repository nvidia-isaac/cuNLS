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

#include <cuda_runtime.h>

#include "cunls/common/helper.h"
#include "cunls/factor/weighted_factor_batch.h"

namespace cunls {

namespace {

/**
 * Factor (measurement) index of evaluation item `item`: factor_ids[item], or
 * item modulo the batch size when factor_ids is null (see FactorBatch::Evaluate).
 */
__device__ __forceinline__ int FactorMeasurementIndex(int item, const int *factor_ids,
                                                      int num_factors) {
  if (factor_ids != nullptr) {
    return factor_ids[item];
  }
  return item < num_factors ? item : item % num_factors;
}

}  // namespace

namespace {

constexpr int kBlockSize = 256;

__global__ void UniformScaleKernel(float weight, float *data, size_t total_elements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    data[idx] *= weight;
  }
}

// Scales each item's `stride` consecutive floats by the weight of the item's
// factor (FactorMeasurementIndex: factor_ids[item] or item % weights_size).
__global__ void PerFactorScaleKernel(const float *weights, float *data, size_t stride,
                                     size_t num_items, const int *factor_ids, int weights_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = num_items * stride;
  if (idx < total) {
    const int item = static_cast<int>(idx / stride);
    data[idx] *= weights[FactorMeasurementIndex(item, factor_ids, weights_size)];
  }
}

}  // namespace

void ApplyUniformWeightToResiduals(float weight, float *residuals, size_t total_elements,
                                   cudaStream_t stream) {
  const int num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;
  UniformScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(weight, residuals, total_elements);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyUniformWeightToJacobians(float weight, float *jacobians, size_t total_elements,
                                   cudaStream_t stream) {
  const int num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;
  UniformScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(weight, jacobians, total_elements);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyPerFactorWeightToResiduals(const float *weights, float *residuals, size_t residual_size,
                                     size_t num_factors, cudaStream_t stream, const int *factor_ids,
                                     size_t weights_size) {
  const size_t total = num_factors * residual_size;
  if (total == 0) {
    return;
  }
  if (weights_size == 0) {
    weights_size = num_factors;
  }
  const int num_blocks = (total + kBlockSize - 1) / kBlockSize;
  PerFactorScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weights, residuals, residual_size, num_factors, factor_ids, static_cast<int>(weights_size));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyPerFactorWeightToJacobians(const float *weights, float *jacobians, size_t residual_size,
                                     size_t jacobian_pitch, size_t num_factors, cudaStream_t stream,
                                     const int *factor_ids, size_t weights_size) {
  const size_t stride = residual_size * jacobian_pitch;
  const size_t total = num_factors * stride;
  if (total == 0) {
    return;
  }
  if (weights_size == 0) {
    weights_size = num_factors;
  }
  const int num_blocks = (total + kBlockSize - 1) / kBlockSize;
  PerFactorScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weights, jacobians, stride, num_factors, factor_ids, static_cast<int>(weights_size));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
