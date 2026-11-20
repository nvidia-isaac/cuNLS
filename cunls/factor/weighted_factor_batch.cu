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

#include <numeric>

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
// factor (FactorMeasurementIndex: factor_ids[item] or item % num_factors).
__global__ void PerFactorScaleKernel(const float *weights, float *data, size_t stride,
                                     size_t num_items, const int *factor_ids, int num_factors) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = num_items * stride;
  if (idx < total) {
    const int item = static_cast<int>(idx / stride);
    data[idx] *= weights[FactorMeasurementIndex(item, factor_ids, num_factors)];
  }
}

/**
 * Scales each item's `stride` consecutive floats of `data` by the weight of the
 * item's factor, weights[f], or by `uniform_weight` when weights is null.
 */
void ScaleItems(float uniform_weight, const float *weights, float *data, size_t stride,
                size_t num_items, const int *factor_ids, size_t num_factors, cudaStream_t stream) {
  const size_t total = num_items * stride;
  if (total == 0) return;
  const unsigned num_blocks = static_cast<unsigned>((total + kBlockSize - 1) / kBlockSize);
  if (weights == nullptr) {
    UniformScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(uniform_weight, data, total);
  } else {
    PerFactorScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(
        weights, data, stride, num_items, factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

bool EvaluateWeighted(const FactorBatch &inner, float uniform_weight,
                      const float *per_factor_weights, float *residuals, float *jacobians,
                      float const *const *state_pointers, cudaStream_t stream,
                      const int *factor_ids, size_t num_factor_ids) {
  const size_t num_factors = inner.NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  if (!inner.Evaluate(residuals, jacobians, state_pointers, stream, factor_ids, num_items)) {
    return false;
  }
  const size_t rsize = inner.ResidualsSize();
  if (residuals != nullptr) {
    ScaleItems(uniform_weight, per_factor_weights, residuals, rsize, num_items, factor_ids,
               num_factors, stream);
  }
  if (jacobians != nullptr) {
    const auto sizes = inner.StateSizes();
    const size_t pitch = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
    ScaleItems(uniform_weight, per_factor_weights, jacobians, rsize * pitch, num_items, factor_ids,
               num_factors, stream);
  }
  return true;
}

}  // namespace cunls
