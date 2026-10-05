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

namespace cunls {

namespace {

/**
 * @brief One thread per state: result = x + delta per component; with bounds,
 * a component with a nonzero delta is clamped into [lower, upper] (bounds
 * entry k % bound_values).
 */
__global__ void vector_plus_kernel(const float *x, const float *delta, float *x_plus_delta,
                                   int num_params, int dim, const float *lower, const float *upper,
                                   size_t bound_values) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < num_params) {
    const float *x_vec = x + tid * dim;
    const float *delta_vec = delta + tid * dim;
    float *result_vec = x_plus_delta + tid * dim;

    for (int i = 0; i < dim; i++) {
      float v = x_vec[i] + delta_vec[i];
      if (lower != nullptr && delta_vec[i] != 0.f) {
        const size_t k = (static_cast<size_t>(tid) * dim + i) % bound_values;
        v = fminf(fmaxf(v, lower[k]), upper[k]);
      }
      result_vec[i] = v;
    }
  }
}

/** @brief Maximum CUDA block size for the vector_plus_kernel. */
constexpr size_t kMaxBlockSize = 256;

}  // namespace

/** @copydoc CalculateVectorPlus */
void CalculateVectorPlus(const float *x, const float *delta, float *x_plus_delta, size_t num_params,
                         int dim, cudaStream_t stream, const float *lower, const float *upper,
                         size_t bound_values) {
  if (num_params == 0) {
    return;
  }
  size_t num_cuda_blocks = (num_params + kMaxBlockSize - 1) / kMaxBlockSize;
  vector_plus_kernel<<<num_cuda_blocks, kMaxBlockSize, 0, stream>>>(
      x, delta, x_plus_delta, num_params, dim, bound_values > 0 ? lower : nullptr, upper,
      bound_values);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
