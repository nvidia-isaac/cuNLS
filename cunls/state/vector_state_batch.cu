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

#include <cuda_runtime.h>

#include "cunls/common/helper.h"

namespace cunls {

namespace {

/**
 * @brief CUDA kernel that performs element-wise vector addition per state block.
 *
 * Each thread processes one state block, iterating over its Dim elements
 * and computing result[i] = x[i] + delta[i].
 *
 * @param x             Device pointer to input state vectors.
 * @param delta         Device pointer to tangent-space update vectors.
 * @param x_plus_delta  Device pointer to output state vectors.
 * @param num_params    Number of state blocks.
 * @param dim           Dimension of each state vector.
 *
 * Grid/block: launched with ceil(num_params / 256) blocks of 256 threads.
 */
__global__ void vector_plus_kernel(const float *x, const float *delta,
                                   float *x_plus_delta, int num_params,
                                   int dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < num_params) {
    const float *x_vec = x + tid * dim;
    const float *delta_vec = delta + tid * dim;
    float *result_vec = x_plus_delta + tid * dim;

    for (int i = 0; i < dim; i++) {
      result_vec[i] = x_vec[i] + delta_vec[i];
    }
  }
}

/** @brief Maximum CUDA block size for the vector_plus_kernel. */
constexpr size_t kMaxBlockSize = 256;

}  // namespace

/** @copydoc CalculateVectorPlus */
void CalculateVectorPlus(const float *x, const float *delta, float *x_plus_delta,
                         size_t num_params, int dim, cudaStream_t stream) {
  size_t num_cuda_blocks = (num_params + kMaxBlockSize - 1) / kMaxBlockSize;
  vector_plus_kernel<<<num_cuda_blocks, kMaxBlockSize, 0, stream>>>(
      x, delta, x_plus_delta, num_params, dim);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
