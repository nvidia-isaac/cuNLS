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

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/factor/prior_vector_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block for prior vector cost kernel launches.
constexpr size_t kBlockSize = 256;

/**
 * @brief CUDA kernel that computes prior vector residuals and Jacobians.
 *
 * For each vector i, computes:
 *   - residual[i] = state[i] - observation[i]   (element-wise)
 *   - jacobian[i] = I  (Dim x Dim identity matrix)
 *
 * Each thread processes one vector independently.
 *
 * @param observations Flattened array of observation vectors on device
 *                     (num_vectors * dim floats).
 * @param state_pointers   Array of device pointers, one per vector, each
 * pointing to dim floats of state data.
 * @param residuals    Output array for residuals (num_vectors * dim floats),
 *                     or nullptr to skip.
 * @param jacobians    Output array for Jacobians (num_vectors * dim * dim
 *                     floats, row-major identity per vector), or nullptr to
 * skip.
 * @param dim          Dimension of each vector.
 * @param num_vectors  Number of vectors (one thread per vector).
 *
 * @note Launch configuration: <<<ceil(num_vectors / kBlockSize), kBlockSize>>>
 */
__global__ void prior_vector_factor_kernel(const float *observations,
                                           float const *const *state_pointers,
                                           float *residuals, float *jacobians,
                                           int dim, int num_vectors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_vectors) {
    return;
  }

  if (residuals != nullptr) {
    auto obs_ptr = observations + tid * dim;
    auto param_ptr = state_pointers[tid];
    assert(param_ptr != nullptr);

    for (int i = 0; i < dim; i++) {
      residuals[dim * tid + i] = param_ptr[i] - obs_ptr[i];
    }
  }

  if (jacobians != nullptr) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        jacobians[(dim * tid + i) * dim + j] = i == j ? 1.f : 0.f;
      }
    }
  }
}

void LaunchPriorVectorFactorKernel(const float *observations,
                                   float const *const *state_pointers,
                                   float *residuals, float *jacobians, int dim,
                                   int num_vectors, cudaStream_t stream) {
  size_t num_blocks = (num_vectors + kBlockSize - 1) / kBlockSize;
  prior_vector_factor_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      observations, state_pointers, residuals, jacobians, dim, num_vectors);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
