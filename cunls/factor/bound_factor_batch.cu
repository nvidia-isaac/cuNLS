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

// Kernels of the vector-state constraint factors: BoundFactorBatch and
// HalfspaceFactorBatch.

#include <cuda_runtime.h>

#include "cunls/common/helper.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/halfspace_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize); }

// One thread per (item, component): writes the upper row i and the lower row
// dim + i of the item, and their Jacobian rows (±e_i).
__global__ void BoundKernel(const float *lower, const float *upper,
                            float const *const *state_pointers, float *values, float *jacobians,
                            int dim, size_t total, const int *factor_ids, int num_factors) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= total) return;
  const int item = static_cast<int>(idx / dim);
  const int i = static_cast<int>(idx % dim);
  const int f = FactorMeasurementIndex(item, factor_ids, num_factors);
  const float x = state_pointers[item][i];
  const size_t bound = static_cast<size_t>(f) * dim + i;
  const size_t row_upper = static_cast<size_t>(item) * 2 * dim + i;
  const size_t row_lower = row_upper + dim;
  values[row_upper] = x - upper[bound];
  values[row_lower] = lower[bound] - x;
  if (jacobians != nullptr) {
    float *ju = jacobians + row_upper * dim;
    float *jl = jacobians + row_lower * dim;
    for (int c = 0; c < dim; ++c) {
      ju[c] = c == i ? 1.f : 0.f;
      jl[c] = c == i ? -1.f : 0.f;
    }
  }
}

// One thread per item: r = a^T x - b, J = a^T.
__global__ void HalfspaceKernel(const float *normals, const float *offsets,
                                float const *const *state_pointers, float *residuals,
                                float *jacobians, int dim, size_t num_items, const int *factor_ids,
                                int num_factors) {
  const size_t item = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (item >= num_items) return;
  const int f = FactorMeasurementIndex(static_cast<int>(item), factor_ids, num_factors);
  const float *x = state_pointers[item];
  const float *a = normals + static_cast<size_t>(f) * dim;
  float r = -offsets[f];
  for (int c = 0; c < dim; ++c) r += a[c] * x[c];
  residuals[item] = r;
  if (jacobians != nullptr) {
    for (int c = 0; c < dim; ++c) jacobians[item * dim + c] = a[c];
  }
}

}  // namespace

void LaunchBoundConstraintKernel(const float *lower, const float *upper,
                                 float const *const *state_pointers, float *values,
                                 float *jacobians, int dim, size_t num_items, const int *factor_ids,
                                 size_t num_factors, cudaStream_t stream) {
  const size_t total = num_items * dim;
  if (total == 0) return;
  BoundKernel<<<Blocks(total), kBlockSize, 0, stream>>>(lower, upper, state_pointers, values,
                                                        jacobians, dim, total, factor_ids,
                                                        static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchHalfspaceKernel(const float *normals, const float *offsets,
                           float const *const *state_pointers, float *residuals, float *jacobians,
                           int dim, size_t num_items, const int *factor_ids, size_t num_factors,
                           cudaStream_t stream) {
  if (num_items == 0) return;
  HalfspaceKernel<<<Blocks(num_items), kBlockSize, 0, stream>>>(
      normals, offsets, state_pointers, residuals, jacobians, dim, num_items, factor_ids,
      static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
