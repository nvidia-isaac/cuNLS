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

/**
 * @file information_factor_batch_kernels.cu
 * @brief Per-item sqrt-information weighting used by InformationFactorBatch.
 *
 * Each item is multiplied by the sqrt-information matrix of its factor
 * (FactorMeasurementIndex), with a fixed per-item summation order, so an item
 * gives bitwise the same result however many items are evaluated together.
 */

#include <cuda_runtime.h>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/factor/information/information_factor_batch.h"

namespace cunls {

namespace {

constexpr int kInformationBlockSize = 128;

/// Largest residual size whose per-thread staging fits in 48 KiB of shared memory;
/// larger sizes take the direct path.
constexpr size_t kMaxInformationResidualSize = 48 * 1024 / (kInformationBlockSize * sizeof(float));

/// Vector v of the layout described at ApplySqrtInformationKernel.
__device__ __forceinline__ float *VectorStart(float *data, int v, int vectors_per_item,
                                              int item_stride) {
  return data + static_cast<size_t>(v / vectors_per_item) * item_stride + (v % vectors_per_item);
}

/**
 * @brief In-place y = S * x for `num_vectors` vectors of length residual_size.
 *
 * Vector v starts at data[(v / vectors_per_item) * item_stride +
 * (v % vectors_per_item)] with element stride `element_stride`; it belongs to
 * item v / vectors_per_item. Residuals: one vector per item, element stride 1.
 * Jacobians: one vector per column, element stride jacobian_pitch.
 */
__global__ void ApplySqrtInformationKernel(const float *sqrt_information, float *data,
                                           int residual_size, int vectors_per_item, int item_stride,
                                           int element_stride, int num_vectors,
                                           const int *factor_ids, int num_factors) {
  extern __shared__ float staging[];
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vectors) {
    return;
  }
  const int m = FactorMeasurementIndex(v / vectors_per_item, factor_ids, num_factors);
  const float *S = sqrt_information + static_cast<size_t>(m) * residual_size * residual_size;
  float *x = VectorStart(data, v, vectors_per_item, item_stride);
  float *buf = staging + threadIdx.x;

  for (int k = 0; k < residual_size; ++k) {
    buf[k * blockDim.x] = x[k * element_stride];
  }
  for (int i = 0; i < residual_size; ++i) {
    float acc = 0.0f;
    for (int k = 0; k < residual_size; ++k) {
      acc += S[i * residual_size + k] * buf[k * blockDim.x];
    }
    x[i * element_stride] = acc;
  }
}

/**
 * @brief Same as ApplySqrtInformationKernel for residual sizes too large to
 * stage in shared memory: reads x and S directly from global memory, writes
 * y = S * x to this vector's slice of `scratch` (residual_size floats per
 * vector), then copies it back over x. Each thread owns its vector, and the
 * summation order is the same as the shared-memory path.
 */
__global__ void ApplySqrtInformationDirectKernel(const float *sqrt_information, float *data,
                                                 int residual_size, int vectors_per_item,
                                                 int item_stride, int element_stride,
                                                 int num_vectors, const int *factor_ids,
                                                 int num_factors, float *scratch) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_vectors) {
    return;
  }
  const int m = FactorMeasurementIndex(v / vectors_per_item, factor_ids, num_factors);
  const float *S = sqrt_information + static_cast<size_t>(m) * residual_size * residual_size;
  float *x = VectorStart(data, v, vectors_per_item, item_stride);
  float *y = scratch + static_cast<size_t>(v) * residual_size;
  for (int i = 0; i < residual_size; ++i) {
    float acc = 0.0f;
    for (int k = 0; k < residual_size; ++k) {
      acc += S[i * residual_size + k] * x[k * element_stride];
    }
    y[i] = acc;
  }
  for (int i = 0; i < residual_size; ++i) {
    x[i * element_stride] = y[i];
  }
}

void LaunchSqrtInformation(const float *sqrt_information, float *data, size_t residual_size,
                           size_t vectors_per_item, size_t item_stride, size_t element_stride,
                           size_t num_items, const int *factor_ids, size_t num_factors,
                           cudaStream_t stream) {
  if (num_items == 0 || residual_size == 0 || vectors_per_item == 0) {
    return;
  }
  if (num_factors == 0) {
    num_factors = num_items;
  }
  const size_t num_vectors = num_items * vectors_per_item;
  const size_t num_blocks = (num_vectors + kInformationBlockSize - 1) / kInformationBlockSize;
  if (residual_size <= kMaxInformationResidualSize) {
    const size_t shared_bytes = kInformationBlockSize * residual_size * sizeof(float);
    ApplySqrtInformationKernel<<<num_blocks, kInformationBlockSize, shared_bytes, stream>>>(
        sqrt_information, data, static_cast<int>(residual_size), static_cast<int>(vectors_per_item),
        static_cast<int>(item_stride), static_cast<int>(element_stride),
        static_cast<int>(num_vectors), factor_ids, static_cast<int>(num_factors));
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return;
  }
  // Stream-ordered scratch: freed once the kernel has run, without a host sync.
  float *scratch = nullptr;
  THROW_ON_CUDA_ERROR(
      cudaMallocAsync(&scratch, num_vectors * residual_size * sizeof(float), stream));
  ApplySqrtInformationDirectKernel<<<num_blocks, kInformationBlockSize, 0, stream>>>(
      sqrt_information, data, static_cast<int>(residual_size), static_cast<int>(vectors_per_item),
      static_cast<int>(item_stride), static_cast<int>(element_stride),
      static_cast<int>(num_vectors), factor_ids, static_cast<int>(num_factors), scratch);
  const cudaError_t launch = cudaGetLastError();
  THROW_ON_CUDA_ERROR(cudaFreeAsync(scratch, stream));
  THROW_ON_CUDA_ERROR(launch);
}

}  // namespace

void ApplyInformationToResidualItems(const float *sqrt_information, float *residuals,
                                     size_t residual_size, size_t num_items, const int *factor_ids,
                                     size_t num_factors, cudaStream_t stream) {
  LaunchSqrtInformation(sqrt_information, residuals, residual_size, 1, residual_size, 1, num_items,
                        factor_ids, num_factors, stream);
}

void ApplyInformationToJacobianItems(const float *sqrt_information, float *jacobians,
                                     size_t residual_size, size_t jacobian_pitch, size_t num_items,
                                     const int *factor_ids, size_t num_factors,
                                     cudaStream_t stream) {
  LaunchSqrtInformation(sqrt_information, jacobians, residual_size, jacobian_pitch,
                        residual_size * jacobian_pitch, jacobian_pitch, num_items, factor_ids,
                        num_factors, stream);
}

}  // namespace cunls
