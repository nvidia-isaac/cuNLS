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
#include "cunls/factor/weighted_factor_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

__global__ void UniformScaleKernel(float weight, float* data,
                                   size_t total_elements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    data[idx] *= weight;
  }
}

__global__ void PerFactorScaleResidualsKernel(const float* weights,
                                              float* residuals,
                                              size_t residual_size,
                                              size_t num_factors) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = num_factors * residual_size;
  if (idx < total) {
    const size_t factor_idx = idx / residual_size;
    residuals[idx] *= weights[factor_idx];
  }
}

__global__ void PerFactorScaleJacobiansKernel(const float* weights,
                                              float* jacobians,
                                              size_t stride,
                                              size_t num_factors) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = num_factors * stride;
  if (idx < total) {
    const size_t factor_idx = idx / stride;
    jacobians[idx] *= weights[factor_idx];
  }
}

}  // namespace

void ApplyUniformWeightToResiduals(float weight, float* residuals,
                                   size_t total_elements,
                                   cudaStream_t stream) {
  const int num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;
  UniformScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weight, residuals, total_elements);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyUniformWeightToJacobians(float weight, float* jacobians,
                                   size_t total_elements,
                                   cudaStream_t stream) {
  const int num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;
  UniformScaleKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weight, jacobians, total_elements);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyPerFactorWeightToResiduals(const float* weights, float* residuals,
                                     size_t residual_size, size_t num_factors,
                                     cudaStream_t stream) {
  const size_t total = num_factors * residual_size;
  const int num_blocks = (total + kBlockSize - 1) / kBlockSize;
  PerFactorScaleResidualsKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weights, residuals, residual_size, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ApplyPerFactorWeightToJacobians(const float* weights, float* jacobians,
                                     size_t residual_size,
                                     size_t jacobian_pitch, size_t num_factors,
                                     cudaStream_t stream) {
  const size_t stride = residual_size * jacobian_pitch;
  const size_t total = num_factors * stride;
  const int num_blocks = (total + kBlockSize - 1) / kBlockSize;
  PerFactorScaleJacobiansKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      weights, jacobians, stride, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
