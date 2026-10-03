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
#include <stdexcept>
#include <string>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize); }

// One thread per constraint row of an item.
__global__ void AugmentedLagrangianKernel(bool inequality, float scale, const float *multipliers,
                                          const float *penalties, float *values, float *jacobians,
                                          int rows, int pitch, size_t num_rows_total,
                                          const int *factor_ids, int num_factors) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= num_rows_total) return;
  const int item = static_cast<int>(idx / rows);
  const int row = static_cast<int>(idx % rows);
  const int f = FactorMeasurementIndex(item, factor_ids, num_factors);
  const float rho = penalties[f];
  const float sqrt_rho = sqrtf(rho);
  // z = c + λ/ρ; an inactive inequality row (z <= 0) contributes nothing.
  const float z = scale * values[idx] + multipliers[static_cast<size_t>(f) * rows + row] / rho;
  const bool active = !inequality || z > 0.f;
  values[idx] = active ? sqrt_rho * z : 0.f;
  if (jacobians != nullptr) {
    const float w = active ? scale * sqrt_rho : 0.f;
    float *j = jacobians + idx * pitch;
    for (int c = 0; c < pitch; ++c) j[c] *= w;
  }
}

__global__ void ScaleKernel(float scale, float *data, size_t n) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx < n) data[idx] *= scale;
}

__global__ void FillKernel(float value, float *data, size_t n) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx < n) data[idx] = value;
}

size_t JacobianPitch(const FactorBatch &batch) {
  const auto sizes = batch.StateSizes();
  return std::accumulate(sizes.begin(), sizes.end(), size_t{0});
}

}  // namespace

void ApplyAugmentedLagrangian(ConstraintKind kind, float scale, const float *multipliers,
                              const float *penalties, float *values, float *jacobians, size_t rows,
                              size_t jacobian_pitch, size_t num_items, const int *factor_ids,
                              size_t num_factors, cudaStream_t stream) {
  const size_t total = num_items * rows;
  if (total == 0) return;
  AugmentedLagrangianKernel<<<Blocks(total), kBlockSize, 0, stream>>>(
      kind == ConstraintKind::kInequality, scale, multipliers, penalties, values, jacobians,
      static_cast<int>(rows), static_cast<int>(jacobian_pitch), total, factor_ids,
      static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void ScaleConstraintRows(float scale, float *values, size_t num_values, float *jacobians,
                         size_t num_jacobian_values, cudaStream_t stream) {
  if (scale == 1.f) return;
  if (num_values > 0) {
    ScaleKernel<<<Blocks(num_values), kBlockSize, 0, stream>>>(scale, values, num_values);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
  if (jacobians != nullptr && num_jacobian_values > 0) {
    ScaleKernel<<<Blocks(num_jacobian_values), kBlockSize, 0, stream>>>(scale, jacobians,
                                                                        num_jacobian_values);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }
}

ConstraintFactorBatchBase::ConstraintFactorBatchBase(ConstraintKind kind, float scale)
    : kind_(kind), scale_(scale) {
  if (!(scale > 0.f)) {
    throw std::invalid_argument("Constraint factor batch: scale must be positive, got " +
                                std::to_string(scale));
  }
}

ConstraintFactorBatchBase::ConstraintFactorBatchBase(ConstraintKind kind, float scale,
                                                     size_t capacity)
    : FactorBatch(capacity), kind_(kind), scale_(scale) {
  if (!(scale > 0.f)) {
    throw std::invalid_argument("Constraint factor batch: scale must be positive, got " +
                                std::to_string(scale));
  }
}

void ConstraintFactorBatchBase::AllocateMultipliers() {
  const size_t capacity = Capacity();
  multipliers_ = dvector<float>(0.f, capacity * ResidualsSize());
  penalties_ = dvector<float>(kDefaultPenalty, capacity);
}

void ConstraintFactorBatchBase::ResetMultipliers(cudaStream_t stream) {
  if (multipliers_.size() == 0) return;
  THROW_ON_CUDA_ERROR(
      cudaMemsetAsync(multipliers_.data(), 0, multipliers_.size() * sizeof(float), stream));
}

void ConstraintFactorBatchBase::SetPenalty(float penalty, cudaStream_t stream) {
  if (!(penalty > 0.f)) {
    throw std::invalid_argument("SetPenalty: the penalty must be positive, got " +
                                std::to_string(penalty));
  }
  if (penalties_.size() == 0) return;
  FillKernel<<<Blocks(penalties_.size()), kBlockSize, 0, stream>>>(penalty, penalties_.data(),
                                                                   penalties_.size());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

bool ConstraintFactorBatchBase::Evaluate(float *residuals, float *jacobians,
                                         float const *const *state_pointers, cudaStream_t stream,
                                         const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  if (!EvaluateConstraintRows(residuals, jacobians, state_pointers, stream, factor_ids,
                              num_items)) {
    return false;
  }
  ApplyAugmentedLagrangian(kind_, scale_, multipliers_.data(), penalties_.data(), residuals,
                           jacobians, ResidualsSize(), JacobianPitch(*this), num_items, factor_ids,
                           num_factors, stream);
  return true;
}

bool ConstraintFactorBatchBase::EvaluateConstraint(float *values, float *jacobians,
                                                   float const *const *state_pointers,
                                                   cudaStream_t stream, const int *factor_ids,
                                                   size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  if (!EvaluateConstraintRows(values, jacobians, state_pointers, stream, factor_ids, num_items)) {
    return false;
  }
  const size_t num_values = num_items * ResidualsSize();
  ScaleConstraintRows(scale_, values, num_values, jacobians, num_values * JacobianPitch(*this),
                      stream);
  return true;
}

ConstraintFactorBatch::ConstraintFactorBatch(FactorBatch *inner, ConstraintKind kind, float scale)
    : ConstraintFactorBatchBase(kind, scale), inner_(inner) {
  if (inner_ == nullptr) {
    const std::string msg = "ConstraintFactorBatch: the wrapped factor batch must not be null";
    LogError(msg);
    throw std::invalid_argument(msg);
  }
  AllocateMultipliers();
}

}  // namespace cunls
