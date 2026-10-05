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

#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/factor/clearance/se3_sphere_clearance_factor_batch.h"

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

constexpr int kBlockSize = 128;
constexpr int kColumns = 6;

/**
 * One thread per item; residual and Jacobian staged in shared memory (odd
 * stride) and written with coalesced stores. kJacobian selects the
 * residual-only variant.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SE3SphereClearanceKernel(const float *__restrict__ obstacles, float margin,
                             float const *const *__restrict__ state_pointers,
                             float *__restrict__ residuals, float *__restrict__ jacobians,
                             int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = 7;  // 1 + 6 floats per item, odd
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const float *o = obstacles + 4 * static_cast<size_t>(f);
    const float *T = state_pointers[t];
    // SE(3) storage: row-major 4x4, t in column 3.
    float d[3], R[9];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      d[i] = __ldg(T + 4 * i + 3) - __ldg(o + i);
#pragma unroll
      for (int j = 0; j < 3; ++j) R[3 * i + j] = __ldg(T + 4 * i + j);
    }
    const float dist = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const float radius = __ldg(o + 3);
    float *out_item = out + threadIdx.x * kStride;
    out_item[0] = radius + margin - dist;
    if constexpr (kJacobian) {
      // Tangent [φ, ρ]: ∂c/∂δφ = 0, ∂c/∂δρ = -(dᵀ / ‖d‖) R.
      const float inv = dist > 0.f ? 1.f / dist : 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        out_item[1 + j] = 0.f;
        out_item[4 + j] = -inv * (d[0] * R[j] + d[1] * R[3 + j] + d[2] * R[6 + j]);
      }
    }
  }
  __syncthreads();

  const int count = min(kBlockSize, num_items - first);
  for (int i = threadIdx.x; i < count; i += kBlockSize) {
    residuals[first + i] = out[i * kStride];
  }
  if constexpr (kJacobian) {
    for (int i = threadIdx.x; i < count * kColumns; i += kBlockSize) {
      jacobians[static_cast<size_t>(first) * kColumns + i] =
          out[(i / kColumns) * kStride + 1 + i % kColumns];
    }
  }
}

}  // namespace

SE3SphereClearanceFactorBatch::SE3SphereClearanceFactorBatch(const float *obstacles, float margin,
                                                             size_t capacity)
    : SizedFactorBatch(capacity), obstacles_(obstacles), margin_(margin) {
  if (obstacles_ == nullptr) {
    throw std::invalid_argument("SE3SphereClearanceFactorBatch: obstacles must not be null");
  }
}

bool SE3SphereClearanceFactorBatch::Evaluate(float *residuals, float *jacobians,
                                             float const *const *state_pointers,
                                             cudaStream_t stream, const int *factor_ids,
                                             size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE3SphereClearanceKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        obstacles_, margin_, state_pointers, residuals, jacobians, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  } else {
    SE3SphereClearanceKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        obstacles_, margin_, state_pointers, residuals, nullptr, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
