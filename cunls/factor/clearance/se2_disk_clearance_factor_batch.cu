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
#include "cunls/factor/clearance/se2_disk_clearance_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"

namespace cunls {
namespace {

constexpr int kBlockSize = 128;
constexpr int kColumns = 3;

/**
 * One thread per item; residual and Jacobian staged in shared memory (odd
 * stride) and written with coalesced stores. kJacobian selects the
 * residual-only variant.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SE2DiskClearanceKernel(const float *__restrict__ obstacles, float margin,
                           float const *const *__restrict__ state_pointers,
                           float *__restrict__ residuals, float *__restrict__ jacobians,
                           int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = 5;  // 1 + 3 floats per item, odd
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const float *o = obstacles + 3 * static_cast<size_t>(f);
    const float *T = state_pointers[t];
    // SE(2) storage: [c, -s, x; s, c, y; 0 0 1].
    const float c = __ldg(T), s = __ldg(T + 3);
    const float d[2] = {__ldg(T + 2) - __ldg(o), __ldg(T + 5) - __ldg(o + 1)};
    const float dist = sqrtf(d[0] * d[0] + d[1] * d[1]);
    const float radius = __ldg(o + 2);
    float *out_item = out + threadIdx.x * kStride;
    out_item[0] = radius + margin - dist;
    if constexpr (kJacobian) {
      // ∂c/∂δv = -(dᵀ / ‖d‖) R, ∂c/∂δθ = 0.
      const float u0 = dist > 0.f ? d[0] / dist : 0.f, u1 = dist > 0.f ? d[1] / dist : 0.f;
      out_item[1] = -(u0 * c + u1 * s);
      out_item[2] = -(-u0 * s + u1 * c);
      out_item[3] = 0.f;
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

SE2DiskClearanceFactorBatch::SE2DiskClearanceFactorBatch(const float *obstacles, float margin,
                                                         size_t capacity)
    : SizedFactorBatch(capacity), obstacles_(obstacles), margin_(margin) {
  if (obstacles_ == nullptr) {
    throw std::invalid_argument("SE2DiskClearanceFactorBatch: obstacles must not be null");
  }
}

bool SE2DiskClearanceFactorBatch::Evaluate(float *residuals, float *jacobians,
                                           float const *const *state_pointers, cudaStream_t stream,
                                           const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE2DiskClearanceKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        obstacles_, margin_, state_pointers, residuals, jacobians, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  } else {
    SE2DiskClearanceKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        obstacles_, margin_, state_pointers, residuals, nullptr, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
