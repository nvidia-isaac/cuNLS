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
#include "cunls/factor/dynamics/se2_kinematics_factor_batch.h"
#include "cunls/math/lie_device.cuh"

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
constexpr int kRows = 3;
constexpr int kColumns = 3 + 3 + 3;              // X_k, ξ_k, X_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 27 floats per item

/**
 * One thread per item; outputs staged in shared memory (odd stride) and written
 * with coalesced stores. kJacobian selects the residual-only variant.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SE2KinematicsKernel(const float *__restrict__ time_steps,
                        float const *const *__restrict__ state_pointers,
                        float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                        const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize + 1 : kRows;  // 31 or 3
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const lie_device::SE2Element X = lie_device::LoadSE2Element(state_pointers[3 * t + 0]);
    const float *xi = state_pointers[3 * t + 1];
    const lie_device::SE2Element Y = lie_device::LoadSE2Element(state_pointers[3 * t + 2]);
    const float dt = __ldg(time_steps + f);
    const float step[3] = {dt * __ldg(xi), dt * __ldg(xi + 1), dt * __ldg(xi + 2)};
    lie_device::SE2Element E;
    float res[3];
    lie_device::SE2StepDefect(X, Y, step, &E, res);

    float *o = out + threadIdx.x * kStride;
    o[0] = res[0];
    o[1] = res[1];
    o[2] = res[2];

    if constexpr (kJacobian) {
      // ∂r/∂X_k = -J_l⁻¹(r) Ad(E⁻¹); ∂r/∂ξ = -dt J_l⁻¹(r) J_r(step); ∂r/∂X_{k+1} = J_r⁻¹(r).
      lie_device::SE2StepDefectJacobian J;
      lie_device::InitSE2StepDefectJacobian(res, step, E, &J);
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float *row = o + kRows + i * kColumns;
        lie_device::SE2StepDefectRow(J, i, dt, row, row + 3, row + 6);
      }
    }
  }
  __syncthreads();

  // Coalesced block writes: items [first, first + count) are contiguous in the outputs.
  const int count = min(kBlockSize, num_items - first);
  for (int i = threadIdx.x; i < count * kRows; i += kBlockSize) {
    residuals[static_cast<size_t>(first) * kRows + i] = out[(i / kRows) * kStride + i % kRows];
  }
  if constexpr (kJacobian) {
    for (int i = threadIdx.x; i < count * kJacobianSize; i += kBlockSize) {
      jacobians[static_cast<size_t>(first) * kJacobianSize + i] =
          out[(i / kJacobianSize) * kStride + kRows + i % kJacobianSize];
    }
  }
}

}  // namespace

SE2KinematicsFactorBatch::SE2KinematicsFactorBatch(const float *time_steps, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE2KinematicsFactorBatch: time_steps must not be null");
  }
}

bool SE2KinematicsFactorBatch::Evaluate(float *residuals, float *jacobians,
                                        float const *const *state_pointers, cudaStream_t stream,
                                        const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE2KinematicsKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, jacobians, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  } else {
    SE2KinematicsKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, nullptr, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
