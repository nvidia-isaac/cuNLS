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
#include "cunls/factor/dynamics/se2_differential_drive_factor_batch.h"
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
constexpr int kRows = 3;                         // residual size
constexpr int kColumns = 3 + 2 + 3;              // T_k, u_k, T_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 24 floats per item

/**
 * One thread per item. The per-item outputs are staged in shared memory (odd
 * strides: no bank conflicts) and the block then writes its contiguous ranges
 * of residuals and Jacobians with coalesced stores. kJacobian selects the
 * residual-only variant at compile time.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SE2DifferentialDriveKernel(const float *__restrict__ time_steps, float wheel_radius,
                               float track_width, float const *const *__restrict__ state_pointers,
                               float *__restrict__ residuals, float *__restrict__ jacobians,
                               int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize : kRows;  // 27 or 3
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const lie_device::SE2Element X = lie_device::LoadSE2Element(state_pointers[3 * t + 0]);
    const float *u = state_pointers[3 * t + 1];
    const lie_device::SE2Element Y = lie_device::LoadSE2Element(state_pointers[3 * t + 2]);
    const float dt = __ldg(time_steps + f);
    const float wl = __ldg(u), wr = __ldg(u + 1);
    const float r = wheel_radius, b = track_width;

    // Step dt ξ, ξ = [r (ω_R + ω_L) / 2, 0, r (ω_R - ω_L) / b], and the defect.
    const float step[3] = {dt * 0.5f * r * (wr + wl), 0.f, dt * r / b * (wr - wl)};
    lie_device::SE2Element E;
    float res[3];
    lie_device::SE2StepDefect(X, Y, step, &E, res);

    float *o = out + threadIdx.x * kStride;
    o[0] = res[0];
    o[1] = res[1];
    o[2] = res[2];

    if constexpr (kJacobian) {
      // Step defect rows (lie_device::SE2StepDefectRow), with
      // ∂ξ/∂u = [[r/2, r/2], [0, 0], [-r/b, r/b]].
      lie_device::SE2StepDefectJacobian J;
      lie_device::InitSE2StepDefectJacobian(res, step, E, &J);
      const float half_r = 0.5f * r, r_over_b = r / b;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float *row = o + kRows + i * kColumns;
        float m[3];
        lie_device::SE2StepDefectRow(J, i, dt, row, m, row + 5);
        row[3] = m[0] * half_r - m[2] * r_over_b;
        row[4] = m[0] * half_r + m[2] * r_over_b;
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

SE2DifferentialDriveFactorBatch::SE2DifferentialDriveFactorBatch(const float *time_steps,
                                                                 float wheel_radius,
                                                                 float track_width, size_t capacity)
    : SizedFactorBatch(capacity),
      time_steps_(time_steps),
      wheel_radius_(wheel_radius),
      track_width_(track_width) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE2DifferentialDriveFactorBatch: time_steps must not be null");
  }
  if (!(wheel_radius_ > 0.f) || !(track_width_ > 0.f)) {
    throw std::invalid_argument(
        "SE2DifferentialDriveFactorBatch: wheel_radius and track_width must be positive");
  }
}

bool SE2DifferentialDriveFactorBatch::Evaluate(float *residuals, float *jacobians,
                                               float const *const *state_pointers,
                                               cudaStream_t stream, const int *factor_ids,
                                               size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE2DifferentialDriveKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheel_radius_, track_width_, state_pointers, residuals, jacobians,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  } else {
    SE2DifferentialDriveKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheel_radius_, track_width_, state_pointers, residuals, nullptr,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
