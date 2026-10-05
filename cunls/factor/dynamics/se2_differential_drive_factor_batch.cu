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

/** SE(2) element as (cos θ, sin θ, x, y): the 4 non-trivial entries of the 3x3 storage. */
struct Pose {
  float c, s, x, y;
};

__device__ __forceinline__ Pose LoadPose(const float *T) {
  return {__ldg(T + 0), __ldg(T + 3), __ldg(T + 2), __ldg(T + 5)};
}

__device__ __forceinline__ Pose Compose(const Pose &a, const Pose &b) {
  return {a.c * b.c - a.s * b.s, a.s * b.c + a.c * b.s, a.c * b.x - a.s * b.y + a.x,
          a.s * b.x + a.c * b.y + a.y};
}

__device__ __forceinline__ Pose Inverse(const Pose &a) {
  return {a.c, -a.s, -(a.c * a.x + a.s * a.y), a.s * a.x - a.c * a.y};
}

__device__ __forceinline__ Pose Exp(const float *xi) {
  float E[9];
  lie_device::ExpSE2(xi, E);
  return {E[0], E[3], E[2], E[5]};
}

__device__ __forceinline__ void Log(const Pose &a, float *xi) {
  const float T[9] = {a.c, -a.s, a.x, a.s, a.c, a.y, 0.f, 0.f, 1.f};
  lie_device::LogSE2(T, xi);
}

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
    const Pose X = LoadPose(state_pointers[3 * t + 0]);
    const float *u = state_pointers[3 * t + 1];
    const Pose Y = LoadPose(state_pointers[3 * t + 2]);
    const float dt = __ldg(time_steps + f);
    const float wl = __ldg(u), wr = __ldg(u + 1);
    const float r = wheel_radius, b = track_width;

    // Step dt ξ, ξ = [r (ω_R + ω_L) / 2, 0, r (ω_R - ω_L) / b], and the defect.
    const float step[3] = {dt * 0.5f * r * (wr + wl), 0.f, dt * r / b * (wr - wl)};
    const Pose E = Exp(step);
    const Pose D = Compose(Inverse(Compose(X, E)), Y);
    float res[3];
    Log(D, res);

    float *o = out + threadIdx.x * kStride;
    o[0] = res[0];
    o[1] = res[1];
    o[2] = res[2];

    if constexpr (kJacobian) {
      // ∂r/∂T_k = -J_l⁻¹(r) Ad(E⁻¹); ∂r/∂u = -dt J_l⁻¹(r) J_r(step) ∂ξ/∂u;
      // ∂r/∂T_{k+1} = J_r⁻¹(r). J_l⁻¹(r) = J_r⁻¹(-r). Last rows are [0 0 1].
      float Jl_inv[9], Jr_inv[9], Jr_step[9];
      lie_device::SE2JrInv(-res[0], -res[1], -res[2], Jl_inv);
      lie_device::SE2JrInv(res[0], res[1], res[2], Jr_inv);
      lie_device::SE2Jr(step[0], step[1], step[2], Jr_step);
      const Pose Ei = Inverse(E);
      const float Ad[9] = {Ei.c, -Ei.s, Ei.y, Ei.s, Ei.c, -Ei.x, 0.f, 0.f, 1.f};
      const float half_r = 0.5f * r, r_over_b = r / b;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float *row = o + kRows + i * kColumns;
        const float *l = Jl_inv + 3 * i;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          row[j] = -(l[0] * Ad[j] + l[1] * Ad[3 + j] + l[2] * Ad[6 + j]);
          row[5 + j] = Jr_inv[3 * i + j];
        }
        // M = -dt J_l⁻¹ J_r(step); ∂ξ/∂u = [[r/2, r/2], [0, 0], [-r/b, r/b]].
        float m[3];
#pragma unroll
        for (int j = 0; j < 3; ++j)
          m[j] = -dt * (l[0] * Jr_step[j] + l[1] * Jr_step[3 + j] + l[2] * Jr_step[6 + j]);
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
