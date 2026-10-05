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
#include "cunls/factor/dynamics/se2_kinematic_bicycle_factor_batch.h"
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
constexpr int kRows = 5;                         // residual size
constexpr int kColumns = 3 + 2 + 2 + 3 + 2;      // T_k, z_k, u_k, T_{k+1}, z_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 60 floats per item

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
    SE2KinematicBicycleKernel(const float *__restrict__ time_steps, float wheelbase,
                              float const *const *__restrict__ state_pointers,
                              float *__restrict__ residuals, float *__restrict__ jacobians,
                              int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize : kRows;  // 65 or 5
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const Pose X = LoadPose(state_pointers[5 * t + 0]);
    const float *z = state_pointers[5 * t + 1];
    const float *u = state_pointers[5 * t + 2];
    const Pose Y = LoadPose(state_pointers[5 * t + 3]);
    const float *z_next = state_pointers[5 * t + 4];
    const float dt = __ldg(time_steps + f);
    const float v = __ldg(z), tan_delta = tanf(__ldg(z + 1));
    const float a = __ldg(u), steer_rate = __ldg(u + 1);

    // Pose: step dt ξ with ξ = [v, 0, v tan δ / L]. Speed and steering: Euler.
    const float step[3] = {dt * v, 0.f, dt * v * tan_delta / wheelbase};
    const Pose E = Exp(step);
    const Pose D = Compose(Inverse(Compose(X, E)), Y);
    float res[3];
    Log(D, res);

    float *o = out + threadIdx.x * kStride;
    o[0] = res[0];
    o[1] = res[1];
    o[2] = res[2];
    o[3] = __ldg(z_next) - v - dt * a;
    o[4] = __ldg(z_next + 1) - __ldg(z + 1) - dt * steer_rate;

    if constexpr (kJacobian) {
      // Pose rows: ∂r/∂T_k = -J_l⁻¹(r) Ad(E⁻¹); ∂r/∂z = -dt J_l⁻¹(r) J_r(step) ∂ξ/∂z;
      // ∂r/∂T_{k+1} = J_r⁻¹(r); nothing on u_k and z_{k+1}. J_l⁻¹(r) = J_r⁻¹(-r).
      float Jl_inv[9], Jr_inv[9], Jr_step[9];
      lie_device::SE2JrInv(-res[0], -res[1], -res[2], Jl_inv);
      lie_device::SE2JrInv(res[0], res[1], res[2], Jr_inv);
      lie_device::SE2Jr(step[0], step[1], step[2], Jr_step);
      const Pose Ei = Inverse(E);
      const float Ad[9] = {Ei.c, -Ei.s, Ei.y, Ei.s, Ei.c, -Ei.x, 0.f, 0.f, 1.f};
      // ∂ξ/∂z = [[1, 0], [0, 0], [tan δ / L, v (1 + tan² δ) / L]].
      const float dw_dv = tan_delta / wheelbase;
      const float dw_ddelta = v * (1.f + tan_delta * tan_delta) / wheelbase;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float *row = o + kRows + i * kColumns;
        const float *l = Jl_inv + 3 * i;
        float m[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          row[j] = -(l[0] * Ad[j] + l[1] * Ad[3 + j] + l[2] * Ad[6 + j]);
          row[7 + j] = Jr_inv[3 * i + j];
          m[j] = -dt * (l[0] * Jr_step[j] + l[1] * Jr_step[3 + j] + l[2] * Jr_step[6 + j]);
        }
        row[3] = m[0] + m[2] * dw_dv;
        row[4] = m[2] * dw_ddelta;
        row[5] = 0.f;
        row[6] = 0.f;
        row[10] = 0.f;
        row[11] = 0.f;
      }
      // Speed and steering rows: z_{k+1} - z_k - dt u_k.
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float *row = o + kRows + (3 + i) * kColumns;
#pragma unroll
        for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
        row[3 + i] = -1.f;
        row[5 + i] = -dt;
        row[10 + i] = 1.f;
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

SE2KinematicBicycleFactorBatch::SE2KinematicBicycleFactorBatch(const float *time_steps,
                                                               float wheelbase, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps), wheelbase_(wheelbase) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE2KinematicBicycleFactorBatch: time_steps must not be null");
  }
  if (!(wheelbase_ > 0.f)) {
    throw std::invalid_argument("SE2KinematicBicycleFactorBatch: wheelbase must be positive");
  }
}

bool SE2KinematicBicycleFactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE2KinematicBicycleKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheelbase_, state_pointers, residuals, jacobians, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  } else {
    SE2KinematicBicycleKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheelbase_, state_pointers, residuals, nullptr, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
