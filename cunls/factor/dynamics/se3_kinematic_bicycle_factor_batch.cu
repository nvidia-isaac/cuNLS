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
#include "cunls/factor/dynamics/se3_kinematic_bicycle_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/math/lie_device.cuh"

namespace cunls {
namespace {

constexpr int kBlockSize = 64;                   // keeps the staging buffer under 48 KB
constexpr int kRows = 6;                         // e[2:6], speed, steering
constexpr int kColumns = 6 + 2 + 2 + 6 + 2;      // T_k, z_k, u_k, T_{k+1}, z_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 108 floats per item

/** Rows 0..2 of an SE(3) storage matrix (row-major 4x4): R and t, pitch 4. */
__device__ __forceinline__ void LoadPose(const float *T, float *P) {
#pragma unroll
  for (int i = 0; i < 12; ++i) P[i] = __ldg(T + i);
}

/** C = A B for SE(3) poses stored as rows 0..2 (pitch 4). */
__device__ __forceinline__ void Compose(const float *A, const float *B, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      C[i * 4 + j] = A[i * 4] * B[j] + A[i * 4 + 1] * B[4 + j] + A[i * 4 + 2] * B[8 + j] +
                     (j == 3 ? A[i * 4 + 3] : 0.f);
    }
  }
}

/**
 * One thread per item; outputs staged in shared memory (odd stride) and written
 * with coalesced stores. kJacobian selects the residual-only variant.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SE3KinematicBicycleKernel(const float *__restrict__ time_steps, float wheelbase,
                              float const *const *__restrict__ state_pointers,
                              float *__restrict__ residuals, float *__restrict__ jacobians,
                              int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize + 1 : kRows + 1;  // 115 or 7
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    float X[16], Y[16];
    LoadPose(state_pointers[5 * t + 0], X);
    const float *z = state_pointers[5 * t + 1];
    const float *u = state_pointers[5 * t + 2];
    LoadPose(state_pointers[5 * t + 3], Y);
    const float *z_next = state_pointers[5 * t + 4];
    const float dt = __ldg(time_steps + f);
    const float v = __ldg(z), delta = __ldg(z + 1), tan_delta = tanf(delta);

    // Pose: step dt ξ, ξ = [0, 0, v tan δ / L, v, 0, 0]. Speed and steering: Euler.
    const float step[6] = {0.f, 0.f, dt * v * tan_delta / wheelbase, dt * v, 0.f, 0.f};
    float E[16], Phi[16], Phi_inv[16], D[16], e[6];
    lie_device::ExpSE3(step, E, 4);
    Compose(X, E, Phi);
    lie_device::InverseSE3(Phi, 4, Phi_inv);
    Compose(Phi_inv, Y, D);
    lie_device::LogSE3(D, 4, e);

    float *o = out + threadIdx.x * kStride;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = e[2 + i];
    o[4] = __ldg(z_next) - v - dt * __ldg(u);
    o[5] = __ldg(z_next + 1) - delta - dt * __ldg(u + 1);

    if constexpr (kJacobian) {
      // Pose rows (e rows 2..5): ∂e/∂T_k = -J_l⁻¹(e) Ad(E⁻¹),
      // ∂e/∂z = -dt J_l⁻¹(e) J_r(step) ∂ξ/∂z, ∂e/∂T_{k+1} = J_l⁻¹(-e); nothing on u, z_{k+1}.
      float Jl_inv[36], Jr_inv[36], Jr_step[36], E_inv[16];
      lie_device::SE3JacobianLeftInverse(e, Jl_inv, 6);
      const float minus_e[6] = {-e[0], -e[1], -e[2], -e[3], -e[4], -e[5]};
      lie_device::SE3JacobianLeftInverse(minus_e, Jr_inv, 6);
      const float minus_step[6] = {0.f, 0.f, -step[2], -step[3], 0.f, 0.f};
      lie_device::SE3JacobianLeft(minus_step, Jr_step, 6);  // J_r(step)
      lie_device::InverseSE3(E, 4, E_inv);
      // ∂ξ/∂z: ξ_2 = v tan δ / L, ξ_3 = v.
      const float dw_dv = tan_delta / wheelbase;
      const float dw_ddelta = v * (1.f + tan_delta * tan_delta) / wheelbase;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float *l = Jl_inv + (2 + i) * 6;
        float *row = o + kRows + i * kColumns;
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 6; ++k) {
          float ad[6];
          lie_device::AdjointSE3RowFromTransform(E_inv, 4, k, ad);
#pragma unroll
          for (int j = 0; j < 6; ++j) acc[j] += l[k] * ad[j];
        }
        float m2 = 0.f, m3 = 0.f;  // (-dt l J_r(step)) columns 2 and 3
#pragma unroll
        for (int k = 0; k < 6; ++k) {
          m2 += l[k] * Jr_step[k * 6 + 2];
          m3 += l[k] * Jr_step[k * 6 + 3];
        }
        m2 *= -dt;
        m3 *= -dt;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          row[j] = -acc[j];
          row[10 + j] = Jr_inv[(2 + i) * 6 + j];
        }
        row[6] = m2 * dw_dv + m3;
        row[7] = m2 * dw_ddelta;
        row[8] = 0.f;
        row[9] = 0.f;
        row[16] = 0.f;
        row[17] = 0.f;
      }
      // Speed and steering rows: z_{k+1} - z_k - dt u_k.
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float *row = o + kRows + (4 + i) * kColumns;
#pragma unroll
        for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
        row[6 + i] = -1.f;
        row[8 + i] = -dt;
        row[16 + i] = 1.f;
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

SE3KinematicBicycleFactorBatch::SE3KinematicBicycleFactorBatch(const float *time_steps,
                                                               float wheelbase, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps), wheelbase_(wheelbase) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE3KinematicBicycleFactorBatch: time_steps must not be null");
  }
  if (!(wheelbase_ > 0.f)) {
    throw std::invalid_argument("SE3KinematicBicycleFactorBatch: wheelbase must be positive");
  }
}

bool SE3KinematicBicycleFactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE3KinematicBicycleKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheelbase_, state_pointers, residuals, jacobians, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  } else {
    SE3KinematicBicycleKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheelbase_, state_pointers, residuals, nullptr, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
