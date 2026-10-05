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
#include "cunls/factor/dynamics/se3_differential_drive_factor_batch.h"
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
constexpr int kRows = 4;                         // e[2:6]
constexpr int kColumns = 6 + 2 + 6;              // T_k, u_k, T_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 56 floats per item

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
    SE3DifferentialDriveKernel(const float *__restrict__ time_steps, float wheel_radius,
                               float track_width, float const *const *__restrict__ state_pointers,
                               float *__restrict__ residuals, float *__restrict__ jacobians,
                               int num_items, const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize + 1 : kRows + 1;  // 61 or 5
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    float X[16], Y[16];
    LoadPose(state_pointers[3 * t + 0], X);
    const float *u = state_pointers[3 * t + 1];
    LoadPose(state_pointers[3 * t + 2], Y);
    const float dt = __ldg(time_steps + f);
    const float wl = __ldg(u), wr = __ldg(u + 1);
    const float r = wheel_radius, b = track_width;

    // Step dt ξ, ξ = [0, 0, ω, v, 0, 0], and e = Log((X Exp(dt ξ))⁻¹ Y).
    const float step[6] = {0.f, 0.f, dt * r / b * (wr - wl), dt * 0.5f * r * (wr + wl), 0.f, 0.f};
    float E[16], Phi[16], Phi_inv[16], D[16], e[6];
    lie_device::ExpSE3(step, E, 4);
    Compose(X, E, Phi);
    lie_device::InverseSE3(Phi, 4, Phi_inv);
    Compose(Phi_inv, Y, D);
    lie_device::LogSE3(D, 4, e);

    float *o = out + threadIdx.x * kStride;
#pragma unroll
    for (int i = 0; i < kRows; ++i) o[i] = e[2 + i];

    if constexpr (kJacobian) {
      // Full-row Jacobians, rows 2..5: ∂e/∂T_k = -J_l⁻¹(e) Ad(E⁻¹),
      // ∂e/∂u = -dt J_l⁻¹(e) J_r(step) ∂ξ/∂u, ∂e/∂T_{k+1} = J_r⁻¹(e) = J_l⁻¹(-e).
      float Jl_inv[36], Jr_inv[36], Jr_step[36], E_inv[16];
      lie_device::SE3JacobianLeftInverse(e, Jl_inv, 6);
      const float minus_e[6] = {-e[0], -e[1], -e[2], -e[3], -e[4], -e[5]};
      lie_device::SE3JacobianLeftInverse(minus_e, Jr_inv, 6);
      const float minus_step[6] = {0.f, 0.f, -step[2], -step[3], 0.f, 0.f};
      lie_device::SE3JacobianLeft(minus_step, Jr_step, 6);  // J_r(step)
      lie_device::InverseSE3(E, 4, E_inv);
      // ∂ξ/∂u: ξ_2 = ω = (r/b)(ω_R - ω_L), ξ_3 = v = (r/2)(ω_R + ω_L).
      const float half_r = 0.5f * r, r_over_b = r / b;
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float *l = Jl_inv + (2 + i) * 6;
        float *row = o + kRows + i * kColumns;
        // -l Ad(E⁻¹), one Ad row at a time.
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
          row[8 + j] = Jr_inv[(2 + i) * 6 + j];
        }
        row[6] = -m2 * r_over_b + m3 * half_r;
        row[7] = m2 * r_over_b + m3 * half_r;
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

SE3DifferentialDriveFactorBatch::SE3DifferentialDriveFactorBatch(const float *time_steps,
                                                                 float wheel_radius,
                                                                 float track_width, size_t capacity)
    : SizedFactorBatch(capacity),
      time_steps_(time_steps),
      wheel_radius_(wheel_radius),
      track_width_(track_width) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE3DifferentialDriveFactorBatch: time_steps must not be null");
  }
  if (!(wheel_radius_ > 0.f) || !(track_width_ > 0.f)) {
    throw std::invalid_argument(
        "SE3DifferentialDriveFactorBatch: wheel_radius and track_width must be positive");
  }
}

bool SE3DifferentialDriveFactorBatch::Evaluate(float *residuals, float *jacobians,
                                               float const *const *state_pointers,
                                               cudaStream_t stream, const int *factor_ids,
                                               size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE3DifferentialDriveKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheel_radius_, track_width_, state_pointers, residuals, jacobians,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  } else {
    SE3DifferentialDriveKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, wheel_radius_, track_width_, state_pointers, residuals, nullptr,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
