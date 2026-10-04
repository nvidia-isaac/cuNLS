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
#include "cunls/factor/dynamics/se3_kinematics_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/math/lie_device.cuh"

namespace cunls {
namespace {

constexpr int kBlockSize = 64;  // keeps the staging buffer under 48 KB
constexpr int kRows = 6;
constexpr int kColumns = 6 + 6 + 6;              // T_k, ξ_k, T_{k+1}
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
    SE3KinematicsKernel(const float *__restrict__ time_steps,
                        float const *const *__restrict__ state_pointers,
                        float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                        const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize + 1 : kRows + 1;  // 115 or 7
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    float X[16], Y[16];
    LoadPose(state_pointers[3 * t + 0], X);
    const float *xi = state_pointers[3 * t + 1];
    LoadPose(state_pointers[3 * t + 2], Y);
    const float dt = __ldg(time_steps + f);
    float step[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) step[i] = dt * __ldg(xi + i);

    float E[16], Phi[16], Phi_inv[16], D[16], e[6];
    lie_device::ExpSE3(step, E, 4);
    Compose(X, E, Phi);
    lie_device::InverseSE3(Phi, 4, Phi_inv);
    Compose(Phi_inv, Y, D);
    lie_device::LogSE3(D, 4, e);

    float *o = out + threadIdx.x * kStride;
#pragma unroll
    for (int i = 0; i < kRows; ++i) o[i] = e[i];

    if constexpr (kJacobian) {
      // ∂e/∂T_k = -J_l⁻¹(e) Ad(E⁻¹); ∂e/∂ξ = -dt J_l⁻¹(e) J_r(step);
      // ∂e/∂T_{k+1} = J_r⁻¹(e) = J_l⁻¹(-e).
      float Jl_inv[36], Jr_inv[36], Jr_step[36], E_inv[16];
      lie_device::SE3JacobianLeftInverse(e, Jl_inv, 6);
      float minus[6];
#pragma unroll
      for (int i = 0; i < 6; ++i) minus[i] = -e[i];
      lie_device::SE3JacobianLeftInverse(minus, Jr_inv, 6);
#pragma unroll
      for (int i = 0; i < 6; ++i) minus[i] = -step[i];
      lie_device::SE3JacobianLeft(minus, Jr_step, 6);  // J_r(step)
      lie_device::InverseSE3(E, 4, E_inv);
#pragma unroll
      for (int i = 0; i < kRows; ++i) {
        const float *l = Jl_inv + i * 6;
        float *row = o + kRows + i * kColumns;
        float acc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float m[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 6; ++k) {
          float ad[6];
          lie_device::AdjointSE3RowFromTransform(E_inv, 4, k, ad);
#pragma unroll
          for (int j = 0; j < 6; ++j) {
            acc[j] += l[k] * ad[j];
            m[j] += l[k] * Jr_step[k * 6 + j];
          }
        }
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          row[j] = -acc[j];
          row[6 + j] = -dt * m[j];
          row[12 + j] = Jr_inv[i * 6 + j];
        }
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

SE3KinematicsFactorBatch::SE3KinematicsFactorBatch(const float *time_steps, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SE3KinematicsFactorBatch: time_steps must not be null");
  }
}

bool SE3KinematicsFactorBatch::Evaluate(float *residuals, float *jacobians,
                                        float const *const *state_pointers, cudaStream_t stream,
                                        const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SE3KinematicsKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, jacobians, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  } else {
    SE3KinematicsKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, nullptr, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
