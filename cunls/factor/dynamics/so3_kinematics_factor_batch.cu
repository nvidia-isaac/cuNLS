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
#include "cunls/factor/dynamics/so3_kinematics_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/math/lie_device.cuh"

namespace cunls {
namespace {

constexpr int kBlockSize = 128;
constexpr int kRows = 3;
constexpr int kColumns = 3 + 3 + 3;              // R_k, ω_k, R_{k+1}
constexpr int kJacobianSize = kRows * kColumns;  // 27 floats per item

/**
 * One thread per item; outputs staged in shared memory (odd stride) and written
 * with coalesced stores. kJacobian selects the residual-only variant.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    SO3KinematicsKernel(const float *__restrict__ time_steps,
                        float const *const *__restrict__ state_pointers,
                        float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                        const int *__restrict__ factor_ids, int num_factors) {
  constexpr int kStride = kJacobian ? kRows + kJacobianSize + 1 : kRows;  // 31 or 3
  __shared__ float out[kBlockSize * kStride];
  const int first = blockIdx.x * kBlockSize;
  const int t = first + threadIdx.x;

  if (t < num_items) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const float *Rk = state_pointers[3 * t + 0];
    const float *w = state_pointers[3 * t + 1];
    const float *Rn = state_pointers[3 * t + 2];
    float X[9], Y[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      X[i] = __ldg(Rk + i);
      Y[i] = __ldg(Rn + i);
    }
    const float dt = __ldg(time_steps + f);
    const float step[3] = {dt * __ldg(w), dt * __ldg(w + 1), dt * __ldg(w + 2)};

    // r = Log((X E)ᵀ Y) with E = Exp(step).
    float E[9], Phi[9], D[9], res[3];
    lie_device::ExpSO3(step, E, 3);
    lie_device::MatMul3x3(X, E, Phi);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int j = 0; j < 3; ++j)
        D[i * 3 + j] = Phi[i] * Y[j] + Phi[3 + i] * Y[3 + j] + Phi[6 + i] * Y[6 + j];
    lie_device::LogSO3(D, 3, res);

    float *o = out + threadIdx.x * kStride;
    o[0] = res[0];
    o[1] = res[1];
    o[2] = res[2];

    if constexpr (kJacobian) {
      // ∂r/∂R_k = -J_l⁻¹(r) Eᵀ (Ad(E⁻¹) = Eᵀ); ∂r/∂ω = -dt J_l⁻¹(r) J_r(step);
      // ∂r/∂R_{k+1} = J_r⁻¹(r) = J_l⁻¹(-r).
      float Jl_inv[9], Jr_inv[9], Jr_step[9];
      lie_device::SO3JacobianLeftInverse(res, Jl_inv, 3);
      const float minus_r[3] = {-res[0], -res[1], -res[2]};
      lie_device::SO3JacobianLeftInverse(minus_r, Jr_inv, 3);
      const float minus_step[3] = {-step[0], -step[1], -step[2]};
      lie_device::SO3JacobianLeft(minus_step, Jr_step, 3);
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float *row = o + kRows + i * kColumns;
        const float *l = Jl_inv + 3 * i;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          // (l Eᵀ)_j = Σ_k l_k E[j][k]
          row[j] = -(l[0] * E[j * 3] + l[1] * E[j * 3 + 1] + l[2] * E[j * 3 + 2]);
          row[3 + j] = -dt * (l[0] * Jr_step[j] + l[1] * Jr_step[3 + j] + l[2] * Jr_step[6 + j]);
          row[6 + j] = Jr_inv[3 * i + j];
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

SO3KinematicsFactorBatch::SO3KinematicsFactorBatch(const float *time_steps, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("SO3KinematicsFactorBatch: time_steps must not be null");
  }
}

bool SO3KinematicsFactorBatch::Evaluate(float *residuals, float *jacobians,
                                        float const *const *state_pointers, cudaStream_t stream,
                                        const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    SO3KinematicsKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, jacobians, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  } else {
    SO3KinematicsKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, state_pointers, residuals, nullptr, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
