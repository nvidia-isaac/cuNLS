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
#include "cunls/factor/dynamics/quadrotor_factor_batch.h"
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

constexpr int kWarpsPerBlock = 4;
constexpr int kBlockSize = 32 * kWarpsPerBlock;
constexpr int kRows = 12;
constexpr int kColumns =
    6 + 3 + 3 + 4 + 6 + 3 + 3;            // T_k, v_k, ω_k, f_k, T_{k+1}, v_{k+1}, ω_{k+1}
constexpr int kRowStride = kColumns + 1;  // 29: odd, no bank conflicts

/**
 * One thread per item, 32 consecutive items per warp. The 12 x 28 Jacobian is
 * produced one row at a time and written through a per-warp staging buffer
 * (coalesced stores, small shared memory per thread). The row loops are fully
 * unrolled so the 6x6 matrices stay in registers (a runtime row index puts
 * them in local memory: 1.8x slower); ~3 ns per item with Jacobians.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    QuadrotorKernel(const float *__restrict__ time_steps, QuadrotorParameters prm,
                    float const *const *__restrict__ state_pointers, float *__restrict__ residuals,
                    float *__restrict__ jacobians, int num_items,
                    const int *__restrict__ factor_ids, int num_factors) {
  __shared__ float stage_all[kWarpsPerBlock * 32 * kRowStride];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float *stage = stage_all + warp * 32 * kRowStride;
  const int warp_first = blockIdx.x * kBlockSize + warp * 32;
  const int count = min(32, num_items - warp_first);
  if (count <= 0) return;  // whole warp past the end
  const int t = warp_first + lane;
  const bool valid = lane < count;

  // --- Inputs (zeros for lanes past the end; their outputs are not written) ---
  float X[16] = {}, Y[16] = {}, v[3] = {}, w[3] = {}, f[4] = {}, vn[3] = {}, wn[3] = {};
  float dt = 0.f;
  if (valid) {
    const int fi = FactorMeasurementIndex(t, factor_ids, num_factors);
    float const *const *s = state_pointers + 7 * static_cast<size_t>(t);
    lie_device::LoadSE3Rows(s[0], X);
    lie_device::LoadSE3Rows(s[4], Y);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      v[i] = __ldg(s[1] + i);
      w[i] = __ldg(s[2] + i);
      vn[i] = __ldg(s[5] + i);
      wn[i] = __ldg(s[6] + i);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __ldg(s[3] + i);
    dt = __ldg(time_steps + fi);
  } else {
    X[0] = X[5] = X[10] = Y[0] = Y[5] = Y[10] = 1.f;
  }
  const float *R = X;  // rows 0..2, pitch 4

  // --- Residual ---
  // Pose twist ξ = [ω; Rᵀ v] (Ṫ = T ξ^ gives ṗ = v, Ṙ = R ω^).
  float rho[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) rho[i] = R[i] * v[0] + R[4 + i] * v[1] + R[8 + i] * v[2];
  const float step[6] = {dt * w[0], dt * w[1], dt * w[2], dt * rho[0], dt * rho[1], dt * rho[2]};
  float E[16], e[6];
  lie_device::SE3StepDefect(X, Y, step, E, e);

  const float inv_m = 1.f / prm.mass;
  const float thrust = f[0] + f[1] + f[2] + f[3];
  const float a = prm.arm_length * 0.70710678f;  // arm / √2
  const float km = prm.torque_coefficient;
  // τ = Σ_i [y_i f_i, -x_i f_i, σ_i k_m f_i] for the X layout of QuadrotorParameters.
  const float tau[3] = {a * (-f[0] + f[1] + f[2] - f[3]), a * (-f[0] + f[1] - f[2] + f[3]),
                        km * (-f[0] - f[1] + f[2] + f[3])};
  const float J[3] = {prm.inertia[0], prm.inertia[1], prm.inertia[2]};
  const float Jw[3] = {J[0] * w[0], J[1] * w[1], J[2] * w[2]};
  const float gyro[3] = {w[1] * Jw[2] - w[2] * Jw[1], w[2] * Jw[0] - w[0] * Jw[2],
                         w[0] * Jw[1] - w[1] * Jw[0]};  // ω × J ω

  float res[kRows];
#pragma unroll
  for (int i = 0; i < 6; ++i) res[i] = e[i];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float accel =
        R[i * 4 + 2] * thrust * inv_m - prm.linear_drag * v[i] - (i == 2 ? prm.gravity : 0.f);
    res[6 + i] = vn[i] - v[i] - dt * accel;
    res[9 + i] = wn[i] - w[i] - dt * (tau[i] - gyro[i]) / J[i];
  }
  lie_device::WriteWarpRows<kColumns, kRowStride>(
      stage, lane, res, kRows, count, residuals + static_cast<size_t>(warp_first) * kRows, kRows);
  if constexpr (!kJacobian) return;

  // --- Jacobian, one row at a time ---
  lie_device::SE3StepDefectJacobian Jd;
  lie_device::InitSE3StepDefectJacobian(e, step, E, &Jd);
  float *jac_base = jacobians + static_cast<size_t>(warp_first) * kRows * kColumns;
  float row[kColumns];

  // Pose rows (lie_device::RigidBodyPoseRow): ∂e/∂T_k, ∂e/∂v, ∂e/∂ω and ∂e/∂T_{k+1};
  // no thrust dependence.
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    lie_device::RigidBodyPoseRow<kColumns, 16>(Jd, i, dt, R, rho, row);
    lie_device::WriteWarpRows<kColumns, kRowStride>(stage, lane, row, kColumns, count,
                                                    jac_base + i * kColumns, kRows * kColumns);
  }

  // Velocity rows: ∂/∂δφ = dt (Σf/m) R [e₃]x = dt (Σf/m) [R[:,1], -R[:,0], 0];
  // ∂/∂v = -(1 - dt D) I; ∂/∂f_k = -dt R[:,2] / m; ∂/∂v_{k+1} = I.
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
    const float s = dt * thrust * inv_m;
    row[0] = s * R[i * 4 + 1];
    row[1] = -s * R[i * 4 + 0];
    row[6 + i] = -(1.f - dt * prm.linear_drag);
#pragma unroll
    for (int k = 0; k < 4; ++k) row[12 + k] = -dt * R[i * 4 + 2] * inv_m;
    row[22 + i] = 1.f;
    lie_device::WriteWarpRows<kColumns, kRowStride>(
        stage, lane, row, kColumns, count, jac_base + (6 + i) * kColumns, kRows * kColumns);
  }

  // Rate rows: ∂/∂ω = -I + dt J⁻¹ ([ω]x J - [Jω]x); ∂/∂f = -dt J⁻¹ M_τ; ∂/∂ω_{k+1} = I.
  const float W[9] = {0.f, -w[2], w[1], w[2], 0.f, -w[0], -w[1], w[0], 0.f};
  const float V[9] = {0.f, -Jw[2], Jw[1], Jw[2], 0.f, -Jw[0], -Jw[1], Jw[0], 0.f};
  const float M_tau[12] = {-a, a, a, -a, -a, a, -a, a, -km, -km, km, km};
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
    const float s = dt / J[i];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      row[9 + j] = (i == j ? -1.f : 0.f) + s * (W[i * 3 + j] * J[j] - V[i * 3 + j]);
      row[25 + j] = i == j ? 1.f : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) row[12 + k] = -s * M_tau[i * 4 + k];
    lie_device::WriteWarpRows<kColumns, kRowStride>(
        stage, lane, row, kColumns, count, jac_base + (9 + i) * kColumns, kRows * kColumns);
  }
}

}  // namespace

QuadrotorFactorBatch::QuadrotorFactorBatch(const float *time_steps,
                                           const QuadrotorParameters &parameters, size_t capacity)
    : SizedFactorBatch(capacity), time_steps_(time_steps), parameters_(parameters) {
  if (time_steps_ == nullptr) {
    throw std::invalid_argument("QuadrotorFactorBatch: time_steps must not be null");
  }
  if (!(parameters_.mass > 0.f) || !(parameters_.inertia[0] > 0.f) ||
      !(parameters_.inertia[1] > 0.f) || !(parameters_.inertia[2] > 0.f) ||
      !(parameters_.arm_length > 0.f)) {
    throw std::invalid_argument(
        "QuadrotorFactorBatch: mass, inertia and arm_length must be positive");
  }
}

bool QuadrotorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                    float const *const *state_pointers, cudaStream_t stream,
                                    const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    QuadrotorKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, parameters_, state_pointers, residuals, jacobians, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  } else {
    QuadrotorKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, parameters_, state_pointers, residuals, nullptr, static_cast<int>(num_items),
        factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
