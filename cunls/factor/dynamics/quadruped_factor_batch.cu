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
#include "cunls/factor/dynamics/quadruped_factor_batch.h"
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
    6 + 3 + 3 + 12 + 6 + 3 + 3;           // T_k, v_k, ω_k, F_k, T_{k+1}, v_{k+1}, ω_{k+1}
constexpr int kRowStride = kColumns + 1;  // 37: odd, no bank conflicts

/** a × b */
__device__ __forceinline__ void Cross(const float *a, const float *b, float *c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * One thread per item, 32 consecutive items per warp. The 12 x 36 Jacobian is
 * produced one row at a time (fully unrolled: the 6x6 matrices stay in
 * registers) and written through a per-warp staging buffer (coalesced stores,
 * small shared memory per thread).
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    QuadrupedKernel(const float *__restrict__ time_steps, const float *__restrict__ contacts,
                    const float *__restrict__ foot_positions, QuadrupedParameters prm,
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
  float X[16] = {}, Y[16] = {}, v[3] = {}, w[3] = {}, F[12] = {}, vn[3] = {}, wn[3] = {};
  float contact[4] = {}, feet[12] = {};
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
    for (int i = 0; i < 12; ++i) {
      F[i] = __ldg(s[3] + i);
      feet[i] = __ldg(foot_positions + 12 * static_cast<size_t>(fi) + i);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) contact[i] = __ldg(contacts + 4 * static_cast<size_t>(fi) + i);
    dt = __ldg(time_steps + fi);
  } else {
    X[0] = X[5] = X[10] = Y[0] = Y[5] = Y[10] = 1.f;
  }
  const float *R = X;  // rows 0..2, pitch 4
  const float p[3] = {X[3], X[7], X[11]};

  // --- Residual ---
  float rho[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) rho[i] = R[i] * v[0] + R[4 + i] * v[1] + R[8 + i] * v[2];
  const float step[6] = {dt * w[0], dt * w[1], dt * w[2], dt * rho[0], dt * rho[1], dt * rho[2]};
  float E[16], e[6];
  lie_device::SE3StepDefect(X, Y, step, E, e);

  // Total force and world torque about the base: Σ s_i f_i, Σ s_i (p_i - p) × f_i.
  float force[3] = {0.f, 0.f, 0.f}, torque_w[3] = {0.f, 0.f, 0.f}, arm[12];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float *f = F + 3 * k;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      arm[3 * k + i] = feet[3 * k + i] - p[i];
      force[i] += contact[k] * f[i];
    }
    float c[3];
    Cross(arm + 3 * k, f, c);
#pragma unroll
    for (int i = 0; i < 3; ++i) torque_w[i] += contact[k] * c[i];
  }
  float torque_b[3], force_b[3];  // Rᵀ τ_w, Rᵀ F
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    torque_b[i] = R[i] * torque_w[0] + R[4 + i] * torque_w[1] + R[8 + i] * torque_w[2];
    force_b[i] = R[i] * force[0] + R[4 + i] * force[1] + R[8 + i] * force[2];
  }
  const float inv_m = 1.f / prm.mass;
  const float J[3] = {prm.inertia[0], prm.inertia[1], prm.inertia[2]};
  const float Jw[3] = {J[0] * w[0], J[1] * w[1], J[2] * w[2]};
  float gyro[3];
  Cross(w, Jw, gyro);

  float res[kRows];
#pragma unroll
  for (int i = 0; i < 6; ++i) res[i] = e[i];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    res[6 + i] = vn[i] - v[i] - dt * (force[i] * inv_m - (i == 2 ? prm.gravity : 0.f));
    res[9 + i] = wn[i] - w[i] - dt * (torque_b[i] - gyro[i]) / J[i];
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
  // no force dependence.
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    lie_device::RigidBodyPoseRow<kColumns, 24>(Jd, i, dt, R, rho, row);
    lie_device::WriteWarpRows<kColumns, kRowStride>(stage, lane, row, kColumns, count,
                                                    jac_base + i * kColumns, kRows * kColumns);
  }

  // Velocity rows: ∂/∂v = -I; ∂/∂f_k = -dt s_k / m I; ∂/∂v_{k+1} = I.
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
    row[6 + i] = -1.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) row[12 + 3 * k + i] = -dt * contact[k] * inv_m;
    row[30 + i] = 1.f;
    lie_device::WriteWarpRows<kColumns, kRowStride>(
        stage, lane, row, kColumns, count, jac_base + (6 + i) * kColumns, kRows * kColumns);
  }

  // Rate rows (s = dt / J_i): ∂/∂δφ = -s [τ_b]x; ∂/∂δρ = -s [Rᵀ F]x (moving the base
  // moves every lever arm); ∂/∂ω = -I + s ([ω]x J - [Jω]x);
  // ∂/∂f_k = -s s_k Rᵀ [p_k - p]x; ∂/∂ω_{k+1} = I.
  const float W[9] = {0.f, -w[2], w[1], w[2], 0.f, -w[0], -w[1], w[0], 0.f};
  const float V[9] = {0.f, -Jw[2], Jw[1], Jw[2], 0.f, -Jw[0], -Jw[1], Jw[0], 0.f};
  const float Tb[9] = {0.f,          -torque_b[2], torque_b[1], torque_b[2], 0.f,
                       -torque_b[0], -torque_b[1], torque_b[0], 0.f};
  const float Fb[9] = {0.f,         -force_b[2], force_b[1], force_b[2], 0.f,
                       -force_b[0], -force_b[1], force_b[0], 0.f};
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
    const float s = dt / J[i];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      row[j] = -s * Tb[i * 3 + j];
      row[3 + j] = -s * Fb[i * 3 + j];
      row[9 + j] = (i == j ? -1.f : 0.f) + s * (W[i * 3 + j] * J[j] - V[i * 3 + j]);
      row[33 + j] = i == j ? 1.f : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float *a = arm + 3 * k;
      // Row i of Rᵀ [a]x: (Rᵀ[a]x)[i][j] = Σ_m R[m][i] [a]x[m][j].
      const float r0 = R[i], r1 = R[4 + i], r2 = R[8 + i];  // column i of R
      const float m0 = r1 * a[2] - r2 * a[1];               // [a]x columns dotted with R[:, i]
      const float m1 = r2 * a[0] - r0 * a[2];
      const float m2 = r0 * a[1] - r1 * a[0];
      const float g = -s * contact[k];
      row[12 + 3 * k + 0] = g * m0;
      row[12 + 3 * k + 1] = g * m1;
      row[12 + 3 * k + 2] = g * m2;
    }
    lie_device::WriteWarpRows<kColumns, kRowStride>(
        stage, lane, row, kColumns, count, jac_base + (9 + i) * kColumns, kRows * kColumns);
  }
}

}  // namespace

QuadrupedFactorBatch::QuadrupedFactorBatch(const float *time_steps, const float *contacts,
                                           const float *foot_positions,
                                           const QuadrupedParameters &parameters, size_t capacity)
    : SizedFactorBatch(capacity),
      time_steps_(time_steps),
      contacts_(contacts),
      foot_positions_(foot_positions),
      parameters_(parameters) {
  if (time_steps_ == nullptr || contacts_ == nullptr || foot_positions_ == nullptr) {
    throw std::invalid_argument(
        "QuadrupedFactorBatch: time_steps, contacts and foot_positions must not be null");
  }
  if (!(parameters_.mass > 0.f) || !(parameters_.inertia[0] > 0.f) ||
      !(parameters_.inertia[1] > 0.f) || !(parameters_.inertia[2] > 0.f)) {
    throw std::invalid_argument("QuadrupedFactorBatch: mass and inertia must be positive");
  }
}

bool QuadrupedFactorBatch::Evaluate(float *residuals, float *jacobians,
                                    float const *const *state_pointers, cudaStream_t stream,
                                    const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    QuadrupedKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, contacts_, foot_positions_, parameters_, state_pointers, residuals, jacobians,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  } else {
    QuadrupedKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        time_steps_, contacts_, foot_positions_, parameters_, state_pointers, residuals, nullptr,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
