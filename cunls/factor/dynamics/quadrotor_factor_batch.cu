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
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/math/lie_device.cuh"

namespace cunls {
namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kBlockSize = 32 * kWarpsPerBlock;
constexpr int kRows = 12;
constexpr int kColumns =
    6 + 3 + 3 + 4 + 6 + 3 + 3;            // T_k, v_k, ω_k, f_k, T_{k+1}, v_{k+1}, ω_{k+1}
constexpr int kRowStride = kColumns + 1;  // 29: odd, no bank conflicts

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
 * Writes one row of n values per item for the warp's `count` items: each lane
 * stages its item's row, then the warp stores the rows of consecutive items
 * (contiguous runs of n floats, `item_stride` apart) with consecutive lanes on
 * consecutive addresses.
 */
__device__ __forceinline__ void WriteWarpRows(float *stage, int lane, const float *values, int n,
                                              int count, float *dst, size_t item_stride) {
#pragma unroll
  for (int c = 0; c < kColumns; ++c) {
    if (c < n) stage[lane * kRowStride + c] = values[c];
  }
  __syncwarp();
  for (int k = lane; k < count * n; k += 32) {
    const int item = k / n, c = k % n;
    dst[item * item_stride + c] = stage[item * kRowStride + c];
  }
  __syncwarp();
}

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
    LoadPose(s[0], X);
    LoadPose(s[4], Y);
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
  float E[16], Phi[16], Phi_inv[16], D[16], e[6];
  lie_device::ExpSE3(step, E, 4);
  Compose(X, E, Phi);
  lie_device::InverseSE3(Phi, 4, Phi_inv);
  Compose(Phi_inv, Y, D);
  lie_device::LogSE3(D, 4, e);

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
  WriteWarpRows(stage, lane, res, kRows, count, residuals + static_cast<size_t>(warp_first) * kRows,
                kRows);
  if constexpr (!kJacobian) return;

  // --- Jacobian, one row at a time ---
  float Jl_inv[36], Jr_inv[36], Jr_step[36], E_inv[16];
  lie_device::SE3JacobianLeftInverse(e, Jl_inv, 6);
  {
    float m[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) m[i] = -e[i];
    lie_device::SE3JacobianLeftInverse(m, Jr_inv, 6);  // J_r⁻¹(e) = J_l⁻¹(-e)
#pragma unroll
    for (int i = 0; i < 6; ++i) m[i] = -step[i];
    lie_device::SE3JacobianLeft(m, Jr_step, 6);  // J_r(step)
  }
  lie_device::InverseSE3(E, 4, E_inv);
  float *jac_base = jacobians + static_cast<size_t>(warp_first) * kRows * kColumns;
  float row[kColumns];

  // Pose rows: ∂e/∂T_k = -J_l⁻¹ (Ad(E⁻¹) + dt J_r ∂ξ/∂δT), with ∂ξ_ρ/∂δφ = [ρ]x;
  // ∂e/∂v = -dt J_l⁻¹ J_r[:, 3:6] Rᵀ; ∂e/∂ω = -dt J_l⁻¹ J_r[:, 0:3]; ∂e/∂T_{k+1} = J_r⁻¹(e).
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const float *l = Jl_inv + i * 6;
    float lad[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, lj[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      float ad[6];
      lie_device::AdjointSE3RowFromTransform(E_inv, 4, k, ad);
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        lad[j] += l[k] * ad[j];
        lj[j] += l[k] * Jr_step[k * 6 + j];
      }
    }
    const float *ar = lj + 3;  // (l J_r)[3:6], against ∂ξ_ρ
    row[0] = -(lad[0] + dt * (ar[1] * rho[2] - ar[2] * rho[1]));
    row[1] = -(lad[1] + dt * (ar[2] * rho[0] - ar[0] * rho[2]));
    row[2] = -(lad[2] + dt * (ar[0] * rho[1] - ar[1] * rho[0]));
    row[3] = -lad[3];
    row[4] = -lad[4];
    row[5] = -lad[5];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      row[6 + j] = -dt * (ar[0] * R[j * 4] + ar[1] * R[j * 4 + 1] + ar[2] * R[j * 4 + 2]);
      row[9 + j] = -dt * lj[j];
    }
#pragma unroll
    for (int j = 12; j < 16; ++j) row[j] = 0.f;
#pragma unroll
    for (int j = 0; j < 6; ++j) row[16 + j] = Jr_inv[i * 6 + j];
#pragma unroll
    for (int j = 22; j < kColumns; ++j) row[j] = 0.f;
    WriteWarpRows(stage, lane, row, kColumns, count, jac_base + i * kColumns, kRows * kColumns);
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
    WriteWarpRows(stage, lane, row, kColumns, count, jac_base + (6 + i) * kColumns,
                  kRows * kColumns);
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
    WriteWarpRows(stage, lane, row, kColumns, count, jac_base + (9 + i) * kColumns,
                  kRows * kColumns);
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
