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
#include "cunls/factor/se3_joint_axis_factor_batch.h"

namespace cunls {
namespace {

constexpr int kBlockSize = 128;

/** Factor (measurement) index of item t (see FactorBatch::Evaluate). */
__device__ __forceinline__ int FactorMeasurementIndex(int t, const int *factor_ids,
                                                      int num_factors) {
  if (factor_ids != nullptr) return factor_ids[t];
  return t < num_factors ? t : t % num_factors;
}

/** R (row-major 3x3) and p of a row-major 4x4 SE(3) state. */
__device__ __forceinline__ void ReadPose(const float *X, float *R, float *p) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) R[3 * i + j] = X[4 * i + j];
    p[i] = X[4 * i + 3];
  }
}

/** The normalized axis of factor f. */
__device__ __forceinline__ void ReadAxis(const float *axes, int f, float *a) {
  a[0] = __ldg(axes + 3 * f);
  a[1] = __ldg(axes + 3 * f + 1);
  a[2] = __ldg(axes + 3 * f + 2);
  const float n = rsqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  a[0] *= n, a[1] *= n, a[2] *= n;
}

/** out = M u for row-major 3x3 M. */
__device__ __forceinline__ void Mul(const float *M, const float *u, float *out) {
#pragma unroll
  for (int i = 0; i < 3; ++i) out[i] = M[3 * i] * u[0] + M[3 * i + 1] * u[1] + M[3 * i + 2] * u[2];
}

/** out = M [u]× (row-major 3x3): column j of [u]× is u × e_j. */
__device__ __forceinline__ void MulSkew(const float *M, const float *u, float *out) {
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float m0 = M[3 * i], m1 = M[3 * i + 1], m2 = M[3 * i + 2];
    out[3 * i + 0] = m1 * u[2] - m2 * u[1];
    out[3 * i + 1] = m2 * u[0] - m0 * u[2];
    out[3 * i + 2] = m0 * u[1] - m1 * u[0];
  }
}

/** One thread per item: r = R_1 a - R_0 a and its 3 x 12 Jacobian. */
__global__ void AxisAlignmentKernel(const float *__restrict__ axes,
                                    float const *const *__restrict__ state_pointers,
                                    float *__restrict__ residuals, float *__restrict__ jacobians,
                                    int num_items, const int *__restrict__ factor_ids,
                                    int num_factors) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_items) return;
  const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
  float a[3], R0[9], p0[3], R1[9], p1[3], a0[3], a1[3];
  ReadAxis(axes, f, a);
  ReadPose(state_pointers[2 * static_cast<size_t>(t)], R0, p0);
  ReadPose(state_pointers[2 * static_cast<size_t>(t) + 1], R1, p1);
  Mul(R0, a, a0);
  Mul(R1, a, a1);
#pragma unroll
  for (int i = 0; i < 3; ++i) residuals[3 * static_cast<size_t>(t) + i] = a1[i] - a0[i];
  if (jacobians == nullptr) return;
  float S0[9], S1[9];
  MulSkew(R0, a, S0);  // ∂r/∂ω_0 = R_0 [a]×
  MulSkew(R1, a, S1);  // ∂r/∂ω_1 = -R_1 [a]×
  float *J = jacobians + static_cast<size_t>(t) * 3 * 12;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      J[12 * i + j] = S0[3 * i + j];
      J[12 * i + 3 + j] = 0.f;
      J[12 * i + 6 + j] = -S1[3 * i + j];
      J[12 * i + 9 + j] = 0.f;
    }
  }
}

/** b_1, b_2: unit vectors perpendicular to the unit axis a (see SE3AxisOffsetFactorBatch). */
__device__ __forceinline__ void Perpendiculars(const float *a, float *b1, float *b2) {
  const float e[3] = {fabsf(a[0]) < 0.9f ? 1.f : 0.f, fabsf(a[0]) < 0.9f ? 0.f : 1.f, 0.f};
  b1[0] = a[1] * e[2] - a[2] * e[1];
  b1[1] = a[2] * e[0] - a[0] * e[2];
  b1[2] = a[0] * e[1] - a[1] * e[0];
  const float n = rsqrtf(b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]);
  b1[0] *= n, b1[1] *= n, b1[2] *= n;
  b2[0] = a[1] * b1[2] - a[2] * b1[1];
  b2[1] = a[2] * b1[0] - a[0] * b1[2];
  b2[2] = a[0] * b1[1] - a[1] * b1[0];
}

/** One thread per item: r_k = (R_0 b_k) · (p_1 - p_0) and its 2 x 12 Jacobian. */
__global__ void AxisOffsetKernel(const float *__restrict__ axes,
                                 float const *const *__restrict__ state_pointers,
                                 float *__restrict__ residuals, float *__restrict__ jacobians,
                                 int num_items, const int *__restrict__ factor_ids,
                                 int num_factors) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_items) return;
  const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
  float a[3], R0[9], p0[3], R1[9], p1[3], b[2][3];
  ReadAxis(axes, f, a);
  ReadPose(state_pointers[2 * static_cast<size_t>(t)], R0, p0);
  ReadPose(state_pointers[2 * static_cast<size_t>(t) + 1], R1, p1);
  Perpendiculars(a, b[0], b[1]);
  const float d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  float dR0[3];  // Δᵀ R_0, i.e. R_0ᵀ Δ
#pragma unroll
  for (int j = 0; j < 3; ++j) dR0[j] = d[0] * R0[j] + d[1] * R0[3 + j] + d[2] * R0[6 + j];
  for (int k = 0; k < 2; ++k) {
    residuals[2 * static_cast<size_t>(t) + k] =
        dR0[0] * b[k][0] + dR0[1] * b[k][1] + dR0[2] * b[k][2];
    if (jacobians == nullptr) continue;
    float w[3];
    Mul(R0, b[k], w);
    float *J = jacobians + (static_cast<size_t>(t) * 2 + k) * 12;
    // ∂r/∂ω_0 = -Δᵀ R_0 [b]×; with u = R_0ᵀΔ, uᵀ[b]× ω = (u × b) · ω, so the row is -(u × b)ᵀ.
    J[0] = -(dR0[1] * b[k][2] - dR0[2] * b[k][1]);
    J[1] = -(dR0[2] * b[k][0] - dR0[0] * b[k][2]);
    J[2] = -(dR0[0] * b[k][1] - dR0[1] * b[k][0]);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      J[3 + j] = -b[k][j];                                            // ∂r/∂v_0 = -b_kᵀ
      J[6 + j] = 0.f;                                                 // ∂r/∂ω_1
      J[9 + j] = w[0] * R1[j] + w[1] * R1[3 + j] + w[2] * R1[6 + j];  // ∂r/∂v_1 = w_kᵀ R_1
    }
  }
}

template <typename Kernel>
bool Launch(Kernel kernel, const float *axes, float *residuals, float *jacobians,
            float const *const *state_pointers, cudaStream_t stream, const int *factor_ids,
            size_t num_factor_ids, size_t num_factors) {
  const size_t n = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (n == 0 || num_factors == 0) return true;
  kernel<<<static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize), kBlockSize, 0, stream>>>(
      axes, state_pointers, residuals, jacobians, static_cast<int>(n), factor_ids,
      static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace

SE3AxisAlignmentFactorBatch::SE3AxisAlignmentFactorBatch(const float *axes, size_t capacity)
    : SizedFactorBatch(capacity), axes_(axes) {
  if (axes == nullptr) {
    throw std::invalid_argument(
        "SE3AxisAlignmentFactorBatch: axes is null (expected a device array of 3 floats per "
        "factor). Fix: pass the axis of every factor.");
  }
}

bool SE3AxisAlignmentFactorBatch::Evaluate(float *residuals, float *jacobians,
                                           float const *const *state_pointers, cudaStream_t stream,
                                           const int *factor_ids, size_t num_factor_ids) const {
  return Launch(AxisAlignmentKernel, axes_, residuals, jacobians, state_pointers, stream,
                factor_ids, num_factor_ids, NumActiveFactors());
}

SE3AxisOffsetFactorBatch::SE3AxisOffsetFactorBatch(const float *axes, size_t capacity)
    : SizedFactorBatch(capacity), axes_(axes) {
  if (axes == nullptr) {
    throw std::invalid_argument(
        "SE3AxisOffsetFactorBatch: axes is null (expected a device array of 3 floats per "
        "factor). Fix: pass the axis of every factor.");
  }
}

bool SE3AxisOffsetFactorBatch::Evaluate(float *residuals, float *jacobians,
                                        float const *const *state_pointers, cudaStream_t stream,
                                        const int *factor_ids, size_t num_factor_ids) const {
  return Launch(AxisOffsetKernel, axes_, residuals, jacobians, state_pointers, stream, factor_ids,
                num_factor_ids, NumActiveFactors());
}

}  // namespace cunls
