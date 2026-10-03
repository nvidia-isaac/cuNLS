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

#pragma once

#include <cstdint>

/**
 * @file lie_device.cuh
 * @brief Per-element device math shared by the batched math kernels
 * (cunls/math/*.cu) and the factor kernels that derive inverses / adjoints of
 * their measurements inline. Header-inline: the library does not use
 * relocatable device code. Internal; not installed with the public headers.
 *
 * Conventions (as in the state batches): right perturbation X Exp(δ); SE(2)
 * tangent [v_x, v_y, θ], SE(3) tangent [φ, ρ] with t = J_l(φ) ρ; matrices
 * row-major.
 */

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstring>

namespace cunls {
namespace lie_device {

// ---------------------------------------------------------------------------
// SE(2) / Sim(2): row-major 3x3, contiguous.
// ---------------------------------------------------------------------------

/** T^{-1} = [R^T, -R^T t; 0 0 1] for T = [R t; 0 0 1]. */
__device__ __forceinline__ void InverseSE2(const float *T, float *Ti) {
  const float r00 = T[0], r01 = T[1], tx = T[2];
  const float r10 = T[3], r11 = T[4], ty = T[5];
  Ti[0] = r00;
  Ti[1] = r10;
  Ti[2] = -(r00 * tx + r10 * ty);
  Ti[3] = r01;
  Ti[4] = r11;
  Ti[5] = -(r01 * tx + r11 * ty);
  Ti[6] = 0.0f;
  Ti[7] = 0.0f;
  Ti[8] = 1.0f;
}

/**
 * (a/2) cot(a/2) and (1 - (a/2) cot(a/2)) / a, accurate in float32 for all a
 * in (-2 pi, 2 pi): a series below |a| = 0.5 (where the second cancels).
 */
__device__ __forceinline__ void HalfCotCoefficients(float a, float *half_cot, float *rest) {
  const float a2 = a * a;
  if (fabsf(a) < 0.5f) {
    const float q =
        1.0f / 12.0f + a2 * (1.0f / 720.0f + a2 * (1.0f / 30240.0f + a2 * (1.0f / 1209600.0f)));
    *half_cot = 1.0f - a2 * q;
    *rest = a * q;
  } else {
    const float h = 0.5f * a;
    *half_cot = h * cosf(h) / sinf(h);
    *rest = (1.0f - *half_cot) / a;
  }
}

/**
 * SE(2) right-Jacobian inverse J_r^{-1}(xi) (row-major 3x3) for xi = [v1, v2, a]:
 * [[c, -a/2, v1 r + v2/2], [a/2, c, v2 r - v1/2], [0, 0, 1]], with c, r from
 * HalfCotCoefficients. J_l^{-1}(xi) = J_r^{-1}(-xi).
 */
__device__ __forceinline__ void SE2JrInv(float v1, float v2, float a, float *J) {
  float c, r;
  HalfCotCoefficients(a, &c, &r);
  J[0] = c;
  J[1] = -0.5f * a;
  J[2] = v1 * r + 0.5f * v2;
  J[3] = 0.5f * a;
  J[4] = c;
  J[5] = v2 * r - 0.5f * v1;
  J[6] = 0.0f;
  J[7] = 0.0f;
  J[8] = 1.0f;
}

/** Exp of SE(2): xi = [v_x, v_y, theta] -> row-major 3x3 T (series near 0, no cancellation). */
__device__ __forceinline__ void ExpSE2(const float *xi, float *T) {
  const float vx = xi[0], vy = xi[1], w = xi[2];
  const float c = cosf(w);
  const float s = sinf(w);
  // sin(w) / w and (1 - cos w) / w = 2 sin^2(w/2) / w (no cancellation).
  float sinw_over_w, one_minus_cosw_over_w;
  if (fabsf(w) < 1e-2f) {
    const float w2 = w * w;
    sinw_over_w = 1.0f - w2 / 6.0f;
    one_minus_cosw_over_w = w * (0.5f - w2 / 24.0f);
  } else {
    const float sh = sinf(0.5f * w);
    sinw_over_w = s / w;
    one_minus_cosw_over_w = 2.0f * sh * sh / w;
  }
  T[0] = c;
  T[1] = -s;
  T[2] = vx * sinw_over_w - vy * one_minus_cosw_over_w;
  T[3] = s;
  T[4] = c;
  T[5] = vx * one_minus_cosw_over_w + vy * sinw_over_w;
  T[6] = 0.0f;
  T[7] = 0.0f;
  T[8] = 1.0f;
}

/**
 * Log of SE(2): row-major 3x3 T -> xi = [v_x, v_y, theta], with
 * V^{-1} = [[h, theta/2], [-theta/2, h]], h = (theta/2) cot(theta/2).
 */
__device__ __forceinline__ void LogSE2(const float *T, float *xi) {
  const float c = T[0], s = T[3], tx = T[2], ty = T[5];  // load first: xi may alias T
  const float theta = atan2f(s, c);
  float h, unused;
  HalfCotCoefficients(theta, &h, &unused);
  xi[0] = h * tx + 0.5f * theta * ty;
  xi[1] = -0.5f * theta * tx + h * ty;
  xi[2] = theta;
}

/**
 * SE(2) right Jacobian J_r(xi) (row-major 3x3), the inverse of SE2JrInv:
 * J_r^{-1} = [[M, m], [0, 1]] gives J_r = [[M^{-1}, -M^{-1} m], [0, 1]], with
 * M = [[c, -a/2], [a/2, c]]. J_l(xi) = J_r(-xi).
 */
__device__ __forceinline__ void SE2Jr(float v1, float v2, float a, float *J) {
  float Ji[9];
  SE2JrInv(v1, v2, a, Ji);
  const float c = Ji[0], h = Ji[3];  // M = [[c, -h], [h, c]]
  const float inv = 1.0f / (c * c + h * h);
  const float m00 = c * inv, m01 = h * inv, m10 = -h * inv, m11 = c * inv;  // M^{-1}
  J[0] = m00;
  J[1] = m01;
  J[2] = -(m00 * Ji[2] + m01 * Ji[5]);
  J[3] = m10;
  J[4] = m11;
  J[5] = -(m10 * Ji[2] + m11 * Ji[5]);
  J[6] = 0.0f;
  J[7] = 0.0f;
  J[8] = 1.0f;
}

/** T^{-1} = [R^T, -s R^T t; 0 0 s] for T = [R t; 0 0 1/s]. */
__device__ __forceinline__ void InverseSim2(const float *T, float *Ti) {
  const float r00 = T[0], r01 = T[1], tx = T[2];
  const float r10 = T[3], r11 = T[4], ty = T[5];
  const float inv_s = T[8];
  const float s = __frcp_rn(inv_s);  // == 1.0f / inv_s, bitwise
  Ti[0] = r00;
  Ti[1] = r10;
  Ti[2] = -s * (r00 * tx + r10 * ty);
  Ti[3] = r01;
  Ti[4] = r11;
  Ti[5] = -s * (r01 * tx + r11 * ty);
  Ti[6] = 0.0f;
  Ti[7] = 0.0f;
  Ti[8] = s;
}

// ---------------------------------------------------------------------------
// SE(3) / Sim(3): row-major 4x4.
// ---------------------------------------------------------------------------

/** T^{-1} = [R^T, -R^T t; 0 1]; T rows `pitch` apart, Ti contiguous 4x4. */
__device__ __forceinline__ void InverseSE3(const float *T, size_t pitch, float *Ti) {
  const float t1 = T[0 * pitch + 3];
  const float t2 = T[1 * pitch + 3];
  const float t3 = T[2 * pitch + 3];
  // R^T
  Ti[0] = T[0 * pitch + 0];
  Ti[1] = T[1 * pitch + 0];
  Ti[2] = T[2 * pitch + 0];
  Ti[4] = T[0 * pitch + 1];
  Ti[5] = T[1 * pitch + 1];
  Ti[6] = T[2 * pitch + 1];
  Ti[8] = T[0 * pitch + 2];
  Ti[9] = T[1 * pitch + 2];
  Ti[10] = T[2 * pitch + 2];
  Ti[3] = -(Ti[0] * t1 + Ti[1] * t2 + Ti[2] * t3);
  Ti[7] = -(Ti[4] * t1 + Ti[5] * t2 + Ti[6] * t3);
  Ti[11] = -(Ti[8] * t1 + Ti[9] * t2 + Ti[10] * t3);
  Ti[12] = 0.0f;
  Ti[13] = 0.0f;
  Ti[14] = 0.0f;
  Ti[15] = 1.0f;
}

/** T^{-1} = [R^T, -s R^T t; 0 s] for T = [R t; 0 1/s], both contiguous 4x4. */
__device__ __forceinline__ void InverseSim3(const float *T, float *Ti) {
  const float r00 = T[0], r01 = T[1], r02 = T[2], tx = T[3];
  const float r10 = T[4], r11 = T[5], r12 = T[6], ty = T[7];
  const float r20 = T[8], r21 = T[9], r22 = T[10], tz = T[11];
  const float inv_s = T[15];
  const float s = __frcp_rn(inv_s);  // == 1.0f / inv_s, bitwise
  Ti[0] = r00;
  Ti[1] = r10;
  Ti[2] = r20;
  Ti[3] = -s * (r00 * tx + r10 * ty + r20 * tz);
  Ti[4] = r01;
  Ti[5] = r11;
  Ti[6] = r21;
  Ti[7] = -s * (r01 * tx + r11 * ty + r21 * tz);
  Ti[8] = r02;
  Ti[9] = r12;
  Ti[10] = r22;
  Ti[11] = -s * (r02 * tx + r12 * ty + r22 * tz);
  Ti[12] = 0.0f;
  Ti[13] = 0.0f;
  Ti[14] = 0.0f;
  Ti[15] = s;
}

/**
 * Row `row` (0..5) of Ad(T) = [[R, 0], [[t]_x R, R]] for rotation R
 * (row-major 3x3) and translation t.
 */
__device__ __forceinline__ void AdjointSE3Row(const float *R, const float *t, int row, float *out) {
  if (row < 3) {
    out[0] = R[row * 3 + 0];
    out[1] = R[row * 3 + 1];
    out[2] = R[row * 3 + 2];
    out[3] = 0.0f;
    out[4] = 0.0f;
    out[5] = 0.0f;
    return;
  }
  const int i = row - 3;
  // Row i of skew(t) = [[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]].
  float skew[9];
  skew[0] = 0;
  skew[1] = -t[2];
  skew[2] = t[1];
  skew[3] = t[2];
  skew[4] = 0;
  skew[5] = -t[0];
  skew[6] = -t[1];
  skew[7] = t[0];
  skew[8] = 0;
  // Explicit rounding order (a*b, then two FMAs), so every caller produces the
  // same bits regardless of how the compiler would contract the expression.
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    out[j] = __fmaf_rn(
        skew[i * 3 + 2], R[2 * 3 + j],
        __fmaf_rn(skew[i * 3 + 1], R[1 * 3 + j], __fmul_rn(skew[i * 3 + 0], R[0 * 3 + j])));
  }
  out[3] = R[i * 3 + 0];
  out[4] = R[i * 3 + 1];
  out[5] = R[i * 3 + 2];
}

/**
 * Loads a 4x4 (16-float) matrix: four 16-byte vector loads when `p` is 16-byte
 * aligned (always true for cudaMalloc'd batches of 16-float blocks), scalar
 * loads otherwise. Memory-bound kernels issue 4x fewer load instructions.
 */
__device__ __forceinline__ void Load16(const float *__restrict__ p, float (&v)[16]) {
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
    const float4 *q = reinterpret_cast<const float4 *>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 a = __ldg(q + k);
      v[4 * k + 0] = a.x;
      v[4 * k + 1] = a.y;
      v[4 * k + 2] = a.z;
      v[4 * k + 3] = a.w;
    }
  } else {
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = p[k];
  }
}

/** Stores a 4x4 matrix; vector stores when `p` is 16-byte aligned (see Load16). */
__device__ __forceinline__ void Store16(float *__restrict__ p, const float (&v)[16]) {
  if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
    float4 *q = reinterpret_cast<float4 *>(p);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      q[k] = make_float4(v[4 * k + 0], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
    }
  } else {
#pragma unroll
    for (int k = 0; k < 16; ++k) p[k] = v[k];
  }
}

/**
 * Makes every value in `v` available before this point. Kernels that load a
 * measurement and a state and then branch (e.g. the rarely taken slow path of
 * __frcp_rn or a singularity check) call it after issuing all their loads:
 * otherwise the compiler may sink part of the loads past the branch, turning one
 * memory round trip per thread into two. Emits no instruction.
 */
template <int N>
__device__ __forceinline__ void LoadsBarrier(const float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    asm volatile("" ::"f"(v[i]));
  }
}

/**
 * Same as AdjointSE3Row, but reads R and t from the transform T (row-major,
 * rows `pitch` floats apart) in global memory. For callers where `row` is a
 * runtime value: no dynamically indexed local array, so nothing spills to local
 * memory. Identical arithmetic to AdjointSE3Row, hence identical bits.
 */
__device__ __forceinline__ void AdjointSE3RowFromTransform(const float *T, int pitch, int row,
                                                           float *out) {
  if (row < 3) {
    out[0] = T[row * pitch + 0];
    out[1] = T[row * pitch + 1];
    out[2] = T[row * pitch + 2];
    out[3] = 0.0f;
    out[4] = 0.0f;
    out[5] = 0.0f;
    return;
  }
  const int i = row - 3;
  const float t0 = T[0 * pitch + 3], t1 = T[1 * pitch + 3], t2 = T[2 * pitch + 3];
  // Row i of skew(t) = [[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]], built with
  // selects so it stays in registers.
  const float k0 = i == 0 ? 0.0f : (i == 1 ? t2 : -t1);
  const float k1 = i == 0 ? -t2 : (i == 1 ? 0.0f : t0);
  const float k2 = i == 0 ? t1 : (i == 1 ? -t0 : 0.0f);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    out[j] = __fmaf_rn(k2, T[2 * pitch + j], __fmaf_rn(k1, T[1 * pitch + j], __fmul_rn(k0, T[j])));
  }
  out[3] = T[i * pitch + 0];
  out[4] = T[i * pitch + 1];
  out[5] = T[i * pitch + 2];
}

/**
 * Sim(3) adjoint (row-major 7x7) of T = [[R, t], [0, 1/s]] (contiguous 4x4):
 * [[R, 0, 0], [s [t]_x R, s R, -s t], [0, 0, 1]].
 */
__device__ __forceinline__ void AdjointSim3(const float *T, float *Ad) {
  const float R00 = T[0], R01 = T[1], R02 = T[2];
  const float R10 = T[4], R11 = T[5], R12 = T[6];
  const float R20 = T[8], R21 = T[9], R22 = T[10];
  const float tx = T[3], ty = T[7], tz = T[11];
  const float s = __frcp_rn(T[15]);  // == 1.0f / T[15], bitwise

  const float A00 = s * (-tz * R10 + ty * R20);
  const float A01 = s * (-tz * R11 + ty * R21);
  const float A02 = s * (-tz * R12 + ty * R22);
  const float A10 = s * (tz * R00 - tx * R20);
  const float A11 = s * (tz * R01 - tx * R21);
  const float A12 = s * (tz * R02 - tx * R22);
  const float A20 = s * (-ty * R00 + tx * R10);
  const float A21 = s * (-ty * R01 + tx * R11);
  const float A22 = s * (-ty * R02 + tx * R12);

  Ad[0] = R00;
  Ad[1] = R01;
  Ad[2] = R02;
  Ad[3] = 0;
  Ad[4] = 0;
  Ad[5] = 0;
  Ad[6] = 0;
  Ad[7] = R10;
  Ad[8] = R11;
  Ad[9] = R12;
  Ad[10] = 0;
  Ad[11] = 0;
  Ad[12] = 0;
  Ad[13] = 0;
  Ad[14] = R20;
  Ad[15] = R21;
  Ad[16] = R22;
  Ad[17] = 0;
  Ad[18] = 0;
  Ad[19] = 0;
  Ad[20] = 0;
  Ad[21] = A00;
  Ad[22] = A01;
  Ad[23] = A02;
  Ad[24] = s * R00;
  Ad[25] = s * R01;
  Ad[26] = s * R02;
  Ad[27] = -s * tx;
  Ad[28] = A10;
  Ad[29] = A11;
  Ad[30] = A12;
  Ad[31] = s * R10;
  Ad[32] = s * R11;
  Ad[33] = s * R12;
  Ad[34] = -s * ty;
  Ad[35] = A20;
  Ad[36] = A21;
  Ad[37] = A22;
  Ad[38] = s * R20;
  Ad[39] = s * R21;
  Ad[40] = s * R22;
  Ad[41] = -s * tz;
  Ad[42] = 0;
  Ad[43] = 0;
  Ad[44] = 0;
  Ad[45] = 0;
  Ad[46] = 0;
  Ad[47] = 0;
  Ad[48] = 1;
}

// ---------------------------------------------------------------------------
// SL(4): row-major 4x4, contiguous.
// ---------------------------------------------------------------------------

/**
 * M^{-1} via adjugate / determinant from the 12 2x2 sub-determinants of rows
 * {0,1} and {2,3}. Returns false (R untouched) if |det M| < 1e-7.
 */
__device__ __forceinline__ bool Inv4(const float *M, float *R) {
  const float s0 = M[0] * M[5] - M[1] * M[4];
  const float s1 = M[0] * M[6] - M[2] * M[4];
  const float s2 = M[0] * M[7] - M[3] * M[4];
  const float s3 = M[1] * M[6] - M[2] * M[5];
  const float s4 = M[1] * M[7] - M[3] * M[5];
  const float s5 = M[2] * M[7] - M[3] * M[6];

  const float c0 = M[8] * M[13] - M[9] * M[12];
  const float c1 = M[8] * M[14] - M[10] * M[12];
  const float c2 = M[8] * M[15] - M[11] * M[12];
  const float c3 = M[9] * M[14] - M[10] * M[13];
  const float c4 = M[9] * M[15] - M[11] * M[13];
  const float c5 = M[10] * M[15] - M[11] * M[14];

  const float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
  if (fabsf(det) < 1e-7f) {
    return false;
  }
  const float d = __frcp_rn(det);  // == 1.0f / det, bitwise

  R[0] = d * (M[5] * c5 - M[6] * c4 + M[7] * c3);
  R[1] = d * (-M[1] * c5 + M[2] * c4 - M[3] * c3);
  R[2] = d * (M[13] * s5 - M[14] * s4 + M[15] * s3);
  R[3] = d * (-M[9] * s5 + M[10] * s4 - M[11] * s3);

  R[4] = d * (-M[4] * c5 + M[6] * c2 - M[7] * c1);
  R[5] = d * (M[0] * c5 - M[2] * c2 + M[3] * c1);
  R[6] = d * (-M[12] * s5 + M[14] * s2 - M[15] * s1);
  R[7] = d * (M[8] * s5 - M[10] * s2 + M[11] * s1);

  R[8] = d * (M[4] * c4 - M[5] * c2 + M[7] * c0);
  R[9] = d * (-M[0] * c4 + M[1] * c2 - M[3] * c0);
  R[10] = d * (M[12] * s4 - M[13] * s2 + M[15] * s0);
  R[11] = d * (-M[8] * s4 + M[9] * s2 - M[11] * s0);

  R[12] = d * (-M[4] * c3 + M[5] * c1 - M[6] * c0);
  R[13] = d * (M[0] * c3 - M[1] * c1 + M[2] * c0);
  R[14] = d * (-M[12] * s3 + M[13] * s1 - M[14] * s0);
  R[15] = d * (M[8] * s3 - M[9] * s1 + M[10] * s0);
  return true;
}

/**
 * Nonzero entries of column `col` of the 16x15 orthonormal VEC_TO_ALG matrix
 * (the sl(4) basis): up to four (row, value) pairs; returns the count (2-4).
 * cols 0-5: (E_ij - E_ji)/sqrt2, 6-11: (E_ij + E_ji)/sqrt2, 12-14: diagonal.
 */
__device__ __forceinline__ int SL4BasisColumnSupport(int col, int *rows, float *vals) {
  constexpr float kInvSqrt2 = 0.7071067811865475244f;
  constexpr float kInvSqrt6 = 0.4082482904638630164f;
  constexpr float kInvSqrt12 = 0.2886751345948128823f;
  constexpr int kPairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  if (col < 6) {
    const int p = col;
    rows[0] = kPairs[p][0] * 4 + kPairs[p][1];
    vals[0] = kInvSqrt2;
    rows[1] = kPairs[p][1] * 4 + kPairs[p][0];
    vals[1] = -kInvSqrt2;
    return 2;
  }
  if (col < 12) {
    const int p = col - 6;
    rows[0] = kPairs[p][0] * 4 + kPairs[p][1];
    vals[0] = kInvSqrt2;
    rows[1] = kPairs[p][1] * 4 + kPairs[p][0];
    vals[1] = kInvSqrt2;
    return 2;
  }
  if (col == 12) {
    rows[0] = 0;
    vals[0] = kInvSqrt2;
    rows[1] = 5;
    vals[1] = -kInvSqrt2;
    return 2;
  }
  if (col == 13) {
    rows[0] = 0;
    vals[0] = kInvSqrt6;
    rows[1] = 5;
    vals[1] = kInvSqrt6;
    rows[2] = 10;
    vals[2] = -2.f * kInvSqrt6;
    return 3;
  }
  rows[0] = 0;
  vals[0] = kInvSqrt12;
  rows[1] = 5;
  vals[1] = kInvSqrt12;
  rows[2] = 10;
  vals[2] = kInvSqrt12;
  rows[3] = 15;
  vals[3] = -3.f * kInvSqrt12;
  return 4;
}

/** Basis supports of all 15 sl(4) columns, typically staged in shared memory. */
struct SL4Basis {
  int nnz[15];
  int rows[15][4];
  float vals[15][4];
};

/**
 * Entry (i, j) of Ad(T) for SL(4): <E_i, T E_j T^{-1}> over the basis
 * supports, with T and T^{-1} row-major 4x4.
 */
__device__ __forceinline__ float SL4AdjointEntry(const SL4Basis &basis, int i, int j,
                                                 const float *T, const float *Tinv) {
  const int ni = basis.nnz[i];
  const int nj = basis.nnz[j];
  const int *ri = basis.rows[i];
  const float *vi = basis.vals[i];
  const int *rj = basis.rows[j];
  const float *vj = basis.vals[j];

  float s = 0.f;
#pragma unroll 4
  for (int ia = 0; ia < ni; ++ia) {
    const int a = ri[ia];
    const float va = vi[ia];
    const float t_row0 = va * T[(a >> 2) * 4];
    const float t_row1 = va * T[(a >> 2) * 4 + 1];
    const float t_row2 = va * T[(a >> 2) * 4 + 2];
    const float t_row3 = va * T[(a >> 2) * 4 + 3];
    const int mi = a & 3;
#pragma unroll 4
    for (int jb = 0; jb < nj; ++jb) {
      const int t = rj[jb];
      const int bj = t >> 2;
      const float tinv = Tinv[(t & 3) * 4 + mi];
      float tij;
      if (bj == 0)
        tij = t_row0;
      else if (bj == 1)
        tij = t_row1;
      else if (bj == 2)
        tij = t_row2;
      else
        tij = t_row3;
      s += tij * tinv * vj[jb];
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// SO(3) / SE(3) exponential and logarithm maps and their Jacobians (shared by
// cunls/math/so_se_lie_math.cu and the factor kernels).
// ---------------------------------------------------------------------------

/** [v]_x (3x3, rows `pitch` apart). */
__device__ __forceinline__ void SkewMatrix(const float *v, float *ptr, const size_t pitch) {
  ptr[0 * pitch + 0] = 0;
  ptr[0 * pitch + 1] = -v[2];
  ptr[0 * pitch + 2] = v[1];
  ptr[1 * pitch + 0] = v[2];
  ptr[1 * pitch + 1] = 0;
  ptr[1 * pitch + 2] = -v[0];
  ptr[2 * pitch + 0] = -v[1];
  ptr[2 * pitch + 1] = v[0];
  ptr[2 * pitch + 2] = 0;
}

/**
 * k1 I + k2 [phi]_x + k3 phi phi^T (rows `pitch` apart); below |phi| = tol,
 * I + k4 [phi]_x (the first-order form of every caller).
 */
__device__ __forceinline__ void RodriguesMatrix(const float *phi, float k1, float k2, float k3,
                                                float k4, float *ptr, const size_t pitch,
                                                float tol = 1e-5f) {
  float theta = norm3df(phi[0], phi[1], phi[2]);
  assert(theta >= 0);
  if (theta < tol) {
    ptr[0 * pitch + 0] = 1;
    ptr[0 * pitch + 1] = -k4 * phi[2];
    ptr[0 * pitch + 2] = k4 * phi[1];
    ptr[1 * pitch + 0] = k4 * phi[2];
    ptr[1 * pitch + 1] = 1;
    ptr[1 * pitch + 2] = -k4 * phi[0];
    ptr[2 * pitch + 0] = -k4 * phi[1];
    ptr[2 * pitch + 1] = k4 * phi[0];
    ptr[2 * pitch + 2] = 1;
    return;
  }
  float a = k2 * phi[2];
  float b = k2 * phi[1];
  float c = k2 * phi[0];
  float phi01 = k3 * phi[0] * phi[1];
  float phi02 = k3 * phi[0] * phi[2];
  float phi12 = k3 * phi[1] * phi[2];
  ptr[0 * pitch + 0] = k1 + k3 * phi[0] * phi[0];
  ptr[0 * pitch + 1] = -a + phi01;
  ptr[0 * pitch + 2] = b + phi02;
  ptr[1 * pitch + 0] = a + phi01;
  ptr[1 * pitch + 1] = k1 + k3 * phi[1] * phi[1];
  ptr[1 * pitch + 2] = -c + phi12;
  ptr[2 * pitch + 0] = -b + phi02;
  ptr[2 * pitch + 1] = c + phi12;
  ptr[2 * pitch + 2] = k1 + k3 * phi[2] * phi[2];
}

/**
 * sin(θ)/θ, (1 - cos θ)/θ² and (θ - sin θ)/θ³ for θ = |phi|, accurate in
 * float32 at every angle: 1 - cos θ = 2 sin²(θ/2) (no cancellation) and a
 * series for the third below θ = 0.3 (the direct forms lose all precision
 * near θ = 1e-4).
 */
__device__ __forceinline__ void SO3Coefficients(float theta, float *sin_over, float *one_minus_cos,
                                                float *theta_minus_sin) {
  const float t2 = theta * theta;
  if (theta < 0.3f) {
    *sin_over = 1.f - t2 / 6.f + t2 * t2 / 120.f;
    *one_minus_cos = 0.5f - t2 / 24.f + t2 * t2 / 720.f;
    *theta_minus_sin = 1.f / 6.f - t2 / 120.f + t2 * t2 / 5040.f;
  } else {
    const float h = sinf(0.5f * theta);
    const float s = sinf(theta);
    *sin_over = s / theta;
    *one_minus_cos = 2.f * h * h / t2;
    *theta_minus_sin = (theta - s) / (t2 * theta);
  }
}

/** R = Exp(phi) (Rodrigues; rows `pitch` apart). */
__device__ __forceinline__ void ExpSO3(const float *phi, float *ptr, const size_t pitch) {
  const float theta = norm3df(phi[0], phi[1], phi[2]);
  float a, b, c;
  SO3Coefficients(theta, &a, &b, &c);
  // I + a [phi]x + b [phi]x² = (1 - b θ²) I + a [phi]x + b phi phi^T
  RodriguesMatrix(phi, 1.f - b * theta * theta, a, b, 1, ptr, pitch);
}

/** J_l(phi) of SO(3) (rows `pitch` apart). J_r(phi) = J_l(-phi). */
__device__ __forceinline__ void SO3JacobianLeft(const float *phi, float *ptr, const size_t pitch) {
  const float theta = norm3df(phi[0], phi[1], phi[2]);
  float a, b, c;
  SO3Coefficients(theta, &a, &b, &c);
  // I + b [phi]x + c [phi]x² = (1 - c θ²) I + b [phi]x + c phi phi^T, 1 - c θ² = a
  RodriguesMatrix(phi, a, b, c, 0.5, ptr, pitch);
}

/** J_l(phi)^{-1} of SO(3) (rows `pitch` apart). J_r^{-1}(phi) = J_l^{-1}(-phi). */
__device__ __forceinline__ void SO3JacobianLeftInverse(const float *phi, float *ptr,
                                                       const size_t pitch) {
  const float theta = norm3df(phi[0], phi[1], phi[2]);
  // I - [phi]x / 2 + k [phi]x², k = (1 - h) / θ² with h = (θ/2) cot(θ/2);
  // HalfCotCoefficients gives h and (1 - h) / θ without cancellation.
  float h, rest;
  HalfCotCoefficients(theta, &h, &rest);
  const float k = theta > 0.f ? rest / theta : 1.f / 12.f;
  RodriguesMatrix(phi, h, -0.5f, k, -0.5, ptr, pitch);
}

/** phi = Log(R) (rows of R `rotation_pitch` apart), accurate at every angle in float32. */
__device__ __forceinline__ void LogSO3(const float *rotation_matrix, const size_t rotation_pitch,
                                       float *twist) {
  float trace = 0;
#pragma unroll
  for (int i = 0; i < 3; i++) {
    trace += rotation_matrix[i * rotation_pitch + i];
  }
  twist[0] = twist[1] = twist[2] = 0.f;

  // v = vee((R - R^T) / 2) = sin(theta) n. theta = atan2(|v|, cos(theta)) is
  // accurate at every angle in float32; acos((trace - 1) / 2) is not near 0,
  // and cutting off near the identity zeroed residuals up to ~3e-3 rad.
  const float v0 =
      0.5f * (rotation_matrix[2 * rotation_pitch + 1] - rotation_matrix[1 * rotation_pitch + 2]);
  const float v1 =
      0.5f * (rotation_matrix[0 * rotation_pitch + 2] - rotation_matrix[2 * rotation_pitch + 0]);
  const float v2 =
      0.5f * (rotation_matrix[1 * rotation_pitch + 0] - rotation_matrix[0 * rotation_pitch + 1]);
  const float sin_theta = norm3df(v0, v1, v2);
  const float cos_theta = 0.5f * (trace - 1.0f);
  const float theta = atan2f(sin_theta, cos_theta);

  if (cos_theta < 0.f && sin_theta < 1e-3f) {
    // Near pi, sin(theta) ~ 0 and v carries no reliable axis. The symmetric
    // part gives it exactly at any angle: (R + R^T) / 2 - cos(theta) I =
    // (1 - cos(theta)) n n^T. Take the column with the largest diagonal
    // (best conditioned) and fix the sign with v = sin(theta) n.
    const float b00 = rotation_matrix[0] - cos_theta;
    const float b11 = rotation_matrix[rotation_pitch + 1] - cos_theta;
    const float b22 = rotation_matrix[2 * rotation_pitch + 2] - cos_theta;
    int best = 0;
    if (b11 > b00 && b11 >= b22) best = 1;
    if (b22 > b00 && b22 > b11) best = 2;
    auto sym = [&](int r, int c) {
      return 0.5f * (rotation_matrix[r * rotation_pitch + c] +
                     rotation_matrix[c * rotation_pitch + r]) -
             (r == c ? cos_theta : 0.f);
    };
    float n0 = sym(0, best), n1 = sym(1, best), n2 = sym(2, best);
    const float sq = n0 * n0 + n1 * n1 + n2 * n2;
    if (sq > 0.0f) {
      float scale = theta * rsqrtf(sq);
      if (n0 * v0 + n1 * v1 + n2 * v2 < 0.f) scale = -scale;
      twist[0] = n0 * scale;
      twist[1] = n1 * scale;
      twist[2] = n2 * scale;
    }
    return;
  }

  // theta / sin(theta), with its series near 0 (theta ~ sin(theta) there).
  const float k = sin_theta > 1e-4f ? theta / sin_theta : 1.f + sin_theta * sin_theta / 6.f;
  twist[0] = k * v0;
  twist[1] = k * v1;
  twist[2] = k * v2;
}

/** C = A B (3x3, contiguous row-major). */
__device__ __forceinline__ void MatMul3x3(const float *A, const float *B, float *C) {
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
#pragma unroll
    for (uint8_t j = 0; j < 3; j++) {
      C[i * 3 + j] =
          A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] + A[i * 3 + 2] * B[2 * 3 + j];
    }
  }
}

/** B += scale A (3x3). */
__device__ __forceinline__ void ScaleAdd3x3(const float *A, float scale, float *B) {
#pragma unroll
  for (uint8_t i = 0; i < 9; i++) B[i] += scale * A[i];
}

/** C += scale A B (3x3). */
__device__ __forceinline__ void MatMulAdd3x3(const float *A, const float *B, float scale,
                                             float *C) {
#pragma unroll
  for (uint8_t i = 0; i < 3; i++) {
#pragma unroll
    for (uint8_t j = 0; j < 3; j++) {
      C[i * 3 + j] += scale * (A[i * 3 + 0] * B[0 * 3 + j] + A[i * 3 + 1] * B[1 * 3 + j] +
                               A[i * 3 + 2] * B[2 * 3 + j]);
    }
  }
}

/**
 * Coefficients A, B, C of Q_left(xi) for |phi| = `phi`: the closed forms
 * cancel catastrophically in float32 for small angles (B loses ~1e-2 relative
 * at phi = 0.05), so their Taylor series is used below 0.3.
 */
__device__ __forceinline__ void QLeftCoefficients(float phi, float *A, float *B, float *C) {
  if (phi < 0.3f) {
    const float p2 = phi * phi, p4 = p2 * p2;
    *A = 1.f / 6.f - p2 / 120.f + p4 / 5040.f;
    *B = 1.f / 24.f - p2 / 720.f + p4 / 40320.f;
    *C = 1.f / 120.f - p2 / 2520.f + p4 / 120960.f;
  } else {
    const float s = sinf(phi), c = cosf(phi);
    const float p2 = phi * phi, p3 = p2 * phi, p4 = p3 * phi, p5 = p4 * phi;
    *A = (phi - s) / p3;
    *B = (p2 * 0.5f + c - 1.f) / p4;
    *C = 0.5f * ((2.f + c) / p4 - 3.f * s / p5);
  }
}

/** Q_left(xi) of SE(3) for xi = [phi, rho] (3x3, rows `Q_pitch` apart). */
__device__ __forceinline__ void QLeftSE3(const float *twist, const size_t Q_pitch, float *Q) {
  float A, B, C;
  QLeftCoefficients(norm3df(twist[0], twist[1], twist[2]), &A, &B, &C);

  float result[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  float temp[9];
  SkewMatrix(&twist[3], temp, 3);
  ScaleAdd3x3(temp, 0.5f, result);

  float W[9];
  SkewMatrix(twist, W, 3);

  float VW[9];
  MatMul3x3(temp, W, VW);
  ScaleAdd3x3(VW, A, result);
  MatMulAdd3x3(VW, W, B, result);

  float WV[9];
  MatMul3x3(W, temp, WV);
  ScaleAdd3x3(WV, A, result);
  MatMulAdd3x3(W, WV, B, result);

  MatMul3x3(WV, W, temp);  // WVW
  ScaleAdd3x3(temp, A - 3.f * B, result);
  MatMulAdd3x3(temp, W, C, result);
  MatMulAdd3x3(W, temp, C, result);

  for (uint8_t i = 0; i < 3; i++)
    for (uint8_t j = 0; j < 3; j++) Q[i * Q_pitch + j] = result[i * 3 + j];
}

/** T = Exp(xi) of SE(3), xi = [phi, rho]: R = Exp(phi), t = J_l(phi) rho (rows `pitch` apart). */
__device__ __forceinline__ void ExpSE3(const float *xi, float *T, const size_t pitch) {
  float J[9];
  SO3JacobianLeft(xi, J, 3);
  ExpSO3(xi, T, pitch);
  for (int i = 0; i < 3; ++i) {
    T[i * pitch + 3] = J[i * 3 + 0] * xi[3] + J[i * 3 + 1] * xi[4] + J[i * 3 + 2] * xi[5];
    T[3 * pitch + i] = 0.f;
  }
  T[3 * pitch + 3] = 1.f;
}

/** xi = Log(T) of SE(3): phi = Log(R), rho = J_l(phi)^{-1} t (rows of T `pitch` apart). */
__device__ __forceinline__ void LogSE3(const float *T, const size_t pitch, float *xi) {
  LogSO3(T, pitch, xi);
  const float t[3] = {T[0 * pitch + 3], T[1 * pitch + 3], T[2 * pitch + 3]};
  float J_inv[9];
  SO3JacobianLeftInverse(xi, J_inv, 3);
  for (int i = 0; i < 3; ++i) {
    xi[3 + i] = J_inv[i * 3 + 0] * t[0] + J_inv[i * 3 + 1] * t[1] + J_inv[i * 3 + 2] * t[2];
  }
}

/** J_l(xi) of SE(3) (6x6, rows `pitch` apart): [[J_l(phi), 0], [Q(xi), J_l(phi)]]. J_r(xi) =
 * J_l(-xi). */
__device__ __forceinline__ void SE3JacobianLeft(const float *xi, float *J, const size_t pitch) {
  float Jl[9], Q[9];
  SO3JacobianLeft(xi, Jl, 3);
  QLeftSE3(xi, 3, Q);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      J[i * pitch + j] = Jl[i * 3 + j];
      J[i * pitch + 3 + j] = 0.f;
      J[(3 + i) * pitch + j] = Q[i * 3 + j];
      J[(3 + i) * pitch + 3 + j] = Jl[i * 3 + j];
    }
  }
}

/**
 * J_l(xi)^{-1} of SE(3) (6x6, rows `pitch` apart):
 * [[J_l^{-1}, 0], [-J_l^{-1} Q J_l^{-1}, J_l^{-1}]]. J_r^{-1}(xi) = J_l^{-1}(-xi).
 */
__device__ __forceinline__ void SE3JacobianLeftInverse(const float *xi, float *J,
                                                       const size_t pitch) {
  float Ji[9], Q[9], T[9];
  SO3JacobianLeftInverse(xi, Ji, 3);
  QLeftSE3(xi, 3, Q);
  MatMul3x3(Ji, Q, T);
  MatMul3x3(T, Ji, Q);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      J[i * pitch + j] = Ji[i * 3 + j];
      J[i * pitch + 3 + j] = 0.f;
      J[(3 + i) * pitch + j] = -Q[i * 3 + j];
      J[(3 + i) * pitch + 3 + j] = Ji[i * 3 + j];
    }
  }
}

}  // namespace lie_device
}  // namespace cunls
