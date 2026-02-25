/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/similarity3_prior_factor_batch.h"

namespace cunls {

constexpr size_t kSim3PriorBlockSize = 256;

// ============================================================================
// 3x3 matrix helpers (row-major, stored in float[9])
// ============================================================================

/// C = A * B  (3x3)
__device__ void mat3_mul(const float* __restrict__ A,
                         const float* __restrict__ B, float* __restrict__ C) {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      float s = 0.0f;
      for (int k = 0; k < 3; ++k) s += A[r * 3 + k] * B[k * 3 + c];
      C[r * 3 + c] = s;
    }
}

/// out = M * v  (3x3 * 3x1)
__device__ void mat3_vec(const float* __restrict__ M,
                         const float* __restrict__ v, float* __restrict__ out) {
  for (int r = 0; r < 3; ++r) {
    float s = 0.0f;
    for (int k = 0; k < 3; ++k) s += M[r * 3 + k] * v[k];
    out[r] = s;
  }
}

/// Invert a 3x3 matrix via cofactors. Returns false if singular.
__device__ bool mat3_inv(const float* __restrict__ M,
                         float* __restrict__ Minv) {
  float c0 = M[4] * M[8] - M[5] * M[7];
  float c1 = -(M[3] * M[8] - M[5] * M[6]);
  float c2 = M[3] * M[7] - M[4] * M[6];
  float det = M[0] * c0 + M[1] * c1 + M[2] * c2;
  if (fabsf(det) < 1e-12f) return false;
  float inv_det = 1.0f / det;
  Minv[0] = c0 * inv_det;
  Minv[1] = (-(M[1] * M[8] - M[2] * M[7])) * inv_det;
  Minv[2] = (M[1] * M[5] - M[2] * M[4]) * inv_det;
  Minv[3] = c1 * inv_det;
  Minv[4] = (M[0] * M[8] - M[2] * M[6]) * inv_det;
  Minv[5] = (-(M[0] * M[5] - M[2] * M[3])) * inv_det;
  Minv[6] = c2 * inv_det;
  Minv[7] = (-(M[0] * M[7] - M[1] * M[6])) * inv_det;
  Minv[8] = (M[0] * M[4] - M[1] * M[3]) * inv_det;
  return true;
}

/// Build skew-symmetric matrix from 3-vector
__device__ void skew3(const float* v, float* S) {
  S[0] = 0.0f;   S[1] = -v[2];  S[2] = v[1];
  S[3] = v[2];   S[4] = 0.0f;   S[5] = -v[0];
  S[6] = -v[1];  S[7] = v[0];   S[8] = 0.0f;
}

// ============================================================================
// Sim(3) kernels
// ============================================================================

/**
 * @brief CUDA kernel to compute inverse of Sim(3) matrices.
 *
 * For T = [R t; 0 1/s], computes T^{-1} = [R^T  -s*R^T*t; 0  s].
 */
__global__ void inverse_sim3_kernel(const float* transforms,
                                    float* inverse_transforms, size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 16;
  float* Ti = inverse_transforms + tid * 16;

  // R in rows 0-2, columns 0-2
  float r00 = T[0], r01 = T[1], r02 = T[2], tx = T[3];
  float r10 = T[4], r11 = T[5], r12 = T[6], ty = T[7];
  float r20 = T[8], r21 = T[9], r22 = T[10], tz = T[11];
  float inv_s = T[15];  // 1/s
  float s = 1.0f / inv_s;

  // T^{-1} = [R^T, -s*R^T*t; 0, s]
  // R^T row 0 = (r00, r10, r20)
  Ti[0]  = r00;   Ti[1]  = r10;   Ti[2]  = r20;
  Ti[3]  = -s * (r00 * tx + r10 * ty + r20 * tz);
  Ti[4]  = r01;   Ti[5]  = r11;   Ti[6]  = r21;
  Ti[7]  = -s * (r01 * tx + r11 * ty + r21 * tz);
  Ti[8]  = r02;   Ti[9]  = r12;   Ti[10] = r22;
  Ti[11] = -s * (r02 * tx + r12 * ty + r22 * tz);
  Ti[12] = 0.0f;  Ti[13] = 0.0f;  Ti[14] = 0.0f;  Ti[15] = s;
}

/**
 * @brief Collect Sim(3) transforms from state pointers.
 */
__global__ void collect_sim3_transforms_kernel(float const* const* state_pointers,
                                               size_t num_factors,
                                               Matrix<4>* transforms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;
  auto p = reinterpret_cast<const Matrix<4>*>(state_pointers[tid]);
  transforms[tid] = *p;
}

/**
 * @brief CUDA kernel: Sim(3) logarithm map.
 *
 * Given T = [R t; 0 1/s], computes xi = [w, u, lambda] where
 * w = Log_SO3(R), lambda = log(s), u = V(w,lambda)^{-1} * t.
 */
__global__ void log_sim3_kernel(const float* transforms, float* tangent,
                                size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* T = transforms + tid * 16;
  float* xi = tangent + tid * 7;

  // Extract R (3x3) and t (3x1)
  float R[9] = {T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]};
  float t[3] = {T[3], T[7], T[11]};
  float inv_s = T[15];
  float s_val = 1.0f / inv_s;
  float lambda = logf(s_val);

  // --- Log_SO3(R) ---
  float trace = R[0] + R[4] + R[8];
  float cos_a = fmaxf(-1.0f, fminf(1.0f, (trace - 1.0f) * 0.5f));
  float theta = acosf(cos_a);
  float theta2 = theta * theta;

  float vee[3] = {R[7] - R[5], R[2] - R[6], R[3] - R[1]};
  float w[3];
  if (fabsf(theta) < 1e-4f) {
    w[0] = 0.5f * vee[0];
    w[1] = 0.5f * vee[1];
    w[2] = 0.5f * vee[2];
  } else {
    float k = theta / (2.0f * sinf(theta));
    w[0] = k * vee[0];
    w[1] = k * vee[1];
    w[2] = k * vee[2];
  }

  // --- SO(3) coefficients ---
  float A1, A2, A3, A4;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    A1 = st / theta;
    A2 = (1.0f - ct) / theta2;
    A3 = (1.0f - A1) / theta2;
    A4 = (0.5f - A2) / theta2;
  } else {
    A1 = 1.0f - theta2 / 6.0f;
    A2 = 0.5f - theta2 / 24.0f;
    A3 = 1.0f / 6.0f - theta2 / 120.0f;
    A4 = 1.0f / 24.0f - theta2 / 720.0f;
  }

  // --- V-matrix coefficients ---
  float lambda2 = lambda * lambda;
  float P_c, Q_c, R_c;
  if (lambda2 > 1e-8f) {
    float e = expf(-lambda);
    P_c = (1.0f - e) / lambda;
    float alpha = lambda2 / (lambda2 + theta2);
    float beta = (e - 1.0f + lambda) / lambda2;
    float mu = (1.0f - lambda + 0.5f * lambda2 - e) / (lambda2 * lambda);
    float one_m_a = 1.0f - alpha;
    Q_c = alpha * beta + one_m_a * (A2 - lambda * A3);
    R_c = alpha * mu + one_m_a * (A3 - lambda * A4);
  } else {
    P_c = 1.0f - lambda / 2.0f + lambda2 / 6.0f;
    Q_c = A2 - lambda * A3;
    R_c = A3 - lambda * A4;
  }

  // --- Build V (3x3) ---
  float diag = P_c - R_c * theta2;
  float V[9];
  float W_mat[9];
  skew3(w, W_mat);
  // V = diag*I + Q*W + R_c*ww^T
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float ij = i * 3 + j;
      V[(int)ij] = R_c * w[i] * w[j] + Q_c * W_mat[(int)ij];
      if (i == j) V[(int)ij] += diag;
    }

  // --- u = V^{-1} * t ---
  float Vinv[9];
  if (!mat3_inv(V, Vinv)) {
    // Fallback for near-identity
    xi[0] = w[0]; xi[1] = w[1]; xi[2] = w[2];
    xi[3] = t[0]; xi[4] = t[1]; xi[5] = t[2];
    xi[6] = lambda;
    return;
  }
  float u[3];
  mat3_vec(Vinv, t, u);

  xi[0] = w[0]; xi[1] = w[1]; xi[2] = w[2];
  xi[3] = u[0]; xi[4] = u[1]; xi[5] = u[2];
  xi[6] = lambda;
}

/**
 * @brief CUDA kernel: inverse right Jacobian of Sim(3).
 *
 * Computes the 7x7 J_r^{-1}(xi) using a truncated Bernoulli series
 * (up to order 6) with efficient block decomposition.
 *
 * The ad matrix of Sim(3) has block structure:
 *   ad(xi) = [[W,   0,      0  ],   3x7
 *             [U,   W+lI,  -u  ],   3x7
 *             [0,   0,      0  ]]   1x7
 *
 * This gives J_r^{-1} the block form:
 *   [[J_r^{-1}_SO3(w),  0,         0     ],
 *    [BL,               BR,        col_u  ],
 *    [0,                0,         1      ]]
 *
 * - Block(0,0) = J_r^{-1}_SO3(w): closed-form SO(3) right Jacobian inverse.
 * - Block(1,1) = BR = Sum B_n^+/n! * M^n  with M = W + lambda*I,
 *   computed via scalar M^n = a_n*I + b_n*W + c_n*W^2 recursion.
 * - Block(1,0) = BL: iterative 3x3 matrix recursion.
 * - Block(1,2) = col_u: scalar recursion + precomputed vectors.
 */
__global__ void jacobian_right_inverse_sim3_kernel(const float* tangent,
                                                   float* jacobians,
                                                   size_t size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  const float* xi = tangent + tid * 7;
  float* J = jacobians + tid * 49;

  float w[3] = {xi[0], xi[1], xi[2]};
  float u[3] = {xi[3], xi[4], xi[5]};
  float lam = xi[6];

  float theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];

  // --- Small-norm fallback: J_r^{-1} ≈ I + 0.5*ad ---
  if (theta2 + lam * lam < 1e-6f) {
    // Build I + 0.5 * ad directly
    // Row 0-2: [I + 0.5*W, 0, 0]
    // Row 3-5: [0.5*U, I + 0.5*(W+lam*I), -0.5*u]
    // Row 6:   [0, 0, 1]
    for (int i = 0; i < 49; ++i) J[i] = 0.0f;
    // Identity diagonal
    J[0] = 1.0f; J[8] = 1.0f; J[16] = 1.0f;
    J[24] = 1.0f; J[32] = 1.0f; J[40] = 1.0f;
    J[48] = 1.0f;
    // 0.5 * W in block(0,0) and block(1,1)
    // W = skew(w)
    // block(0,0): rows 0-2, cols 0-2
    J[1]  += -0.5f * w[2];  J[2]  +=  0.5f * w[1];
    J[7]  +=  0.5f * w[2];  J[9]  += -0.5f * w[0];
    J[14] += -0.5f * w[1];  J[15] +=  0.5f * w[0];
    // block(1,1): rows 3-5, cols 3-5  plus 0.5*lam*I
    J[24] += 0.5f * lam;    J[25] += -0.5f * w[2];  J[26] +=  0.5f * w[1];
    J[31] += 0.5f * w[2];   J[32] += 0.5f * lam;    J[33] += -0.5f * w[0];
    J[38] += -0.5f * w[1];  J[39] +=  0.5f * w[0];  J[40] += 0.5f * lam;
    // block(1,0): 0.5*U  rows 3-5, cols 0-2, U = skew(u)
    J[21] += -0.5f * u[2];  J[22] +=  0.5f * u[1];
    J[28] +=  0.5f * u[2];  J[30] += -0.5f * u[0];
    J[35] += -0.5f * u[1];  J[36] +=  0.5f * u[0];
    // block(1,2): -0.5*u  rows 3-5, col 6
    J[27] = -0.5f * u[0];  J[34] = -0.5f * u[1];  J[41] = -0.5f * u[2];
    return;
  }

  float theta = sqrtf(theta2);

  // ========== Block(0,0): J_r^{-1}_SO3(w) = a0*I + a1*W + a2*W^2 ==========
  float a0 = 1.0f, a1 = 0.5f, a2;
  if (theta2 > 1e-6f) {
    float st = sinf(theta), ct = cosf(theta);
    a2 = 1.0f / theta2 - (1.0f + ct) / (2.0f * theta * st);
  } else {
    a2 = 1.0f / 12.0f + theta2 / 720.0f;
  }

  // ========== Precompute W, W^2, U, U*W, U*W^2 ==========
  float W[9], W2[9], U_mat[9], UW[9], UW2[9];
  skew3(w, W);
  skew3(u, U_mat);
  mat3_mul(W, W, W2);
  mat3_mul(U_mat, W, UW);
  mat3_mul(U_mat, W2, UW2);

  // Precompute cross-product and dot-product vectors for block(1,2)
  float wu_cross[3] = {w[1]*u[2]-w[2]*u[1], w[2]*u[0]-w[0]*u[2], w[0]*u[1]-w[1]*u[0]};
  float dot_wu = w[0]*u[0] + w[1]*u[1] + w[2]*u[2];
  float W2u[3] = {w[0]*dot_wu - theta2*u[0],
                   w[1]*dot_wu - theta2*u[1],
                   w[2]*dot_wu - theta2*u[2]};

  // ========== Bernoulli coefficients for n=1..6 ==========
  // B_n^+/n!: c[1]=1/2, c[2]=1/12, c[3]=0, c[4]=-1/720, c[5]=0, c[6]=1/30240
  // Shifted: d[0]=1/2, d[1]=1/12, d[2]=0, d[3]=-1/720, d[4]=0, d[5]=1/30240
  constexpr float c1 = 0.5f, c2 = 1.0f/12.0f, c4 = -1.0f/720.0f, c6 = 1.0f/30240.0f;
  constexpr float d0 = 0.5f, d1 = 1.0f/12.0f, d3 = -1.0f/720.0f, d5 = 1.0f/30240.0f;

  // ========== M^n scalar recursion: M^n = ma*I + mb*W + mc*W^2 ==========
  // Recursion: ma' = lam*ma, mb' = lam*mb + ma - theta2*mc, mc' = lam*mc + mb
  float ma = lam, mb = 1.0f, mc = 0.0f;  // M^1

  // W^n scalar recursion: W^n = wa*I + wb*W + wc*W^2  (wa always 0 for n>=1)
  // W^0 = I, W^1 = W
  // For U*W^{n-1}: uses wa_{n-1}, wb_{n-1}, wc_{n-1}
  float wa_prev = 1.0f, wb_prev = 0.0f, wc_prev = 0.0f;  // W^0

  // Accumulators for block(1,1): S_C = sa*I + sb*W + sc*W^2
  float sa = 0.0f, sb = 0.0f, sc = 0.0f;
  // (block(1,2) is computed separately via recomputed M^n scalars below)

  // Block(1,0): B_n is a full 3x3, and we accumulate S_B
  float Bn[9], SB[9];
  for (int i = 0; i < 9; ++i) { Bn[i] = U_mat[i]; SB[i] = 0.0f; }

  // --- n = 1 ---
  sa += c1 * ma; sb += c1 * mb; sc += c1 * mc;
  for (int i = 0; i < 9; ++i) SB[i] += c1 * Bn[i];

  // Advance M^n: n=1->2
  { float na = lam*ma;
    float nb = lam*mb + ma - theta2*mc;
    float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  // Advance W^n: need W^1 coeffs for n=2 step
  float wa_cur = 0.0f, wb_cur = 1.0f, wc_cur = 0.0f;  // W^1
  // Advance B_n: B_2 = U*W^1 + M*B_1
  // U*W^1 = wb_cur*UW (wa=0, wc=0 for W^1)
  {
    float UWn[9];
    for (int i = 0; i < 9; ++i)
      UWn[i] = wa_cur * U_mat[i] + wb_cur * UW[i] + wc_cur * UW2[i];
    // M*B_1 = lam*B_1 + W*B_1
    float WB[9];
    mat3_mul(W, Bn, WB);
    for (int i = 0; i < 9; ++i)
      Bn[i] = UWn[i] + lam * Bn[i] + WB[i];
    // w_prev = w_cur for next step
    wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = 1.0f;  // W^2
  }

  // --- n = 2 ---
  sa += c2 * ma; sb += c2 * mb; sc += c2 * mc;
  for (int i = 0; i < 9; ++i) SB[i] += c2 * Bn[i];

  // Advance M^n: n=2->3
  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  // Advance W^n: W^2 -> W^3 = -theta2*W
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = -theta2; wc_cur = 0.0f; }
  // Advance B_n
  { float UWn[9]; float WB[9];
    for (int i = 0; i < 9; ++i)
      UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i];
    mat3_mul(W, Bn, WB);
    for (int i = 0; i < 9; ++i) Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; }

  // --- n = 3: B_3^+ = 0, skip accumulation ---
  // Advance M^n: n=3->4
  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  // Advance W^n: W^3 -> W^4 = -theta2*W^2
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = -theta2; }
  // Advance B_n
  { float UWn[9]; float WB[9];
    for (int i = 0; i < 9; ++i)
      UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i];
    mat3_mul(W, Bn, WB);
    for (int i = 0; i < 9; ++i) Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; }

  // --- n = 4 ---
  sa += c4 * ma; sb += c4 * mb; sc += c4 * mc;
  for (int i = 0; i < 9; ++i) SB[i] += c4 * Bn[i];

  // Advance M^n: n=4->5
  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  // Advance W^n: W^4 -> W^5 = theta^4*W
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    float t4 = theta2*theta2;
    wa_cur = 0.0f; wb_cur = t4; wc_cur = 0.0f; }
  // Advance B_n
  { float UWn[9]; float WB[9];
    for (int i = 0; i < 9; ++i)
      UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i];
    mat3_mul(W, Bn, WB);
    for (int i = 0; i < 9; ++i) Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; }

  // --- n = 5: B_5^+ = 0, skip ---
  // Advance M^n: n=5->6
  { float na = lam*ma; float nb = lam*mb + ma - theta2*mc; float nc = lam*mc + mb;
    ma = na; mb = nb; mc = nc; }
  // Advance W^n: W^5 -> W^6 = theta^4*W^2
  { wa_prev = wa_cur; wb_prev = wb_cur; wc_prev = wc_cur;
    float t4 = theta2*theta2;
    wa_cur = 0.0f; wb_cur = 0.0f; wc_cur = t4; }
  // Advance B_n
  { float UWn[9]; float WB[9];
    for (int i = 0; i < 9; ++i)
      UWn[i] = wa_prev * U_mat[i] + wb_prev * UW[i] + wc_prev * UW2[i];
    mat3_mul(W, Bn, WB);
    for (int i = 0; i < 9; ++i) Bn[i] = UWn[i] + lam * Bn[i] + WB[i]; }

  // --- n = 6 ---
  sa += c6 * ma; sb += c6 * mb; sc += c6 * mc;
  for (int i = 0; i < 9; ++i) SB[i] += c6 * Bn[i];

  // ========== Assemble the 7x7 Jacobian (row-major) ==========
  // Zero-initialize
  for (int i = 0; i < 49; ++i) J[i] = 0.0f;

  // --- Block(0,0): J_r^{-1}_SO3(w) = a0*I + a1*W + a2*W^2 ---
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      int ij = r * 3 + c;
      float val = a1 * W[ij] + a2 * W2[ij];
      if (r == c) val += a0;
      J[r * 7 + c] = val;
    }

  // --- Block(0,1) = 0, Block(0,2) = 0  (already zero) ---

  // --- Block(1,1): I + sa*I + sb*W + sc*W^2 ---
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      int ij = r * 3 + c;
      float val = sb * W[ij] + sc * W2[ij];
      if (r == c) val += 1.0f + sa;
      J[(r + 3) * 7 + (c + 3)] = val;
    }

  // --- Block(1,0) = SB ---
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      J[(r + 3) * 7 + c] = SB[r * 3 + c];

  // --- Block(1,2) = -(ta*u + tb*wu_cross + tc*W2u) ---
  // tb and tc come from the shifted Bernoulli sums of beta_n and gamma_n
  // However, for the shifted series we need the M^{n-1} coefficients applied
  // to u. Using the recursion:
  // S_d = -Sum d_n * M^n * u = -(ta*u + tb*(w x u) + tc*(W^2*u))
  // where (ta, tb, tc) are the scalar sums of d_n * (ma_n, mb_n, mc_n).
  //
  // We need the *shifted* M^n coefficients: for d[k], we use M^k.
  // Let's recompute: d[0]*M^0=0.5*I, d[1]*M^1=(1/12)*M, d[3]*M^3=(-1/720)*M^3,
  // d[5]*M^5=(1/30240)*M^5

  // Recompute tb, tc properly via M^n scalar coefficients for the shifted series
  // M^0 = I: (1, 0, 0)
  // M^1 = (lam, 1, 0)
  // M^3: use recursion from M^1
  float ma_s[6], mb_s[6], mc_s[6];
  ma_s[0] = 1.0f; mb_s[0] = 0.0f; mc_s[0] = 0.0f;
  ma_s[1] = lam;  mb_s[1] = 1.0f; mc_s[1] = 0.0f;
  for (int n = 1; n < 5; ++n) {
    ma_s[n+1] = lam * ma_s[n];
    mb_s[n+1] = lam * mb_s[n] + ma_s[n] - theta2 * mc_s[n];
    mc_s[n+1] = lam * mc_s[n] + mb_s[n];
  }
  // Shifted Bernoulli sum: Sum d[k] * M^k for k=0,1,3,5
  float sd_a = d0*ma_s[0] + d1*ma_s[1] + d3*ma_s[3] + d5*ma_s[5];
  float sd_b = d0*mb_s[0] + d1*mb_s[1] + d3*mb_s[3] + d5*mb_s[5];
  float sd_c = d0*mc_s[0] + d1*mc_s[1] + d3*mc_s[3] + d5*mc_s[5];

  J[(3) * 7 + 6] = -(sd_a * u[0] + sd_b * wu_cross[0] + sd_c * W2u[0]);
  J[(4) * 7 + 6] = -(sd_a * u[1] + sd_b * wu_cross[1] + sd_c * W2u[1]);
  J[(5) * 7 + 6] = -(sd_a * u[2] + sd_b * wu_cross[2] + sd_c * W2u[2]);

  // --- Block(2,*) = [0, 0, 1] ---
  J[48] = 1.0f;
}

// ============================================================================
// Constructor & Evaluate
// ============================================================================

Similarity3PriorFactorBatch::Similarity3PriorFactorBatch(
    cuBLASHandle& cublas_handle, const Matrix<4>* observations_ptr,
    size_t num_factors)
    : observations_ptr_(observations_ptr),
      num_factors_(num_factors),
      observations_inverse_(num_factors),
      cublas_handle_(cublas_handle),
      transforms_current_(num_factors),
      transforms_error_(num_factors) {
  CudaStream stream;
  size_t num_blocks =
      (num_factors_ + kSim3PriorBlockSize - 1) / kSim3PriorBlockSize;
  inverse_sim3_kernel<<<num_blocks, kSim3PriorBlockSize, 0,
                        stream.GetStream()>>>(
      reinterpret_cast<const float*>(observations_ptr_),
      reinterpret_cast<float*>(observations_inverse_.data()),
      num_factors_);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
}

bool Similarity3PriorFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  size_t num_factors = NumFactors();
  size_t num_blocks =
      (num_factors + kSim3PriorBlockSize - 1) / kSim3PriorBlockSize;

  // Step 1: Collect transforms
  collect_sim3_transforms_kernel<<<num_blocks, kSim3PriorBlockSize, 0,
                                   stream>>>(
      state_pointers, num_factors, transforms_current_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 2: T_error = T_target^{-1} * T_current via cuBLAS
  auto handle = cublas_handle_.GetHandle(stream);
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 4;
  constexpr int stride = 16;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      reinterpret_cast<const float*>(transforms_current_.data()), mat_size,
      stride,
      reinterpret_cast<const float*>(observations_inverse_.data()), mat_size,
      stride, &beta,
      reinterpret_cast<float*>(transforms_error_.data()), mat_size, stride,
      num_factors));

  // Step 3: residual = Log(T_error)
  log_sim3_kernel<<<num_blocks, kSim3PriorBlockSize, 0, stream>>>(
      reinterpret_cast<const float*>(transforms_error_.data()), residuals,
      num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 4: Jacobian = J_r^{-1}(residual)
  if (jacobians != nullptr) {
    jacobian_right_inverse_sim3_kernel<<<num_blocks, kSim3PriorBlockSize, 0,
                                         stream>>>(residuals, jacobians,
                                                    num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
