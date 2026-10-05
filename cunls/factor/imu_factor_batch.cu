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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/factor/imu_factor_batch.h"
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
constexpr int kRows = 15;                        // 9 chain + 6 bias random walk
constexpr int kColumns = 6 + 3 + 6 + 6 + 3 + 6;  // T_a, v_a, b_a, T_b, v_b, b_b
constexpr int kRowStride = kColumns + 1;         // 31: odd, no bank conflicts
constexpr int kSampleSize = 7;                   // ω (3), a (3), Δt
constexpr int kMaxGroupSize = 32;                // lanes per factor, at most a warp
// Work split (tuned on an RTX A6000 over 16..65536 factors of 20..1000 samples):
// resident threads to aim for per SM (2 blocks of the Jacobian kernel at 255
// registers), and the fewest samples worth a lane of their own.
constexpr int kTargetThreadsPerSM = 256;
constexpr int kMinSamplesPerLane = 4;
// Floor for variances (Σ's Cholesky pivots, the bias random walk σ_b² T) of
// valid chains: keeps a numerically singular Σ finite instead of NaN or inf.
// Chains without duration are skipped instead (zero rows), see ImuKernel.
constexpr float kMinVariance = 1e-36f;

/**
 * Effect of a run of samples, in the frame of its first state (rotation I,
 * no gravity): x_end = x_start ⊕ (ΔR, Δv, Δp) with v_end = v_start + g T +
 * R_start Δv and p_end = p_start + v_start T + ½ g T² + R_start Δp. The
 * transition of the tangent [δφ, δv, δp] over the run is
 * Φ = [[ΔRᵀ, 0, 0], [-[Δv]x, I, 0], [-[Δp]x, T I, I]] (in this frame), Q its
 * accumulated noise and G = ∂x_end/∂(b_g, b_a). Runs compose associatively
 * (Combine), which is what lets several lanes share one chain.
 */
struct Summary {
  float R[9], v[3], p[3], T;
  float QRR[9], QRV[9], QRP[9], QVV[9], QVP[9], QPP[9];
  float GR[9], GVg[9], GPg[9], GVa[9], GPa[9];  // G_R of b_a is zero
};
constexpr int kSummaryFloats = sizeof(Summary) / sizeof(float);                // 115
constexpr int kResidualSummaryFloats = offsetof(Summary, GR) / sizeof(float);  // 70: no G

// --- 3x3 blocks (row-major) ---------------------------------------------------

/** C = A B. */
__device__ __forceinline__ void Mul(const float *A, const float *B, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
}

/** C = A Bᵀ. */
__device__ __forceinline__ void MulBt(const float *A, const float *B, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[i * 3 + j] =
          A[i * 3] * B[j * 3] + A[i * 3 + 1] * B[j * 3 + 1] + A[i * 3 + 2] * B[j * 3 + 2];
}

/** C = Aᵀ B. */
__device__ __forceinline__ void MulAt(const float *A, const float *B, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[i * 3 + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j];
}

/** C = A R for 3x3 A and the rotation R of T. */
__device__ __forceinline__ void MulRotation(const float *A, const SE3Transform &T, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[i * 3 + j] = A[i * 3] * T[j] + A[i * 3 + 1] * T[4 + j] + A[i * 3 + 2] * T[8 + j];
}

/** C = Rᵀ A for the rotation R of T and 3x3 A. */
__device__ __forceinline__ void MulRotationT(const SE3Transform &T, const float *A, float *C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[i * 3 + j] = T[i] * A[j] + T[4 + i] * A[3 + j] + T[8 + i] * A[6 + j];
}

/** C = R A Rᵀ. */
__device__ __forceinline__ void Rotate(const float *R, const float *A, float *C) {
  float X[9];
  Mul(R, A, X);
  MulBt(X, R, C);
}

/** Row vector times 3x3: out_j = Σ_k l_k A_kj. */
__device__ __forceinline__ void RowMul(const float *l, const float *A, float *out) {
#pragma unroll
  for (int j = 0; j < 3; ++j) out[j] = l[0] * A[j] + l[1] * A[3 + j] + l[2] * A[6 + j];
}

/** C = s A [t]x for 3x3 A. */
__device__ __forceinline__ void MulSkew(const float *A, const float *t, float s, float *C) {
  const float K[9] = {0.f, -s * t[2], s * t[1], s * t[2], 0.f, -s * t[0], -s * t[1], s * t[0], 0.f};
  Mul(A, K, C);
}

/** Packed lower-triangular index of (i, j), i >= j. */
__host__ __device__ constexpr int Tri(int i, int j) { return i * (i + 1) / 2 + j; }

__device__ __forceinline__ void Identity(Summary &s) {
#pragma unroll
  for (int i = 0; i < kSummaryFloats; ++i) reinterpret_cast<float *>(&s)[i] = 0.f;
  s.R[0] = s.R[4] = s.R[8] = 1.f;
}

/**
 * Q ← Φ Q Φᵀ for Φ = [[F, 0, 0], [U, I, 0], [W, τ I, I]] by 3x3 blocks, row
 * blocks P, V, R in that order so that each overwrites only old blocks no
 * longer needed.
 */
template <bool kScaledW = false>
__device__ __forceinline__ void Propagate(const float *F, const float *U, const float *W, float tau,
                                          Summary &s, float w_scale = 0.f) {
  float UR[9], UV[9], UP[9], X[9];
  Mul(U, s.QRR, UR);
  Mul(U, s.QRV, UV);
  Mul(U, s.QRP, UP);
  {
    // Row block P: T_P* = W Q_R* + τ Q_V* + Q_P*; Q'_PP = T_PR Wᵀ + τ T_PV + T_PP.
    // One Euler step has W = ½Δt U: W Q_R* = ½Δt U Q_R*, already computed.
    float WR[9], WV[9], WP[9], TPR[9], TPV[9];
    if constexpr (kScaledW) {
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        WR[i] = w_scale * UR[i];
        WV[i] = w_scale * UV[i];
        WP[i] = w_scale * UP[i];
      }
    } else {
      Mul(W, s.QRR, WR);
      Mul(W, s.QRV, WV);
      Mul(W, s.QRP, WP);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int ij = i * 3 + j, ji = j * 3 + i;
        TPR[ij] = WR[ij] + tau * s.QRV[ji] + s.QRP[ji];
        TPV[ij] = WV[ij] + tau * s.QVV[ij] + s.QVP[ji];
        s.QPP[ij] += WP[ij] + tau * s.QVP[ij];  // T_PP
      }
    MulBt(TPR, W, X);
#pragma unroll
    for (int i = 0; i < 9; ++i) s.QPP[i] += X[i] + tau * TPV[i];
  }
  {
    // Row block V: T_V* = U Q_R* + Q_V*; Q'_VV = T_VR Uᵀ + T_VV, Q'_VP = T_VR Wᵀ + τ T_VV + T_VP.
    float TVR[9];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int j = 0; j < 3; ++j) TVR[i * 3 + j] = UR[i * 3 + j] + s.QRV[j * 3 + i];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      s.QVV[i] += UV[i];                   // T_VV
      s.QVP[i] += UP[i] + tau * s.QVV[i];  // T_VP + τ T_VV
    }
    MulBt(TVR, W, X);
#pragma unroll
    for (int i = 0; i < 9; ++i) s.QVP[i] += X[i];
    MulBt(TVR, U, X);
#pragma unroll
    for (int i = 0; i < 9; ++i) s.QVV[i] += X[i];
  }
  {
    // Row block R: T_R* = F Q_R*; Q'_RR = T_RR Fᵀ, Q'_RV = T_RR Uᵀ + T_RV, Q'_RP = T_RR Wᵀ + τ T_RV
    // + T_RP.
    float TRR[9], TRV[9], TRP[9];
    Mul(F, s.QRR, TRR);
    Mul(F, s.QRV, TRV);
    Mul(F, s.QRP, TRP);
    MulBt(TRR, F, s.QRR);
    MulBt(TRR, U, s.QRV);
    MulBt(TRR, W, s.QRP);
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      s.QRV[i] += TRV[i];
      s.QRP[i] += tau * TRV[i] + TRP[i];
    }
  }
}

/** Appends one Euler step (sample `smp`: ω, a, Δt) with bias (bg, ba) to s. */
template <bool kJacobian>
__device__ __forceinline__ void Step(const float *smp, const float *bg, const float *ba,
                                     const ImuParameters &prm, Summary &s) {
  const float h = smp[6];
  const float th[3] = {(smp[0] - bg[0]) * h, (smp[1] - bg[1]) * h, (smp[2] - bg[2]) * h};
  const float a[3] = {smp[3] - ba[0], smp[4] - ba[1], smp[5] - ba[2]};
  // E = Exp(θ), J_r(θ) = J_l(-θ), sharing the trigonometric coefficients.
  float E[9], Jr[9];
  {
    const float n = norm3df(th[0], th[1], th[2]);
    float c1, c2, c3;
    lie_device::SO3Coefficients(n, &c1, &c2, &c3);
    lie_device::RodriguesMatrix(th, 1.f - c2 * n * n, c1, c2, 1.f, E, 3);
    const float mth[3] = {-th[0], -th[1], -th[2]};
    lie_device::RodriguesMatrix(mth, c1, c2, c3, 0.5f, Jr, 3);
  }
  const float half_h = 0.5f * h;
  // u = ΔR a; one step has Φ = [[Eᵀ, 0, 0], [U, I, 0], [½Δt U, Δt I, I]], U = -ΔR [a]x Δt.
  float u[3], U[9], W[9], F[9];
#pragma unroll
  for (int i = 0; i < 3; ++i)
    u[i] = s.R[i * 3] * a[0] + s.R[i * 3 + 1] * a[1] + s.R[i * 3 + 2] * a[2];
  MulSkew(s.R, a, -h, U);
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    W[i] = half_h * U[i];
    F[i] = E[(i % 3) * 3 + i / 3];
  }
  if constexpr (kJacobian) {
    // G ← Φ G + ∂(step)/∂b: gyro [-J_r Δt; 0; 0], accel [0; -ΔR Δt; -½ ΔR Δt²].
    float X[9];
    Mul(U, s.GR, X);  // W G_R = ½Δt U G_R
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      s.GPg[i] += h * s.GVg[i] + half_h * X[i];
      s.GVg[i] += X[i];
    }
    Mul(F, s.GR, X);
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      s.GR[i] = X[i] - h * Jr[i];
      s.GPa[i] += h * s.GVa[i] - half_h * h * s.R[i];
      s.GVa[i] -= h * s.R[i];
    }
  }
  Propagate<true>(F, U, W, h, s, half_h);
  // Step noise: rotation σ_g² Δt J_r J_rᵀ; velocity / position σ_a² Δt [[1, ½Δt], [½Δt, ¼Δt²]] ⊗ I
  // plus σ_i² Δt on the position (isotropic: the same in every frame).
  {
    float X[9];
    MulBt(Jr, Jr, X);
    const float qr = prm.gyro_noise_density * prm.gyro_noise_density * h;
    const float qv = prm.accel_noise_density * prm.accel_noise_density * h;
    const float qi = prm.integration_noise_density * prm.integration_noise_density * h;
#pragma unroll
    for (int i = 0; i < 9; ++i) s.QRR[i] += qr * X[i];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      s.QVV[i * 4] += qv;
      s.QVP[i * 4] += qv * half_h;
      s.QPP[i * 4] += qv * half_h * half_h + qi;
    }
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    s.p[i] += h * s.v[i] + half_h * h * u[i];
    s.v[i] += h * u[i];
  }
  float Rn[9];
  Mul(s.R, E, Rn);
#pragma unroll
  for (int i = 0; i < 9; ++i) s.R[i] = Rn[i];
  s.T += h;
}

/** a ← a then b (b in shared memory), in the frame of a's first state. */
template <bool kJacobian>
__device__ __forceinline__ void Combine(Summary &a, const Summary *b) {
  float RA[9], U[9], W[9], F[9], X[9], Y[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    RA[i] = a.R[i];
    F[i] = b->R[(i % 3) * 3 + i / 3];
  }
  const float tau = b->T;
  const float bv[3] = {b->v[0], b->v[1], b->v[2]}, bp[3] = {b->p[0], b->p[1], b->p[2]};
  MulSkew(RA, bv, -1.f, U);  // RA Φ_b,VR = -RA [Δv_b]x
  MulSkew(RA, bp, -1.f, W);
  if constexpr (kJacobian) {
    // G_ab = D Φ_b Dᵀ G_a + D G_b, D = diag(I, RA, RA).
    Mul(W, a.GR, X);
    Mul(RA, b->GPg, Y);
#pragma unroll
    for (int i = 0; i < 9; ++i) a.GPg[i] += X[i] + tau * a.GVg[i] + Y[i];
    Mul(U, a.GR, X);
    Mul(RA, b->GVg, Y);
#pragma unroll
    for (int i = 0; i < 9; ++i) a.GVg[i] += X[i] + Y[i];
    Mul(F, a.GR, X);
#pragma unroll
    for (int i = 0; i < 9; ++i) a.GR[i] = X[i] + b->GR[i];
    Mul(RA, b->GPa, Y);
#pragma unroll
    for (int i = 0; i < 9; ++i) a.GPa[i] += tau * a.GVa[i] + Y[i];
    Mul(RA, b->GVa, Y);
#pragma unroll
    for (int i = 0; i < 9; ++i) a.GVa[i] += Y[i];
  }
  Propagate(F, U, W, tau, a);
  // + D Q_b Dᵀ.
  MulBt(b->QRV, RA, X);
  MulBt(b->QRP, RA, Y);
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    a.QRR[i] += b->QRR[i];
    a.QRV[i] += X[i];
    a.QRP[i] += Y[i];
  }
  Rotate(RA, b->QVV, X);
  Rotate(RA, b->QVP, Y);
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    a.QVV[i] += X[i];
    a.QVP[i] += Y[i];
  }
  Rotate(RA, b->QPP, X);
#pragma unroll
  for (int i = 0; i < 9; ++i) a.QPP[i] += X[i];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    a.p[i] += tau * a.v[i] + (RA[i * 3] * bp[0] + RA[i * 3 + 1] * bp[1] + RA[i * 3 + 2] * bp[2]);
    a.v[i] += RA[i * 3] * bv[0] + RA[i * 3 + 1] * bv[1] + RA[i * 3 + 2] * bv[2];
  }
  float Rb[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) Rb[i] = b->R[i];
  Mul(RA, Rb, a.R);
  a.T += tau;
}

/**
 * Shared floats per warp: the output staging (32 rows), or the tree's
 * summaries (16 slots) when larger; the two are never live together.
 */
__host__ __device__ constexpr int WarpRegionFloats(bool jacobian, int group_size) {
  const int stage = 32 * kRowStride;
  const int tree = group_size > 1 ? 16 * (jacobian ? kSummaryFloats : kResidualSummaryFloats) : 0;
  return stage > tree ? stage : tree;
}

/**
 * Stages one row of n values per group leader of the warp and stores the rows
 * of the warp's items with coalesced writes: row `row` of item t goes to
 * dst + t * item_stride + row * n. items[g] is group g's item, -1 if none.
 */
__device__ __forceinline__ void WriteGroupRows(float *stage, const int *items, int lane, int groups,
                                               bool leader, int group, const float *values, int n,
                                               int row, float *dst, size_t item_stride) {
  if (leader) {
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
      if (c < n) stage[group * kRowStride + c] = values[c];
    }
  }
  __syncwarp();
  for (int k = lane; k < groups * n; k += 32) {
    const int g = k / n, c = k % n;
    if (items[g] >= 0) dst[items[g] * item_stride + row * n + c] = stage[g * kRowStride + c];
  }
  __syncwarp();
}

/**
 * One group of `group_size` lanes (a power of 2 up to 32) per item. Lane l
 * integrates its share of the item's samples into a Summary (frame of its
 * first state), the group combines the summaries in a binary tree through
 * shared memory, and the leader (lane 0) turns the total into the residual and
 * Jacobian: world frame, defect e of keyframe b against the prediction,
 * Σ = L Lᵀ, rows L⁻¹ D⁻¹ ∂e/∂x. Outputs are written row by row through a
 * per-warp staging buffer (coalesced stores). With group_size 1 this is one
 * sequential sweep per thread; larger groups cut the sequential depth from N
 * steps to N / group_size steps plus log2(group_size) combines, for batches
 * too small to fill the GPU.
 */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize, kJacobian ? 2 : 3)
    ImuKernel(const float *__restrict__ samples, const int *__restrict__ offsets, ImuParameters prm,
              float const *const *__restrict__ state_pointers, float *__restrict__ residuals,
              float *__restrict__ jacobians, int num_items, const int *__restrict__ factor_ids,
              int num_factors, int group_size) {
  // Per warp: the tree's summaries (16 slots) and, once the tree is done,
  // the output staging (32 rows) in the same region; then 32 item indices.
  extern __shared__ float smem[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int region = WarpRegionFloats(kJacobian, group_size);
  float *stage = smem + warp * region;
  float *tree = stage;
  int *items = reinterpret_cast<int *>(smem + (kBlockSize / 32) * region) + warp * 32;
  const int groups_per_block = kBlockSize / group_size;
  const int group_lane = threadIdx.x & (group_size - 1);
  const int group = threadIdx.x / group_size;  // in the block
  const int t = blockIdx.x * groups_per_block + group;
  const bool valid = t < num_items;
  const bool leader = group_lane == 0;
  const int warp_groups = 32 / group_size;
  const int group_in_warp = (threadIdx.x & 31) / group_size;
  if (leader) items[group_in_warp] = valid ? t : -1;
  // Lanes of items past the end read item 0's states and no samples (outputs not written).
  float const *const *s = state_pointers + 6 * static_cast<size_t>(valid ? t : 0);

  // The item's samples are split over its first `active` lanes, each taking
  // at least kMinSamplesPerLane: below that, combining costs more than it saves.
  int first_sample = 0, end_sample = 0, active = 1;
  if (valid) {
    const int f = FactorMeasurementIndex(t, factor_ids, num_factors);
    const int begin = __ldg(offsets + f), n = __ldg(offsets + f + 1) - begin;
    active = group_size;
    while (active > 1 && active * kMinSamplesPerLane > n) active >>= 1;
    if (group_lane < active) {
      first_sample = begin + (group_lane * n) / active;
      end_sample = begin + ((group_lane + 1) * n) / active;
    }
  }
  const float bg[3] = {__ldg(s[2] + 0), __ldg(s[2] + 1), __ldg(s[2] + 2)};
  const float ba[3] = {__ldg(s[2] + 3), __ldg(s[2] + 4), __ldg(s[2] + 5)};

  Summary sum;
  Identity(sum);
  // The next sample is loaded while the current one is integrated: the load
  // latency is otherwise on the critical path of the (sequential) chain.
  float next[kSampleSize];
  if (first_sample < end_sample) {
#pragma unroll
    for (int i = 0; i < kSampleSize; ++i)
      next[i] = __ldg(samples + static_cast<size_t>(kSampleSize) * first_sample + i);
  }
  for (int k = first_sample; k < end_sample; ++k) {
    float cur[kSampleSize];
#pragma unroll
    for (int i = 0; i < kSampleSize; ++i) cur[i] = next[i];
    if (k + 1 < end_sample) {
#pragma unroll
      for (int i = 0; i < kSampleSize; ++i)
        next[i] = __ldg(samples + static_cast<size_t>(kSampleSize) * (k + 1) + i);
    }
    Step<kJacobian>(cur, bg, ba, prm, sum);
  }

  // Binary tree over the group: at level `half`, lane l + half hands its summary to lane l.
  constexpr int kCopied = kJacobian ? kSummaryFloats : kResidualSummaryFloats;
  for (int half = 1; half < group_size; half <<= 1) {
    const int rel = group_lane & (2 * half - 1);
    // Slot of the receiver group_lane - rel: (receiver / 2) is unique per level.
    float *slot = tree + (group_in_warp * (group_size / 2) + (group_lane - rel) / 2) * kCopied;
    const bool pair = group_lane - rel + half < active;  // receiver and sender both active
    if (rel == half && pair) {
#pragma unroll
      for (int i = 0; i < kCopied; ++i) slot[i] = reinterpret_cast<const float *>(&sum)[i];
    }
    __syncwarp();
    if (rel == 0 && pair) Combine<kJacobian>(sum, reinterpret_cast<const Summary *>(slot));
    __syncwarp();
  }

  // --- Leader: world frame, defect, Σ = L Lᵀ, residual ---
  float res[kRows] = {}, Li[45] = {};
  // States are world_from_rig T = (R, t), the pose convention of cuNLS (as the
  // reprojection and PnP factors). The IMU in the world: R_imu = R R_bi, p_imu = R t_bi + t.
  const float tbi[3] = {prm.body_from_imu[3], prm.body_from_imu[7], prm.body_from_imu[11]};
  float Ra[9] = {}, Rb[9] = {}, R0[9] = {}, Rh[9] = {}, pa[3] = {}, pb[3] = {}, Tsum = 0.f,
        sigma_inv[6] = {};
  if (leader) {
    const float *Ta = s[0], *Tb = s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        Ra[i * 3 + j] = __ldg(Ta + i * 4 + j);
        Rb[i * 3 + j] = __ldg(Tb + i * 4 + j);
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      pa[i] = __ldg(Ta + i * 4 + 3) + Ra[i * 3] * tbi[0] + Ra[i * 3 + 1] * tbi[1] +
              Ra[i * 3 + 2] * tbi[2];
      pb[i] = __ldg(Tb + i * 4 + 3) + Rb[i * 3] * tbi[0] + Rb[i * 3 + 1] * tbi[1] +
              Rb[i * 3 + 2] * tbi[2];
    }
    MulRotation(Ra, prm.body_from_imu, R0);  // IMU rotation at keyframe a
    Mul(R0, sum.R, Rh);                      // R̂
    Tsum = sum.T;
    float e[9];
    {
      float RbI[9], Rt[9];
      MulRotation(Rb, prm.body_from_imu, RbI);
      MulAt(Rh, RbI, Rt);  // R̂ᵀ R_b,imu
      lie_device::LogSO3(Rt, 3, e);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      // v̂ - v_a = g T + R0 Δv; p̂ - p_a - v_a T = ½ g T² + R0 Δp.
      const float va = __ldg(s[1] + i), vb = __ldg(s[4] + i);
      const float dv = prm.gravity[i] * Tsum +
                       (R0[i * 3] * sum.v[0] + R0[i * 3 + 1] * sum.v[1] + R0[i * 3 + 2] * sum.v[2]);
      const float dp = 0.5f * prm.gravity[i] * Tsum * Tsum +
                       (R0[i * 3] * sum.p[0] + R0[i * 3 + 1] * sum.p[1] + R0[i * 3 + 2] * sum.p[2]);
      e[3 + i] = (vb - va) - dv;
      e[6 + i] = (pb[i] - pa[i]) - va * Tsum - dp;
    }

    // Σ in the world frame: C Q Cᵀ, C = diag(I, R0, R0); packed lower, then L and L⁻¹.
    float L[45];
    {
      float SRV[9], SRP[9], SVV[9], SVP[9], SPP[9];
      MulBt(sum.QRV, R0, SRV);
      MulBt(sum.QRP, R0, SRP);
      Rotate(R0, sum.QVV, SVV);
      Rotate(R0, sum.QVP, SVP);
      Rotate(R0, sum.QPP, SPP);
#pragma unroll
      for (int i = 0; i < 9; ++i)
#pragma unroll
        for (int j = 0; j <= i; ++j) {
          const int bi = i / 3, bj = j / 3, ii = i % 3, jj = j % 3;
          float v;
          if (bi == bj) {
            v = bi == 0 ? sum.QRR[ii * 3 + jj] : (bi == 1 ? SVV[ii * 3 + jj] : SPP[ii * 3 + jj]);
          } else if (bj == 0) {  // (V, R) or (P, R): Σ_RXᵀ
            v = bi == 1 ? SRV[jj * 3 + ii] : SRP[jj * 3 + ii];
          } else {  // (P, V)
            v = SVP[jj * 3 + ii];
          }
          L[Tri(i, j)] = v;
        }
    }
#pragma unroll
    for (int j = 0; j < 9; ++j) {
      float d = L[Tri(j, j)];
#pragma unroll
      for (int k = 0; k < j; ++k) d -= L[Tri(j, k)] * L[Tri(j, k)];
      d = sqrtf(fmaxf(d, kMinVariance));
      L[Tri(j, j)] = d;
      const float inv = 1.f / d;
#pragma unroll
      for (int i = j + 1; i < 9; ++i) {
        float v = L[Tri(i, j)];
#pragma unroll
        for (int k = 0; k < j; ++k) v -= L[Tri(i, k)] * L[Tri(j, k)];
        L[Tri(i, j)] = v * inv;
      }
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const float inv = 1.f / L[Tri(i, i)];
      Li[Tri(i, i)] = inv;
#pragma unroll
      for (int j = 0; j < i; ++j) {
        float v = 0.f;
#pragma unroll
        for (int k = j; k < i; ++k) v += L[Tri(i, k)] * Li[Tri(k, j)];
        Li[Tri(i, j)] = -v * inv;
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      // The floor applies to the variance, not to T: σ_b² times a tiny T underflows to 0.
      const float wg = prm.gyro_bias_random_walk, wa = prm.accel_bias_random_walk;
      sigma_inv[i] = rsqrtf(fmaxf(wg * wg * Tsum, kMinVariance));
      sigma_inv[3 + i] = rsqrtf(fmaxf(wa * wa * Tsum, kMinVariance));
    }
    // A chain without duration (no samples: offsets[f] == offsets[f + 1], or
    // only Δt <= 0) has zero covariance and carries no information: skip it
    // with all-zero rows (a zero whitening) rather than whitening by the floor,
    // which would give it weights near 1e18 and let it dominate the solve.
    if (!(Tsum > 0.f)) {
#pragma unroll
      for (int i = 0; i < 45; ++i) Li[i] = 0.f;
#pragma unroll
      for (int i = 0; i < 6; ++i) sigma_inv[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      float v = 0.f;
#pragma unroll
      for (int j = 0; j <= i; ++j) v += Li[Tri(i, j)] * e[j];
      res[i] = v;
    }
#pragma unroll
    for (int i = 0; i < 6; ++i) res[9 + i] = sigma_inv[i] * (__ldg(s[5] + i) - __ldg(s[2] + i));
  }
  WriteGroupRows(stage, items, lane, warp_groups, leader, group_in_warp, res, kRows, 0, residuals,
                 kRows);
  if constexpr (!kJacobian) return;

  // --- Jacobian rows L⁻¹ D⁻¹ ∂e/∂x (D = diag(J_l⁻¹(e_R), I, I)), one row at a time ---
  // Sensitivities of the prediction in the world frame: the IMU rotation at a, perturbed
  // R_imu Exp(φ_imu), changes e by -Ψ_φ φ_imu with Ψ_φ = [ΔRᵀ; -R0 [Δv]x; -R0 [Δp]x]; gyro
  // bias [G_R; R0 G_Vg; R0 G_Pg], accel [0; R0 G_Va; R0 G_Pa]. A state perturbation
  // T Exp([φ; ρ]) (rig frame) moves the IMU by φ_imu = R_biᵀ φ and
  // δp_imu = -R [t_bi]x φ + R ρ. Unwhitened rows by block:
  //   T_a: [-Ψ_φ R_biᵀ + [0; 0; R_a [t_bi]x] | [0; 0; -R_a]]   v_a: -[0; I; T I]
  //   b_a: -[G_g | G_a]   T_b: [[R̂ᵀ R_b; 0; -R_b [t_bi]x] | [0; 0; R_b]]   v_b: [0; I; 0]
  float PR[9] = {}, PV[9] = {}, PP[9] = {}, GVg[9] = {}, GPg[9] = {}, GVa[9] = {}, GPa[9] = {};
  float Sa[9] = {}, Sb[9] = {}, RhTRb[9] = {};
  if (leader) {
    float X[9], Y[9], BX[9];
    const float I[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    MulRotationT(prm.body_from_imu, I, BX);  // R_biᵀ
    MulAt(sum.R, BX, PR);                    // ΔRᵀ R_biᵀ
    MulSkew(R0, sum.v, -1.f, X);
    Mul(X, BX, PV);
    MulSkew(R0, sum.p, -1.f, Y);
    Mul(Y, BX, PP);
    Mul(R0, sum.GVg, GVg);
    Mul(R0, sum.GPg, GPg);
    Mul(R0, sum.GVa, GVa);
    Mul(R0, sum.GPa, GPa);
    MulSkew(Ra, tbi, 1.f, Sa);  // R_a [t_bi]x
    MulSkew(Rb, tbi, 1.f, Sb);
    MulAt(Rh, Rb, RhTRb);  // R̂ᵀ R_b
  }
  float row[kColumns];
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    if (leader) {
      float l[9];
#pragma unroll
      for (int j = 0; j < 9; ++j) l[j] = j <= i ? Li[Tri(i, j)] : 0.f;
      const float *lR = l, *lV = l + 3, *lP = l + 6;
      float x[3], y[3], z[3], w[3], u[3];
      // T_a
      RowMul(lR, PR, x);
      RowMul(lV, PV, y);
      RowMul(lP, PP, z);
      RowMul(lP, Sa, w);
      RowMul(lP, Ra, u);
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        row[j] = w[j] - (x[j] + y[j] + z[j]);
        row[3 + j] = -u[j];
        row[6 + j] = -(lV[j] + Tsum * lP[j]);  // v_a
      }
      // b_a: gyro, accel
      RowMul(lR, sum.GR, x);
      RowMul(lV, GVg, y);
      RowMul(lP, GPg, z);
#pragma unroll
      for (int j = 0; j < 3; ++j) row[9 + j] = -(x[j] + y[j] + z[j]);
      RowMul(lV, GVa, x);
      RowMul(lP, GPa, y);
#pragma unroll
      for (int j = 0; j < 3; ++j) row[12 + j] = -(x[j] + y[j]);
      // T_b, v_b, b_b
      RowMul(lR, RhTRb, x);
      RowMul(lP, Sb, y);
      RowMul(lP, Rb, u);
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        row[15 + j] = x[j] - y[j];
        row[18 + j] = u[j];
        row[21 + j] = lV[j];
        row[24 + j] = 0.f;
        row[27 + j] = 0.f;
      }
    }
    WriteGroupRows(stage, items, lane, warp_groups, leader, group_in_warp, row, kColumns, i,
                   jacobians, kRows * kColumns);
  }
  // Bias random walk rows.
#pragma unroll
  for (int i = 0; i < 6; ++i) {
#pragma unroll
    for (int j = 0; j < kColumns; ++j) row[j] = 0.f;
    row[9 + i] = -sigma_inv[i];
    row[24 + i] = sigma_inv[i];
    WriteGroupRows(stage, items, lane, warp_groups, leader, group_in_warp, row, kColumns, 9 + i,
                   jacobians, kRows * kColumns);
  }
}

/** Dynamic shared memory of a launch with `group_size` lanes per item. */
size_t SharedBytes(bool jacobian, int group_size) {
  const int warps = kBlockSize / 32;
  return (static_cast<size_t>(warps) * WarpRegionFloats(jacobian, group_size) + kBlockSize) *
         sizeof(float);
}

/**
 * Lanes per factor: enough to fill the device when the batch alone cannot
 * (one sequential chain per thread leaves most SMs idle below ~10^4 factors),
 * with at least kMinSamplesPerLane samples per lane for the typical chain. A function of
 * the batch size and the buffers only, not of the items of one call, so that
 * item evaluations match plain ones bitwise.
 */
int GroupSize(size_t num_factors, size_t samples_per_factor) {
  int device = 0, sms = 1;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const size_t target = static_cast<size_t>(sms) * kTargetThreadsPerSM;
  int group_size = 1;
  while (group_size < kMaxGroupSize && num_factors * group_size * 2 <= target &&
         static_cast<size_t>(group_size) * 2 * kMinSamplesPerLane <= samples_per_factor) {
    group_size *= 2;
  }
  return group_size;
}

}  // namespace

ImuFactorBatch::ImuFactorBatch(const float *imu_samples, const int *sample_offsets,
                               size_t num_samples, const ImuParameters &parameters, size_t capacity)
    : SizedFactorBatch(capacity),
      imu_samples_(imu_samples),
      sample_offsets_(sample_offsets),
      num_samples_(num_samples),
      parameters_(parameters) {
  if (imu_samples_ == nullptr || sample_offsets_ == nullptr) {
    throw std::invalid_argument("ImuFactorBatch: imu_samples and sample_offsets must not be null");
  }
  const ImuParameters &p = parameters_;
  if (!(p.gyro_noise_density > 0.f) || !(p.accel_noise_density > 0.f) ||
      !(p.integration_noise_density > 0.f) || !(p.gyro_bias_random_walk > 0.f) ||
      !(p.accel_bias_random_walk > 0.f)) {
    throw std::invalid_argument("ImuFactorBatch: noise densities must be positive");
  }
}

bool ImuFactorBatch::Evaluate(float *residuals, float *jacobians,
                              float const *const *state_pointers, cudaStream_t stream,
                              const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (num_items == 0 || num_factors == 0) return true;
  const int group_size = GroupSize(num_factors, num_samples_ / std::max<size_t>(1, Capacity()));
  const size_t groups_per_block = kBlockSize / group_size;
  const unsigned blocks =
      static_cast<unsigned>((num_items + groups_per_block - 1) / groups_per_block);
  if (jacobians != nullptr) {
    ImuKernel<true><<<blocks, kBlockSize, SharedBytes(true, group_size), stream>>>(
        imu_samples_, sample_offsets_, parameters_, state_pointers, residuals, jacobians,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors), group_size);
  } else {
    ImuKernel<false><<<blocks, kBlockSize, SharedBytes(false, group_size), stream>>>(
        imu_samples_, sample_offsets_, parameters_, state_pointers, residuals, nullptr,
        static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors), group_size);
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
