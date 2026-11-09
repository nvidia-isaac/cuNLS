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

// y = H x for a damped bundle-adjustment Gauss-Newton matrix H, stored as full symmetric scalar CSR
// or as 3x3 block storage (BSR), with n_c = 6 * (number of cameras) pose rows followed by 3 rows
// per point.
//
// Only the point rows are read in full. Point p (k = pair count) owns 3 rows of length L = 6k + 3
// whose 3L values are contiguous at pbase + 3 * cs(p), cs(p) = 6 * pair_ptr[p] + 3p. Row t holds
// W_j[t][0..5] for each pair j (H column 6 * body_j + u), then the point's 3x3 block V[t][0..2]. In
// block storage the same 3L values are the point's 2k + 1 row-major tiles. The point has k + 1
// "units" (its k pairs and its V block).
//
// One lane handles one unit and reads its columns of all 3 rows (18 values):
// clang-format off
//   y_c[6b + u] += sum_t W[t][u] * x_p[t]      (W^T x_p: the pose rows' transposed copy of W is
//                                               never read)
//   y_p[t]      += sum_u W[t][u] * x[6b + u]   (or V x_p for the V unit), one segmented warp
//                                               reduction per 32 units, pulled into the lane that
//                                               owns the point.
// clang-format on
// A warp processes a window of consecutive points at a time (lane j: point q + j). Windows come
// from a global queue; y_c is accumulated exactly in fixed point in one shared table per block (see
// hvp_dyn), so the result is bit-identical run to run for any window-to-warp assignment. One
// cooperative launch: a grid barrier separates the point pass from the pose-row finish (block b
// finishes pose rows [b n_c / G, (b + 1) n_c / G), adding the pose-pose prefix).
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <climits>
#include <string>

#include "cunls/common/helper.h"
#include "cunls/linear_solver/ba_hessian_vector_product.h"

namespace cunls {
#define FULL 0xffffffffu
namespace {
constexpr int kSmemBudget = 97 * 1024;
constexpr int kMaxSlice = 256;
/** @brief Acquire load at GPU scope (grid barrier). */
__device__ __forceinline__ unsigned long long ld_acquire64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
/** @brief Relaxed load at GPU scope. */
__device__ __forceinline__ unsigned long long ld_relaxed64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
/** @brief Store kept in L2 (evict-last) so it is not written back mid-kernel (sm_80+). */
__device__ __forceinline__ void st_keep(float* p, float v) {
#if __CUDA_ARCH__ >= 800
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("st.global.L2::cache_hint.f32 [%0], %1, %2;" ::"l"(p), "f"(v), "l"(pol) : "memory");
#else
  *p = v;
#endif
}
/**
 * @brief Offset of value (row t, column u) of a unit from the unit's first value: row-major rows of
 * length L in scalar CSR, two row-major 3x3 tiles in 3x3 block storage.
 */
template <bool kTiles>
__device__ __forceinline__ int unit_value(int t, int u, int L) {
  return kTiles ? 9 * (u / 3) + 3 * t + u % 3 : t * L + u;
}
/** @brief A warp's window of consecutive points (lane j: point q + j). */
struct Window {
  int q, npts, pp, pn, up, upn, uend;
  float x0, x1, x2;
};
/** @brief One lane's unit: its 3x6 values and pose. */
struct Step {
  int g0, b, jl;
  float m0, m1, m2;
  float w[3][6];
};
/** @brief Window queue: heavy windows, then main windows, then light windows. */
struct Sched {
  int P2;
  int nh, nm, nwin;
};
constexpr int kHeavyK = 16;
// Window size and warps per SM set how many value bytes are being read at once and how far apart:
// DRAM serves this op best when the windows in flight span a narrow address range (8 warps per SM
// x 5 points here, vs 7 x 8), while each window still carries enough units to amortize its claim
// and metadata round trips.
constexpr int kMainWin = 5;
// The light windows that end the queue stay at 8 points: there a window's claim and metadata round
// trips are not hidden behind other warps' streaming, so fewer, larger windows finish the drain
// sooner.
constexpr int kLightWin = 8;
/** @brief Points [x, y) and class z (0 main, 1 heavy, 2 light) of window i of the queue. */
__device__ __forceinline__ int3 window_range(const Sched& sc, int i, int P) {
  if (i < sc.nh) return make_int3(sc.P2 + 32 * i, min(sc.P2 + 32 * i + 32, P), 1);
  i -= sc.nh;
  if (i < sc.nm) return make_int3(kMainWin * i, min(kMainWin * i + kMainWin, sc.P2), 0);
  i -= sc.nm;
  return make_int3(sc.P2 + kLightWin * i, min(sc.P2 + kLightWin * i + kLightWin, P), 2);
}
constexpr int kDynWarps = 8;
constexpr int kDynUS = 4;
// The largest sampled W^T x_p term maps to [2^40, 2^41) fixed-point units: terms up to 8-16x larger
// stay on the chunk path, and a row needs the scalar recompute below only when its total is below
// 2^-17 of that term.
constexpr int kSampleScaleLog2 = 40;
// Fixed point: v = rint(c * 2^S). |v| < 2^44 goes straight into the three signed 15-bit chunk
// tables. 2^44 <= |v| < 2^76 (rare) is split exactly as v = hi * 2^32 + lo, |lo| <= 2^31: lo takes
// the chunk path, hi (|hi| <= 2^44) is added to a second global int64 accumulator. A pose entry has
// at most P < 2^19 contributions, so neither int64 accumulator can wrap. Only |v| >= 2^76 or
// non-finite values take the CSR fallback.
constexpr float kFixedMax = 17592186044416.f;  // 2^44
constexpr float kSpillMax = 0x1p76f;
constexpr float kMagic = 12582912.f;
constexpr int kMagicBits = 0x4B400000;
constexpr int kMaxBlockPoints = (1 << 17) - 64;
// y_p is staged per warp in shared memory (point index + 3 values, up to kStageCap points) and
// written once the warp runs out of windows, so the value stream stays read-only: y_p stores issued
// during the stream leave partially written lines in L2, some of which are written back to DRAM
// mid-kernel.
constexpr int kStageCap = 384;
/**
 * @brief y = H x in one cooperative launch (see the file comment); kTiles selects 3x3 block
 *        storage over scalar CSR.
 */
template <int NWARPS, int US, bool kTiles>
__global__ void __launch_bounds__(NWARPS * 32, 1)
    hvp_dyn(const float* __restrict__ values, const int* __restrict__ pair_body,
            const int* __restrict__ pair_ptr, const int* __restrict__ row_ptr,
            const int* __restrict__ col_ids, const int* __restrict__ cc_row_end,
            const float* __restrict__ x, float* __restrict__ y,
            unsigned long long* __restrict__ gacc, int n_c, int P, long long pbase, Sched sc,
            int scap) {
  extern __shared__ __align__(16) unsigned sfix[];
  __shared__ int s_npts;
  __shared__ int s_scale[2];
  __shared__ float s_pre[kMaxSlice];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = blockDim.x >> 5;
  // gacc: [0, n_c) fixed-point sums, [n_c, 2 n_c) spill sums, then this instance's window counter,
  // grid barrier and overflow epoch.
  unsigned long long* __restrict__ gsync = gacc + 2 * n_c;
  const unsigned long long bar0 = ld_relaxed64(gsync + 1);
  const float* __restrict__ vb = values + pbase;
  const float* __restrict__ xpt = x + n_c;
  float* __restrict__ ypt = y + n_c;
  unsigned long long* __restrict__ ghi = gacc + n_c;
  const int gw = blockIdx.x * nwarps + warp;
  const int NWT = gridDim.x * nwarps;
  int cur = gw;
  int3 cr = cur < sc.nwin ? window_range(sc, cur, P) : make_int3(0, 0, 0);
  int w_pp = 0, w_pn = 0;
  float w_x0 = 0.f, w_x1 = 0.f, w_x2 = 0.f;
  /** @brief Loads lane j's point q + j (pair range and x_p) of window [q, qe). */
  auto load_window = [&](int q, int qe) {
    if (q + lane < qe) {
      const int p = q + lane;
      w_pp = __ldg(pair_ptr + p);
      w_pn = __ldg(pair_ptr + p + 1);
      w_x0 = __ldg(xpt + 3 * p);
      w_x1 = __ldg(xpt + 3 * p + 1);
      w_x2 = __ldg(xpt + 3 * p + 2);
    }
  };
  if (cur < sc.nwin) load_window(cr.x, cr.y);
  // Scale sample: the first pair of 64 points spread evenly over all P points.
  int sm_pp[2] = {0, 0}, sm_pn[2] = {0, 0};
  float sm_x[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
  if (warp == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int p = static_cast<int>((static_cast<long long>(2 * (lane + 32 * h) + 1) * P) >> 7);
      sm_pp[h] = __ldg(pair_ptr + p);
      sm_pn[h] = __ldg(pair_ptr + p + 1);
#pragma unroll
      for (int t = 0; t < 3; ++t) sm_x[h][t] = __ldg(xpt + 3 * p + t);
    }
  }
  const int nb = n_c / 6;
  float4* __restrict__ s_stage =
      reinterpret_cast<float4*>(reinterpret_cast<char*>(sfix) + ((12 * n_c + 15) & ~15)) +
      warp * scap;
  int st_cnt = 0;
  if (threadIdx.x == 0) s_npts = 0, s_scale[1] = 0;
  {
    uint2* s2 = reinterpret_cast<uint2*>(sfix);
    for (int i = threadIdx.x; i < 3 * n_c / 2; i += blockDim.x) s2[i] = make_uint2(0u, 0u);
  }
  __syncthreads();
  const int G = gridDim.x;
  const int r0 = static_cast<int>(static_cast<long long>(blockIdx.x) * n_c / G);
  const int r1 = static_cast<int>(static_cast<long long>(blockIdx.x + 1) * n_c / G);
  /** @brief Pose-pose part of pose row i (its leading pose columns) times x. */
  auto prefix = [&](int i) {
    if constexpr (kTiles) {
      // Pose row i is row i % 3 of block row i / 3; its pose-pose tiles come first.
      const float* __restrict__ vr = values + i % 3 * 3;
      float s = 0.f;
      for (int e = __ldg(row_ptr + i / 3), e1 = __ldg(cc_row_end + i / 3); e < e1; ++e) {
        const float* __restrict__ xe = x + 3 * __ldg(col_ids + e);
#pragma unroll
        for (int l = 0; l < 3; ++l) s = fmaf(__ldg(vr + 9LL * e + l), __ldg(xe + l), s);
      }
      return s;
    }
    const int e0 = __ldg(row_ptr + i), e1 = __ldg(cc_row_end + i);
    // The six rows of body b hold the same 6x6 pose-pose blocks, so their prefixes share one column
    // list: read the col_ids of row 6b (the lanes of one body then request the same sectors) unless
    // the prefix lengths differ.
    const int i0 = i - i % 6;
    const int f0 = __ldg(row_ptr + i0), f1 = __ldg(cc_row_end + i0);
    const int* __restrict__ cp = (f1 - f0 == e1 - e0 ? col_ids + f0 : col_ids + e0) - e0;
    float s = 0.f;
    for (int e = e0; e < e1; e += 18) {
      int c[18];
      float v[18];
#pragma unroll
      for (int j = 0; j < 18; ++j) {
        c[j] = e + j < e1 ? __ldg(cp + e + j) : -1;
        v[j] = e + j < e1 ? __ldg(values + e + j) : 0.f;
      }
#pragma unroll
      for (int j = 0; j < 18; ++j)
        if (c[j] >= 0) s = fmaf(v[j], __ldg(x + c[j]), s);
    }
    return s;
  };
  int S = kSampleScaleLog2;
  float scale = 0.f;
  bool have_scale = false;
  /** @brief Warp 0: sets the call's fixed-point scale 2^S from the sampled W^T x_p terms. */
  auto get_scale = [&]() {
    float m = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int p = static_cast<int>((static_cast<long long>(2 * (lane + 32 * h) + 1) * P) >> 7);
      const int pp = sm_pp[h], k = sm_pn[h] - pp;
      if (k > 0) {
        const long long a = 18LL * pp + 9LL * p;
        const int L = 6 * k + 3;
        const float m0 = sm_x[h][0], m1 = sm_x[h][1], m2 = sm_x[h][2];
#pragma unroll
        for (int u = 0; u < 6; ++u) {
          const float c = fmaf(__ldg(vb + a + unit_value<kTiles>(2, u, L)), m2,
                               fmaf(__ldg(vb + a + unit_value<kTiles>(1, u, L)), m1,
                                    __ldg(vb + a + unit_value<kTiles>(0, u, L)) * m0));
          m = fmaxf(m, fabsf(c));
        }
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
    if (m > 0.f && m <= 3.0e38f) S = kSampleScaleLog2 - ilogbf(m);
    S = min(max(S, -126), 127);
    scale = ldexpf(1.f, S);
    have_scale = true;
  };
  if (warp == 0) {
    get_scale();
    if (lane == 0) {
      atomicExch(&s_scale[0], S);
      __threadfence_block();
      atomicExch(&s_scale[1], 1);
    }
    for (int i = r0 + lane; i < min(r1, r0 + kMaxSlice); i += 32) s_pre[i - r0] = prefix(i);
  }
  /** @brief Waits for warp 0 to publish S and loads the scale. */
  auto wait_scale = [&]() {
    while (atomicAdd(&s_scale[1], 0) == 0) {
    }
    __threadfence_block();
    S = atomicAdd(&s_scale[0], 0);
    scale = ldexpf(1.f, S);
    have_scale = true;
  };
  bool ovf = false;
  /**
   * @brief Window [q, qe) with unit offsets; filt keeps all (0), heavy (1) or light (2) points,
   *        incl tells whether the lane's point is kept.
   */
  auto make_window = [&](int q, int qe, int filt, bool& incl) {
    Window W;
    W.q = q;
    W.npts = qe - q;
    const bool pl = lane < W.npts;
    W.pp = w_pp;
    W.pn = w_pn;
    W.x0 = w_x0;
    W.x1 = w_x1;
    W.x2 = w_x2;
    const int k = W.pn - W.pp;
    incl = pl && (filt == 0 || (filt == 1 ? k >= kHeavyK : k < kHeavyK));
    int us = incl ? k + 1 : 0;
    // Inclusive scan of unit counts; main and light windows have at most 8 points, so 3 levels
    // cover them.
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const int t = __shfl_up_sync(FULL, us, d);
      if (lane >= d) us += t;
    }
    if (W.npts > 8) {
#pragma unroll
      for (int d = 8; d < 32; d <<= 1) {
        const int t = __shfl_up_sync(FULL, us, d);
        if (lane >= d) us += t;
      }
    }
    W.upn = pl ? us : INT_MAX;
    W.up = pl ? us - (incl ? k + 1 : 0) : INT_MAX;
    W.uend = __shfl_sync(FULL, us, W.npts - 1);
    return W;
  };
  /** @brief Loads the 3x6 values and body of unit g0 + lane of window W. */
  auto fetch = [&](const Window& W, int g0) {
    Step S_;
    S_.g0 = g0;
    const int g = g0 + lane;
    // Point of unit g: binary search over the window's unit offsets (lanes >= npts hold INT_MAX).
    int idx = 0;
    if (W.npts > 8) {
      if (__shfl_sync(FULL, W.up, 16) <= g) idx = 16;
      if (__shfl_sync(FULL, W.up, idx + 8) <= g) idx += 8;
    }
#pragma unroll
    for (int s = 4; s >= 1; s >>= 1)
      if (__shfl_sync(FULL, W.up, idx + s) <= g) idx += s;
    const int mpp = __shfl_sync(FULL, W.pp, idx);
    const int mpn = __shfl_sync(FULL, W.pn, idx);
    const int vst = __shfl_sync(FULL, W.up, idx);
    S_.m0 = __shfl_sync(FULL, W.x0, idx);
    S_.m1 = __shfl_sync(FULL, W.x1, idx);
    S_.m2 = __shfl_sync(FULL, W.x2, idx);
    const int k = mpn - mpp;
    S_.jl = g - vst;
    const bool isV = S_.jl == k;
    const int L = 6 * k + 3;
    const int A0 = 3 * (6 * mpp + 3 * (W.q + idx)) + (kTiles ? 18 : 6) * S_.jl;
    S_.b = -1;
#pragma unroll
    for (int t = 0; t < 3; ++t)
#pragma unroll
      for (int u = 0; u < 6; ++u) S_.w[t][u] = 0.f;
    if (g < W.uend) {
#pragma unroll
      for (int t = 0; t < 3; ++t)
#pragma unroll
        for (int u = 0; u < 6; ++u)
          if (u < 3 || !isV) S_.w[t][u] = __ldg(vb + A0 + unit_value<kTiles>(t, u, L));
      if (!isV) S_.b = __ldg(pair_body + mpp + S_.jl);
    }
    return S_;
  };
  /**
   * @brief Adds a unit's W^T x_p to the fixed-point pose table and its W x_c (or V x_p) to the
   *        owning point's acc0..acc2.
   */
  auto process = [&](const Step& St, const Window& W, float& acc0, float& acc1, float& acc2) {
    float xc[6];
    if (St.b >= 0) {
      const float2* xb = reinterpret_cast<const float2*>(x + 6 * St.b);
      const float2 v0 = __ldg(xb), v1 = __ldg(xb + 1), v2 = __ldg(xb + 2);
      xc[0] = v0.x;
      xc[1] = v0.y;
      xc[2] = v1.x;
      xc[3] = v1.y;
      xc[4] = v2.x;
      xc[5] = v2.y;
      unsigned* t0 = sfix + St.b;
#pragma unroll
      for (int u = 0; u < 6; ++u) {
        const float c = fmaf(St.w[2][u], St.m2, fmaf(St.w[1][u], St.m1, St.w[0][u] * St.m0));
        float f = rintf(c * scale);
        if (!(fabsf(f) < kFixedMax)) {
          if (fabsf(f) < kSpillMax) {
            const float fh = rintf(f * 0x1p-32f);
            f = fmaf(-fh, 0x1p32f, f);
            atomicAdd(ghi + u * nb + St.b, static_cast<unsigned long long>(__float2ll_rn(fh)));
          } else {
            ovf = true;
          }
        }
        const float h2 = fmaf(f, 0x1p-30f, kMagic);
        const float q2 = h2 - kMagic;
        const float r1 = fmaf(-q2, 0x1p30f, f);
        const float h1 = fmaf(r1, 0x1p-15f, kMagic);
        const float q1 = h1 - kMagic;
        const float r0 = fmaf(-q1, 0x1p15f, r1);
        const float h0 = r0 + kMagic;
        atomicAdd(t0 + u * nb, static_cast<unsigned>(__float_as_int(h0) - kMagicBits));
        atomicAdd(t0 + (6 + u) * nb, static_cast<unsigned>(__float_as_int(h1) - kMagicBits));
        atomicAdd(t0 + (12 + u) * nb, static_cast<unsigned>(__float_as_int(h2) - kMagicBits));
      }
    } else {
      xc[0] = St.m0;
      xc[1] = St.m1;
      xc[2] = St.m2;
      xc[3] = 0.f;
      xc[4] = 0.f;
      xc[5] = 0.f;
    }
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
#pragma unroll
    for (int u = 0; u < 6; ++u) {
      a0 = fmaf(St.w[0][u], xc[u], a0);
      a1 = fmaf(St.w[1][u], xc[u], a1);
      a2 = fmaf(St.w[2][u], xc[u], a2);
    }
    const unsigned heads = __ballot_sync(FULL, lane == 0 || St.jl == 0);
    const unsigned above = heads & ~((2u << lane) - 1u);
    const int e = above ? (__ffs(above) - 2) : 31;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float t0 = __shfl_down_sync(FULL, a0, d);
      const float t1 = __shfl_down_sync(FULL, a1, d);
      const float t2 = __shfl_down_sync(FULL, a2, d);
      if (lane + d <= e) {
        a0 += t0;
        a1 += t1;
        a2 += t2;
      }
    }
    const bool present = W.up < St.g0 + 32 && W.upn > St.g0;
    const int src = present ? max(W.up - St.g0, 0) : 0;
    const float s0 = __shfl_sync(FULL, a0, src);
    const float s1 = __shfl_sync(FULL, a1, src);
    const float s2 = __shfl_sync(FULL, a2, src);
    if (present) {
      acc0 += s0;
      acc1 += s1;
      acc2 += s2;
    }
  };
  while (cur < sc.nwin) {
    bool incl;
    const Window W = make_window(cr.x, cr.y, cr.z, incl);
    if (lane == 0 && atomicAdd(&s_npts, W.npts) + W.npts > kMaxBlockPoints) ovf = true;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    int nn = sc.nwin;
    for (int g0 = 0; g0 < W.uend; g0 += 32 * US) {
      Step St[US];
#pragma unroll
      for (int m = 0; m < US; ++m) St[m] = fetch(W, g0 + 32 * m);
      if (!have_scale) wait_scale();
#pragma unroll
      for (int m = 0; m < US; ++m)
        if (g0 + 32 * m < W.uend) process(St[m], W, acc0, acc1, acc2);
    }
    if (lane == 0) nn = NWT + static_cast<int>(atomicAdd(gsync, 1ull));
    {
      // The window claimed about one round from now: pull its pair_ptr / x_p lines into L2, so its
      // metadata load is an L2 hit instead of a DRAM round trip on that warp's critical path.
      const int ahead = __shfl_sync(FULL, nn, 0) + NWT / 2;
      if (ahead < sc.nwin && lane < 3) {
        const int q = window_range(sc, ahead, P).x;
        if (lane == 0)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(pair_ptr + q));
        else if (3 * q + 32 * (lane - 1) < 3 * P)
          asm volatile("prefetch.global.L2 [%0];" ::"l"(xpt + 3 * q + 32 * (lane - 1)));
      }
    }
    {
      const unsigned im = __ballot_sync(FULL, incl);
      const int nin = __popc(im);
      if (st_cnt + nin <= scap) {
        if (incl)
          s_stage[st_cnt + __popc(im & ((1u << lane) - 1u))] =
              make_float4(__int_as_float(W.q + lane), acc0, acc1, acc2);
        st_cnt += nin;
      } else if (incl) {
        float* yp = ypt + 3 * (W.q + lane);
        st_keep(yp, acc0);
        st_keep(yp + 1, acc1);
        st_keep(yp + 2, acc2);
      }
    }
    cur = __shfl_sync(FULL, nn, 0);
    if (cur < sc.nwin) {
      cr = window_range(sc, cur, P);
      load_window(cr.x, cr.y);
    }
  }
  const int nc = 6 * nb;
  {
    // Out of windows: write this warp's staged y_p (windows past the staging capacity were stored
    // directly). Component-major: consecutive lanes store consecutive floats of consecutive staged
    // points (contiguous within a window), so each store instruction covers ~128 contiguous bytes
    // instead of three strided columns.
    __syncwarp();
    for (int k = lane; k < 3 * st_cnt; k += 32) {
      const int e = k / 3, c = k - 3 * e;
      const float4 v = s_stage[e];
      ypt[3 * __float_as_int(v.x) + c] = c == 0 ? v.y : (c == 1 ? v.z : v.w);
    }
  }
  if (!have_scale) wait_scale();
  const unsigned long long epoch = bar0 / gridDim.x;
  if (__any_sync(FULL, ovf) && lane == 0) atomicMax(gsync + 2, epoch + 1);
  __syncthreads();
  {
#pragma unroll 4
    for (int j = threadIdx.x; j < nc; j += blockDim.x) {
      const long long v = static_cast<long long>(static_cast<int>(sfix[j])) +
                          static_cast<long long>(static_cast<int>(sfix[nc + j])) * (1LL << 15) +
                          static_cast<long long>(static_cast<int>(sfix[2 * nc + j])) * (1LL << 30);
      if (v != 0)
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(gacc + j), "l"(v) : "memory");
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(gsync + 1) : "memory");
    const unsigned long long target = (epoch + 1) * gridDim.x;
    while (ld_acquire64(gsync + 1) < target) {
    }
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) gsync[0] = 0ull;
  const unsigned long long ov = ld_relaxed64(gsync + 2);
  const double rscale = __longlong_as_double(static_cast<long long>(1023 - S) << 52);
  // Rows below 2^23 fixed-point units (including zero rows, whose terms may all have rounded to
  // zero), where rounding each term to the grid (half a unit) could exceed float's worst-case error
  // for the row, and every row after an overflow, are recomputed with the scalar product.
  const float small = ldexpf(1.f, 23 - S);
  bool redo = ov == epoch + 1;
  for (int i = r0 + threadIdx.x; i < r1; i += blockDim.x) {
    const int b = i / 6, j = (i - 6 * b) * nb + b;
    const long long s = static_cast<long long>(__ldcg(gacc + j));
    const long long sh = static_cast<long long>(__ldcg(ghi + j));
    gacc[j] = 0ull;
    double t = static_cast<double>(s);
    if (sh != 0) {
      ghi[j] = 0ull;
      t = fma(static_cast<double>(sh), 0x1p32, t);
    }
    const float pre = i - r0 < kMaxSlice ? s_pre[i - r0] : prefix(i);
    const float v = pre + static_cast<float>(t * rscale);
    y[i] = v;
    redo = redo || !(fabsf(v) >= small);
  }
  if (__syncthreads_or(redo)) {
    for (int i = r0 + warp; i < r1; i += nwarps) {
      if (ov != epoch + 1 && fabsf(y[i]) >= small) continue;
      float s = 0.f;
      if constexpr (kTiles) {
        const int e1 = __ldg(row_ptr + i / 3 + 1);
        for (int e = __ldg(row_ptr + i / 3) + lane; e < e1; e += 32) {
          const float* __restrict__ ve = values + 9LL * e + i % 3 * 3;
          const float* __restrict__ xe = x + 3 * __ldg(col_ids + e);
#pragma unroll
          for (int l = 0; l < 3; ++l) s = fmaf(__ldg(ve + l), __ldg(xe + l), s);
        }
      } else {
        const int e0 = __ldg(row_ptr + i), e1 = __ldg(row_ptr + i + 1);
        for (int e = e0 + lane; e < e1; e += 32)
          s = fmaf(__ldg(values + e), __ldg(x + __ldg(col_ids + e)), s);
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(FULL, s, o);
      if (lane == 0) y[i] = s;
    }
  }
}
/** @brief Most cameras whose pose table (3 x 4 bytes per pose row) fits in kSmemBudget. */
constexpr int kMaxCameras = kSmemBudget / 72;
/** @brief Most points: per-row point counts stay below 2^19 so the int64 sums cannot wrap. */
constexpr int kMaxPoints = (1 << 19) - 1;

/** @brief Dynamic shared memory of hvp_dyn for n_c pose rows; *scap = staged points per warp. */
template <int NWARPS>
size_t dyn_smem(int n_c, int* scap) {
  const int stage_off = (12 * n_c + 15) & ~15;
  *scap = std::max(0, std::min(kStageCap, (kSmemBudget - stage_off) / (16 * NWARPS)));
  return static_cast<size_t>(stage_off) + static_cast<size_t>(16) * NWARPS * *scap;
}

/** @brief Launches hvp_dyn on one cooperative wave of @p num_sms blocks. */
template <int NWARPS, int US, bool kTiles>
cudaError_t launch_dyn(const float* values, const int* pair_body, const int* pair_ptr,
                       const int* row_ptr, const int* col_ids, const int* cc_row_end,
                       const float* x, float* y, unsigned long long* gacc, int n_c, int P,
                       long long pbase, int num_sms, cudaStream_t stream) {
  const int G = num_sms, NWT = G * NWARPS;
  Sched sc;
  // The last T points are queued heavy-first (start of the queue) and light-last (end): ~2 light
  // windows per warp at the end of the queue absorb the cost variance of the main windows, so
  // blocks reach the grid barrier together.
  const int T = min(P, 16 * NWT);
  sc.P2 = P - T;
  sc.nh = (T + 31) / 32;
  sc.nm = (sc.P2 + kMainWin - 1) / kMainWin;
  sc.nwin = sc.nh + sc.nm + (T + kLightWin - 1) / kLightWin;
  int scap = 0;
  const size_t smem = dyn_smem<NWARPS>(n_c, &scap);
  void* args[] = {&values, &pair_body, &pair_ptr, &row_ptr, &col_ids, &cc_row_end, &x,
                  &y,      &gacc,      &n_c,      &P,       &pbase,   &sc,         &scap};
  return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(hvp_dyn<NWARPS, US, kTiles>), dim3(G),
                                     dim3(NWARPS * 32), args, smem, stream);
}

// Pattern checks, in units of the storage block (bs = 1: scalars, bs = 3: 3x3 tiles). A camera
// spans cb = 6 / bs block columns and a point pb = 3 / bs. check[0] counts violations, check[1] the
// point-column entries of the pose rows (must be cb * pb per pair: the transposed copies of the
// point rows' pose blocks).

/** @brief pair_count[p] = number of camera blocks in point p's rows; flags rows of unequal or
 * invalid length. */
__global__ void BaPointPairCountKernel(const int* __restrict__ row_off, int pose_rows, int bs,
                                       int num_points, int* __restrict__ pair_count,
                                       unsigned long long* __restrict__ check) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_points) {
    return;
  }
  const int cb = 6 / bs;
  const int pb = 3 / bs;
  const int* r = row_off + pose_rows + pb * p;
  const int len = r[1] - r[0];
  bool ok = len >= pb && (len - pb) % cb == 0;
  for (int t = 1; t < pb; ++t) {
    ok = ok && r[t + 1] - r[t] == len;
  }
  pair_count[p] = ok ? (len - pb) / cb : 0;
  if (!ok) {
    atomicAdd(check, 1ull);
  }
}

/** @brief pair_body of point p's camera blocks; flags any column or row offset off the pattern. */
__global__ void BaPointPairBodyKernel(const int* __restrict__ row_off,
                                      const int* __restrict__ col_idx,
                                      const int* __restrict__ pair_ptr, int pose_rows, int bs,
                                      int num_points, int point_base, int* __restrict__ pair_body,
                                      unsigned long long* __restrict__ check) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_points) {
    return;
  }
  const int cb = 6 / bs;
  const int pb = 3 / bs;
  const int pp = pair_ptr[p];
  const int k = pair_ptr[p + 1] - pp;
  const int len = cb * k + pb;
  const int r0 = row_off[pose_rows + pb * p];
  bool ok =
      r0 == point_base + static_cast<long long>(cb * pb) * pp + pb * pb * static_cast<long long>(p);
  for (int j = 0; j < k && ok; ++j) {
    const int c0 = col_idx[r0 + cb * j];
    ok = c0 >= 0 && c0 < pose_rows && c0 % cb == 0;
    for (int t = 0; t < pb && ok; ++t) {
      for (int u = 0; u < cb && ok; ++u) {
        ok = col_idx[r0 + t * len + cb * j + u] == c0 + u;
      }
    }
    pair_body[pp + j] = c0 / cb;
  }
  for (int t = 0; t < pb && ok; ++t) {
    for (int u = 0; u < pb && ok; ++u) {
      ok = col_idx[r0 + t * len + cb * k + u] == pose_rows + pb * p + u;
    }
  }
  if (!ok) {
    atomicAdd(check, 1ull);
  }
}

/**
 * @brief One warp per pose row: cc_row_end = end of its leading run of pose columns; flags a pose
 * column after it and counts the point columns.
 */
__global__ void BaPoseRowPrefixKernel(const int* __restrict__ row_off,
                                      const int* __restrict__ col_idx, int pose_rows,
                                      int block_size, int* __restrict__ cc_row_end,
                                      unsigned long long* __restrict__ check) {
  const int i =
      static_cast<int>((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int lane = threadIdx.x & 31;
  if (i >= pose_rows) {
    return;
  }
  const int e0 = row_off[i];
  const int e1 = row_off[i + 1];
  int pose = 0;
  for (int e = e0 + lane; e < e1; e += 32) {
    pose += col_idx[e] < pose_rows;
  }
  for (int d = 16; d > 0; d >>= 1) {
    pose += __shfl_xor_sync(FULL, pose, d);
  }
  int bad = 0;
  for (int e = e0 + lane; e < e0 + pose; e += 32) {
    bad += col_idx[e] >= pose_rows;
  }
  // Scalar CSR: hvp_dyn reads the pose columns of row 6b for each row of body b whose pose prefix
  // has the same length, so those columns must match.
  if (block_size == 1 && i % 6 != 0) {
    const int f0 = row_off[i - i % 6];
    const int f1 = row_off[i - i % 6 + 1];
    int pose0 = 0;
    for (int e = f0 + lane; e < f1; e += 32) {
      pose0 += col_idx[e] < pose_rows;
    }
    for (int d = 16; d > 0; d >>= 1) {
      pose0 += __shfl_xor_sync(FULL, pose0, d);
    }
    if (pose0 == pose) {
      for (int j = lane; j < pose; j += 32) {
        bad += col_idx[e0 + j] != col_idx[f0 + j];
      }
    }
  }
  for (int d = 16; d > 0; d >>= 1) {
    bad += __shfl_xor_sync(FULL, bad, d);
  }
  if (lane == 0) {
    cc_row_end[i] = e0 + pose;
    if (bad != 0) {
      atomicAdd(check, 1ull);
    }
    atomicAdd(check + 1, static_cast<unsigned long long>(e1 - e0 - pose));
  }
}

}  // namespace

/** @brief Validates the pattern and builds pair_ptr, pair_body and cc_row_end (see header). */
bool BaHessianProduct::Build(cudaStream_t stream, const int* row_offsets, const int* col_ids,
                             int block_size, int num_cameras, int num_points) {
  ready_ = false;
  /** @brief Records why the product cannot be used and returns false. */
  const auto fail = [this](std::string reason) {
    reason_ = std::move(reason);
    return false;
  };
  if (num_cameras <= 0 || num_points <= 0) {
    return fail("layout is not {{cameras, 6}, {points, 3}}");
  }
  if (block_size != 1 && block_size != 3) {
    return fail("storage is not 1x1 or 3x3 blocks");
  }
  int device = 0;
  int cc_major = 0;
  int cooperative = 0;
  THROW_ON_CUDA_ERROR(cudaGetDevice(&device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  if (cc_major < 8) {
    return fail("GPU older than sm_80");
  }
  if (cooperative == 0) {
    return fail("no cooperative launch");
  }
  // Size limits are checked before any of the pattern is read.
  if (num_cameras > kMaxCameras) {
    return fail(std::to_string(num_cameras) + " cameras > " + std::to_string(kMaxCameras));
  }
  if (num_points > kMaxPoints) {
    return fail(std::to_string(num_points) + " points > " + std::to_string(kMaxPoints));
  }
  const int n_c = 6 * num_cameras;
  THROW_ON_CUDA_ERROR(cudaDeviceGetAttribute(&num_sms_, cudaDevAttrMultiProcessorCount, device));
  const auto kernel =
      block_size == 3 ? hvp_dyn<kDynWarps, kDynUS, true> : hvp_dyn<kDynWarps, kDynUS, false>;
  THROW_ON_CUDA_ERROR(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBudget));
  int scap = 0;
  int blocks_per_sm = 0;
  THROW_ON_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, kDynWarps * 32, dyn_smem<kDynWarps>(n_c, &scap)));
  if (blocks_per_sm < 1) {
    return fail("kernel does not fit one block per SM");
  }

  const int pose_rows = n_c / block_size;
  const int threads = 256;
  check_.resize(2);
  pair_ptr_.resize(num_points + 1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(check_.data(), 0, 2 * sizeof(unsigned long long), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(pair_ptr_.data() + num_points, 0, sizeof(int), stream));
  BaPointPairCountKernel<<<(num_points + threads - 1) / threads, threads, 0, stream>>>(
      row_offsets, pose_rows, block_size, num_points, pair_ptr_.data(), check_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  thrust::exclusive_scan(thrust::cuda::par_nosync.on(stream), pair_ptr_.data(),
                         pair_ptr_.data() + num_points + 1, pair_ptr_.data());

  unsigned long long check[2] = {0, 0};
  int num_pairs = 0;
  int point_base = 0;
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(check, check_.data(), sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&num_pairs, pair_ptr_.data() + num_points, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(&point_base, row_offsets + pose_rows, sizeof(int),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  if (check[0] != 0) {
    return fail("pattern not bundle-adjustment shaped");
  }
  // In-kernel value offsets are 32-bit and relative to the first point row.
  if (18LL * num_pairs + 9LL * num_points > INT_MAX) {
    return fail(std::to_string(num_pairs) + " camera-point blocks exceed 32-bit offsets");
  }

  pair_body_.resize(std::max(num_pairs, 1));
  cc_row_end_.resize(pose_rows);
  BaPointPairBodyKernel<<<(num_points + threads - 1) / threads, threads, 0, stream>>>(
      row_offsets, col_ids, pair_ptr_.data(), pose_rows, block_size, num_points, point_base,
      pair_body_.data(), check_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  BaPoseRowPrefixKernel<<<static_cast<int>((32LL * pose_rows + threads - 1) / threads), threads, 0,
                          stream>>>(row_offsets, col_ids, pose_rows, block_size, cc_row_end_.data(),
                                    check_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(check, check_.data(), 2 * sizeof(unsigned long long),
                                      cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  if (check[0] != 0 || check[1] != 2ull * (9 / block_size / block_size) * num_pairs) {
    return fail("pattern not bundle-adjustment shaped");
  }

  // accumulators_[0, n_c): low fixed-point sums; [n_c, 2 n_c): spill sums in units of 2^32;
  // then the window counter, the grid barrier (G per call, so its value / G is the call's epoch)
  // and the epoch + 1 of the last call that overflowed. The kernel returns the sums and the window
  // counter to zero on every call.
  accumulators_.resize(2 * static_cast<size_t>(n_c) + 3);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(accumulators_.data(), 0,
                                      accumulators_.size() * sizeof(unsigned long long), stream));
  block_size_ = block_size;
  num_pose_rows_ = n_c;
  num_points_ = num_points;
  point_base_ = static_cast<long long>(block_size) * block_size * point_base;
  reason_.clear();
  ready_ = true;
  return true;
}

/** @brief Launches hvp_dyn for scalar CSR or 3x3 block storage. */
bool BaHessianProduct::Multiply(cudaStream_t stream, const int* row_offsets, const int* col_ids,
                                const float* values, const float* x, float* y) {
  const auto launch =
      block_size_ == 3 ? launch_dyn<kDynWarps, kDynUS, true> : launch_dyn<kDynWarps, kDynUS, false>;
  const cudaError_t err = launch(values, pair_body_.data(), pair_ptr_.data(), row_offsets, col_ids,
                                 cc_row_end_.data(), x, y, accumulators_.data(), num_pose_rows_,
                                 num_points_, point_base_, num_sms_, stream);
  if (err == cudaErrorCooperativeLaunchTooLarge) {
    // Not sticky: clear it so the caller's next CUDA call does not see it.
    (void)cudaGetLastError();
    ready_ = false;
    reason_ = "cooperative launch did not fit";
    return false;
  }
  THROW_ON_CUDA_ERROR(err);
  return true;
}

/** @brief Whether the overflow word flags the last launch's epoch (synchronizes the device). */
bool BaHessianProduct::LastMultiplyUsedFallback() const {
  if (num_sms_ <= 0 || accumulators_.size() < 3) {
    return false;
  }
  unsigned long long sync[2] = {0, 0};  // grid barrier, overflow epoch + 1
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
  THROW_ON_CUDA_ERROR(cudaMemcpy(sync, accumulators_.data() + accumulators_.size() - 2,
                                 sizeof(sync), cudaMemcpyDeviceToHost));
  return sync[1] != 0 && sync[1] == sync[0] / num_sms_;
}

}  // namespace cunls
