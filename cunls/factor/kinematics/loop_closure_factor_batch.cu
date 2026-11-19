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
#include <string>
#include <type_traits>

#include "cunls/common/helper.h"
#include "cunls/factor/kinematics/loop_closure_factor_batch.h"
#include "cunls/math/lie_device.cuh"

namespace cunls {

namespace {

constexpr int kBlockSize = 128;

/** Factor (measurement) index of item t (see FactorBatch::Evaluate). */
__device__ __forceinline__ int FactorIndex(int t, const int *factor_ids, int num_factors) {
  if (factor_ids != nullptr) return factor_ids[t];
  return t < num_factors ? t : t % num_factors;
}

/** R (row-major 3x3) of the quaternion (x, y, z, w), normalized. */
__device__ __forceinline__ void QuatToMatrix(const float *q, float *R) {
  const float n = rsqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float x = q[0] * n, y = q[1] * n, z = q[2] * n, w = q[3] * n;
  R[0] = 1.f - 2.f * (y * y + z * z);
  R[1] = 2.f * (x * y - z * w);
  R[2] = 2.f * (x * z + y * w);
  R[3] = 2.f * (x * y + z * w);
  R[4] = 1.f - 2.f * (x * x + z * z);
  R[5] = 2.f * (y * z - x * w);
  R[6] = 2.f * (x * z - y * w);
  R[7] = 2.f * (y * z + x * w);
  R[8] = 1.f - 2.f * (x * x + y * y);
}

__device__ __forceinline__ void Cross(const float *a, const float *b, float *c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ float Dot(const float *a, const float *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** y = R v (row-major 3x3). */
__device__ __forceinline__ void Rot(const float *R, const float *v, float *y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) y[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
}

/**
 * World frame F = X_link · (p, q): rotation RF (row-major 3x3) and origin pF.
 * `link_pose` is the 3x4 world pose of the link, or null for the world.
 */
__device__ __forceinline__ void JointFrame(const float *link_pose, const float *frame, float *RF,
                                           float *pF) {
  float Rj[9];
  QuatToMatrix(frame + 3, Rj);
  if (link_pose == nullptr) {
#pragma unroll
    for (int i = 0; i < 9; ++i) RF[i] = Rj[i];
#pragma unroll
    for (int i = 0; i < 3; ++i) pF[i] = frame[i];
    return;
  }
  const float RL[9] = {link_pose[0], link_pose[1], link_pose[2], link_pose[4], link_pose[5],
                       link_pose[6], link_pose[8], link_pose[9], link_pose[10]};
  lie_device::MatMul3x3(RL, Rj, RF);
  Rot(RL, frame, pF);
  pF[0] += link_pose[3];
  pF[1] += link_pose[7];
  pF[2] += link_pose[11];
}

/** Two unit vectors b1, b2 with (a, b1, b2) orthonormal, for a unit a. */
__device__ __forceinline__ void Perpendiculars(const float *a, float *b1, float *b2) {
  const float e[3] = {fabsf(a[0]) < 0.9f ? 1.f : 0.f, fabsf(a[0]) < 0.9f ? 0.f : 1.f, 0.f};
  Cross(a, e, b1);
  const float n = rsqrtf(Dot(b1, b1));
  b1[0] *= n;
  b1[1] *= n;
  b1[2] *= n;
  Cross(a, b1, b2);
}

/**
 * One group of G lanes per item; every lane derives the frames, lane 0
 * writes the residual, lane l writes columns l, l + G, ... of every row.
 * kType: the loop-closing joint type (1 REVOLUTE, 0 PRISMATIC, 2 BALL, 3 FIXED).
 */
template <int kType, int G, bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    LoopClosureKernel(ArticulationTopology::DeviceView v, const int *__restrict__ parent_links,
                      const int *__restrict__ child_links, const float *__restrict__ frames,
                      const float *__restrict__ axes, float const *const *__restrict__ state_ptrs,
                      float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                      const int *__restrict__ factor_ids, int num_factors) {
  constexpr bool kRevolute = kType == 1, kPrismatic = kType == 0, kBall = kType == 2,
                 kFixed = kType == 3;
  constexpr int m = kBall ? 3 : ((kFixed || kRevolute) ? 6 : 5);
  const int t = static_cast<int>((blockIdx.x * static_cast<size_t>(kBlockSize) + threadIdx.x) / G);
  const int lane = threadIdx.x % G;
  if (t >= num_items) return;
  const int f = FactorIndex(t, factor_ids, num_factors);
  const float *x = state_ptrs[t];
  const int P = __ldg(parent_links + f), C = __ldg(child_links + f);
  if (P < -1 || P >= v.num_joints || C < 0 || C >= v.num_joints) {
    // Not links of the topology: NaN residual (diagnose reports it), zero Jacobian, and no read
    // past the instance's cache.
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < m; ++i) residuals[static_cast<size_t>(t) * m + i] = NAN;
    }
    if constexpr (kJacobian) {
      float *J = jacobians + static_cast<size_t>(t) * m * v.num_dofs;
      for (int j = lane; j < m * v.num_dofs; j += G) J[j] = 0.f;
    }
    return;
  }
  float fr[14];
#pragma unroll
  for (int i = 0; i < 14; ++i) fr[i] = __ldg(frames + 14 * f + i);
  const float *poses = x + v.num_coords;
  float RP[9], pP[3], RC[9], pC[3];
  JointFrame(P >= 0 ? poses + 12 * P : nullptr, fr, RP, pP);
  JointFrame(poses + 12 * C, fr + 7, RC, pC);
  const float dp[3] = {pC[0] - pP[0], pC[1] - pP[1], pC[2] - pP[2]};

  float r[m];
  float aC[3], aP[3], bw[2][3], phi[3], JlInv[9], JrInv[9];
  if constexpr (!kPrismatic) {
#pragma unroll
    for (int i = 0; i < 3; ++i) r[i] = dp[i];
  }
  if constexpr (kRevolute || kPrismatic) {
    float a[3] = {__ldg(axes + 3 * f), __ldg(axes + 3 * f + 1), __ldg(axes + 3 * f + 2)};
    const float n = rsqrtf(Dot(a, a));
    a[0] *= n;
    a[1] *= n;
    a[2] *= n;
    if constexpr (kRevolute) {
      // The full difference, not its projections ⊥ a_P: those also vanish for opposite axes.
      Rot(RC, a, aC);
      Rot(RP, a, aP);
#pragma unroll
      for (int i = 0; i < 3; ++i) r[3 + i] = aC[i] - aP[i];
    } else {
      float b1[3], b2[3];
      Perpendiculars(a, b1, b2);
      Rot(RP, b1, bw[0]);
      Rot(RP, b2, bw[1]);
      r[3] = Dot(bw[0], dp);
      r[4] = Dot(bw[1], dp);
    }
  }
  if constexpr (kPrismatic || kFixed) {
    float M[9];  // RC RPᵀ
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        M[3 * a + b] =
            RC[3 * a] * RP[3 * b] + RC[3 * a + 1] * RP[3 * b + 1] + RC[3 * a + 2] * RP[3 * b + 2];
      }
    }
    lie_device::LogSO3(M, 3, phi);
    if constexpr (kJacobian) {
      lie_device::SO3JacobianLeftInverse(phi, JlInv, 3);
      const float minus[3] = {-phi[0], -phi[1], -phi[2]};
      lie_device::SO3JacobianLeftInverse(minus, JrInv, 3);  // J_r⁻¹(φ) = J_l⁻¹(-φ)
    }
    const int at = kFixed ? 3 : 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) r[at + i] = phi[i];
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < m; ++i) residuals[static_cast<size_t>(t) * m + i] = r[i];
  }
  if constexpr (kJacobian) {
    const int T = v.num_dofs, W = v.ancestor_words;
    const float *columns = x + v.num_coords + 12 * v.num_joints;
    const unsigned *ancC = v.ancestors + static_cast<size_t>(C) * W;
    const unsigned *ancP = P >= 0 ? v.ancestors + static_cast<size_t>(P) * W : nullptr;
    float *J = jacobians + static_cast<size_t>(t) * m * T;
    for (int j = lane; j < T; j += G) {
      const float sC = ((__ldg(ancC + (j >> 5)) >> (j & 31)) & 1u) ? 1.f : 0.f;
      const float sP = (ancP != nullptr && ((__ldg(ancP + (j >> 5)) >> (j & 31)) & 1u)) ? 1.f : 0.f;
      float col[m];
#pragma unroll
      for (int i = 0; i < m; ++i) col[i] = 0.f;
      if (sC != 0.f || sP != 0.f) {
        const float *c = columns + 6 * j;
        const float om[3] = {c[0], c[1], c[2]}, vl[3] = {c[3], c[4], c[5]};
        // dp_C = sC (ω × p_C + v), dp_P = sP (ω × p_P + v); dω_C = sC ω, dω_P = sP ω.
        float xC[3], xP[3], ddp[3];
        Cross(om, pC, xC);
        Cross(om, pP, xP);
#pragma unroll
        for (int i = 0; i < 3; ++i) ddp[i] = sC * (xC[i] + vl[i]) - sP * (xP[i] + vl[i]);
        if constexpr (!kPrismatic) {
#pragma unroll
          for (int i = 0; i < 3; ++i) col[i] = ddp[i];
        }
        if constexpr (kRevolute) {
          // d(a_C - a_P) = dω_C × a_C - dω_P × a_P
          float wxC[3], wxP[3];
          Cross(om, aC, wxC);
          Cross(om, aP, wxP);
#pragma unroll
          for (int i = 0; i < 3; ++i) col[3 + i] = sC * wxC[i] - sP * wxP[i];
        }
        if constexpr (kPrismatic) {
          // d(b·Δp) = dω_P·(b × Δp) + b·dΔp
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            float bxd[3];
            Cross(bw[k], dp, bxd);
            col[3 + k] = sP * Dot(om, bxd) + Dot(bw[k], ddp);
          }
        }
        if constexpr (kPrismatic || kFixed) {
          // dφ = J_l⁻¹(φ) dω_C - J_r⁻¹(φ) dω_P
          const int at = kFixed ? 3 : 0;
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            col[at + i] =
                sC * (JlInv[3 * i] * om[0] + JlInv[3 * i + 1] * om[1] + JlInv[3 * i + 2] * om[2]) -
                sP * (JrInv[3 * i] * om[0] + JrInv[3 * i + 1] * om[1] + JrInv[3 * i + 2] * om[2]);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < m; ++i) J[static_cast<size_t>(i) * T + j] = col[i];
    }
  }
}

template <int kType>
void LaunchLoopClosure(const ArticulationTopology::DeviceView &v, const int *parents,
                       const int *children, const float *frames, const float *axes,
                       float const *const *state_ptrs, float *residuals, float *jacobians, size_t n,
                       const int *factor_ids, size_t num_factors, cudaStream_t stream) {
  auto launch = [&](auto group) {
    constexpr int G = decltype(group)::value;
    const unsigned blocks = static_cast<unsigned>((n * G + kBlockSize - 1) / kBlockSize);
    if (jacobians != nullptr) {
      LoopClosureKernel<kType, G, true><<<blocks, kBlockSize, 0, stream>>>(
          v, parents, children, frames, axes, state_ptrs, residuals, jacobians, static_cast<int>(n),
          factor_ids, static_cast<int>(num_factors));
    } else {
      LoopClosureKernel<kType, G, false><<<blocks, kBlockSize, 0, stream>>>(
          v, parents, children, frames, axes, state_ptrs, residuals, nullptr, static_cast<int>(n),
          factor_ids, static_cast<int>(num_factors));
    }
  };
  if (jacobians == nullptr) {
    launch(std::integral_constant<int, 1>());  // residuals only: one thread per item
  } else if (v.num_dofs <= 8) {
    launch(std::integral_constant<int, 8>());
  } else if (v.num_dofs <= 16) {
    launch(std::integral_constant<int, 16>());
  } else {
    launch(std::integral_constant<int, 32>());
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

size_t LoopClosureFactorBatch::RowsOf(ArticulationJointType joint_type) {
  switch (joint_type) {
    case ArticulationJointType::kBall:
      return 3;
    case ArticulationJointType::kPrismatic:
      return 5;
    case ArticulationJointType::kRevolute:
    case ArticulationJointType::kFixed:
      return 6;
    default:
      throw std::invalid_argument(
          "LoopClosureFactorBatch: unsupported loop-closing joint type (got " +
          std::to_string(static_cast<int>(joint_type)) +
          ", expected BALL=2, REVOLUTE=1, PRISMATIC=0 or FIXED=3). Fix: model the loop with one "
          "of these joint types.");
  }
}

LoopClosureFactorBatch::LoopClosureFactorBatch(const ArticulationTopology &topology,
                                               ArticulationJointType joint_type,
                                               const int *parent_links, const int *child_links,
                                               const float *frames, const float *axes,
                                               size_t capacity)
    : ArticulationFactorBatch(topology, RowsOf(joint_type), capacity),
      joint_type_(joint_type),
      parent_links_(parent_links),
      child_links_(child_links),
      frames_(frames),
      axes_(axes) {
  if (parent_links == nullptr || child_links == nullptr || frames == nullptr) {
    throw std::invalid_argument(
        "LoopClosureFactorBatch: parent_links, child_links and frames must be device arrays "
        "(got a null pointer). Fix: pass all three arrays.");
  }
  if (axes == nullptr && (joint_type == ArticulationJointType::kRevolute ||
                          joint_type == ArticulationJointType::kPrismatic)) {
    throw std::invalid_argument(
        "LoopClosureFactorBatch: axes is null for a REVOLUTE/PRISMATIC loop (expected 3 floats "
        "per factor). Fix: pass the joint axes.");
  }
}

bool LoopClosureFactorBatch::Evaluate(float *residuals, float *jacobians,
                                      float const *const *state_pointers, cudaStream_t stream,
                                      const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t n = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (n == 0 || num_factors == 0) return true;
  const auto &v = Topology().View();
  switch (joint_type_) {
    case ArticulationJointType::kPrismatic:
      LaunchLoopClosure<0>(v, parent_links_, child_links_, frames_, axes_, state_pointers,
                           residuals, jacobians, n, factor_ids, num_factors, stream);
      break;
    case ArticulationJointType::kRevolute:
      LaunchLoopClosure<1>(v, parent_links_, child_links_, frames_, axes_, state_pointers,
                           residuals, jacobians, n, factor_ids, num_factors, stream);
      break;
    case ArticulationJointType::kBall:
      LaunchLoopClosure<2>(v, parent_links_, child_links_, frames_, axes_, state_pointers,
                           residuals, jacobians, n, factor_ids, num_factors, stream);
      break;
    default:
      LaunchLoopClosure<3>(v, parent_links_, child_links_, frames_, axes_, state_pointers,
                           residuals, jacobians, n, factor_ids, num_factors, stream);
      break;
  }
  return true;
}

}  // namespace cunls
