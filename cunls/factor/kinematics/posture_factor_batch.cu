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
#include <type_traits>

#include "cunls/common/helper.h"
#include "cunls/factor/kinematics/posture_factor_batch.h"
#include "cunls/math/lie_device.cuh"

namespace cunls {

namespace {

constexpr int kBlockSize = 128;
constexpr int kBall = 2, kFree = 4;

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

/** phi = Log(R(q) R(q*)ᵀ) for quaternions q, q* (x, y, z, w). */
__device__ __forceinline__ void QuatError(const float *q, const float *qt, float *phi) {
  float R[9], Rt[9], M[9];
  QuatToMatrix(q, R);
  QuatToMatrix(qt, Rt);
#pragma unroll
  for (int a = 0; a < 3; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      M[3 * a + b] =
          R[3 * a] * Rt[3 * b] + R[3 * a + 1] * Rt[3 * b + 1] + R[3 * a + 2] * Rt[3 * b + 2];
    }
  }
  lie_device::LogSO3(M, 3, phi);
}

/**
 * One group of G lanes per item. Lane 0 writes the residual; lane l writes
 * Jacobian column l, l + G, ...: the column's only nonzero entries are in the
 * rows of the same joint (block diagonal).
 */
template <int G, bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    PostureKernel(ArticulationTopology::DeviceView v, const float *__restrict__ targets,
                  const float *__restrict__ weights, float const *const *__restrict__ state_ptrs,
                  float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                  const int *__restrict__ factor_ids, int num_factors) {
  const int t = static_cast<int>((blockIdx.x * static_cast<size_t>(kBlockSize) + threadIdx.x) / G);
  const int lane = threadIdx.x % G;
  if (t >= num_items) return;
  const int f = FactorIndex(t, factor_ids, num_factors);
  const int T = v.num_dofs, C = v.num_coords;
  const float *x = state_ptrs[t];
  const float *qt = targets + static_cast<size_t>(f) * C;
  const float *w = weights != nullptr ? weights + static_cast<size_t>(f) * T : nullptr;

  if (lane == 0) {
    float *res = residuals + static_cast<size_t>(t) * T;
    for (int j = 0; j < v.num_joints; ++j) {
      const int q0 = v.q_start[j], d0 = v.qd_start[j], n = v.qd_start[j + 1] - d0;
      const int type = v.type[j];
      float r[6];
      if (type == kBall) {
        QuatError(x + q0, qt + q0, r);
      } else if (type == kFree) {
        for (int i = 0; i < 3; ++i) r[i] = x[q0 + i] - __ldg(qt + q0 + i);
        QuatError(x + q0 + 3, qt + q0 + 3, r + 3);
      } else {
        for (int i = 0; i < n; ++i) r[i] = x[q0 + i] - __ldg(qt + q0 + i);
      }
      for (int i = 0; i < n; ++i) res[d0 + i] = (w != nullptr ? __ldg(w + d0 + i) : 1.f) * r[i];
    }
  }
  if constexpr (kJacobian) {
    float *Jt = jacobians + static_cast<size_t>(t) * T * T;
    for (int c = lane; c < T; c += G) {
      const int j = v.dof_joint[c];
      const int q0 = v.q_start[j], d0 = v.qd_start[j], d1 = v.qd_start[j + 1];
      const int type = v.type[j];
      // Rows of joint j: 3x3 J_l⁻¹ for rotation blocks, identity otherwise.
      float Jinv[9];
      int rot0 = -1;  // first row/column of the rotation block, if any
      if (type == kBall || type == kFree) {
        rot0 = type == kBall ? d0 : d0 + 3;
        float phi[3];
        QuatError(x + (type == kBall ? q0 : q0 + 3), qt + (type == kBall ? q0 : q0 + 3), phi);
        lie_device::SO3JacobianLeftInverse(phi, Jinv, 3);
      }
      for (int r = 0; r < T; ++r) {
        float value = 0.f;
        if (r >= d0 && r < d1) {
          if (rot0 >= 0 && r >= rot0 && r < rot0 + 3 && c >= rot0 && c < rot0 + 3) {
            value = Jinv[3 * (r - rot0) + (c - rot0)];
          } else if (r == c) {
            value = 1.f;
          }
          if (w != nullptr) value *= __ldg(w + r);
        }
        Jt[static_cast<size_t>(r) * T + c] = value;
      }
    }
  }
}

}  // namespace

PostureFactorBatch::PostureFactorBatch(const ArticulationTopology &topology, const float *targets,
                                       const float *weights, size_t capacity)
    : ArticulationFactorBatch(topology, static_cast<size_t>(topology.DofCount()), capacity),
      targets_(targets),
      weights_(weights) {
  if (targets == nullptr) {
    throw std::invalid_argument(
        "PostureFactorBatch: targets is null (expected a device array of CoordCount() floats "
        "per factor). Fix: pass the nominal joint_q.");
  }
}

bool PostureFactorBatch::Evaluate(float *residuals, float *jacobians,
                                  float const *const *state_pointers, cudaStream_t stream,
                                  const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t n = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (n == 0 || num_factors == 0 || Topology().DofCount() == 0) return true;
  const auto &v = Topology().View();
  auto launch = [&](auto group) {
    constexpr int G = decltype(group)::value;
    const unsigned blocks = static_cast<unsigned>((n * G + kBlockSize - 1) / kBlockSize);
    if (jacobians != nullptr) {
      PostureKernel<G, true><<<blocks, kBlockSize, 0, stream>>>(
          v, targets_, weights_, state_pointers, residuals, jacobians, static_cast<int>(n),
          factor_ids, static_cast<int>(num_factors));
    } else {
      PostureKernel<G, false><<<blocks, kBlockSize, 0, stream>>>(
          v, targets_, weights_, state_pointers, residuals, nullptr, static_cast<int>(n),
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
  return true;
}

}  // namespace cunls
