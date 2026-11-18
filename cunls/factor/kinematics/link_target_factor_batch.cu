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
#include "cunls/factor/kinematics/link_target_factor_batch.h"
#include "cunls/math/lie_device.cuh"

namespace cunls {

namespace {

constexpr int kBlockSize = 128;

/** Which rows a link-target factor has. */
enum class Rows { kPosition, kOrientation, kPose };

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

/**
 * One group of G lanes per item: every lane computes the residual (a few
 * dozen flops), lane 0 writes it, and lane l writes Jacobian columns
 * l, l + G, ... of every row, so each row is written with consecutive
 * stores. `frames` holds the per-factor point (kPosition, 3 floats), rotation
 * offset (kOrientation, 4) or frame offset (kPose, 7); null: none.
 */
template <Rows kRows, int G, bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    LinkTargetKernel(ArticulationTopology::DeviceView v, const int *__restrict__ links,
                     const float *__restrict__ frames, const float *__restrict__ targets,
                     const float *__restrict__ weights, float const *const *__restrict__ state_ptrs,
                     float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                     const int *__restrict__ factor_ids, int num_factors) {
  constexpr int m = kRows == Rows::kPose ? 6 : 3;
  constexpr int kPos = kRows == Rows::kOrientation ? -1 : 0;  // first position row
  constexpr int kOri = kRows == Rows::kPosition ? -1 : (kRows == Rows::kPose ? 3 : 0);
  const int t = static_cast<int>((blockIdx.x * static_cast<size_t>(kBlockSize) + threadIdx.x) / G);
  const int lane = threadIdx.x % G;
  if (t >= num_items) return;
  const int f = FactorIndex(t, factor_ids, num_factors);
  const float *x = state_ptrs[t];
  const int link = __ldg(links + f);
  const float *P = x + v.num_coords + 12 * link;
  const float RL[9] = {P[0], P[1], P[2], P[4], P[5], P[6], P[8], P[9], P[10]};

  float w[m];
#pragma unroll
  for (int r = 0; r < m; ++r) w[r] = weights != nullptr ? __ldg(weights + m * f + r) : 1.f;

  float r[m];
  float xw[3] = {0.f, 0.f, 0.f};  // world point (position rows)
  float Jinv[9];                  // J_l⁻¹ of the rotation residual (orientation rows)
  if constexpr (kPos >= 0) {
    float c[3] = {0.f, 0.f, 0.f};
    if (frames != nullptr) {
      const int stride = kRows == Rows::kPose ? 7 : 3;
#pragma unroll
      for (int i = 0; i < 3; ++i) c[i] = __ldg(frames + stride * f + i);
    }
    const int tstride = kRows == Rows::kPose ? 7 : 3;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      xw[i] = RL[3 * i] * c[0] + RL[3 * i + 1] * c[1] + RL[3 * i + 2] * c[2] + P[4 * i + 3];
      r[kPos + i] = xw[i] - __ldg(targets + tstride * f + i);
    }
  }
  if constexpr (kOri >= 0) {
    float Rcur[9];
    if (frames != nullptr) {
      const int stride = kRows == Rows::kPose ? 7 : 4;
      const int at = kRows == Rows::kPose ? 3 : 0;
      float q[4], Roff[9];
#pragma unroll
      for (int i = 0; i < 4; ++i) q[i] = __ldg(frames + stride * f + at + i);
      QuatToMatrix(q, Roff);
      lie_device::MatMul3x3(RL, Roff, Rcur);
    } else {
#pragma unroll
      for (int i = 0; i < 9; ++i) Rcur[i] = RL[i];
    }
    const int tstride = kRows == Rows::kPose ? 7 : 4;
    const int tat = kRows == Rows::kPose ? 3 : 0;
    float q[4], Rt[9];
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = __ldg(targets + tstride * f + tat + i);
    QuatToMatrix(q, Rt);
    float M[9];  // Rcur Rtᵀ
#pragma unroll
    for (int a = 0; a < 3; ++a) {
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        M[3 * a + b] = Rcur[3 * a] * Rt[3 * b] + Rcur[3 * a + 1] * Rt[3 * b + 1] +
                       Rcur[3 * a + 2] * Rt[3 * b + 2];
      }
    }
    float phi[3];
    lie_device::LogSO3(M, 3, phi);
    if constexpr (kJacobian) lie_device::SO3JacobianLeftInverse(phi, Jinv, 3);
#pragma unroll
    for (int i = 0; i < 3; ++i) r[kOri + i] = phi[i];
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < m; ++i) residuals[static_cast<size_t>(t) * m + i] = w[i] * r[i];
  }
  if constexpr (kJacobian) {
    const int T = v.num_dofs;
    const unsigned *anc = v.ancestors + static_cast<size_t>(link) * v.ancestor_words;
    const float *columns = x + v.num_coords + 12 * v.num_joints;
    float *J = jacobians + static_cast<size_t>(t) * m * T;
    for (int j = lane; j < T; j += G) {
      float col[m];
      if ((__ldg(anc + (j >> 5)) >> (j & 31)) & 1u) {
        const float *c = columns + 6 * j;
        const float om[3] = {c[0], c[1], c[2]};
        if constexpr (kPos >= 0) {
          col[kPos + 0] = om[1] * xw[2] - om[2] * xw[1] + c[3];
          col[kPos + 1] = om[2] * xw[0] - om[0] * xw[2] + c[4];
          col[kPos + 2] = om[0] * xw[1] - om[1] * xw[0] + c[5];
        }
        if constexpr (kOri >= 0) {
#pragma unroll
          for (int i = 0; i < 3; ++i) {
            col[kOri + i] = Jinv[3 * i] * om[0] + Jinv[3 * i + 1] * om[1] + Jinv[3 * i + 2] * om[2];
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < m; ++i) col[i] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < m; ++i) J[static_cast<size_t>(i) * T + j] = w[i] * col[i];
    }
  }
}

/** Launches the kernel with a lane group sized to the DOF count. */
template <Rows kRows>
void LaunchLinkTarget(const ArticulationTopology::DeviceView &v, const int *links,
                      const float *frames, const float *targets, const float *weights,
                      float const *const *state_ptrs, float *residuals, float *jacobians,
                      size_t num_items, const int *factor_ids, size_t num_factors,
                      cudaStream_t stream) {
  auto launch = [&](auto group) {
    constexpr int G = decltype(group)::value;
    const size_t threads = num_items * G;
    const unsigned blocks = static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize);
    if (jacobians != nullptr) {
      LinkTargetKernel<kRows, G, true><<<blocks, kBlockSize, 0, stream>>>(
          v, links, frames, targets, weights, state_ptrs, residuals, jacobians,
          static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
    } else {
      LinkTargetKernel<kRows, G, false><<<blocks, kBlockSize, 0, stream>>>(
          v, links, frames, targets, weights, state_ptrs, residuals, nullptr,
          static_cast<int>(num_items), factor_ids, static_cast<int>(num_factors));
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

void RequireNonNull(const void *p, const char *cls, const char *name) {
  if (p == nullptr) {
    throw std::invalid_argument(std::string(cls) + ": " + name +
                                " is null (expected a device array). Fix: pass the " + name +
                                " array.");
  }
}

size_t NumItems(size_t num_factors, size_t num_factor_ids) {
  return num_factor_ids == 0 ? num_factors : num_factor_ids;
}

}  // namespace

LinkPositionFactorBatch::LinkPositionFactorBatch(const ArticulationTopology &topology,
                                                 const int *links, const float *points,
                                                 const float *targets, const float *weights,
                                                 size_t capacity)
    : ArticulationFactorBatch(topology, 3, capacity),
      links_(links),
      points_(points),
      targets_(targets),
      weights_(weights) {
  RequireNonNull(links, "LinkPositionFactorBatch", "links");
  RequireNonNull(points, "LinkPositionFactorBatch", "points");
  RequireNonNull(targets, "LinkPositionFactorBatch", "targets");
}

bool LinkPositionFactorBatch::Evaluate(float *residuals, float *jacobians,
                                       float const *const *state_pointers, cudaStream_t stream,
                                       const int *factor_ids, size_t num_factor_ids) const {
  const size_t n = NumItems(NumActiveFactors(), num_factor_ids);
  if (n == 0 || NumActiveFactors() == 0) return true;
  LaunchLinkTarget<Rows::kPosition>(Topology().View(), links_, points_, targets_, weights_,
                                    state_pointers, residuals, jacobians, n, factor_ids,
                                    NumActiveFactors(), stream);
  return true;
}

LinkOrientationFactorBatch::LinkOrientationFactorBatch(const ArticulationTopology &topology,
                                                       const int *links, const float *offsets,
                                                       const float *targets, const float *weights,
                                                       size_t capacity)
    : ArticulationFactorBatch(topology, 3, capacity),
      links_(links),
      offsets_(offsets),
      targets_(targets),
      weights_(weights) {
  RequireNonNull(links, "LinkOrientationFactorBatch", "links");
  RequireNonNull(targets, "LinkOrientationFactorBatch", "targets");
}

bool LinkOrientationFactorBatch::Evaluate(float *residuals, float *jacobians,
                                          float const *const *state_pointers, cudaStream_t stream,
                                          const int *factor_ids, size_t num_factor_ids) const {
  const size_t n = NumItems(NumActiveFactors(), num_factor_ids);
  if (n == 0 || NumActiveFactors() == 0) return true;
  LaunchLinkTarget<Rows::kOrientation>(Topology().View(), links_, offsets_, targets_, weights_,
                                       state_pointers, residuals, jacobians, n, factor_ids,
                                       NumActiveFactors(), stream);
  return true;
}

LinkPoseFactorBatch::LinkPoseFactorBatch(const ArticulationTopology &topology, const int *links,
                                         const float *offsets, const float *targets,
                                         const float *weights, size_t capacity)
    : ArticulationFactorBatch(topology, 6, capacity),
      links_(links),
      offsets_(offsets),
      targets_(targets),
      weights_(weights) {
  RequireNonNull(links, "LinkPoseFactorBatch", "links");
  RequireNonNull(targets, "LinkPoseFactorBatch", "targets");
}

bool LinkPoseFactorBatch::Evaluate(float *residuals, float *jacobians,
                                   float const *const *state_pointers, cudaStream_t stream,
                                   const int *factor_ids, size_t num_factor_ids) const {
  const size_t n = NumItems(NumActiveFactors(), num_factor_ids);
  if (n == 0 || NumActiveFactors() == 0) return true;
  LaunchLinkTarget<Rows::kPose>(Topology().View(), links_, offsets_, targets_, weights_,
                                state_pointers, residuals, jacobians, n, factor_ids,
                                NumActiveFactors(), stream);
  return true;
}

}  // namespace cunls
