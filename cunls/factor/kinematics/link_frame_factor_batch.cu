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
#include <stdexcept>
#include <string>

#include "cunls/common/helper.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/kinematics/link_frame_factor_batch.h"

namespace cunls {
namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxSlots = LinkFrameFactorBatch::kMaxInnerSlots;

/**
 * How the wrapped factor's slots map to this batch's (by value, for kernels):
 * pose slot s is inner slot pose_inner[s]; inner slot k is outer slot
 * outer_of_inner[k] (-1: a pose slot); inner slot k's Jacobian columns start
 * at inner_col[k]; outer slot o (>= 1) starts at outer_col[o] and has
 * outer_size[o] columns.
 */
struct SlotMap {
  int num_inner = 0, num_outer = 0, num_pose_slots = 0;
  int inner_cols = 0, outer_cols = 0;
  int pose_inner[kMaxSlots] = {};
  int outer_of_inner[kMaxSlots] = {};
  int inner_col[kMaxSlots] = {};
  int inner_of_outer[kMaxSlots] = {};
  int outer_col[kMaxSlots] = {};
  int outer_size[kMaxSlots] = {};
};

/**
 * One thread per item: F_s = X_L X_off of each pose slot into `poses`
 * (row-major 4x4; NaN for a link outside the topology), and the inner
 * factor's state pointers (pose slots: the scratch poses; others: this
 * item's pointers).
 */
__global__ void GatherLinkFramesKernel(ArticulationTopology::DeviceView v, SlotMap map,
                                       const int *__restrict__ topology_links,
                                       const float *__restrict__ frame_offsets,
                                       float const *const *__restrict__ state_ptrs,
                                       float *__restrict__ poses, float **__restrict__ inner_ptrs,
                                       int num_items, const int *__restrict__ factor_ids,
                                       int num_factors) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_items) return;
  const int f = factor_ids != nullptr ? factor_ids[t] : (t < num_factors ? t : t % num_factors);
  const float *x = state_ptrs[static_cast<size_t>(t) * map.num_outer];
  const int nP = map.num_pose_slots;
  for (int s = 0; s < nP; ++s) {
    const int link = __ldg(topology_links + static_cast<size_t>(f) * nP + s);
    float *F = poses + (static_cast<size_t>(t) * nP + s) * 16;
    if (link < 0 || link >= v.num_joints) {
      for (int i = 0; i < 16; ++i) F[i] = NAN;
    } else {
      const float *P = x + v.num_coords + 12 * link;  // [R | p] row-major 3x4
      float Ro[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, po[3] = {0, 0, 0};
      if (frame_offsets != nullptr) {
        const float *o = frame_offsets + (static_cast<size_t>(f) * nP + s) * 7;
        po[0] = __ldg(o);
        po[1] = __ldg(o + 1);
        po[2] = __ldg(o + 2);
        float qx = __ldg(o + 3), qy = __ldg(o + 4), qz = __ldg(o + 5), qw = __ldg(o + 6);
        const float n = rsqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
        qx *= n, qy *= n, qz *= n, qw *= n;
        Ro[0] = 1.f - 2.f * (qy * qy + qz * qz);
        Ro[1] = 2.f * (qx * qy - qz * qw);
        Ro[2] = 2.f * (qx * qz + qy * qw);
        Ro[3] = 2.f * (qx * qy + qz * qw);
        Ro[4] = 1.f - 2.f * (qx * qx + qz * qz);
        Ro[5] = 2.f * (qy * qz - qx * qw);
        Ro[6] = 2.f * (qx * qz - qy * qw);
        Ro[7] = 2.f * (qy * qz + qx * qw);
        Ro[8] = 1.f - 2.f * (qx * qx + qy * qy);
      }
#pragma unroll
      for (int a = 0; a < 3; ++a) {
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          F[4 * a + b] = P[4 * a] * Ro[b] + P[4 * a + 1] * Ro[3 + b] + P[4 * a + 2] * Ro[6 + b];
        }
        F[4 * a + 3] =
            P[4 * a] * po[0] + P[4 * a + 1] * po[1] + P[4 * a + 2] * po[2] + P[4 * a + 3];
      }
      F[12] = F[13] = F[14] = 0.f;
      F[15] = 1.f;
    }
    inner_ptrs[static_cast<size_t>(t) * map.num_inner + map.pose_inner[s]] = F;
  }
  for (int k = 0; k < map.num_inner; ++k) {
    const int o = map.outer_of_inner[k];
    if (o >= 0) {
      inner_ptrs[static_cast<size_t>(t) * map.num_inner + k] =
          const_cast<float *>(state_ptrs[static_cast<size_t>(t) * map.num_outer + o]);
    }
  }
}

/**
 * One thread per (item, output column): column c < T is DOF c, the sum over
 * pose slots of the inner columns of the slot times the body twist of its
 * frame per unit of DOF c; a later column copies the inner column of its
 * passed-through slot. Threads of one item write consecutive columns.
 */
__global__ void CarryJacobianKernel(ArticulationTopology::DeviceView v, SlotMap map, int m,
                                    const int *__restrict__ topology_links,
                                    float const *const *__restrict__ state_ptrs,
                                    const float *__restrict__ poses,
                                    const float *__restrict__ inner_jac, float *__restrict__ jac,
                                    int num_items, const int *__restrict__ factor_ids,
                                    int num_factors) {
  const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const int Jo = map.outer_cols, Ji = map.inner_cols, T = v.num_dofs;
  const int t = static_cast<int>(gid / Jo), c = static_cast<int>(gid % Jo);
  if (t >= num_items) return;
  const float *Jin = inner_jac + static_cast<size_t>(t) * m * Ji;
  float *Jout = jac + static_cast<size_t>(t) * m * Jo;
  if (c >= T) {  // a passed-through slot
    int o = 1;
    while (o + 1 < map.num_outer && c >= map.outer_col[o + 1]) ++o;
    const int ci = map.inner_col[map.inner_of_outer[o]] + (c - map.outer_col[o]);
    for (int r = 0; r < m; ++r)
      Jout[static_cast<size_t>(r) * Jo + c] = Jin[static_cast<size_t>(r) * Ji + ci];
    return;
  }
  const int f = factor_ids != nullptr ? factor_ids[t] : (t < num_factors ? t : t % num_factors);
  const float *x = state_ptrs[static_cast<size_t>(t) * map.num_outer];
  const float *col = x + v.num_coords + 12 * v.num_joints + 6 * c;
  const float om[3] = {col[0], col[1], col[2]}, vl[3] = {col[3], col[4], col[5]};
  float b[kMaxSlots][6];
  bool moves[kMaxSlots];
  const int nP = map.num_pose_slots;
  for (int s = 0; s < nP; ++s) {
    const int link = __ldg(topology_links + static_cast<size_t>(f) * nP + s);
    moves[s] = link >= 0 && link < v.num_joints &&
               ((__ldg(v.ancestors + static_cast<size_t>(link) * v.ancestor_words + (c >> 5)) >>
                 (c & 31)) &
                1u);
    if (!moves[s]) continue;
    const float *F = poses + (static_cast<size_t>(t) * nP + s) * 16;
    const float p[3] = {F[3], F[7], F[11]};
    // u = ω × p + v: the world velocity of the frame origin per unit of DOF c.
    const float u[3] = {om[1] * p[2] - om[2] * p[1] + vl[0], om[2] * p[0] - om[0] * p[2] + vl[1],
                        om[0] * p[1] - om[1] * p[0] + vl[2]};
#pragma unroll
    for (int i = 0; i < 3; ++i) {  // Rᵀ: column i of R is (F[i], F[4 + i], F[8 + i])
      b[s][i] = F[i] * om[0] + F[4 + i] * om[1] + F[8 + i] * om[2];
      b[s][3 + i] = F[i] * u[0] + F[4 + i] * u[1] + F[8 + i] * u[2];
    }
  }
  for (int r = 0; r < m; ++r) {
    const float *row = Jin + static_cast<size_t>(r) * Ji;
    float acc = 0.f;
    for (int s = 0; s < nP; ++s) {
      if (!moves[s]) continue;
      const float *g = row + map.inner_col[map.pose_inner[s]];
#pragma unroll
      for (int k = 0; k < 6; ++k) acc += g[k] * b[s][k];
    }
    Jout[static_cast<size_t>(r) * Jo + c] = acc;
  }
}

std::string SlotList(const std::vector<size_t> &sizes) {
  std::string out = "[";
  for (size_t i = 0; i < sizes.size(); ++i) out += (i ? ", " : "") + std::to_string(sizes[i]);
  return out + "]";
}

}  // namespace

LinkFrameFactorBatch::LinkFrameFactorBatch(const ArticulationTopology &topology, FactorBatch *inner,
                                           const std::vector<int> &pose_slots,
                                           const int *topology_links, const float *frame_offsets)
    : ArticulationFactorBatch(topology, inner != nullptr ? inner->ResidualsSize() : 0,
                              inner != nullptr ? inner->Capacity() : 0),
      inner_(inner),
      pose_slots_(pose_slots),
      topology_links_(topology_links),
      frame_offsets_(frame_offsets) {
  const std::string cls = "LinkFrameFactorBatch: ";
  if (inner == nullptr) {
    throw std::invalid_argument(cls +
                                "inner is null (expected a factor batch over SE(3) "
                                "poses). Fix: pass the factor to wrap.");
  }
  if (topology_links == nullptr) {
    throw std::invalid_argument(cls +
                                "topology_links is null (expected a device array of "
                                "pose_slots.size() ints per factor). Fix: pass the "
                                "topology link of every pose slot.");
  }
  if (dynamic_cast<const ConstraintFactorBatchBase *>(inner) != nullptr) {
    throw std::invalid_argument(
        cls +
        "inner is a constraint batch (got a ConstraintFactorBatchBase, expected a plain "
        "factor: a wrapped constraint is hidden from AugmentedLagrangianMinimizer). Fix: "
        "wrap the plain factor here, then wrap this batch in ConstraintFactorBatch.");
  }
  const auto sizes = inner->StateSizes();
  if (sizes.size() > static_cast<size_t>(kMaxInnerSlots)) {
    throw std::invalid_argument(cls + "the inner factor has " + std::to_string(sizes.size()) +
                                " state slots (expected at most " + std::to_string(kMaxInnerSlots) +
                                "). Fix: wrap a factor with fewer states.");
  }
  if (pose_slots.empty()) {
    throw std::invalid_argument(cls +
                                "pose_slots is empty (expected at least one state slot of "
                                "the inner factor). Fix: list the inner factor's SE(3) "
                                "slots that receive a link frame pose.");
  }
  std::vector<int> seen;
  for (int s : pose_slots) {
    if (s < 0 || static_cast<size_t>(s) >= sizes.size() || sizes[s] != 6) {
      throw std::invalid_argument(
          cls + "pose slot " + std::to_string(s) + " is not an SE(3) slot of the inner factor " +
          "(got state sizes " + SlotList(sizes) + ", expected a slot of size 6). Fix: list only " +
          "the inner factor's SE(3) slots in pose_slots.");
    }
    if (std::find(seen.begin(), seen.end(), s) != seen.end()) {
      throw std::invalid_argument(
          cls + "pose slot " + std::to_string(s) +
          " is listed twice (expected distinct slots). Fix: list each "
          "slot once; two frames need two SE(3) slots in the inner factor.");
    }
    seen.push_back(s);
  }
}

std::vector<size_t> LinkFrameFactorBatch::StateSizes() const {
  std::vector<size_t> out = {static_cast<size_t>(Topology().DofCount())};
  const auto sizes = inner_->StateSizes();
  for (size_t k = 0; k < sizes.size(); ++k) {
    if (std::find(pose_slots_.begin(), pose_slots_.end(), static_cast<int>(k)) ==
        pose_slots_.end()) {
      out.push_back(sizes[k]);
    }
  }
  return out;
}

bool LinkFrameFactorBatch::Evaluate(float *residuals, float *jacobians,
                                    float const *const *state_pointers, cudaStream_t stream,
                                    const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t n = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (n == 0 || num_factors == 0) return true;

  SlotMap map;
  const auto sizes = inner_->StateSizes();
  map.num_inner = static_cast<int>(sizes.size());
  map.num_pose_slots = static_cast<int>(pose_slots_.size());
  for (int s = 0; s < map.num_pose_slots; ++s) map.pose_inner[s] = pose_slots_[s];
  map.num_outer = 1;
  map.outer_cols = Topology().DofCount();
  for (int k = 0; k < map.num_inner; ++k) {
    map.inner_col[k] = map.inner_cols;
    map.inner_cols += static_cast<int>(sizes[k]);
    if (std::find(pose_slots_.begin(), pose_slots_.end(), k) != pose_slots_.end()) {
      map.outer_of_inner[k] = -1;
      continue;
    }
    const int o = map.num_outer++;
    map.outer_of_inner[k] = o;
    map.inner_of_outer[o] = k;
    map.outer_col[o] = map.outer_cols;
    map.outer_size[o] = static_cast<int>(sizes[k]);
    map.outer_cols += static_cast<int>(sizes[k]);
  }

  const size_t m = ResidualsSize();
  poses_.resize(std::max(poses_.size(), n * map.num_pose_slots * 16));
  inner_pointers_.resize(std::max(inner_pointers_.size(), n * map.num_inner));
  const auto view = Topology().View();
  const unsigned blocks = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
  GatherLinkFramesKernel<<<blocks, kBlockSize, 0, stream>>>(
      view, map, topology_links_, frame_offsets_, state_pointers, poses_.data(),
      inner_pointers_.data(), static_cast<int>(n), factor_ids, static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  float *inner_jac = nullptr;
  if (jacobians != nullptr) {
    inner_jacobians_.resize(std::max(inner_jacobians_.size(), n * m * map.inner_cols));
    inner_jac = inner_jacobians_.data();
  }
  if (!inner_->Evaluate(residuals, inner_jac, inner_pointers_.data(), stream, factor_ids,
                        num_factor_ids)) {
    return false;
  }
  if (jacobians == nullptr) return true;
  const size_t threads = n * static_cast<size_t>(map.outer_cols);
  CarryJacobianKernel<<<static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize), kBlockSize,
                        0, stream>>>(
      view, map, static_cast<int>(m), topology_links_, state_pointers, poses_.data(), inner_jac,
      jacobians, static_cast<int>(n), factor_ids, static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
