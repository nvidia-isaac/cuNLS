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
 * Scratch floats per item and frame slot: the view in the layout of its state
 * type (kPose 16, kPosition 3, kOrientation 9). Each view holds what its
 * derivative needs (kPose R and p, kPosition p, kOrientation R), so the
 * Jacobian step reads the frame back from it.
 */
constexpr int kFrameFloats = 16;

/** Tangent size of a view's slot (kPose 6, kPosition and kOrientation 3). */
__host__ __device__ constexpr int ViewTangent(FrameView view) {
  return view == FrameView::kPose ? 6 : 3;
}

/**
 * How the wrapped factor's slots map to this batch's (by value, for kernels):
 * frame slot s is inner slot frame_inner[s] with view frame_view[s]; inner
 * slot k is outer slot outer_of_inner[k] (-1: a frame slot); inner slot k's
 * Jacobian columns start at inner_col[k]; outer slot o (>= 1) starts at
 * outer_col[o].
 */
struct SlotMap {
  int num_inner = 0, num_outer = 0, num_frame_slots = 0;
  int inner_cols = 0, outer_cols = 0;
  int frame_inner[kMaxSlots] = {};
  FrameView frame_view[kMaxSlots] = {};
  int outer_of_inner[kMaxSlots] = {};
  int inner_col[kMaxSlots] = {};
  int inner_of_outer[kMaxSlots] = {};
  int outer_col[kMaxSlots] = {};
};

/**
 * One thread per item: for each frame slot, the view of F_s = X_L X_off =
 * (R, p) into scratch (`frames`, kFrameFloats per slot, in the layout of the
 * view's state type; link -1 is the world; NaN for a link outside the
 * topology), and the inner
 * factor's state pointers (frame slots: the view in scratch; others: this
 * item's pointers).
 */
__global__ void GatherLinkFramesKernel(ArticulationTopology::DeviceView v, SlotMap map,
                                       const int *__restrict__ topology_links,
                                       const float *__restrict__ frame_offsets,
                                       float const *const *__restrict__ state_ptrs,
                                       float *__restrict__ frames, float **__restrict__ inner_ptrs,
                                       int num_items, const int *__restrict__ factor_ids,
                                       int num_factors) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= num_items) return;
  const int f = factor_ids != nullptr ? factor_ids[t] : (t < num_factors ? t : t % num_factors);
  const float *x = state_ptrs[static_cast<size_t>(t) * map.num_outer];
  const int nF = map.num_frame_slots;
  for (int s = 0; s < nF; ++s) {
    const int link = __ldg(topology_links + static_cast<size_t>(f) * nF + s);
    float *F = frames + (static_cast<size_t>(t) * nF + s) * kFrameFloats;
    float R[9], p[3];
    if (link < -1 || link >= v.num_joints) {
      for (int i = 0; i < 9; ++i) R[i] = NAN;
      p[0] = p[1] = p[2] = NAN;
    } else {
      // Link -1 is the world: its pose is the identity, so the frame is X_off itself.
      constexpr float kWorld[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
      const float *P = link >= 0 ? x + v.num_coords + 12 * link : kWorld;  // [R | p] 3x4
      float Ro[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, po[3] = {0, 0, 0};
      if (frame_offsets != nullptr) {
        const float *o = frame_offsets + (static_cast<size_t>(f) * nF + s) * 7;
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
          R[3 * a + b] = P[4 * a] * Ro[b] + P[4 * a + 1] * Ro[3 + b] + P[4 * a + 2] * Ro[6 + b];
        }
        p[a] = P[4 * a] * po[0] + P[4 * a + 1] * po[1] + P[4 * a + 2] * po[2] + P[4 * a + 3];
      }
    }
    switch (map.frame_view[s]) {
      case FrameView::kPose:  // row-major 4x4
#pragma unroll
        for (int a = 0; a < 3; ++a) {
          F[4 * a] = R[3 * a];
          F[4 * a + 1] = R[3 * a + 1];
          F[4 * a + 2] = R[3 * a + 2];
          F[4 * a + 3] = p[a];
        }
        F[12] = F[13] = F[14] = 0.f;
        F[15] = 1.f;
        break;
      case FrameView::kPosition:
#pragma unroll
        for (int i = 0; i < 3; ++i) F[i] = p[i];
        break;
      case FrameView::kOrientation:  // row-major 3x3
#pragma unroll
        for (int i = 0; i < 9; ++i) F[i] = R[i];
        break;
    }
    inner_ptrs[static_cast<size_t>(t) * map.num_inner + map.frame_inner[s]] = F;
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
 * frame slots of the inner columns of the slot times the motion of its view
 * per unit of DOF c (see the class comment); a later column copies the inner
 * column of its passed-through slot. An item with a link outside the topology
 * gets zero columns. Threads of one item write consecutive columns.
 */
__global__ void CarryJacobianKernel(ArticulationTopology::DeviceView v, SlotMap map, int m,
                                    const int *__restrict__ topology_links,
                                    float const *const *__restrict__ state_ptrs,
                                    const float *__restrict__ frames,
                                    const float *__restrict__ inner_jac, float *__restrict__ jac,
                                    int num_items, const int *__restrict__ factor_ids,
                                    int num_factors) {
  const size_t gid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const int Jo = map.outer_cols, Ji = map.inner_cols, T = v.num_dofs;
  const int t = static_cast<int>(gid / Jo), c = static_cast<int>(gid % Jo);
  if (t >= num_items) return;
  const float *Jin = inner_jac + static_cast<size_t>(t) * m * Ji;
  float *Jout = jac + static_cast<size_t>(t) * m * Jo;
  const int f = factor_ids != nullptr ? factor_ids[t] : (t < num_factors ? t : t % num_factors);
  const int nF = map.num_frame_slots;
  for (int s = 0; s < nF; ++s) {
    const int link = __ldg(topology_links + static_cast<size_t>(f) * nF + s);
    if (link < -1 || link >= v.num_joints) {
      // A link outside the topology: the residual is NaN, so the whole Jacobian row block is
      // zero, the inner factor's columns for the other slots included.
      for (int r = 0; r < m; ++r) Jout[static_cast<size_t>(r) * Jo + c] = 0.f;
      return;
    }
  }
  if (c >= T) {  // a passed-through slot
    int o = 1;
    while (o + 1 < map.num_outer && c >= map.outer_col[o + 1]) ++o;
    const int ci = map.inner_col[map.inner_of_outer[o]] + (c - map.outer_col[o]);
    for (int r = 0; r < m; ++r)
      Jout[static_cast<size_t>(r) * Jo + c] = Jin[static_cast<size_t>(r) * Ji + ci];
    return;
  }
  const float *x = state_ptrs[static_cast<size_t>(t) * map.num_outer];
  const float *col = x + v.num_coords + 12 * v.num_joints + 6 * c;
  const float om[3] = {col[0], col[1], col[2]}, vl[3] = {col[3], col[4], col[5]};
  float b[kMaxSlots][6];
  bool moves[kMaxSlots];
  for (int s = 0; s < nF; ++s) {
    const int link = __ldg(topology_links + static_cast<size_t>(f) * nF + s);
    moves[s] = link >= 0 && link < v.num_joints &&
               ((__ldg(v.ancestors + static_cast<size_t>(link) * v.ancestor_words + (c >> 5)) >>
                 (c & 31)) &
                1u);
    if (!moves[s]) continue;
    const float *F = frames + (static_cast<size_t>(t) * nF + s) * kFrameFloats;
    const FrameView view = map.frame_view[s];
    // Row stride of R within the view (4x4 pose: 4; 3x3 orientation: 3).
    const int rs = view == FrameView::kPose ? 4 : 3;
    if (view != FrameView::kOrientation) {
      // u = ω × p + v: the world velocity of the frame origin per unit of DOF c.
      const float p[3] = {view == FrameView::kPose ? F[3] : F[0],
                          view == FrameView::kPose ? F[7] : F[1],
                          view == FrameView::kPose ? F[11] : F[2]};
      const float u[3] = {om[1] * p[2] - om[2] * p[1] + vl[0], om[2] * p[0] - om[0] * p[2] + vl[1],
                          om[0] * p[1] - om[1] * p[0] + vl[2]};
      if (view == FrameView::kPosition) {
#pragma unroll
        for (int i = 0; i < 3; ++i) b[s][i] = u[i];
        continue;
      }
#pragma unroll
      for (int i = 0; i < 3; ++i) {  // Rᵀu: column i of R is (F[i], F[rs + i], F[2 rs + i])
        b[s][3 + i] = F[i] * u[0] + F[rs + i] * u[1] + F[2 * rs + i] * u[2];
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) b[s][i] = F[i] * om[0] + F[rs + i] * om[1] + F[2 * rs + i] * om[2];
  }
  for (int r = 0; r < m; ++r) {
    const float *row = Jin + static_cast<size_t>(r) * Ji;
    float acc = 0.f;
    for (int s = 0; s < nF; ++s) {
      if (!moves[s]) continue;
      const float *g = row + map.inner_col[map.frame_inner[s]];
      const int k_end = ViewTangent(map.frame_view[s]);
      for (int k = 0; k < k_end; ++k) acc += g[k] * b[s][k];
    }
    Jout[static_cast<size_t>(r) * Jo + c] = acc;
  }
}

std::string SlotList(const std::vector<size_t> &sizes) {
  std::string out = "[";
  for (size_t i = 0; i < sizes.size(); ++i) out += (i ? ", " : "") + std::to_string(sizes[i]);
  return out + "]";
}

const char *ViewName(FrameView view) {
  switch (view) {
    case FrameView::kPose:
      return "kPose";
    case FrameView::kPosition:
      return "kPosition";
    case FrameView::kOrientation:
      return "kOrientation";
  }
  return "an unknown view";
}

}  // namespace

LinkFrameFactorBatch::LinkFrameFactorBatch(const ArticulationTopology &topology, FactorBatch *inner,
                                           const std::vector<LinkFrameSlot> &frame_slots,
                                           const int *topology_links, const float *frame_offsets)
    : ArticulationFactorBatch(topology, inner != nullptr ? inner->ResidualsSize() : 0,
                              inner != nullptr ? inner->Capacity() : 0),
      inner_(inner),
      frame_slots_(frame_slots),
      topology_links_(topology_links),
      frame_offsets_(frame_offsets) {
  const std::string cls = "LinkFrameFactorBatch: ";
  if (inner == nullptr) {
    throw std::invalid_argument(cls +
                                "inner is null (expected a factor batch over link frame "
                                "views). Fix: pass the factor to wrap.");
  }
  if (topology_links == nullptr) {
    throw std::invalid_argument(cls +
                                "topology_links is null (expected a device array of "
                                "frame_slots.size() ints per factor). Fix: pass the "
                                "topology link of every frame slot.");
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
  if (frame_slots.empty()) {
    throw std::invalid_argument(cls +
                                "frame_slots is empty (expected at least one state slot of "
                                "the inner factor). Fix: list the inner factor's slots that "
                                "receive a view of a link frame.");
  }
  std::vector<int> seen;
  for (const LinkFrameSlot &slot : frame_slots) {
    const int s = slot.inner_slot;
    const int view = static_cast<int>(slot.view);
    if (view < 0 || view > static_cast<int>(FrameView::kOrientation)) {
      throw std::invalid_argument(cls + "frame slot " + std::to_string(s) + " has view " +
                                  std::to_string(view) +
                                  " (expected kPose, kPosition or kOrientation). Fix: pass a "
                                  "FrameView value.");
    }
    const size_t tangent = static_cast<size_t>(ViewTangent(slot.view));
    if (s < 0 || static_cast<size_t>(s) >= sizes.size() || sizes[s] != tangent) {
      throw std::invalid_argument(
          cls + "frame slot " + std::to_string(s) + " cannot take view " + ViewName(slot.view) +
          " (got inner state sizes " + SlotList(sizes) + ", expected slot " + std::to_string(s) +
          " of size " + std::to_string(tangent) +
          "). Fix: name an inner slot of the view's size: kPose 6 (SE(3)), kPosition 3 (a "
          "3-vector), kOrientation 3 (SO(3)).");
    }
    if (std::find(seen.begin(), seen.end(), s) != seen.end()) {
      throw std::invalid_argument(cls + "frame slot " + std::to_string(s) +
                                  " is listed twice (expected distinct slots). Fix: list each "
                                  "slot once; two frames need two slots in the inner factor.");
    }
    seen.push_back(s);
  }
}

bool LinkFrameFactorBatch::IsFrameSlot(int k) const {
  return std::any_of(frame_slots_.begin(), frame_slots_.end(),
                     [k](const LinkFrameSlot &slot) { return slot.inner_slot == k; });
}

std::vector<size_t> LinkFrameFactorBatch::StateSizes() const {
  std::vector<size_t> out = {static_cast<size_t>(Topology().DofCount())};
  const auto sizes = inner_->StateSizes();
  for (size_t k = 0; k < sizes.size(); ++k) {
    if (!IsFrameSlot(static_cast<int>(k))) out.push_back(sizes[k]);
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
  map.num_frame_slots = static_cast<int>(frame_slots_.size());
  for (int s = 0; s < map.num_frame_slots; ++s) {
    map.frame_inner[s] = frame_slots_[s].inner_slot;
    map.frame_view[s] = frame_slots_[s].view;
  }
  map.num_outer = 1;
  map.outer_cols = Topology().DofCount();
  for (int k = 0; k < map.num_inner; ++k) {
    map.inner_col[k] = map.inner_cols;
    map.inner_cols += static_cast<int>(sizes[k]);
    if (IsFrameSlot(k)) {
      map.outer_of_inner[k] = -1;
      continue;
    }
    const int o = map.num_outer++;
    map.outer_of_inner[k] = o;
    map.inner_of_outer[o] = k;
    map.outer_col[o] = map.outer_cols;
    map.outer_cols += static_cast<int>(sizes[k]);
  }

  const size_t m = ResidualsSize();
  frames_.resize(std::max(frames_.size(), n * map.num_frame_slots * kFrameFloats));
  inner_pointers_.resize(std::max(inner_pointers_.size(), n * map.num_inner));
  const auto view = Topology().View();
  const unsigned blocks = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
  GatherLinkFramesKernel<<<blocks, kBlockSize, 0, stream>>>(
      view, map, topology_links_, frame_offsets_, state_pointers, frames_.data(),
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
  // No output columns (a zero-DOF topology whose inner slots are all frame slots): nothing to
  // carry, and a zero-size launch would fail.
  if (jacobians == nullptr || map.outer_cols == 0) return true;
  const size_t threads = n * static_cast<size_t>(map.outer_cols);
  CarryJacobianKernel<<<static_cast<unsigned>((threads + kBlockSize - 1) / kBlockSize), kBlockSize,
                        0, stream>>>(
      view, map, static_cast<int>(m), topology_links_, state_pointers, frames_.data(), inner_jac,
      jacobians, static_cast<int>(n), factor_ids, static_cast<int>(num_factors));
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
