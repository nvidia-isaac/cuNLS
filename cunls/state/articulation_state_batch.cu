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

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/helper.h"
#include "cunls/state/articulation_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 128;

// Joint type values (ArticulationJointType).
constexpr int kPrismatic = 0, kRevolute = 1, kBall = 2, kFixed = 3, kFree = 4, kD6 = 6;

/** R (row-major 3x3) of the unit quaternion (x, y, z, w). */
__device__ __forceinline__ void QuatToMatrix(const float *q, float *R) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
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

/** q ← normalize(Exp(phi) ⊗ q), quaternions (x, y, z, w), phi a rotation vector. */
__device__ __forceinline__ void QuatLeftExp(const float *phi, float *q) {
  const float theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  const float theta = sqrtf(theta2);
  // e = (sin(θ/2) φ/θ, cos(θ/2)), series below 1e-4.
  const float s = theta < 1e-4f ? 0.5f - theta2 / 48.f : sinf(0.5f * theta) / theta;
  const float c = cosf(0.5f * theta);
  const float ex = s * phi[0], ey = s * phi[1], ez = s * phi[2], ew = c;
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  float r[4] = {ew * x + ex * w + ey * z - ez * y, ew * y - ex * z + ey * w + ez * x,
                ew * z + ex * y - ey * x + ez * w, ew * w - ex * x - ey * y - ez * z};
  const float inv = rsqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = r[i] * inv;
}

/** R (row-major 3x3) = rotation about the unit axis a by angle t. */
__device__ __forceinline__ void AxisAngleMatrix(const float *a, float t, float *R) {
  float s, c;
  sincosf(t, &s, &c);
  const float k = 1.f - c;
  R[0] = c + k * a[0] * a[0];
  R[1] = k * a[0] * a[1] - s * a[2];
  R[2] = k * a[0] * a[2] + s * a[1];
  R[3] = k * a[1] * a[0] + s * a[2];
  R[4] = c + k * a[1] * a[1];
  R[5] = k * a[1] * a[2] - s * a[0];
  R[6] = k * a[2] * a[0] - s * a[1];
  R[7] = k * a[2] * a[1] + s * a[0];
  R[8] = c + k * a[2] * a[2];
}

/** C = A B for row-major 3x3. */
__device__ __forceinline__ void Mul33(const float *A, const float *B, float *C) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
    }
  }
}

/** y = R v for row-major 3x3 R. */
__device__ __forceinline__ void Rot(const float *R, const float *v, float *y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) y[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
}

/** C = A B for 3x4 row-major transforms [R | t] (C may not alias A or B). */
__device__ __forceinline__ void Compose(const float *A, const float *B, float *C) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      C[4 * r + c] = A[4 * r] * B[c] + A[4 * r + 1] * B[4 + c] + A[4 * r + 2] * B[8 + c] +
                     (c == 3 ? A[4 * r + 3] : 0.f);
    }
  }
}

/** Writes the 3x4 transform [R | t] (R row-major 3x3) into M. */
__device__ __forceinline__ void MakeTransform(const float *R, const float *t, float *M) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    M[4 * r] = R[3 * r];
    M[4 * r + 1] = R[3 * r + 1];
    M[4 * r + 2] = R[3 * r + 2];
    M[4 * r + 3] = t[r];
  }
}

/** Motion column (ω, v = p × ω) of a rotation about world axis ω through world point p. */
__device__ __forceinline__ void WriteAngularColumn(const float *w, const float *p, float *col) {
  col[0] = w[0];
  col[1] = w[1];
  col[2] = w[2];
  col[3] = p[1] * w[2] - p[2] * w[1];
  col[4] = p[2] * w[0] - p[0] * w[2];
  col[5] = p[0] * w[1] - p[1] * w[0];
}

/** Motion column (0, v) of a translation along world direction v. */
__device__ __forceinline__ void WriteLinearColumn(const float *v, float *col) {
  col[0] = col[1] = col[2] = 0.f;
  col[3] = v[0];
  col[4] = v[1];
  col[5] = v[2];
}

/**
 * Kinematics of joint j of one instance: from the parent's world pose
 * (3x4) and the joint's coordinates q, writes the child's world pose (12
 * floats) to `pose` and the world motion columns of the joint's DOFs (6
 * floats each, in DOF order) to `cols`. See ArticulationTopology.
 */
__device__ void JointKinematics(const ArticulationTopology::DeviceView &v, int j,
                                const float *parent_pose, const float *q, float *pose,
                                float *cols) {
  float A[12];
  Compose(parent_pose, v.X_p + 12 * j, A);
  const float RA[9] = {A[0], A[1], A[2], A[4], A[5], A[6], A[8], A[9], A[10]};
  const float pA[3] = {A[3], A[7], A[11]};
  const int d0 = v.qd_start[j];
  float Rj[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
  float tj[3] = {0.f, 0.f, 0.f};
  switch (v.type[j]) {
    case kPrismatic: {
      const float *a = v.axis + 3 * d0;
      tj[0] = a[0] * q[0];
      tj[1] = a[1] * q[0];
      tj[2] = a[2] * q[0];
      float w[3];
      Rot(RA, a, w);
      WriteLinearColumn(w, cols);
      break;
    }
    case kRevolute: {
      const float *a = v.axis + 3 * d0;
      AxisAngleMatrix(a, q[0], Rj);
      float w[3];
      Rot(RA, a, w);
      WriteAngularColumn(w, pA, cols);
      break;
    }
    case kBall: {
      QuatToMatrix(q, Rj);
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const float w[3] = {RA[i], RA[3 + i], RA[6 + i]};  // R_A e_i
        WriteAngularColumn(w, pA, cols + 6 * i);
      }
      break;
    }
    case kFree: {
      tj[0] = q[0];
      tj[1] = q[1];
      tj[2] = q[2];
      QuatToMatrix(q + 3, Rj);
      float p[3];
      Rot(RA, tj, p);
      p[0] += pA[0];
      p[1] += pA[1];
      p[2] += pA[2];
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        const float e[3] = {RA[i], RA[3 + i], RA[6 + i]};
        WriteLinearColumn(e, cols + 6 * i);
        WriteAngularColumn(e, p, cols + 6 * (3 + i));
      }
      break;
    }
    case kD6: {
      const int lin = v.dof_dim[2 * j], ang = v.dof_dim[2 * j + 1];
      for (int i = 0; i < lin; ++i) {
        const float *a = v.axis + 3 * (d0 + i);
        tj[0] += a[0] * q[i];
        tj[1] += a[1] * q[i];
        tj[2] += a[2] * q[i];
        float w[3];
        Rot(RA, a, w);
        WriteLinearColumn(w, cols + 6 * i);
      }
      // Angular axes pivot about the translated origin.
      float pivot[3];
      Rot(RA, tj, pivot);
      pivot[0] += pA[0];
      pivot[1] += pA[1];
      pivot[2] += pA[2];
      // Intrinsic chain: a_k = R_acc axis_k, R_acc ← R(a_k, q_k) R_acc.
      for (int i = 0; i < ang; ++i) {
        float a[3];
        Rot(Rj, v.axis + 3 * (d0 + lin + i), a);
        float Rk[9], Rn[9];
        AxisAngleMatrix(a, q[lin + i], Rk);
        Mul33(Rk, Rj, Rn);
#pragma unroll
        for (int e = 0; e < 9; ++e) Rj[e] = Rn[e];
        float w[3];
        Rot(RA, a, w);
        WriteAngularColumn(w, pivot, cols + 6 * (lin + i));
      }
      break;
    }
    default:  // kFixed
      break;
  }
  float Xj[12], AXj[12];
  MakeTransform(Rj, tj, Xj);
  Compose(A, Xj, AXj);
  Compose(AXj, v.X_c_inv + 12 * j, pose);
}

/**
 * The coordinates of joint j after the step: x ⊕ δ per joint type (δ null:
 * x's coordinates), moved scalar coordinates clamped into [lo, hi] (null:
 * no bounds). Writes q_start[j+1] - q_start[j] floats to `q`.
 */
__device__ void JointCoordinates(const ArticulationTopology::DeviceView &v, int j, const float *xs,
                                 const float *ds, const float *lo, const float *hi, float *q) {
  const int q0 = v.q_start[j], d0 = v.qd_start[j];
  const int nc = v.q_start[j + 1] - q0;
  if (ds == nullptr) {
    for (int i = 0; i < nc; ++i) q[i] = xs[q0 + i];
    return;
  }
  switch (v.type[j]) {
    case kBall: {
#pragma unroll
      for (int i = 0; i < 4; ++i) q[i] = xs[q0 + i];
      QuatLeftExp(ds + d0, q);
      break;
    }
    case kFree: {
#pragma unroll
      for (int i = 0; i < 3; ++i) q[i] = xs[q0 + i] + ds[d0 + i];
#pragma unroll
      for (int i = 0; i < 4; ++i) q[3 + i] = xs[q0 + 3 + i];
      QuatLeftExp(ds + d0 + 3, q + 3);
      break;
    }
    case kFixed:
      break;
    default: {  // scalar coordinates
      for (int i = 0; i < nc; ++i) {
        const float d = ds[d0 + i];
        float value = xs[q0 + i] + d;
        if (lo != nullptr && d != 0.f) value = fminf(fmaxf(value, lo[d0 + i]), hi[d0 + i]);
        q[i] = value;
      }
      break;
    }
  }
}

/** Shared ints of the staged topology: order, type, parent, child (J each), q/qd_start (J+1),
 * dof_dim (2J). */
__host__ __device__ constexpr int TopologyIntCount(int J) { return 8 * J + 2; }

/** Shared floats of the staged topology: X_p, X_c_inv (12 J each), axis (3 T), root (12), padded
 * to 4. */
__host__ __device__ constexpr int TopologyFloatCount(int J, int T) {
  return ((24 * J + 3 * T + 12) + 3) / 4 * 4;
}

/**
 * One thread per instance of `num_states` (replicas included). The thread
 * builds its whole instance in a shared-memory row (stride `stride`, odd, so
 * rows sit on distinct banks), joint by joint in parents-first order: the
 * joint's coordinates (x ⊕ δ; δ null: x's; moved scalar coordinates clamped
 * into [lower, upper], bounds entry (s % bound_states) * T + k), the child's
 * world pose and the joint's motion columns. The block then writes its
 * instances, contiguous in `out`, with coalesced stores (full sectors).
 * x and out may be the same array only when delta is null. The topology's
 * arrays are staged in shared memory once per block.
 */
__global__ void ArticulationPlusKernel(ArticulationTopology::DeviceView v, const float *x,
                                       const float *delta, float *out, size_t num_states,
                                       const float *lower, const float *upper, size_t bound_states,
                                       int stride) {
  extern __shared__ float shared[];
  const int J = v.num_joints, T = v.num_dofs, A = v.ambient, C = v.num_coords;
  int *ints = reinterpret_cast<int *>(shared);
  float *floats = shared + TopologyIntCount(J);
  ArticulationTopology::DeviceView view = v;
  view.order = ints;
  view.type = ints + J;
  view.parent = ints + 2 * J;
  view.child = ints + 3 * J;
  view.q_start = ints + 4 * J;
  view.qd_start = ints + 5 * J + 1;
  view.dof_dim = ints + 6 * J + 2;
  view.X_p = floats;
  view.X_c_inv = floats + 12 * J;
  view.axis = floats + 24 * J;
  view.root = floats + 24 * J + 3 * T;
  for (int i = threadIdx.x; i < J; i += blockDim.x) {
    ints[i] = v.order[i];
    ints[J + i] = v.type[i];
    ints[2 * J + i] = v.parent[i];
    ints[3 * J + i] = v.child[i];
    ints[6 * J + 2 + 2 * i] = v.dof_dim[2 * i];
    ints[6 * J + 3 + 2 * i] = v.dof_dim[2 * i + 1];
  }
  for (int i = threadIdx.x; i <= J; i += blockDim.x) {
    ints[4 * J + i] = v.q_start[i];
    ints[5 * J + 1 + i] = v.qd_start[i];
  }
  for (int i = threadIdx.x; i < 12 * J; i += blockDim.x) {
    floats[i] = v.X_p[i];
    floats[12 * J + i] = v.X_c_inv[i];
  }
  for (int i = threadIdx.x; i < 3 * T; i += blockDim.x) floats[24 * J + i] = v.axis[i];
  for (int i = threadIdx.x; i < 12; i += blockDim.x) floats[24 * J + 3 * T + i] = v.root[i];
  __syncthreads();
  float *staged = floats + TopologyFloatCount(J, T);

  const size_t first = blockIdx.x * static_cast<size_t>(blockDim.x);
  const size_t s = first + threadIdx.x;
  if (s < num_states) {
    const float *xs = x + s * A;
    float *row = staged + threadIdx.x * stride;
    const float *ds = delta != nullptr ? delta + s * T : nullptr;
    const float *lo = lower != nullptr ? lower + (s % bound_states) * T : nullptr;
    const float *hi = upper != nullptr ? upper + (s % bound_states) * T : nullptr;
    for (int k = 0; k < J; ++k) {
      const int j = view.order[k];
      const int q0 = view.q_start[j], parent = view.parent[j];
      JointCoordinates(view, j, xs, ds, lo, hi, row + q0);
      JointKinematics(view, j, parent < 0 ? view.root : row + C + 12 * parent, row + q0,
                      row + C + 12 * view.child[j], row + C + 12 * J + 6 * view.qd_start[j]);
    }
  }
  __syncthreads();
  // Coalesced copy of the block's instances (contiguous in out).
  const size_t count = min(static_cast<size_t>(blockDim.x), num_states - min(first, num_states));
  float *dst = out + first * A;
  const int total = static_cast<int>(count) * A;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    dst[i] = staged[(i / A) * stride + i % A];
  }
}

/** Writes the zero configuration (scalars 0, identity quaternions) into `num_states` instances. */
__global__ void ZeroConfigurationKernel(ArticulationTopology::DeviceView v, float *x,
                                        size_t num_states) {
  const size_t s = blockIdx.x * static_cast<size_t>(kBlockSize) + threadIdx.x;
  if (s >= num_states) return;
  float *xs = x + s * v.ambient;
  for (int i = 0; i < v.num_coords; ++i) xs[i] = 0.f;
  for (int j = 0; j < v.num_joints; ++j) {
    if (v.type[j] == kBall) xs[v.q_start[j] + 3] = 1.f;
    if (v.type[j] == kFree) xs[v.q_start[j] + 6] = 1.f;
  }
}

unsigned Blocks(size_t n) { return static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize); }

/** Launches ArticulationPlusKernel with as many instances per block as fit 48 KB of shared memory.
 */
void LaunchPlus(const ArticulationTopology::DeviceView &v, const float *x, const float *delta,
                float *out, size_t num_states, const float *lower, const float *upper,
                size_t bound_states, cudaStream_t stream) {
  if (num_states == 0) return;
  const int stride = v.ambient | 1;  // odd: per-thread rows on distinct banks
  const size_t topology_bytes = sizeof(float) * (TopologyIntCount(v.num_joints) +
                                                 TopologyFloatCount(v.num_joints, v.num_dofs));
  constexpr size_t kSharedBytes = 48 * 1024;
  int per_block = static_cast<int>((kSharedBytes - topology_bytes) / (sizeof(float) * stride));
  per_block = per_block >= 64 ? 64 : (per_block >= 32 ? 32 : (per_block > 0 ? per_block : 1));
  const unsigned blocks = static_cast<unsigned>((num_states + per_block - 1) / per_block);
  ArticulationPlusKernel<<<blocks, per_block, topology_bytes + sizeof(float) * stride * per_block,
                           stream>>>(v, x, delta, out, num_states, lower, upper, bound_states,
                                     stride);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace

ArticulationStateBatch::ArticulationStateBatch(const ArticulationTopology &topology,
                                               size_t capacity, bool apply_joint_limits)
    : topology_(topology), capacity_(capacity) {
  Initialize(apply_joint_limits);
}

ArticulationStateBatch::ArticulationStateBatch(const ArticulationTopology &topology,
                                               size_t capacity,
                                               const int *device_constant_state_ids,
                                               size_t const_capacity, bool apply_joint_limits)
    : topology_(topology),
      capacity_(capacity),
      const_ids_(device_constant_state_ids),
      const_capacity_(const_capacity) {
  Initialize(apply_joint_limits);
}

void ArticulationStateBatch::Initialize(bool apply_joint_limits) {
  storage_.resize(capacity_ * AmbientSize());
  if (capacity_ > 0) {
    ZeroConfigurationKernel<<<Blocks(capacity_), kBlockSize>>>(topology_.View(), storage_.data(),
                                                               capacity_);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    LaunchPlus(topology_.View(), storage_.data(), nullptr, storage_.data(), capacity_, nullptr,
               nullptr, 1, nullptr);
  }
  if (topology_.HasLimits() && capacity_ > 0) {
    const size_t T = TangentSize();
    std::vector<float> lo(capacity_ * T), hi(capacity_ * T);
    for (size_t s = 0; s < capacity_; ++s) {
      std::copy(topology_.LowerLimits().begin(), topology_.LowerLimits().end(), lo.begin() + s * T);
      std::copy(topology_.UpperLimits().begin(), topology_.UpperLimits().end(), hi.begin() + s * T);
    }
    limit_lower_ = dvector<float>(lo);
    limit_upper_ = dvector<float>(hi);
  }
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());  // default-stream initialization done
  if (apply_joint_limits) UseJointLimits();
}

void ArticulationStateBatch::UseJointLimits() {
  lower_ = limit_lower_.size() > 0 ? limit_lower_.data() : nullptr;
  upper_ = limit_upper_.size() > 0 ? limit_upper_.data() : nullptr;
}

void ArticulationStateBatch::SetBounds(const float *lower, const float *upper) {
  if ((lower == nullptr) != (upper == nullptr)) {
    throw std::invalid_argument(
        "ArticulationStateBatch::SetBounds: got one bound and nullptr (expected both arrays, or "
        "both nullptr to remove the bounds). Fix: pass -inf / +inf for an unbounded side.");
  }
  lower_ = lower;
  upper_ = upper;
}

void ArticulationStateBatch::SetNumActiveStates(size_t num_active_states, size_t num_const_states) {
  if (num_active_states > capacity_ || num_const_states > const_capacity_) {
    throw std::invalid_argument(
        "ArticulationStateBatch::SetNumActiveStates: size exceeds the capacity (got " +
        std::to_string(num_active_states) + " states / " + std::to_string(num_const_states) +
        " constant, expected at most " + std::to_string(capacity_) + " / " +
        std::to_string(const_capacity_) + "). Fix: construct the batch with a larger capacity.");
  }
  num_active_states_ = num_active_states;
  num_const_states_ = num_const_states;
}

float *ArticulationStateBatch::StateDevicePtr(size_t state_idx) {
  return state_idx < capacity_ ? storage_.data() + state_idx * AmbientSize() : nullptr;
}

const float *ArticulationStateBatch::StateDevicePtr(size_t state_idx) const {
  return state_idx < capacity_ ? storage_.data() + state_idx * AmbientSize() : nullptr;
}

void ArticulationStateBatch::Plus(const float *x, const float *delta, float *x_plus_delta,
                                  cudaStream_t stream, size_t num_replicas) {
  const size_t n = num_active_states_ * num_replicas;
  if (n == 0) return;
  LaunchPlus(topology_.View(), x, delta, x_plus_delta, n, lower_, upper_,
             num_active_states_ > 0 ? num_active_states_ : 1, stream);
}

void ArticulationStateBatch::PlusIgnoringBounds(const float *x, const float *delta,
                                                float *x_plus_delta, cudaStream_t stream) {
  if (num_active_states_ == 0) return;
  LaunchPlus(topology_.View(), x, delta, x_plus_delta, num_active_states_, nullptr, nullptr, 1,
             stream);
}

void ArticulationStateBatch::RefreshDerivedData(cudaStream_t stream) {
  if (num_active_states_ == 0) return;
  LaunchPlus(topology_.View(), storage_.data(), nullptr, storage_.data(), num_active_states_,
             nullptr, nullptr, 1, stream);
}

void ArticulationStateBatch::SetJointQ(const float *joint_q, size_t num_states,
                                       cudaStream_t stream) {
  if (num_states > capacity_) {
    throw std::invalid_argument("ArticulationStateBatch::SetJointQ: got " +
                                std::to_string(num_states) + " states, expected at most " +
                                std::to_string(capacity_) +
                                " (the capacity). Fix: pass fewer states or a larger capacity.");
  }
  if (num_states == 0) return;
  const size_t C = topology_.CoordCount(), A = AmbientSize();
  if (C > 0) {
    THROW_ON_CUDA_ERROR(cudaMemcpy2DAsync(storage_.data(), A * sizeof(float), joint_q,
                                          C * sizeof(float), C * sizeof(float), num_states,
                                          cudaMemcpyDeviceToDevice, stream));
  }
  LaunchPlus(topology_.View(), storage_.data(), nullptr, storage_.data(), num_states, nullptr,
             nullptr, 1, stream);
}

void ArticulationStateBatch::GetJointQ(float *joint_q, size_t num_states,
                                       cudaStream_t stream) const {
  if (num_states > capacity_) {
    throw std::invalid_argument("ArticulationStateBatch::GetJointQ: got " +
                                std::to_string(num_states) + " states, expected at most " +
                                std::to_string(capacity_) + " (the capacity). Fix: pass fewer.");
  }
  const size_t C = topology_.CoordCount(), A = AmbientSize();
  if (num_states == 0 || C == 0) return;
  THROW_ON_CUDA_ERROR(cudaMemcpy2DAsync(joint_q, C * sizeof(float), storage_.data(),
                                        A * sizeof(float), C * sizeof(float), num_states,
                                        cudaMemcpyDeviceToDevice, stream));
}

}  // namespace cunls
