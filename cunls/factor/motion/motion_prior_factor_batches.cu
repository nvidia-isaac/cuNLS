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

/**
 * @file motion_prior_factor_batches.cu
 * @brief The constant-velocity and constant-acceleration motion priors of
 * SO(2), SO(3), SE(2) and SE(3): one fused kernel, templated on the group and
 * the order (2: velocity, 3: acceleration).
 *
 * States per item: pose_k, pose_{k+1}, vel_k, vel_{k+1} (and accel_k,
 * accel_{k+1}); with twist = Log(pose_k^{-1} pose_{k+1}), J_l^{-1} and J_r^{-1}
 * of the twist and per-factor dt:
 *   r_pose  = twist - dt vel_k (- dt²/2 accel_k)
 *   r_vel   = J_l^{-1} vel_{k+1} - vel_k (- dt accel_k)
 *   r_accel = J_l^{-1} accel_{k+1} - accel_k
 * Jacobian columns [pose_k, pose_{k+1}, vel_k, vel_{k+1}(, accel_k,
 * accel_{k+1})]: ∂r_pose/∂pose_k = -J_l^{-1}, ∂r_pose/∂pose_{k+1} = J_r^{-1};
 * the pose blocks of r_vel and r_accel are treated as zero (the documented
 * simplification of the class comments).
 */

#include "cunls/common/helper.h"
#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"
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

// Group policies: kDim (tangent size), Twist (Log(L^{-1} R) of two pose
// states) and the D x D J_l^{-1} / J_r^{-1} of a twist.

/** SO(2), 2x2 rotations: abelian, J_l^{-1} = J_r^{-1} = 1. */
struct SO2Group {
  static constexpr int kDim = 1;
  __device__ static void Twist(const float *L, const float *R, float *tw) {
    // L^T R; Log = atan2 of its column 0.
    const float c = L[0] * R[0] + L[2] * R[2];
    const float s = L[1] * R[0] + L[3] * R[2];
    tw[0] = atan2f(s, c);
  }
  __device__ static void LeftInverse(const float *, float *J) { J[0] = 1.f; }
  __device__ static void RightInverse(const float *, float *J) { J[0] = 1.f; }
};

/** SO(3), row-major 3x3 rotations. */
struct SO3Group {
  static constexpr int kDim = 3;
  __device__ static void Twist(const float *L, const float *R, float *tw) {
    float rel[9];  // L^T R
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 3; k++) acc += L[k * 3 + i] * R[k * 3 + j];
        rel[i * 3 + j] = acc;
      }
    }
    lie_device::LogSO3(rel, 3, tw);
  }
  __device__ static void LeftInverse(const float *tw, float *J) {
    lie_device::SO3JacobianLeftInverse(tw, J, 3);
  }
  __device__ static void RightInverse(const float *tw, float *J) {
    const float m[3] = {-tw[0], -tw[1], -tw[2]};  // J_r^{-1}(x) = J_l^{-1}(-x)
    lie_device::SO3JacobianLeftInverse(m, J, 3);
  }
};

/** SE(2), row-major 3x3 transforms. */
struct SE2Group {
  static constexpr int kDim = 3;
  __device__ static void Twist(const float *L, const float *R, float *tw) {
    const float l00 = L[0], l01 = L[1], l02 = L[2];
    const float l10 = L[3], l11 = L[4], l12 = L[5];
    const float i02 = -(l00 * l02 + l10 * l12);
    const float i12 = -(l01 * l02 + l11 * l12);
    const float r00 = R[0], r01 = R[1], r02 = R[2];
    const float r10 = R[3], r11 = R[4], r12 = R[5];
    const float rel[9] = {l00 * r00 + l10 * r10,
                          l00 * r01 + l10 * r11,
                          l00 * r02 + l10 * r12 + i02,
                          l01 * r00 + l11 * r10,
                          l01 * r01 + l11 * r11,
                          l01 * r02 + l11 * r12 + i12,
                          0.0f,
                          0.0f,
                          1.0f};
    lie_device::LogSE2(rel, tw);
  }
  __device__ static void LeftInverse(const float *tw, float *J) {
    lie_device::SE2JrInv(-tw[0], -tw[1], -tw[2], J);  // J_l^{-1}(x) = J_r^{-1}(-x)
  }
  __device__ static void RightInverse(const float *tw, float *J) {
    lie_device::SE2JrInv(tw[0], tw[1], tw[2], J);
  }
};

/** SE(3), row-major 4x4 transforms (last row [0 0 0 1] exploited). */
struct SE3Group {
  static constexpr int kDim = 6;
  __device__ static void Twist(const float *L, const float *R, float *tw) {
    const float l00 = L[0], l01 = L[1], l02 = L[2], l03 = L[3];
    const float l10 = L[4], l11 = L[5], l12 = L[6], l13 = L[7];
    const float l20 = L[8], l21 = L[9], l22 = L[10], l23 = L[11];
    const float i03 = -(l00 * l03 + l10 * l13 + l20 * l23);
    const float i13 = -(l01 * l03 + l11 * l13 + l21 * l23);
    const float i23 = -(l02 * l03 + l12 * l13 + l22 * l23);
    const float r00 = R[0], r01 = R[1], r02 = R[2], r03 = R[3];
    const float r10 = R[4], r11 = R[5], r12 = R[6], r13 = R[7];
    const float r20 = R[8], r21 = R[9], r22 = R[10], r23 = R[11];
    const float rel[12] = {
        l00 * r00 + l10 * r10 + l20 * r20, l00 * r01 + l10 * r11 + l20 * r21,
        l00 * r02 + l10 * r12 + l20 * r22, l00 * r03 + l10 * r13 + l20 * r23 + i03,
        l01 * r00 + l11 * r10 + l21 * r20, l01 * r01 + l11 * r11 + l21 * r21,
        l01 * r02 + l11 * r12 + l21 * r22, l01 * r03 + l11 * r13 + l21 * r23 + i13,
        l02 * r00 + l12 * r10 + l22 * r20, l02 * r01 + l12 * r11 + l22 * r21,
        l02 * r02 + l12 * r12 + l22 * r22, l02 * r03 + l12 * r13 + l22 * r23 + i23};
    lie_device::LogSE3(rel, 4, tw);  // reads rows 0..2 only
  }
  __device__ static void LeftInverse(const float *tw, float *J) {
    lie_device::SE3JacobianLeftInverse(tw, J, 6);
  }
  __device__ static void RightInverse(const float *tw, float *J) {
    const float m[6] = {-tw[0], -tw[1], -tw[2], -tw[3], -tw[4], -tw[5]};
    lie_device::SE3JacobianLeftInverse(m, J, 6);  // J_r^{-1}(x) = J_l^{-1}(-x)
  }
};

constexpr int kWarpsPerBlock = 4;
constexpr int kBlockSize = 32 * kWarpsPerBlock;

/**
 * One thread per item, 32 consecutive items per warp. Residual rows and the
 * kOrder*D x 2*kOrder*D Jacobian are written one row at a time through a
 * per-warp staging buffer (lie_device::WriteWarpRows: coalesced stores). The
 * row loops are fully unrolled so the D x D matrices stay in registers.
 */
template <class G, int kOrder, bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    MotionPriorKernel(const float *__restrict__ dt_ptr,
                      float const *const *__restrict__ state_pointers,
                      float *__restrict__ residuals, float *__restrict__ jacobians, int num_items,
                      const int *__restrict__ factor_ids, int num_factors) {
  constexpr int D = G::kDim;
  constexpr int kRows = kOrder * D;
  constexpr int kCols = 2 * kOrder * D;
  constexpr int kRowStride = kCols | 1;  // odd: no bank conflicts
  __shared__ float stage_all[kWarpsPerBlock * 32 * kRowStride];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float *stage = stage_all + warp * 32 * kRowStride;
  const int warp_first = blockIdx.x * kBlockSize + warp * 32;
  const int count = min(32, num_items - warp_first);
  if (count <= 0) return;  // whole warp past the end
  const int t = warp_first + lane;

  // --- Inputs (zeros for lanes past the end; their outputs are not written) ---
  float tw[D] = {}, vel_k[D] = {}, vel_k1[D] = {}, acc_k[D] = {}, acc_k1[D] = {};
  float dt = 0.f;
  if (lane < count) {
    float const *const *s = state_pointers + 2 * kOrder * static_cast<size_t>(t);
    G::Twist(s[0], s[1], tw);
#pragma unroll
    for (int i = 0; i < D; ++i) {
      vel_k[i] = s[2][i];
      vel_k1[i] = s[3][i];
      if constexpr (kOrder == 3) {
        acc_k[i] = s[4][i];
        acc_k1[i] = s[5][i];
      }
    }
    dt = dt_ptr[FactorMeasurementIndex(t, factor_ids, num_factors)];
  }
  const float half_dt2 = 0.5f * dt * dt;

  // --- Residual ---
  float Jli[D * D];
  G::LeftInverse(tw, Jli);
  float res[kRows];
#pragma unroll
  for (int i = 0; i < D; i++) {
    if constexpr (kOrder == 3) {
      res[i] = tw[i] - dt * vel_k[i] - half_dt2 * acc_k[i];
      float acc_vel = -vel_k[i] - dt * acc_k[i];
      float acc_accel = -acc_k[i];
#pragma unroll
      for (int j = 0; j < D; j++) {
        acc_vel += Jli[i * D + j] * vel_k1[j];
        acc_accel += Jli[i * D + j] * acc_k1[j];
      }
      res[D + i] = acc_vel;
      res[2 * D + i] = acc_accel;
    } else {
      res[i] = tw[i] - dt * vel_k[i];
      float acc = -vel_k[i];
#pragma unroll
      for (int j = 0; j < D; j++) acc += Jli[i * D + j] * vel_k1[j];
      res[D + i] = acc;
    }
  }
  lie_device::WriteWarpRows<kCols, kRowStride>(
      stage, lane, res, kRows, count, residuals + static_cast<size_t>(warp_first) * kRows, kRows);
  if constexpr (!kJacobian) return;

  // --- Jacobian, one row at a time ---
  float Jri[D * D];
  G::RightInverse(tw, Jri);
  float *jac_base = jacobians + static_cast<size_t>(warp_first) * kRows * kCols;
  float row[kCols];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int block = r / D, i = r % D;
#pragma unroll
    for (int j = 0; j < kCols; ++j) row[j] = 0.f;
#pragma unroll
    for (int j = 0; j < D; ++j) {
      if (block == 0) {  // r_pose
        row[j] = -Jli[i * D + j];
        row[D + j] = Jri[i * D + j];
        row[2 * D + j] = i == j ? -dt : 0.f;
        if constexpr (kOrder == 3) row[4 * D + j] = i == j ? -half_dt2 : 0.f;
      } else if (block == 1) {  // r_vel
        row[2 * D + j] = i == j ? -1.f : 0.f;
        row[3 * D + j] = Jli[i * D + j];
        if constexpr (kOrder == 3) row[4 * D + j] = i == j ? -dt : 0.f;
      } else {  // r_accel
        row[4 * D + j] = i == j ? -1.f : 0.f;
        row[5 * D + j] = Jli[i * D + j];
      }
    }
    lie_device::WriteWarpRows<kCols, kRowStride>(stage, lane, row, kCols, count,
                                                 jac_base + r * kCols, kRows * kCols);
  }
}

/** Launches MotionPriorKernel<G, kOrder> over `num_items` items (FactorBatch item contract). */
template <class G, int kOrder>
bool EvaluateMotionPrior(const float *dt_ptr, size_t num_active_factors, float *residuals,
                         float *jacobians, float const *const *state_pointers, cudaStream_t stream,
                         const int *factor_ids, size_t num_factor_ids) {
  const size_t num_items = num_factor_ids == 0 ? num_active_factors : num_factor_ids;
  if (num_items == 0 || num_active_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((num_items + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    MotionPriorKernel<G, kOrder, true><<<blocks, kBlockSize, 0, stream>>>(
        dt_ptr, state_pointers, residuals, jacobians, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_active_factors));
  } else {
    MotionPriorKernel<G, kOrder, false><<<blocks, kBlockSize, 0, stream>>>(
        dt_ptr, state_pointers, residuals, nullptr, static_cast<int>(num_items), factor_ids,
        static_cast<int>(num_active_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Constant velocity
// ---------------------------------------------------------------------------

ConstantVelocitySO2FactorBatch::ConstantVelocitySO2FactorBatch(const float *dt_ptr, size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantVelocitySO2FactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  return EvaluateMotionPrior<SO2Group, 2>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantVelocitySO3FactorBatch::ConstantVelocitySO3FactorBatch(const float *dt_ptr, size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantVelocitySO3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  return EvaluateMotionPrior<SO3Group, 2>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantVelocitySE2FactorBatch::ConstantVelocitySE2FactorBatch(const float *dt_ptr, size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantVelocitySE2FactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  return EvaluateMotionPrior<SE2Group, 2>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantVelocitySE3FactorBatch::ConstantVelocitySE3FactorBatch(const float *dt_ptr, size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantVelocitySE3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                              float const *const *state_pointers,
                                              cudaStream_t stream, const int *factor_ids,
                                              size_t num_factor_ids) const {
  return EvaluateMotionPrior<SE3Group, 2>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

// ---------------------------------------------------------------------------
// Constant acceleration
// ---------------------------------------------------------------------------

ConstantAccelerationSO2FactorBatch::ConstantAccelerationSO2FactorBatch(const float *dt_ptr,
                                                                       size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantAccelerationSO2FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream, const int *factor_ids,
                                                  size_t num_factor_ids) const {
  return EvaluateMotionPrior<SO2Group, 3>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantAccelerationSO3FactorBatch::ConstantAccelerationSO3FactorBatch(const float *dt_ptr,
                                                                       size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantAccelerationSO3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream, const int *factor_ids,
                                                  size_t num_factor_ids) const {
  return EvaluateMotionPrior<SO3Group, 3>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantAccelerationSE2FactorBatch::ConstantAccelerationSE2FactorBatch(const float *dt_ptr,
                                                                       size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantAccelerationSE2FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream, const int *factor_ids,
                                                  size_t num_factor_ids) const {
  return EvaluateMotionPrior<SE2Group, 3>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

ConstantAccelerationSE3FactorBatch::ConstantAccelerationSE3FactorBatch(const float *dt_ptr,
                                                                       size_t capacity)
    : SizedFactorBatch(capacity), dt_ptr_(dt_ptr) {}

bool ConstantAccelerationSE3FactorBatch::Evaluate(float *residuals, float *jacobians,
                                                  float const *const *state_pointers,
                                                  cudaStream_t stream, const int *factor_ids,
                                                  size_t num_factor_ids) const {
  return EvaluateMotionPrior<SE3Group, 3>(dt_ptr_, NumActiveFactors(), residuals, jacobians,
                                          state_pointers, stream, factor_ids, num_factor_ids);
}

}  // namespace cunls
