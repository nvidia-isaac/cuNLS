/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/math/lie_device.cuh"
#include "cunls/math/sim_lie_math.h"

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

constexpr size_t kBlockSize = 256;

constexpr size_t kSim3TransformStride = 16;
constexpr size_t kSim3TangentStride = 7;
constexpr size_t kSim3JacobianPitch = 14;
constexpr size_t kSim3JacobianStride = 98;

/**
 * @brief Fused kernel: collect L/R Sim(3), compute (L^{-1} * R) * Delta.
 *
 * Full 4x4 row-major multiply (128 FMAs for 2 matmuls). Inline Sim(3) inverse.
 */
__global__ void collect_and_compute_sim3_between_error_kernel(float const *const *state_pointers,
                                                              const Matrix<4> *deltas,
                                                              size_t num_items, Matrix<4> *errors,
                                                              const int *factor_ids,
                                                              int num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_items) return;
  const int m = FactorMeasurementIndex(tid, factor_ids, num_factors);

  const float *__restrict__ L = state_pointers[2 * tid];
  const float *__restrict__ R = state_pointers[2 * tid + 1];
  const float *__restrict__ D = deltas[m].data();
  float *__restrict__ out = errors[tid].data();

  // Load L: 4x4, bottom-right = 1/s
  const float l00 = L[0], l01 = L[1], l02 = L[2], l03 = L[3];
  const float l10 = L[4], l11 = L[5], l12 = L[6], l13 = L[7];
  const float l20 = L[8], l21 = L[9], l22 = L[10], l23 = L[11];
  const float l_inv_s = L[15];
  const float s = 1.0f / l_inv_s;

  // L = [R t; 0 1/s]  =>  L^{-1} = [R^T, -s*R^T*t; 0, s]
  const float i00 = l00, i01 = l10, i02 = l20;
  const float i10 = l01, i11 = l11, i12 = l21;
  const float i20 = l02, i21 = l12, i22 = l22;
  const float i03 = -s * (i00 * l03 + i01 * l13 + i02 * l23);
  const float i13 = -s * (i10 * l03 + i11 * l13 + i12 * l23);
  const float i23 = -s * (i20 * l03 + i21 * l13 + i22 * l23);
  const float i33 = s;

  // Load R
  const float r0 = R[0], r1 = R[1], r2 = R[2], r3 = R[3];
  const float r4 = R[4], r5 = R[5], r6 = R[6], r7 = R[7];
  const float r8 = R[8], r9 = R[9], r10 = R[10], r11 = R[11];
  const float r12 = R[12], r13 = R[13], r14 = R[14], r15 = R[15];

  // temp = L^{-1} * R  (full 4x4)
  const float t00 = i00 * r0 + i01 * r4 + i02 * r8 + i03 * r12;
  const float t01 = i00 * r1 + i01 * r5 + i02 * r9 + i03 * r13;
  const float t02 = i00 * r2 + i01 * r6 + i02 * r10 + i03 * r14;
  const float t03 = i00 * r3 + i01 * r7 + i02 * r11 + i03 * r15;
  const float t10 = i10 * r0 + i11 * r4 + i12 * r8 + i13 * r12;
  const float t11 = i10 * r1 + i11 * r5 + i12 * r9 + i13 * r13;
  const float t12 = i10 * r2 + i11 * r6 + i12 * r10 + i13 * r14;
  const float t13 = i10 * r3 + i11 * r7 + i12 * r11 + i13 * r15;
  const float t20 = i20 * r0 + i21 * r4 + i22 * r8 + i23 * r12;
  const float t21 = i20 * r1 + i21 * r5 + i22 * r9 + i23 * r13;
  const float t22 = i20 * r2 + i21 * r6 + i22 * r10 + i23 * r14;
  const float t23 = i20 * r3 + i21 * r7 + i22 * r11 + i23 * r15;
  const float t30 = i33 * r12;
  const float t31 = i33 * r13;
  const float t32 = i33 * r14;
  const float t33 = i33 * r15;

  // error = D * temp  (row-major)
  const float d0 = D[0], d1 = D[1], d2 = D[2], d3 = D[3];
  const float d4 = D[4], d5 = D[5], d6 = D[6], d7 = D[7];
  const float d8 = D[8], d9 = D[9], d10 = D[10], d11 = D[11];
  const float d12 = D[12], d13 = D[13], d14 = D[14], d15 = D[15];

  out[0] = d0 * t00 + d1 * t10 + d2 * t20 + d3 * t30;
  out[1] = d0 * t01 + d1 * t11 + d2 * t21 + d3 * t31;
  out[2] = d0 * t02 + d1 * t12 + d2 * t22 + d3 * t32;
  out[3] = d0 * t03 + d1 * t13 + d2 * t23 + d3 * t33;
  out[4] = d4 * t00 + d5 * t10 + d6 * t20 + d7 * t30;
  out[5] = d4 * t01 + d5 * t11 + d6 * t21 + d7 * t31;
  out[6] = d4 * t02 + d5 * t12 + d6 * t22 + d7 * t32;
  out[7] = d4 * t03 + d5 * t13 + d6 * t23 + d7 * t33;
  out[8] = d8 * t00 + d9 * t10 + d10 * t20 + d11 * t30;
  out[9] = d8 * t01 + d9 * t11 + d10 * t21 + d11 * t31;
  out[10] = d8 * t02 + d9 * t12 + d10 * t22 + d11 * t32;
  out[11] = d8 * t03 + d9 * t13 + d10 * t23 + d11 * t33;
  out[12] = d12 * t00 + d13 * t10 + d14 * t20 + d15 * t30;
  out[13] = d12 * t01 + d13 * t11 + d14 * t21 + d15 * t31;
  out[14] = d12 * t02 + d13 * t12 + d14 * t22 + d15 * t32;
  out[15] = d12 * t03 + d13 * t13 + d14 * t23 + d15 * t33;
}

// Fused kernel: computes the full 7x14 Jacobian for Sim(3) between factor.
// Left block (cols 0-6):  H_left  = -J_l^{-1}(r) * Ad(Delta) = -J_r^{-1}(-r) * Ad
// Right block (cols 7-13): H_right =  J_r^{-1}(r)
__global__ void __launch_bounds__(128, 2)
    sim3_between_fused_jacobians_kernel(const float *__restrict__ residuals,
                                        const Matrix<4> *__restrict__ deltas, int num_items,
                                        float *__restrict__ jacobians,
                                        const int *__restrict__ factor_ids, int num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_items) return;

  const float *r = residuals + tid * kSim3TangentStride;

  // Compute J_r^{-1}(r) for right block
  float Jr[49];
  lie_device::Sim3JrInv(r, Jr);

  // Compute J_r^{-1}(-r) = J_l^{-1}(r) for left block
  float neg_r[7];
#pragma unroll
  for (int i = 0; i < 7; ++i) neg_r[i] = -r[i];
  float Jl[49];
  lie_device::Sim3JrInv(neg_r, Jl);

  // Ad(Delta) 7x7, derived from the measurement in place (nothing cached)
  float Ad[49];
  lie_device::AdjointSim3(deltas[FactorMeasurementIndex(tid, factor_ids, num_factors)].data(), Ad);

  float *out = jacobians + tid * kSim3JacobianStride;

  // Write both blocks row by row
#pragma unroll
  for (int row = 0; row < 7; ++row) {
    // Left block: -(Jl * Ad)[row][col] = -sum_k Jl[row][k] * Ad[k][col]
#pragma unroll
    for (int col = 0; col < 7; ++col) {
      float s = 0.0f;
#pragma unroll
      for (int k = 0; k < 7; ++k) {
        s += Jl[row * 7 + k] * Ad[k * 7 + col];
      }
      out[row * kSim3JacobianPitch + col] = -s;
    }
    // Right block: Jr[row][col]
#pragma unroll
    for (int col = 0; col < 7; ++col) {
      out[row * kSim3JacobianPitch + 7 + col] = Jr[row * 7 + col];
    }
  }
}

Similarity3BetweenFactorBatch::Similarity3BetweenFactorBatch(
    const Similarity3Transform *pose_deltas_ptr, size_t capacity)
    : SizedFactorBatch(capacity), pose_deltas_ptr_(pose_deltas_ptr), errors_(capacity) {}

constexpr size_t kSim3FusedBlockSize = 128;

bool Similarity3BetweenFactorBatch::Evaluate(float *residuals, float *jacobians,
                                             float const *const *state_pointers,
                                             cudaStream_t stream, const int *factor_ids,
                                             size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  const int num_factors = static_cast<int>(NumActiveFactors());
  errors_.resize(num_items);  // keeps capacity: allocates at most once per size
  size_t num_blocks = (num_items + kBlockSize - 1) / kBlockSize;

  collect_and_compute_sim3_between_error_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, pose_deltas_ptr_, num_items, errors_.data(), factor_ids, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  ComputeLogSim3(stream, reinterpret_cast<const float *>(errors_.data()), kSim3TransformStride,
                 kSim3TangentStride, num_items, residuals);

  if (jacobians != nullptr) {
    size_t fused_blocks = (num_items + kSim3FusedBlockSize - 1) / kSim3FusedBlockSize;
    sim3_between_fused_jacobians_kernel<<<fused_blocks, kSim3FusedBlockSize, 0, stream>>>(
        residuals, pose_deltas_ptr_, static_cast<int>(num_items), jacobians, factor_ids,
        num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
