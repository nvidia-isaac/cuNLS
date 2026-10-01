/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/indexed_evaluation.cuh"
#include "cunls/math/lie_device.cuh"
#include "cunls/math/sl_lie_math.h"

namespace cunls {

constexpr size_t kBlockSize = 256;

/**
 * @brief Fused kernel: collect L/R SL(4), compute (L^{-1} * R) * Delta.
 *
 * SL(4) uses general 4x4 matrices, so we must compute the full inverse
 * using a helper kernel or precompute it. Since InverseSL4 already exists,
 * we fuse the collection, the two matrix multiplies, but call InverseSL4
 * externally. This kernel does: collect + 2x matmul(4x4) in one pass.
 */
__global__ void collect_and_matmul_sl4_between_kernel(float const *const *state_pointers,
                                                      const SL4Transform *left_inverse,
                                                      const SL4Transform *deltas, size_t num_items,
                                                      SL4Transform *errors, const int *factor_ids,
                                                      int num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_items) return;
  const int m = FactorMeasurementIndex(tid, factor_ids, num_factors);

  const float *__restrict__ R = state_pointers[2 * tid + 1];
  const float *__restrict__ I = left_inverse[tid].data();
  const float *__restrict__ D = deltas[m].data();
  float *__restrict__ out = errors[tid].data();

  // temp = I * R  (full 4x4)
  float t[16];
  const float i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3];
  const float i4 = I[4], i5 = I[5], i6 = I[6], i7 = I[7];
  const float i8 = I[8], i9 = I[9], i10 = I[10], i11 = I[11];
  const float i12 = I[12], i13 = I[13], i14 = I[14], i15 = I[15];

  const float r0 = R[0], r1 = R[1], r2 = R[2], r3 = R[3];
  const float r4 = R[4], r5 = R[5], r6 = R[6], r7 = R[7];
  const float r8 = R[8], r9 = R[9], r10 = R[10], r11 = R[11];
  const float r12 = R[12], r13 = R[13], r14 = R[14], r15 = R[15];

  t[0] = i0 * r0 + i1 * r4 + i2 * r8 + i3 * r12;
  t[1] = i0 * r1 + i1 * r5 + i2 * r9 + i3 * r13;
  t[2] = i0 * r2 + i1 * r6 + i2 * r10 + i3 * r14;
  t[3] = i0 * r3 + i1 * r7 + i2 * r11 + i3 * r15;
  t[4] = i4 * r0 + i5 * r4 + i6 * r8 + i7 * r12;
  t[5] = i4 * r1 + i5 * r5 + i6 * r9 + i7 * r13;
  t[6] = i4 * r2 + i5 * r6 + i6 * r10 + i7 * r14;
  t[7] = i4 * r3 + i5 * r7 + i6 * r11 + i7 * r15;
  t[8] = i8 * r0 + i9 * r4 + i10 * r8 + i11 * r12;
  t[9] = i8 * r1 + i9 * r5 + i10 * r9 + i11 * r13;
  t[10] = i8 * r2 + i9 * r6 + i10 * r10 + i11 * r14;
  t[11] = i8 * r3 + i9 * r7 + i10 * r11 + i11 * r15;
  t[12] = i12 * r0 + i13 * r4 + i14 * r8 + i15 * r12;
  t[13] = i12 * r1 + i13 * r5 + i14 * r9 + i15 * r13;
  t[14] = i12 * r2 + i13 * r6 + i14 * r10 + i15 * r14;
  t[15] = i12 * r3 + i13 * r7 + i14 * r11 + i15 * r15;

  // error = D * temp  (row-major: D_row * temp_row)
  const float d0 = D[0], d1 = D[1], d2 = D[2], d3 = D[3];
  const float d4 = D[4], d5 = D[5], d6 = D[6], d7 = D[7];
  const float d8 = D[8], d9 = D[9], d10 = D[10], d11 = D[11];
  const float d12 = D[12], d13 = D[13], d14 = D[14], d15 = D[15];

  out[0] = d0 * t[0] + d1 * t[4] + d2 * t[8] + d3 * t[12];
  out[1] = d0 * t[1] + d1 * t[5] + d2 * t[9] + d3 * t[13];
  out[2] = d0 * t[2] + d1 * t[6] + d2 * t[10] + d3 * t[14];
  out[3] = d0 * t[3] + d1 * t[7] + d2 * t[11] + d3 * t[15];
  out[4] = d4 * t[0] + d5 * t[4] + d6 * t[8] + d7 * t[12];
  out[5] = d4 * t[1] + d5 * t[5] + d6 * t[9] + d7 * t[13];
  out[6] = d4 * t[2] + d5 * t[6] + d6 * t[10] + d7 * t[14];
  out[7] = d4 * t[3] + d5 * t[7] + d6 * t[11] + d7 * t[15];
  out[8] = d8 * t[0] + d9 * t[4] + d10 * t[8] + d11 * t[12];
  out[9] = d8 * t[1] + d9 * t[5] + d10 * t[9] + d11 * t[13];
  out[10] = d8 * t[2] + d9 * t[6] + d10 * t[10] + d11 * t[14];
  out[11] = d8 * t[3] + d9 * t[7] + d10 * t[11] + d11 * t[15];
  out[12] = d12 * t[0] + d13 * t[4] + d14 * t[8] + d15 * t[12];
  out[13] = d12 * t[1] + d13 * t[5] + d14 * t[9] + d15 * t[13];
  out[14] = d12 * t[2] + d13 * t[6] + d14 * t[10] + d15 * t[14];
  out[15] = d12 * t[3] + d13 * t[7] + d14 * t[11] + d15 * t[15];
}

__global__ void collect_sl4_left_poses_kernel(float const *const *state_pointers, size_t num_items,
                                              SL4Transform *pose_left) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_items) return;
  pose_left[tid] = *reinterpret_cast<const SL4Transform *>(state_pointers[2 * tid]);
}

// Fused Jacobian kernel: one warp per item writes the full 15x30 Jacobian.
//   Left block (cols 0-14):  -Ad(D), derived from the measurement D in place
//   Right block (cols 15-29): identity
// Lane 0 inverts D (Inv4); then the 450 outputs of the item are striped over
// the 32 lanes in memory order, so stores are coalesced. Ad entries use the
// same basis-support arithmetic as AdjointSL4Kernel (lie_device helpers).
constexpr int kSL4JacWarpsPerBlock = 8;
constexpr int kSL4JacBlockSize = kSL4JacWarpsPerBlock * 32;

__global__ void __launch_bounds__(kSL4JacBlockSize)
    sl4_between_fused_jacobian_kernel(const SL4Transform *__restrict__ deltas,
                                      float *__restrict__ jacobians, int num_items,
                                      const int *__restrict__ factor_ids, int num_factors) {
  __shared__ lie_device::SL4Basis s_basis;
  __shared__ float s_T[kSL4JacWarpsPerBlock][16];
  __shared__ float s_Tinv[kSL4JacWarpsPerBlock][16];
  __shared__ int s_ok[kSL4JacWarpsPerBlock];

  if (threadIdx.x < 15) {
    s_basis.nnz[threadIdx.x] = lie_device::SL4BasisColumnSupport(
        threadIdx.x, s_basis.rows[threadIdx.x], s_basis.vals[threadIdx.x]);
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int item = blockIdx.x * kSL4JacWarpsPerBlock + w;
  if (item >= num_items) return;

  if (lane < 16) {
    s_T[w][lane] = deltas[FactorMeasurementIndex(item, factor_ids, num_factors)].data()[lane];
  }
  __syncwarp();
  if (lane == 0) {
    s_ok[w] = lie_device::Inv4(s_T[w], s_Tinv[w]) ? 1 : 0;
  }
  __syncwarp();
  const bool ok = s_ok[w] != 0;

  float *J = jacobians + static_cast<size_t>(item) * 450;
  for (int o = lane; o < 450; o += 32) {
    const int row = o / 30;
    const int col = o - row * 30;
    float v;
    if (col < 15) {
      // Singular D: Ad = identity (as AdjointSL4Kernel), negated below.
      const float ad = ok ? lie_device::SL4AdjointEntry(s_basis, row, col, s_T[w], s_Tinv[w])
                          : (row == col ? 1.f : 0.f);
      v = -ad;
    } else {
      v = (col - 15 == row) ? 1.0f : 0.0f;
    }
    J[o] = v;
  }
}

SL4BetweenFactorBatch::SL4BetweenFactorBatch(const SL4Transform *pose_deltas_ptr, size_t capacity)
    : SizedFactorBatch(capacity),
      pose_deltas_ptr_(pose_deltas_ptr),
      poses_left_(capacity),
      poses_right_(capacity),
      poses_left_inverse_(capacity) {}

bool SL4BetweenFactorBatch::Evaluate(float *residuals, float *jacobians,
                                     float const *const *state_pointers, cudaStream_t stream,
                                     const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  const int num_factors = static_cast<int>(NumActiveFactors());
  poses_left_.resize(num_items);  // keeps capacity: allocates at most once per size
  poses_left_inverse_.resize(num_items);
  size_t num_blocks = (num_items + kBlockSize - 1) / kBlockSize;
  constexpr size_t pitch = 4;
  constexpr size_t stride = 16;

  // SL(4) inverse is non-trivial (general 4x4). Pipeline:
  // 1. Gather left poses into contiguous buffer
  // 2. Compute L^{-1} via ComputeInverseSL4
  // 3. Fused kernel: read L^{-1} + R from state_pointers, compute (L^{-1}*R)*D
  collect_sl4_left_poses_kernel<<<num_blocks, kBlockSize, 0, stream>>>(state_pointers, num_items,
                                                                       poses_left_.data());
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  ComputeInverseSL4(stream, reinterpret_cast<const float *>(poses_left_.data()), pitch, stride,
                    pitch, stride, num_items,
                    reinterpret_cast<float *>(poses_left_inverse_.data()));

  collect_and_matmul_sl4_between_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
      state_pointers, poses_left_inverse_.data(), pose_deltas_ptr_, num_items, poses_left_.data(),
      factor_ids, num_factors);
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t twist_stride = 15;
  ComputeLogSL4(stream, reinterpret_cast<const float *>(poses_left_.data()), pitch, stride,
                twist_stride, num_items, residuals);

  if (jacobians != nullptr) {
    const size_t jac_blocks = (num_items + kSL4JacWarpsPerBlock - 1) / kSL4JacWarpsPerBlock;
    sl4_between_fused_jacobian_kernel<<<jac_blocks, kSL4JacBlockSize, 0, stream>>>(
        pose_deltas_ptr_, jacobians, static_cast<int>(num_items), factor_ids, num_factors);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
  }

  return true;
}

}  // namespace cunls
