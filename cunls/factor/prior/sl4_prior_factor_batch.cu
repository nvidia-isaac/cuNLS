/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/math/lie_device.cuh"
#include "cunls/math/sl_lie_math.h"

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

constexpr size_t kSL4PriorBlockSize = 256;

/**
 * @brief Fused kernel: collect SL(4) transform and compute T_inv * T_current.
 * Full 4x4, fully unrolled.
 */
__global__ void collect_and_multiply_sl4_prior_kernel(float const *const *state_pointers,
                                                      const SL4Transform *observations,
                                                      size_t num_items, SL4Transform *errors,
                                                      const int *factor_ids, int num_factors) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_items) return;

  const float *__restrict__ C = state_pointers[tid];
  const float *__restrict__ T =
      observations[FactorMeasurementIndex(tid, factor_ids, num_factors)].data();
  // Load the measurement and the state together, then derive T^{-1} from
  // registers (LoadsBarrier keeps the compiler from splitting the loads).
  float tv[16], cv[16];
  lie_device::Load16(T, tv);
  lie_device::Load16(C, cv);
  lie_device::LoadsBarrier(tv);
  lie_device::LoadsBarrier(cv);
  const float c0 = cv[0], c1 = cv[1], c2 = cv[2], c3 = cv[3];
  const float c4 = cv[4], c5 = cv[5], c6 = cv[6], c7 = cv[7];
  const float c8 = cv[8], c9 = cv[9], c10 = cv[10], c11 = cv[11];
  const float c12 = cv[12], c13 = cv[13], c14 = cv[14], c15 = cv[15];
  // T_target^{-1}, derived from the measurement in place (zeros if singular,
  // as ComputeInverseSL4 does).
  float I[16];
  if (!lie_device::Inv4(tv, I)) {
#pragma unroll
    for (int k = 0; k < 16; ++k) I[k] = 0.f;
  }
  float out[16];

  const float i0 = I[0], i1 = I[1], i2 = I[2], i3 = I[3];
  const float i4 = I[4], i5 = I[5], i6 = I[6], i7 = I[7];
  const float i8 = I[8], i9 = I[9], i10 = I[10], i11 = I[11];
  const float i12 = I[12], i13 = I[13], i14 = I[14], i15 = I[15];

  out[0] = i0 * c0 + i1 * c4 + i2 * c8 + i3 * c12;
  out[1] = i0 * c1 + i1 * c5 + i2 * c9 + i3 * c13;
  out[2] = i0 * c2 + i1 * c6 + i2 * c10 + i3 * c14;
  out[3] = i0 * c3 + i1 * c7 + i2 * c11 + i3 * c15;
  out[4] = i4 * c0 + i5 * c4 + i6 * c8 + i7 * c12;
  out[5] = i4 * c1 + i5 * c5 + i6 * c9 + i7 * c13;
  out[6] = i4 * c2 + i5 * c6 + i6 * c10 + i7 * c14;
  out[7] = i4 * c3 + i5 * c7 + i6 * c11 + i7 * c15;
  out[8] = i8 * c0 + i9 * c4 + i10 * c8 + i11 * c12;
  out[9] = i8 * c1 + i9 * c5 + i10 * c9 + i11 * c13;
  out[10] = i8 * c2 + i9 * c6 + i10 * c10 + i11 * c14;
  out[11] = i8 * c3 + i9 * c7 + i10 * c11 + i11 * c15;
  out[12] = i12 * c0 + i13 * c4 + i14 * c8 + i15 * c12;
  out[13] = i12 * c1 + i13 * c5 + i14 * c9 + i15 * c13;
  out[14] = i12 * c2 + i13 * c6 + i14 * c10 + i15 * c14;
  out[15] = i12 * c3 + i13 * c7 + i14 * c11 + i15 * c15;
  lie_device::Store16(errors[tid].data(), out);
}

SL4PriorFactorBatch::SL4PriorFactorBatch(const SL4Transform *observations_ptr, size_t capacity)
    : SizedFactorBatch(capacity),
      observations_ptr_(observations_ptr),
      transforms_error_(capacity) {}

bool SL4PriorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                   float const *const *state_pointers, cudaStream_t stream,
                                   const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  transforms_error_.resize(num_items);  // keeps capacity: allocates at most once per size
  size_t num_blocks = (num_items + kSL4PriorBlockSize - 1) / kSL4PriorBlockSize;
  // Fused: collect T_current + compute T_inv * T_current
  collect_and_multiply_sl4_prior_kernel<<<num_blocks, kSL4PriorBlockSize, 0, stream>>>(
      state_pointers, observations_ptr_, num_items, transforms_error_.data(), factor_ids,
      static_cast<int>(NumActiveFactors()));
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  constexpr size_t twist_stride = 15;
  ComputeLogSL4(stream, reinterpret_cast<const float *>(transforms_error_.data()), transform_pitch,
                transform_stride, twist_stride, num_items, residuals);

  if (jacobians != nullptr) {
    constexpr size_t jacobian_pitch = 15;
    constexpr size_t jacobian_stride = 225;
    FillIdentity15x15(stream, num_items, jacobians, jacobian_pitch, jacobian_stride);
  }

  return true;
}

}  // namespace cunls
