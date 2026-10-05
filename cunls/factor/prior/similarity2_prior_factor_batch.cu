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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
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

constexpr size_t kSim2PriorBlockSize = 256;

// Sim(2) layout constants
constexpr size_t kSim2TransformStride = 9;
constexpr size_t kSim2TangentStride = 4;
constexpr size_t kSim2JacobianStride = 16;

/**
 * @brief Fused kernel: collect Sim(2) transform and compute T_inv * T_current.
 *
 * 3x3 row-major multiply, fully unrolled. Last row structure not assumed
 * since Sim(2) bottom-right is 1/s not 1.
 */
__global__ void collect_and_multiply_sim2_prior_kernel(float const *const *state_pointers,
                                                       const Matrix<3> *observations,
                                                       size_t num_items, float *errors,
                                                       const int *factor_ids, int num_factors) {
  // Each thread computes its 3x3 error into shared memory; the block then
  // writes its contiguous run of errors with 16-byte stores (9 floats per item
  // is not 16-byte aligned per thread, but the run of a full block is).
  __shared__ __align__(16) float s_out[kSim2PriorBlockSize * kSim2TransformStride];
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t block_first = static_cast<size_t>(blockIdx.x) * blockDim.x;
  const int block_items =
      static_cast<int>(num_items - block_first < blockDim.x ? num_items - block_first : blockDim.x);
  if (tid < num_items) {
    const float *__restrict__ C = state_pointers[tid];
    const float *__restrict__ T =
        observations[FactorMeasurementIndex(tid, factor_ids, num_factors)].data();
    // Load the measurement and the state together, then derive T^{-1} from
    // registers (LoadsBarrier keeps the compiler from splitting the loads).
    float tv[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) tv[k] = T[k];
    const float c0 = C[0], c1 = C[1], c2 = C[2];
    const float c3 = C[3], c4 = C[4], c5 = C[5];
    const float c6 = C[6], c7 = C[7], c8 = C[8];
    lie_device::LoadsBarrier(tv);
    lie_device::LoadsBarrier<9>({c0, c1, c2, c3, c4, c5, c6, c7, c8});
    float I[9];  // T_target^{-1}, derived from the measurement in place
    lie_device::InverseSim2(tv, I);
    float out[9];

    const float i0 = I[0], i1 = I[1], i2 = I[2];
    const float i3 = I[3], i4 = I[4], i5 = I[5];
    const float i6 = I[6], i7 = I[7], i8 = I[8];

    out[0] = i0 * c0 + i1 * c3 + i2 * c6;
    out[1] = i0 * c1 + i1 * c4 + i2 * c7;
    out[2] = i0 * c2 + i1 * c5 + i2 * c8;
    out[3] = i3 * c0 + i4 * c3 + i5 * c6;
    out[4] = i3 * c1 + i4 * c4 + i5 * c7;
    out[5] = i3 * c2 + i4 * c5 + i5 * c8;
    out[6] = i6 * c0 + i7 * c3 + i8 * c6;
    out[7] = i6 * c1 + i7 * c4 + i8 * c7;
    out[8] = i6 * c2 + i7 * c5 + i8 * c8;
#pragma unroll
    for (int k = 0; k < 9; ++k) s_out[threadIdx.x * 9 + k] = out[k];
  }
  __syncthreads();
  float *dst = errors + block_first * 9;
  const int count = block_items * 9;
  if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0 && count % 4 == 0) {
    float4 *dst4 = reinterpret_cast<float4 *>(dst);
    const float4 *src4 = reinterpret_cast<const float4 *>(s_out);
    for (int i = threadIdx.x; i < count / 4; i += blockDim.x) dst4[i] = src4[i];
  } else {
    for (int i = threadIdx.x; i < count; i += blockDim.x) dst[i] = s_out[i];
  }
}

Similarity2PriorFactorBatch::Similarity2PriorFactorBatch(
    const Similarity2Transform *observations_ptr, size_t capacity)
    : SizedFactorBatch(capacity),
      observations_ptr_(observations_ptr),
      transforms_error_(capacity * kSim2TransformStride) {}

bool Similarity2PriorFactorBatch::Evaluate(float *residuals, float *jacobians,
                                           float const *const *state_pointers, cudaStream_t stream,
                                           const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  if (num_items == 0 || NumActiveFactors() == 0) {
    return true;
  }
  transforms_error_.resize(num_items * kSim2TransformStride);  // keeps capacity
  size_t num_blocks = (num_items + kSim2PriorBlockSize - 1) / kSim2PriorBlockSize;

  // Fused: collect T_current + compute T_inv * T_current
  collect_and_multiply_sim2_prior_kernel<<<num_blocks, kSim2PriorBlockSize, 0, stream>>>(
      state_pointers, observations_ptr_, num_items, transforms_error_.data(), factor_ids,
      static_cast<int>(NumActiveFactors()));
  THROW_ON_CUDA_ERROR(cudaGetLastError());

  // Step 3: residual = Log(T_error)
  ComputeLogSim2(stream, transforms_error_.data(), kSim2TransformStride, kSim2TangentStride,
                 num_items, residuals);

  // Step 4: Jacobian = J_r^{-1}(residual) if requested
  if (jacobians != nullptr) {
    ComputeJacobianRightInverseSim2(stream, residuals, kSim2TangentStride, kSim2JacobianStride,
                                    num_items, jacobians);
  }

  return true;
}

}  // namespace cunls
