/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cunls/common/helper.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/state/sl4_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

/** Row-major 4x4 multiply: C = A * B per batch element (64 FMAs, fully unrolled). */
__global__ void sl4_matmul_kernel(const float* __restrict__ x,
                                  const float* __restrict__ exp_delta,
                                  float* __restrict__ result, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  const float* A = x + tid * 16;
  const float* B = exp_delta + tid * 16;
  float* C = result + tid * 16;

#define DOT4(row, col) \
  (A[row * 4 + 0] * B[0 * 4 + col] + A[row * 4 + 1] * B[1 * 4 + col] + \
   A[row * 4 + 2] * B[2 * 4 + col] + A[row * 4 + 3] * B[3 * 4 + col])

  C[0] = DOT4(0, 0);
  C[1] = DOT4(0, 1);
  C[2] = DOT4(0, 2);
  C[3] = DOT4(0, 3);
  C[4] = DOT4(1, 0);
  C[5] = DOT4(1, 1);
  C[6] = DOT4(1, 2);
  C[7] = DOT4(1, 3);
  C[8] = DOT4(2, 0);
  C[9] = DOT4(2, 1);
  C[10] = DOT4(2, 2);
  C[11] = DOT4(2, 3);
  C[12] = DOT4(3, 0);
  C[13] = DOT4(3, 1);
  C[14] = DOT4(3, 2);
  C[15] = DOT4(3, 3);
#undef DOT4
}

}  // namespace

SL4StateBatch::SL4StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 15) {}

SL4StateBatch::SL4StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 15) {}

void SL4StateBatch::Plus(const float* x, const float* delta, float* x_plus_delta,
                         cudaStream_t stream) {
  const int n = static_cast<int>(NumStateBlocks());

  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(twists_.data(), delta,
                                      static_cast<size_t>(n) * 15 * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  constexpr size_t twist_stride = 15;
  constexpr size_t transform_pitch = 4;
  constexpr size_t transform_stride = 16;
  ComputeExpSL4(stream, reinterpret_cast<const float*>(twists_.data()),
                twist_stride, transform_pitch, transform_stride,
                static_cast<size_t>(n),
                reinterpret_cast<float*>(delta_transforms_.data()));

  const int grid = (n + kBlockSize - 1) / kBlockSize;
  sl4_matmul_kernel<<<grid, kBlockSize, 0, stream>>>(
      x, reinterpret_cast<const float*>(delta_transforms_.data()), x_plus_delta, n);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  static_cast<void>(cublas_handle_);
}

}  // namespace cunls
