/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include "cunls/common/helper.h"
#include "cunls/math/so_se_lie_math.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

SO3StateBatch::SO3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      twists_(num_blocks * 3) {}

SO3StateBatch::SO3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      twists_(num_blocks * 3) {}

/** @copydoc SO3StateBatch::ApplyUpdate */
void SO3StateBatch::ApplyUpdate(const float* x, const float* delta,
                              float* result, bool invert_delta,
                              cudaStream_t stream) {
  size_t num_rotations = NumStateBlocks();

  auto twists_ptr =
      reinterpret_cast<const float*>(twists_.data());
  auto delta_rotations_ptr = reinterpret_cast<float*>(
      delta_rotations_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> twists_device_ptr(twists_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_rotations * 3, twists_device_ptr);

  // Negate twists if computing Exp(-delta) instead of Exp(delta)
  if (invert_delta) {
    thrust::transform(stream_policy, twists_device_ptr, twists_device_ptr + twists_.size(),
                      twists_device_ptr, thrust::negate<float>());
  }

  constexpr size_t twist_stride = 3;
  constexpr size_t rotation_stride = 9;
  constexpr size_t rotation_pitch = 3;
  // Compute update matrices: delta_rotations = Exp(±delta)
  ComputeExpSO3(stream, twists_ptr, twist_stride, rotation_pitch, rotation_stride,
         num_rotations, delta_rotations_ptr);
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr int mat_size = 3;
  constexpr int stride = 9;

  // Perform batched matrix multiplication: result = x * Exp(±delta)
  // Note: cuBLAS uses column-major, but CUBLAS_OP_N for both operands computes
  // the equivalent of row-major right-multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha, delta_rotations_ptr,
      mat_size, stride, x, mat_size, stride, &beta, result,
      mat_size, stride, num_rotations));
}

/**
 * @brief Performs the Plus operation: x_plus_delta = x * Exp(skew(delta))
 *
 * Computes the right-multiplication update for SO(3) rotations.
 * First computes the update matrix Exp(skew(delta)) using Exp,
 * then performs batched matrix multiplication using cuBLAS.
 *
 * Note: cuBLAS uses column-major storage, but our matrices are row-major.
 * For right-multiplication (x * update), we use CUBLAS_OP_N for both operands.
 * cuBLAS interprets the matrices as column-major, so this computes the
 * equivalent of the desired row-major result.
 *
 * @param x Input rotation matrices (device pointer, row-major)
 * @param delta Tangent space updates (3D rotation vectors, device pointer)
 * @param x_plus_delta Output rotation matrices (device pointer,
 * row-major)
 * @param stream CUDA stream for asynchronous execution
 */
void SO3StateBatch::Plus(const float* x, const float* delta,
                         float* x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
