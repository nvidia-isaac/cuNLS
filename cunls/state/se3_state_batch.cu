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
#include "cunls/state/se3_state_batch.h"

namespace cunls {

SE3StateBatch::SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 6) {}

SE3StateBatch::SE3StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      twists_(num_blocks * 6) {}

/**
 * @brief Performs the Plus operation: x_plus_delta = x * Exp(skew(delta))
 *
 * Computes the right-multiplication update for SE(3) transformations.
 * First computes the update matrix Exp(skew(delta)) using Exp,
 * then performs batched matrix multiplication using cuBLAS.
 *
 * Note: cuBLAS uses column-major storage, but our matrices are row-major.
 * For right-multiplication (x * update), we use CUBLAS_OP_N for both operands.
 * cuBLAS interprets the matrices as column-major, so this computes the
 * equivalent of the desired row-major result.
 *
 * @param x Input transformation matrices (device pointer, row-major)
 * @param delta Tangent space updates (6D twist vectors, device pointer)
 * @param x_plus_delta Output transformation matrices (device pointer,
 * row-major)
 * @param stream CUDA stream for asynchronous execution
 */
void SE3StateBatch::Plus(const float* x, const float* delta,
                         float* x_plus_delta, cudaStream_t stream) {
  size_t num_transforms = NumStateBlocks();

  auto twists_ptr = reinterpret_cast<const float*>(twists_.data());
  auto delta_transforms_ptr =
      reinterpret_cast<float*>(delta_transforms_.data());

  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(twists_.data(), delta,
                                      num_transforms * 6 * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  constexpr size_t twist_stride = 6;
  constexpr size_t transform_stride = 16;
  constexpr size_t transform_pitch = 4;
  // Compute update matrices: delta_transforms = Exp(±delta)
  ComputeExpSE3(stream, twists_ptr, twist_stride, transform_pitch,
                transform_stride, num_transforms, delta_transforms_ptr);
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr int mat_size = 4;
  constexpr int stride = 16;

  // Perform batched matrix multiplication: x_plus_delta = x * Exp(±delta)
  // Note: cuBLAS uses column-major, but CUBLAS_OP_N for both operands computes
  // the equivalent of row-major right-multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_transforms_ptr, mat_size, stride, x, mat_size, stride, &beta,
      x_plus_delta, mat_size, stride, num_transforms));
}

}  // namespace cunls
