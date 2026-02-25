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
#include "cunls/state/so2_state_batch.h"

namespace cunls {

/**
 * @brief CUDA kernel that computes the SO(2) exponential map for a batch of angles.
 *
 * For each angle theta, computes the 2x2 rotation matrix in row-major order:
 *   [cos(theta), -sin(theta), sin(theta), cos(theta)]
 *
 * @param angles Input angles in radians (device pointer, one per rotation)
 * @param rotations Output 2x2 rotation matrices (device pointer, row-major,
 *                  4 floats per matrix)
 * @param size Number of angles to process
 */
__global__ void ExpSO2Kernel(const float* angles, float* rotations,
                             size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  float theta = angles[idx];
  float c = cosf(theta);
  float s = sinf(theta);

  float* R = rotations + idx * 4;
  R[0] = c;   // [0,0]
  R[1] = -s;  // [0,1]
  R[2] = s;   // [1,0]
  R[3] = c;   // [1,1]
}

/** @copydoc SO2StateBatch::ApplyUpdate */
void SO2StateBatch::ApplyUpdate(const float* x, const float* delta,
                                         float* result, bool invert_delta,
                                         cudaStream_t stream) {
  size_t num_rotations = NumStateBlocks();

  dvector<Matrix<2>> delta_rotations_(num_rotations);

  dvector<float> angles_(num_rotations);

  auto angles_ptr =
      reinterpret_cast<const float*>(angles_.data());
  auto delta_rotations_ptr = reinterpret_cast<float*>(
      delta_rotations_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> angles_device_ptr(angles_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_rotations, angles_device_ptr);

  // Negate angles if computing Exp(-delta) instead of Exp(delta)
  if (invert_delta) {
    thrust::transform(stream_policy, angles_device_ptr, angles_device_ptr + angles_.size(),
                      angles_device_ptr, thrust::negate<float>());
  }

  // Compute update matrices: delta_rotations = Exp(±delta)
  constexpr int block_size = 256;
  int num_blocks = (num_rotations + block_size - 1) / block_size;
  ExpSO2Kernel<<<num_blocks, block_size, 0, stream>>>(
      angles_ptr, delta_rotations_ptr, num_rotations);

  auto handle = cublas_handle_.GetHandle(stream);

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr int mat_size = 2;
  constexpr int stride = 4;

  // Perform batched matrix multiplication: result = x * Exp(±delta)
  // Note: cuBLAS uses column-major, but CUBLAS_OP_N for both operands computes
  // the equivalent of row-major right-multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha, delta_rotations_ptr,
      mat_size, stride, x, mat_size, stride, &beta, result,
      mat_size, stride, num_rotations));
}

/**
 * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
 *
 * Computes the right-multiplication update for SO(2) rotations.
 * First computes the update matrix Exp(delta) from the angle,
 * then performs batched matrix multiplication using cuBLAS.
 *
 * Note: cuBLAS uses column-major storage, but our matrices are row-major.
 * For right-multiplication (x * update), we use CUBLAS_OP_N for both operands.
 * cuBLAS interprets the matrices as column-major, so this computes the
 * equivalent of the desired row-major result.
 *
 * @param x Input rotation matrices (device pointer, row-major)
 * @param delta Tangent space updates (scalar angles in radians, device pointer)
 * @param x_plus_delta Output rotation matrices (device pointer,
 * row-major)
 * @param stream CUDA stream for asynchronous execution
 */
void SO2StateBatch::Plus(const float* x, const float* delta,
                                  float* x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
