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
#include "cunls/state/se2_state_batch.h"

namespace cunls {

/**
 * @brief CUDA kernel that computes the SE(2) exponential map for a batch of tangent vectors.
 *
 * For each tangent vector [v_x, v_y, theta], computes the 3x3 transformation matrix
 * in row-major order:
 *   [cos(theta), -sin(theta), tx]
 *   [sin(theta),  cos(theta), ty]
 *   [    0,           0,       1]
 *
 * where [tx, ty] = V * [v_x, v_y] with the V matrix from SE(2) exponential map:
 *   V = [[sin(theta)/theta, -(1-cos(theta))/theta],
 *        [(1-cos(theta))/theta, sin(theta)/theta]]
 *
 * For small theta (|theta| < 1e-5), V approaches identity and [tx, ty] ≈ [v_x, v_y].
 *
 * @param tangent Input tangent vectors (3D, device pointer, stride 3)
 * @param transforms Output 3x3 transformation matrices (device pointer, row-major,
 *                   9 floats per matrix)
 * @param size Number of tangent vectors to process
 */
__global__ void ExpSE2Kernel(const float* tangent, float* transforms,
                             size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  const float* xi = tangent + idx * 3;
  float vx = xi[0];
  float vy = xi[1];
  float w = xi[2];

  float c = cosf(w);
  float s = sinf(w);

  float tx, ty;
  if (fabsf(w) < 1e-3f) {
    // For small angles, V ≈ I, so translation ≈ [v_x, v_y]
    // Note: threshold must be large enough that (1-cos(w))/w is
    // numerically stable in single-precision float.
    tx = vx;
    ty = vy;
  } else {
    float sinw_over_w = s / w;
    float one_minus_cosw_over_w = (1.0f - c) / w;
    tx = vx * sinw_over_w - vy * one_minus_cosw_over_w;
    ty = vx * one_minus_cosw_over_w + vy * sinw_over_w;
  }

  // Row-major 3x3 transformation matrix
  float* T = transforms + idx * 9;
  T[0] = c;     T[1] = -s;    T[2] = tx;
  T[3] = s;     T[4] = c;     T[5] = ty;
  T[6] = 0.0f;  T[7] = 0.0f;  T[8] = 1.0f;
}

/** @copydoc SE2StateBatch::ApplyUpdate */
void SE2StateBatch::ApplyUpdate(const float* x, const float* delta,
                                         float* result, bool invert_delta,
                                         cudaStream_t stream) {
  size_t num_transforms = NumStateBlocks();

  dvector<Matrix<3>> delta_transforms_(num_transforms);

  dvector<float> tangents_(num_transforms * 3);

  auto tangents_ptr =
      reinterpret_cast<const float*>(tangents_.data());
  auto delta_transforms_ptr = reinterpret_cast<float*>(
      delta_transforms_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> tangents_device_ptr(tangents_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_transforms * 3,
               tangents_device_ptr);

  // Negate tangent vectors if computing Exp(-delta) instead of Exp(delta)
  if (invert_delta) {
    thrust::transform(stream_policy, tangents_device_ptr,
                      tangents_device_ptr + tangents_.size(),
                      tangents_device_ptr, thrust::negate<float>());
  }

  // Compute update matrices: delta_transforms = Exp(±delta)
  constexpr int block_size = 256;
  int num_blocks = (num_transforms + block_size - 1) / block_size;
  ExpSE2Kernel<<<num_blocks, block_size, 0, stream>>>(
      tangents_ptr, delta_transforms_ptr, num_transforms);

  auto handle = cublas_handle_.GetHandle(stream);

  // cuBLAS uses column-major storage, but our matrices are row-major
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  constexpr int mat_size = 3;
  constexpr int stride = 9;

  // Perform batched matrix multiplication: result = x * Exp(±delta)
  // Note: cuBLAS uses column-major, but CUBLAS_OP_N for both operands computes
  // the equivalent of row-major right-multiplication
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_transforms_ptr, mat_size, stride, x, mat_size, stride, &beta,
      result, mat_size, stride, num_transforms));
}

/**
 * @brief Performs the Plus operation: x_plus_delta = x * Exp(delta)
 *
 * Computes the right-multiplication update for SE(2) transformations.
 * First computes the update matrix Exp(delta) from the tangent vector,
 * then performs batched matrix multiplication using cuBLAS.
 *
 * Note: cuBLAS uses column-major storage, but our matrices are row-major.
 * For right-multiplication (x * update), we use CUBLAS_OP_N for both operands.
 * cuBLAS interprets the matrices as column-major, so this computes the
 * equivalent of the desired row-major result.
 *
 * @param x Input transformation matrices (device pointer, row-major)
 * @param delta Tangent space updates (3D vectors [v_x, v_y, theta], device pointer)
 * @param x_plus_delta Output transformation matrices (device pointer,
 * row-major)
 * @param stream CUDA stream for asynchronous execution
 */
void SE2StateBatch::Plus(const float* x, const float* delta,
                                  float* x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
