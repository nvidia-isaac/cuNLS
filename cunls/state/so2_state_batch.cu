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
#include "cunls/state/so2_state_batch.h"

namespace cunls {

SO2StateBatch::SO2StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      angles_(num_blocks) {}

SO2StateBatch::SO2StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                             size_t num_blocks, const int* device_constant_state_ids,
                             size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_rotations_(num_blocks),
      angles_(num_blocks) {}

void SO2StateBatch::ApplyUpdate(const float* x, const float* delta,
                                float* result, bool invert_delta,
                                cudaStream_t stream) {
  size_t num_rotations = NumStateBlocks();

  auto angles_ptr = reinterpret_cast<const float*>(angles_.data());
  auto delta_rotations_ptr =
      reinterpret_cast<float*>(delta_rotations_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> angles_device_ptr(angles_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_rotations, angles_device_ptr);

  if (invert_delta) {
    thrust::transform(stream_policy, angles_device_ptr,
                      angles_device_ptr + angles_.size(),
                      angles_device_ptr, thrust::negate<float>());
  }

  // Exp(delta): angles -> SO(2) rotation matrices
  constexpr size_t kAngleStride = 1;
  constexpr size_t kRotationStride = 4;
  ComputeExpSO2(stream, angles_ptr, kAngleStride, kRotationStride,
                num_rotations, delta_rotations_ptr);

  // result = x * Exp(delta) via batched matrix multiplication
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 2;
  constexpr int stride = 4;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_rotations_ptr, mat_size, stride, x, mat_size, stride, &beta,
      result, mat_size, stride, num_rotations));
}

void SO2StateBatch::Plus(const float* x, const float* delta,
                         float* x_plus_delta, cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
