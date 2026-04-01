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
#include "cunls/math/sim_lie_math.h"
#include "cunls/state/similarity2_state_batch.h"

namespace cunls {

Similarity2StateBatch::Similarity2StateBatch(cuBLASHandle& cublas_handle,
                                             const float* device_ptr,
                                             size_t num_blocks)
    : Base(device_ptr, num_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      tangents_(num_blocks * 4) {}

Similarity2StateBatch::Similarity2StateBatch(
    cuBLASHandle& cublas_handle, const float* device_ptr, size_t num_blocks,
    const int* device_constant_state_ids, size_t num_const_state_blocks)
    : Base(device_ptr, num_blocks, device_constant_state_ids,
           num_const_state_blocks),
      cublas_handle_(cublas_handle),
      delta_transforms_(num_blocks),
      tangents_(num_blocks * 4) {}

void Similarity2StateBatch::ApplyUpdate(const float* x, const float* delta,
                                      float* result,
                                      bool invert_delta,
                                      cudaStream_t stream) {
  size_t num_transforms = NumStateBlocks();

  auto tangents_ptr = reinterpret_cast<const float*>(tangents_.data());
  auto delta_transforms_ptr =
      reinterpret_cast<float*>(delta_transforms_.data());

  thrust::device_ptr<const float> ptr = thrust::device_pointer_cast(delta);
  thrust::device_ptr<float> tangents_device_ptr(tangents_.data());
  auto stream_policy = thrust::cuda::par_nosync.on(stream);
  thrust::copy(stream_policy, ptr, ptr + num_transforms * 4,
               tangents_device_ptr);

  if (invert_delta) {
    thrust::transform(stream_policy, tangents_device_ptr,
                      tangents_device_ptr + tangents_.size(),
                      tangents_device_ptr, thrust::negate<float>());
  }

  // Exp(delta): tangent vectors -> Sim(2) matrices
  constexpr size_t kTangentStride = 4;
  constexpr size_t kTransformStride = 9;
  ComputeExpSim2(stream, tangents_ptr, kTangentStride, kTransformStride,
                 num_transforms, delta_transforms_ptr);

  // result = x * Exp(delta) via batched matrix multiplication
  auto handle = static_cast<cublasHandle_t>(cublas_handle_.GetHandle(stream));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int mat_size = 3;
  constexpr int stride = 9;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, mat_size, mat_size, mat_size, &alpha,
      delta_transforms_ptr, mat_size, stride, x, mat_size, stride, &beta,
      result, mat_size, stride, num_transforms));
}

void Similarity2StateBatch::Plus(const float* x, const float* delta,
                               float* x_plus_delta,
                               cudaStream_t stream) {
  ApplyUpdate(x, delta, x_plus_delta, false, stream);
}

}  // namespace cunls
