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

#include <cuda/std/limits>

#include "cunls/common/helper.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"

namespace cunls {

constexpr size_t kArctanBlockSize = 256;

__global__ void arctan_loss_kernel(float a, float b, float* s, float3* out,
                                   int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  const float sq_error = s[tid];
  const float sum = 1.0f + sq_error * sq_error * b;
  const float inv = 1.0f / sum;

  float3& rho = out[tid];
  rho.x = a * atanf(sq_error / a);
  rho.y = fmaxf(cuda::std::numeric_limits<float>::min(), inv);
  rho.z = -2.0f * sq_error * b * (inv * inv);
}

ArctanLossFunctionBatch::ArctanLossFunctionBatch(float a, float b)
    : a_(a), b_(b) {}

bool ArctanLossFunctionBatch::Evaluate(float* s, float3* out, int num_losses,
                                       cudaStream_t stream) const {
  if (num_losses <= 0) {
    return true;
  }

  size_t num_blocks = (num_losses + kArctanBlockSize - 1) / kArctanBlockSize;
  arctan_loss_kernel<<<num_blocks, kArctanBlockSize, 0, stream>>>(
      a_, b_, s, out, num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
