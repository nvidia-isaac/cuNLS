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

#include "cunls/common/helper.h"
#include "cunls/robustifier/scaled_loss_function_batch.h"

namespace cunls {

constexpr size_t kScaledBlockSize = 256;

__global__ void scaled_loss_apply_kernel(float a, float3 *out, int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  float3 &rho = out[tid];
  rho.x *= a;
  rho.y *= a;
  rho.z *= a;
}

void ApplyScaling(float a, float3 *out, int num_losses, cudaStream_t stream) {
  if (num_losses <= 0) {
    return;
  }
  size_t num_blocks = (num_losses + kScaledBlockSize - 1) / kScaledBlockSize;
  scaled_loss_apply_kernel<<<num_blocks, kScaledBlockSize, 0, stream>>>(
      a, out, num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

} // namespace cunls
