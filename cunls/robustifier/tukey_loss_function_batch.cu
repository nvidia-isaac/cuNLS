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
#include "cunls/robustifier/tukey_loss_function_batch.h"

namespace cunls {

constexpr size_t kTukeyBlockSize = 256;

__global__ void tukey_loss_kernel(float a_squared, float *s, float3 *out,
                                  int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  const float sq_error = s[tid];
  float3 &rho = out[tid];

  if (sq_error <= a_squared) {
    const float value = 1.0f - sq_error / a_squared;
    const float value_sq = value * value;
    rho.x = (a_squared / 3.0f) * (1.0f - value_sq * value);
    rho.y = value_sq;
    rho.z = -2.0f / a_squared * value;
  } else {
    rho.x = a_squared / 3.0f;
    rho.y = 0.0f;
    rho.z = 0.0f;
  }
}

TukeyLossFunctionBatch::TukeyLossFunctionBatch(float a) : a_squared_(a * a) {}

bool TukeyLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                      cudaStream_t stream) const {
  if (num_losses <= 0) {
    return true;
  }
  size_t num_blocks = (num_losses + kTukeyBlockSize - 1) / kTukeyBlockSize;
  tukey_loss_kernel<<<num_blocks, kTukeyBlockSize, 0, stream>>>(
      a_squared_, s, out, num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

} // namespace cunls
