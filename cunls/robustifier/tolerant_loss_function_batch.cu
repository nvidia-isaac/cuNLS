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
#include "cunls/robustifier/tolerant_loss_function_batch.h"

namespace cunls {

constexpr size_t kTolerantBlockSize = 256;
constexpr float kLog2Pow53 = 36.7f;

__global__ void tolerant_loss_kernel(float a, float b, float c, float* s,
                                     float3* out, int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  const float sq_error = s[tid];
  const float x = (sq_error - a) / b;

  float3& rho = out[tid];

  if (x > kLog2Pow53) {
    rho.x = sq_error - a - c;
    rho.y = 1.0f;
    rho.z = 0.0f;
  } else {
    const float e_x = expf(x);
    const float one_plus_e_x = 1.0f + e_x;
    rho.x = b * logf(one_plus_e_x) - c;
    rho.y = fmaxf(cuda::std::numeric_limits<float>::min(), e_x / one_plus_e_x);
    rho.z = 0.5f / (b * (1.0f + coshf(x)));
  }
}

TolerantLossFunctionBatch::TolerantLossFunctionBatch(float a, float b)
    : a_(a), b_(b), c_(b * logf(1.0f + expf(-a / b))) {}

bool TolerantLossFunctionBatch::Evaluate(float* s, float3* out, int num_losses,
                                         cudaStream_t stream) const {
  if (num_losses <= 0) {
    return true;
  }
  size_t num_blocks =
      (num_losses + kTolerantBlockSize - 1) / kTolerantBlockSize;
  tolerant_loss_kernel<<<num_blocks, kTolerantBlockSize, 0, stream>>>(
      a_, b_, c_, s, out, num_losses);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
