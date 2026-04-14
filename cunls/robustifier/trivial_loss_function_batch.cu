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

#include <stdint.h>

#include "cunls/common/helper.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"

namespace cunls {

/** @brief CUDA block size for the trivial loss kernel. */
constexpr size_t block_size = 256;

/**
 * @brief CUDA kernel that computes the trivial (identity) loss for each squared
 * residual.
 *
 * Sets out[tid] = {s[tid], 1.0, 0} for each thread, representing
 * rho(s) = s, rho'(s) = 1, rho''(s) = 0.
 *
 * @param s          Device array of squared residuals (input).
 * @param out        Device array of float3 output triplets (rho, rho', rho'').
 * @param num_losses Total number of residuals.
 *
 * Grid/block: launched with ceil(num_losses / 256) blocks of 256 threads.
 */
__global__ void trivial_loss_kernel(float *s, float3 *out, int num_losses) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_losses) {
    return;
  }

  out[tid] = {s[tid], 1.0, 0};
}

/** @copydoc TrivialLossFunctionBatch::Evaluate */
bool TrivialLossFunctionBatch::Evaluate(float *s, float3 *out, int num_losses,
                                        cudaStream_t stream) const {
  if (num_losses <= 0) {
    return true;
  }
  size_t num_blocks = (num_losses + block_size - 1) / block_size;
  trivial_loss_kernel<<<num_blocks, block_size, 0, stream>>>(s, out,
                                                             num_losses);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}
} // namespace cunls
