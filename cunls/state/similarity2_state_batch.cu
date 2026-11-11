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
#include "cunls/math/lie_device.cuh"
#include "cunls/state/similarity2_state_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 256;

__global__ void sim2_apply_update_fused_kernel(const float *__restrict__ x,
                                               const float *__restrict__ delta,
                                               float *__restrict__ result, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  const float *xi = delta + tid * 4;
  float ux = xi[0], uy = xi[1], theta = xi[2], lambda = xi[3];

  const float c = cosf(theta);
  const float s = sinf(theta);
  const float inv_scale = expf(-lambda);

  float X, thetaY;
  lie_device::Sim2VCoefficients(theta, lambda, &X, &thetaY);
  const float tx = X * ux - thetaY * uy;
  const float ty = thetaY * ux + X * uy;

  const float e00 = c, e01 = -s, e02 = tx;
  const float e10 = s, e11 = c, e12 = ty;
  const float e20 = 0.0f, e21 = 0.0f, e22 = inv_scale;

  const float *xm = x + tid * 9;
  float *rm = result + tid * 9;

#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const float xr0 = xm[r * 3 + 0];
    const float xr1 = xm[r * 3 + 1];
    const float xr2 = xm[r * 3 + 2];
    rm[r * 3 + 0] = xr0 * e00 + xr1 * e10 + xr2 * e20;
    rm[r * 3 + 1] = xr0 * e01 + xr1 * e11 + xr2 * e21;
    rm[r * 3 + 2] = xr0 * e02 + xr1 * e12 + xr2 * e22;
  }
}

}  // namespace

Similarity2StateBatch::Similarity2StateBatch(const float *device_ptr, size_t capacity)
    : Base(device_ptr, capacity) {}

Similarity2StateBatch::Similarity2StateBatch(const float *device_ptr, size_t capacity,
                                             const int *device_constant_state_ids,
                                             size_t const_capacity)
    : Base(device_ptr, capacity, device_constant_state_ids, const_capacity) {}

void Similarity2StateBatch::Plus(const float *x, const float *delta, float *x_plus_delta,
                                 cudaStream_t stream, size_t num_replicas) {
  const size_t num_transforms = NumActiveStates() * num_replicas;
  if (num_transforms == 0) {
    return;
  }
  const int n = static_cast<int>(num_transforms);
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  sim2_apply_update_fused_kernel<<<grid, kBlockSize, 0, stream>>>(x, delta, x_plus_delta, n);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace cunls
