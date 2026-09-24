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
#include "cunls/factor/motion_prior_information.h"

namespace cunls {

constexpr size_t kMotionPriorInfoBlockSize = 256;

// Fills sqrt_information[tid] = L_M(dt[tid]) (x) diag(1/sqrt(qc)), the 2x2
// block-Kronecker structure described in motion_prior_information.h.
template <int Dim>
__global__ void constant_velocity_sqrt_information_kernel(const float *dt, const float *qc_diag,
                                                          size_t num_factors,
                                                          float *sqrt_information_out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  constexpr int N = 2 * Dim;
  const float t = dt[tid];
  const float inv_sqrt_t = rsqrtf(t);
  const float l11 = 2.0f * 1.7320508f * inv_sqrt_t * inv_sqrt_t * inv_sqrt_t;  // 2*sqrt(3)/t^1.5
  const float l21 = -1.7320508f * inv_sqrt_t;                                  // -sqrt(3)/sqrt(t)
  const float l22 = inv_sqrt_t;                                                // 1/sqrt(t)
  const float lm[2][2] = {{l11, 0.0f}, {l21, l22}};

  float *out = sqrt_information_out + (size_t)tid * N * N;
#pragma unroll
  for (int r = 0; r < N * N; r++) out[r] = 0.0f;

    // M(dt) = lm * lm^T (lm lower-triangular). InformationFactorBatch needs S
    // with S^T S == M, i.e. S = lm^T (upper-triangular): write lm[p][q] to
    // block position (q, p), not (p, q).
#pragma unroll
  for (int p = 0; p < 2; p++)
#pragma unroll
    for (int q = 0; q < 2; q++) {
      if (lm[p][q] == 0.0f) continue;
      for (int i = 0; i < Dim; i++) {
        const float inv_sqrt_qc = rsqrtf(qc_diag[i]);
        out[(q * Dim + i) * N + (p * Dim + i)] = lm[p][q] * inv_sqrt_qc;
      }
    }
}

// Fills sqrt_information[tid] = L_M(dt[tid]) (x) diag(1/sqrt(qc)), the 3x3
// block-Kronecker structure described in motion_prior_information.h.
template <int Dim>
__global__ void constant_acceleration_sqrt_information_kernel(const float *dt, const float *qc_diag,
                                                              size_t num_factors,
                                                              float *sqrt_information_out) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (int)num_factors) return;

  constexpr int N = 3 * Dim;
  const float t = dt[tid];
  const float inv_sqrt_t = rsqrtf(t);
  const float t_m1 = inv_sqrt_t * inv_sqrt_t;  // 1/t
  const float t_m1_5 = t_m1 * inv_sqrt_t;      // 1/t^1.5
  const float t_m2_5 = t_m1 * t_m1_5;          // 1/t^2.5
  const float sqrt5 = 2.2360680f;
  const float sqrt3 = 1.7320508f;

  const float l11 = 12.0f * sqrt5 * t_m2_5;
  const float l21 = -6.0f * sqrt5 * t_m1_5;
  const float l22 = 2.0f * sqrt3 * t_m1_5;
  const float l31 = sqrt5 * inv_sqrt_t;
  const float l32 = -sqrt3 * inv_sqrt_t;
  const float l33 = inv_sqrt_t;
  const float lm[3][3] = {{l11, 0.0f, 0.0f}, {l21, l22, 0.0f}, {l31, l32, l33}};

  float *out = sqrt_information_out + (size_t)tid * N * N;
#pragma unroll
  for (int r = 0; r < N * N; r++) out[r] = 0.0f;

    // Same transpose reasoning as the CV kernel above: write lm[p][q] to
    // block position (q, p).
#pragma unroll
  for (int p = 0; p < 3; p++)
#pragma unroll
    for (int q = 0; q < 3; q++) {
      if (lm[p][q] == 0.0f) continue;
      for (int i = 0; i < Dim; i++) {
        const float inv_sqrt_qc = rsqrtf(qc_diag[i]);
        out[(q * Dim + i) * N + (p * Dim + i)] = lm[p][q] * inv_sqrt_qc;
      }
    }
}

template <int Dim>
void ComputeConstantVelocitySqrtInformation(cudaStream_t stream, const float *dt_ptr,
                                            const float *qc_diag_ptr, size_t num_factors,
                                            float *sqrt_information_out) {
  const size_t num_blocks =
      (num_factors + kMotionPriorInfoBlockSize - 1) / kMotionPriorInfoBlockSize;
  constant_velocity_sqrt_information_kernel<Dim>
      <<<num_blocks, kMotionPriorInfoBlockSize, 0, stream>>>(dt_ptr, qc_diag_ptr, num_factors,
                                                             sqrt_information_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

template <int Dim>
void ComputeConstantAccelerationSqrtInformation(cudaStream_t stream, const float *dt_ptr,
                                                const float *qc_diag_ptr, size_t num_factors,
                                                float *sqrt_information_out) {
  const size_t num_blocks =
      (num_factors + kMotionPriorInfoBlockSize - 1) / kMotionPriorInfoBlockSize;
  constant_acceleration_sqrt_information_kernel<Dim>
      <<<num_blocks, kMotionPriorInfoBlockSize, 0, stream>>>(dt_ptr, qc_diag_ptr, num_factors,
                                                             sqrt_information_out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

// Explicit instantiations for the four supported pose tangent dimensions:
// SE(3)=6, SE(2)/SO(3)=3, SO(2)=1.
template void ComputeConstantVelocitySqrtInformation<6>(cudaStream_t, const float *, const float *,
                                                        size_t, float *);
template void ComputeConstantVelocitySqrtInformation<3>(cudaStream_t, const float *, const float *,
                                                        size_t, float *);
template void ComputeConstantVelocitySqrtInformation<1>(cudaStream_t, const float *, const float *,
                                                        size_t, float *);

template void ComputeConstantAccelerationSqrtInformation<6>(cudaStream_t, const float *,
                                                            const float *, size_t, float *);
template void ComputeConstantAccelerationSqrtInformation<3>(cudaStream_t, const float *,
                                                            const float *, size_t, float *);
template void ComputeConstantAccelerationSqrtInformation<1>(cudaStream_t, const float *,
                                                            const float *, size_t, float *);

}  // namespace cunls
