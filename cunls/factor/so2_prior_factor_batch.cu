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

#include <cassert>

#include "cunls/common/helper.h"
#include "cunls/factor/so2_prior_factor_batch.h"

namespace cunls {

/// Number of threads per CUDA block for SO2 prior cost kernel launches.
constexpr size_t kSO2BlockSize = 256;

/**
 * @brief CUDA kernel that computes SO(2) prior residuals and Jacobians.
 *
 * For each rotation i, computes:
 *   - R_error = R_target^T * R_current
 *   - residual[i] = atan2(sin(error), cos(error))  (Log map for SO(2))
 *   - jacobian[i] = 1.0  (SO(2) is abelian, so the Jacobian is always 1)
 *
 * @param observations Target rotation matrices (4 floats each, row-major: [c, -s, s, c])
 * @param state_pointers Array of state block pointers (each points to a 2x2 rotation)
 * @param residuals Output residuals (1 float per factor), or nullptr to skip
 * @param jacobians Output Jacobians (1 float per factor), or nullptr to skip
 * @param num_factors Number of factors to process
 */
__global__ void so2_prior_cost_kernel(const float* observations,
                                      float const* const* state_pointers,
                                      float* residuals, float* jacobians,
                                      int num_factors) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_factors) return;

  // Read current rotation matrix [c, -s, s, c]
  const float* R = state_pointers[tid];
  assert(R != nullptr);
  float c_curr = R[0];
  float s_curr = R[2];

  // Read target rotation matrix
  const float* R_target = observations + tid * 4;
  float c_tgt = R_target[0];
  float s_tgt = R_target[2];

  // Compute R_error = R_target^T * R_current
  // R_target^T = [[c_tgt, s_tgt], [-s_tgt, c_tgt]]
  float c_err = c_tgt * c_curr + s_tgt * s_curr;
  float s_err = c_tgt * s_curr - s_tgt * c_curr;

  // Residual = Log(R_error) = atan2(sin, cos)
  if (residuals != nullptr) {
    residuals[tid] = atan2f(s_err, c_err);
  }

  // Jacobian = 1.0 (SO(2) is abelian, all Jacobians are identity)
  if (jacobians != nullptr) {
    jacobians[tid] = 1.0f;
  }
}

bool SO2PriorFactorBatch::Evaluate(float* residuals, float* jacobians,
                                         float const* const* state_pointers,
                                         cudaStream_t stream) const {
  auto data_ptr = reinterpret_cast<const float*>(observations_ptr_);
  int num_factors = static_cast<int>(NumFactors());

  size_t num_blocks = (num_factors + kSO2BlockSize - 1) / kSO2BlockSize;
  so2_prior_cost_kernel<<<num_blocks, kSO2BlockSize, 0, stream>>>(
      data_ptr, state_pointers, residuals, jacobians, num_factors);

  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
