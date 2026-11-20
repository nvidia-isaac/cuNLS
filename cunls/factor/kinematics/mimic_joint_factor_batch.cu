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

#include <cmath>
#include <stdexcept>

#include "cunls/common/helper.h"
#include "cunls/factor/kinematics/mimic_joint_factor_batch.h"

namespace cunls {

namespace {

constexpr int kBlockSize = 128;
constexpr int kGroup = 8;

/** Factor (measurement) index of item t (see FactorBatch::Evaluate). */
__device__ __forceinline__ int FactorIndex(int t, const int *factor_ids, int num_factors) {
  if (factor_ids != nullptr) return factor_ids[t];
  return t < num_factors ? t : t % num_factors;
}

/** One group of 8 lanes per item; lane l writes columns l, l + 8, ... of the one row. */
template <bool kJacobian>
__global__ void __launch_bounds__(kBlockSize)
    MimicJointKernel(ArticulationTopology::DeviceView v, const int *__restrict__ mimic_dofs,
                     const int *__restrict__ reference_dofs, const float *__restrict__ coefficients,
                     float const *const *__restrict__ state_ptrs, float *__restrict__ residuals,
                     float *__restrict__ jacobians, int num_items,
                     const int *__restrict__ factor_ids, int num_factors) {
  const int t =
      static_cast<int>((blockIdx.x * static_cast<size_t>(kBlockSize) + threadIdx.x) / kGroup);
  const int lane = threadIdx.x % kGroup;
  if (t >= num_items) return;
  const int f = FactorIndex(t, factor_ids, num_factors);
  const int dm = __ldg(mimic_dofs + f), dr = __ldg(reference_dofs + f);
  const float a = __ldg(coefficients + 2 * f), b = __ldg(coefficients + 2 * f + 1);
  // A DOF outside the topology, or of a BALL / FREE joint, has no scalar coordinate: NaN residual.
  const bool in_range = dm >= 0 && dm < v.num_dofs && dr >= 0 && dr < v.num_dofs;
  const int cm = in_range ? v.scalar_coord[dm] : -1, cr = in_range ? v.scalar_coord[dr] : -1;
  const float *x = state_ptrs[t];
  if (lane == 0) {
    residuals[t] = (cm < 0 || cr < 0) ? NAN : x[cm] - (a + b * x[cr]);
  }
  if constexpr (kJacobian) {
    float *J = jacobians + static_cast<size_t>(t) * v.num_dofs;
    // A DOF without a scalar coordinate (out of range, BALL or FREE): NaN residual, zero row.
    const bool valid = cm >= 0 && cr >= 0;
    for (int c = lane; c < v.num_dofs; c += kGroup) {
      J[c] = valid ? (c == dm ? 1.f : 0.f) - (c == dr ? b : 0.f) : 0.f;
    }
  }
}

}  // namespace

MimicJointFactorBatch::MimicJointFactorBatch(const ArticulationTopology &topology,
                                             const int *mimic_dofs, const int *reference_dofs,
                                             const float *coefficients, size_t capacity)
    : ArticulationFactorBatch(topology, 1, capacity),
      mimic_dofs_(mimic_dofs),
      reference_dofs_(reference_dofs),
      coefficients_(coefficients) {
  if (mimic_dofs == nullptr || reference_dofs == nullptr || coefficients == nullptr) {
    throw std::invalid_argument(
        "MimicJointFactorBatch: mimic_dofs, reference_dofs and coefficients must be device "
        "arrays (got a null pointer). Fix: pass all three arrays.");
  }
}

bool MimicJointFactorBatch::Evaluate(float *residuals, float *jacobians,
                                     float const *const *state_pointers, cudaStream_t stream,
                                     const int *factor_ids, size_t num_factor_ids) const {
  const size_t num_factors = NumActiveFactors();
  const size_t n = num_factor_ids == 0 ? num_factors : num_factor_ids;
  if (n == 0 || num_factors == 0) return true;
  const unsigned blocks = static_cast<unsigned>((n * kGroup + kBlockSize - 1) / kBlockSize);
  if (jacobians != nullptr) {
    MimicJointKernel<true><<<blocks, kBlockSize, 0, stream>>>(
        Topology().View(), mimic_dofs_, reference_dofs_, coefficients_, state_pointers, residuals,
        jacobians, static_cast<int>(n), factor_ids, static_cast<int>(num_factors));
  } else {
    MimicJointKernel<false><<<blocks, kBlockSize, 0, stream>>>(
        Topology().View(), mimic_dofs_, reference_dofs_, coefficients_, state_pointers, residuals,
        nullptr, static_cast<int>(n), factor_ids, static_cast<int>(num_factors));
  }
  THROW_ON_CUDA_ERROR(cudaGetLastError());
  return true;
}

}  // namespace cunls
