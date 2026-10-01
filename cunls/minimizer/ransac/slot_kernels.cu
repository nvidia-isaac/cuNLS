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

/**
 * @file slot_kernels.cu
 * @brief Elementwise kernels of the RANSAC minimizers: minimal samples,
 * state-pointer tables, replica copies, step scatter and accept / reset.
 * One thread per output element, no atomics.
 */

#include <cmath>

#include "cunls/common/helper.h"
#include "cunls/minimizer/ransac/kernel_common.cuh"

namespace cunls {
namespace ransac_internal {

namespace {

constexpr int kThreads = 256;

__global__ void DrawSamplesKernel(int n, int sample_size, int num_slots, uint64_t seed,
                                  uint64_t round, int *samples) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_slots * sample_size) {
    return;
  }
  const int slot = idx / sample_size;
  const uint64_t key = PermutationKey(seed, round, static_cast<uint64_t>(slot));
  samples[idx] = static_cast<int>(
      PermuteIndex(static_cast<uint32_t>(idx % sample_size), static_cast<uint32_t>(n), key));
}

/** Replica of `block` for `slot` (current or candidate), or user storage if slot < 0. */
__device__ float *ReplicaPointer(const StateView &state, int block, int slot, bool candidate) {
  float *replicas = candidate ? state.rep_cand : state.rep_cur;
  if (slot < 0 || replicas == nullptr) {
    return const_cast<float *>(state.base) + static_cast<size_t>(block) * state.ambient;
  }
  return replicas + (static_cast<size_t>(slot) * state.num_blocks + block) * state.ambient;
}

__global__ void SampleTablesKernel(const StateView *states, const int2 *blocks, int nb,
                                   int num_factors, int u_offset, const int *samples,
                                   int sample_size, int num_items, bool candidate, float **table,
                                   int *factor_ids) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_items * nb) {
    return;
  }
  const int item = idx / nb;
  const int b = idx % nb;
  const int f = samples[item] - u_offset;
  const int factor = (f >= 0 && f < num_factors) ? f : 0;  // other batches' entries: unused rows
  const int2 sb = blocks[static_cast<size_t>(factor) * nb + b];
  table[idx] = ReplicaPointer(states[sb.x], sb.y, item / sample_size, candidate);
  if (b == 0) {
    factor_ids[item] = factor;
  }
}

__global__ void SlotTablesKernel(const StateView *states, const int2 *blocks, int num_factors,
                                 int nb, int num_slots, int slot_offset, bool candidate,
                                 float **tables) {
  const size_t per_slot = static_cast<size_t>(num_factors) * nb;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= per_slot * num_slots) {
    return;
  }
  const int2 sb = blocks[idx % per_slot];
  tables[idx] =
      ReplicaPointer(states[sb.x], sb.y, slot_offset + static_cast<int>(idx / per_slot), candidate);
}

__global__ void IndexedSlotTablesKernel(const StateView *states, const int2 *blocks,
                                        int num_factors, int nb, const int *slot_index,
                                        int num_slots, float **tables) {
  const size_t per_slot = static_cast<size_t>(num_factors) * nb;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= per_slot * num_slots) {
    return;
  }
  const int2 sb = blocks[idx % per_slot];
  tables[idx] = ReplicaPointer(states[sb.x], sb.y, slot_index[idx / per_slot], false);
}

__global__ void SubsetTablesKernel(const StateView *states, const int2 *blocks, int nb,
                                   const int *ids, int count, int num_slots, int slot_offset,
                                   float **tables, int *item_ids) {
  const size_t items = static_cast<size_t>(count) * num_slots;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= items * nb) {
    return;
  }
  const size_t t = idx / nb;
  const int b = static_cast<int>(idx % nb);
  const int f = ids[t % count];
  const int2 sb = blocks[static_cast<size_t>(f) * nb + b];
  const int slot = slot_offset + static_cast<int>(t / count);
  tables[idx] = ReplicaPointer(states[sb.x], sb.y, slot, false);
  if (b == 0) {
    item_ids[t] = f;
  }
}

__global__ void PermutationPrefixKernel(int n, int count, uint64_t key, int *ids) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < count) {
    ids[j] =
        static_cast<int>(PermuteIndex(static_cast<uint32_t>(j), static_cast<uint32_t>(n), key));
  }
}

__global__ void GatherLocalColumnsKernel(const int *local_col, const int *ids, int count, int nb,
                                         int *out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count * nb) {
    out[idx] = local_col[static_cast<size_t>(ids[idx / nb]) * nb + idx % nb];
  }
}

__global__ void FillKernel(float *data, size_t n, float value) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = value;
  }
}

__global__ void ScatterDeltaKernel(int num_slots, int dim, const float *delta, int num_blocks,
                                   int tangent, const int *block_col, float *delta_full) {
  const size_t per_slot = static_cast<size_t>(num_blocks) * tangent;
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= per_slot * num_slots) {
    return;
  }
  const size_t slot = idx / per_slot;
  const int col = block_col[(idx % per_slot) / tangent];
  delta_full[idx] = col >= 0 ? delta[slot * dim + col + idx % tangent] : 0.f;
}

__global__ void CopyAcceptedKernel(int num_slots, size_t slot_floats, const int *accept,
                                   const float *cand, float *cur) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < slot_floats * num_slots && accept[idx / slot_floats]) {
    cur[idx] = cand[idx];
  }
}

__global__ void CopyReplicasKernel(int num_slots, size_t slot_floats, const float *src,
                                   const int *src_slot, const int *only_if, float *dst) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if ((only_if != nullptr && *only_if == 0) || idx >= slot_floats * num_slots) {
    return;
  }
  const size_t s = src_slot != nullptr ? static_cast<size_t>(*src_slot) : 0;
  dst[idx] = src[s * slot_floats + idx % slot_floats];
}

/** Gauss-Newton rule: accept a cost decrease; stop otherwise or when converged. */
__device__ void GaussNewtonDecision(const DevicePolicy &pol, bool ok, float cc, float cn,
                                    float step_sq, int &accept, bool &stop) {
  accept = (ok && isfinite(cn) && cn < cc) ? 1 : 0;
  stop = !accept || (cc - cn) <= pol.cost_tolerance * cc || step_sq <= pol.state_tolerance;
}

/** Levenberg-Marquardt rule: gain-ratio test with damping updates. */
__device__ void LevenbergMarquardtDecision(const DevicePolicy &pol, bool ok, float cc, float cn,
                                           float predicted, float step_sq, float &lambda,
                                           int &accept, bool &stop) {
  const float rho = predicted > 0.f ? (cc - cn) / predicted : (cn < cc ? 1.f : -1.f);
  accept = (ok && isfinite(cn) && isfinite(rho) && rho >= pol.step_accept_threshold) ? 1 : 0;
  if (accept) {
    if (rho > pol.lambda_downscale_threshold) {
      lambda = fmaxf(lambda * pol.lambda_downscale, pol.lambda_min);
    }
    stop = (cc - cn) <= pol.cost_tolerance * cc || step_sq <= pol.state_tolerance;
  } else {
    lambda = fminf(lambda * pol.lambda_upscale, 2.f * pol.lambda_max);
    stop = lambda > pol.lambda_max;
  }
}

__global__ void AcceptKernel(int num_slots, DevicePolicy policy, const float *cost_cur,
                             const float *cost_cand, const float *predicted, const float *step_sq,
                             const int *solve_ok, int *active, float *lambda, int *accept,
                             int *valid, int *iterations, int *num_accepted) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_slots) {
    return;
  }
  if (active[p] == 0) {
    accept[p] = 0;
    return;
  }
  const bool ok = solve_ok[p] != 0;
  if (iterations[p]++ == 0 && valid != nullptr) {
    valid[p] = ok ? 1 : 0;
  }
  int acc = 0;
  bool stop = false;
  if (policy.levenberg_marquardt) {
    LevenbergMarquardtDecision(policy, ok, cost_cur[p], cost_cand[p], predicted[p], step_sq[p],
                               lambda[p], acc, stop);
  } else {
    GaussNewtonDecision(policy, ok, cost_cur[p], cost_cand[p], step_sq[p], acc, stop);
  }
  accept[p] = acc;
  num_accepted[p] += acc;
  if (stop) {
    active[p] = 0;
  }
}

__global__ void ResetSlotsKernel(int num_slots, float initial_lambda, int *active, float *lambda,
                                 int *valid, int *iterations, int *num_accepted) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_slots) {
    return;
  }
  active[p] = 1;
  lambda[p] = initial_lambda;
  iterations[p] = 0;
  num_accepted[p] = 0;
  if (valid != nullptr) {
    valid[p] = 1;
  }
}

__global__ void CountActiveKernel(int num_slots, const int *active, int *count) {
  __shared__ int scratch[kThreads / kWarpSize];
  int sum = 0;
  for (int p = threadIdx.x; p < num_slots; p += blockDim.x) {
    sum += active[p] != 0 ? 1 : 0;
  }
  sum = BlockSum(sum, scratch);
  if (threadIdx.x == 0) {
    *count = sum;
  }
}

}  // namespace

void LaunchCountActive(cudaStream_t stream, int num_slots, const int *active, int *count) {
  CountActiveKernel<<<1, kThreads, 0, stream>>>(num_slots, active, count);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchDrawSamples(cudaStream_t stream, int n, int sample_size, int num_slots, uint64_t seed,
                       uint64_t round, int *samples) {
  const int count = num_slots * sample_size;
  if (count <= 0) {
    return;
  }
  DrawSamplesKernel<<<GridFor(count), kThreads, 0, stream>>>(n, sample_size, num_slots, seed, round,
                                                             samples);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchSampleTables(cudaStream_t stream, const StateView *states, const int2 *blocks, int nb,
                        int num_factors, int u_offset, const int *samples, int sample_size,
                        int num_slots, bool candidate, float **table, int *factor_ids) {
  const int items = num_slots * sample_size;
  if (items <= 0) {
    return;
  }
  SampleTablesKernel<<<GridFor(static_cast<size_t>(items) * nb), kThreads, 0, stream>>>(
      states, blocks, nb, num_factors, u_offset, samples, sample_size, items, candidate, table,
      factor_ids);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchSlotTables(cudaStream_t stream, const StateView *states, const int2 *blocks,
                      int num_factors, int nb, int num_slots, int slot_offset, bool candidate,
                      float **tables) {
  const size_t count = static_cast<size_t>(num_factors) * nb * num_slots;
  if (count == 0) {
    return;
  }
  SlotTablesKernel<<<GridFor(count), kThreads, 0, stream>>>(
      states, blocks, num_factors, nb, num_slots, slot_offset, candidate, tables);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchIndexedSlotTables(cudaStream_t stream, const StateView *states, const int2 *blocks,
                             int num_factors, int nb, const int *slot_index, int num_slots,
                             float **tables) {
  const size_t count = static_cast<size_t>(num_factors) * nb * num_slots;
  if (count == 0) {
    return;
  }
  IndexedSlotTablesKernel<<<GridFor(count), kThreads, 0, stream>>>(states, blocks, num_factors, nb,
                                                                   slot_index, num_slots, tables);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchSubsetTables(cudaStream_t stream, const StateView *states, const int2 *blocks, int nb,
                        const int *ids, int count, int num_slots, int slot_offset, float **tables,
                        int *item_ids) {
  const size_t total = static_cast<size_t>(count) * num_slots * nb;
  if (total == 0) {
    return;
  }
  SubsetTablesKernel<<<GridFor(total), kThreads, 0, stream>>>(
      states, blocks, nb, ids, count, num_slots, slot_offset, tables, item_ids);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchPermutationPrefix(cudaStream_t stream, int n, int count, uint64_t key, int *ids) {
  if (count <= 0) {
    return;
  }
  PermutationPrefixKernel<<<GridFor(count), kThreads, 0, stream>>>(n, count, key, ids);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchGatherLocalColumns(cudaStream_t stream, const int *local_col, const int *ids, int count,
                              int nb, int *out) {
  if (count * nb <= 0) {
    return;
  }
  GatherLocalColumnsKernel<<<GridFor(count * nb), kThreads, 0, stream>>>(local_col, ids, count, nb,
                                                                         out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchFill(cudaStream_t stream, float *data, size_t n, float value) {
  if (n == 0) {
    return;
  }
  FillKernel<<<GridFor(n), kThreads, 0, stream>>>(data, n, value);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchScatterDelta(cudaStream_t stream, int num_slots, int dim, const float *delta,
                        int num_blocks, int tangent, const int *block_col, float *delta_full) {
  const size_t count = static_cast<size_t>(num_slots) * num_blocks * tangent;
  if (count == 0) {
    return;
  }
  ScatterDeltaKernel<<<GridFor(count), kThreads, 0, stream>>>(num_slots, dim, delta, num_blocks,
                                                              tangent, block_col, delta_full);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchCopyAccepted(cudaStream_t stream, int num_slots, size_t slot_floats, const int *accept,
                        const float *cand, float *cur) {
  const size_t count = slot_floats * num_slots;
  if (count == 0) {
    return;
  }
  CopyAcceptedKernel<<<GridFor(count), kThreads, 0, stream>>>(num_slots, slot_floats, accept, cand,
                                                              cur);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchCopyReplicas(cudaStream_t stream, int num_slots, size_t slot_floats, const float *src,
                        const int *src_slot, const int *only_if, float *dst) {
  const size_t count = slot_floats * num_slots;
  if (count == 0) {
    return;
  }
  CopyReplicasKernel<<<GridFor(count), kThreads, 0, stream>>>(num_slots, slot_floats, src, src_slot,
                                                              only_if, dst);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchAccept(cudaStream_t stream, int num_slots, const DevicePolicy &policy,
                  const float *cost_cur, const float *cost_cand, const float *predicted,
                  const float *step_sq, const int *solve_ok, int *active, float *lambda,
                  int *accept, int *valid, int *iterations, int *num_accepted) {
  if (num_slots <= 0) {
    return;
  }
  AcceptKernel<<<GridFor(num_slots), kThreads, 0, stream>>>(
      num_slots, policy, cost_cur, cost_cand, predicted, step_sq, solve_ok, active, lambda, accept,
      valid, iterations, num_accepted);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchResetSlots(cudaStream_t stream, int num_slots, float initial_lambda, int *active,
                      float *lambda, int *valid, int *iterations, int *num_accepted) {
  if (num_slots <= 0) {
    return;
  }
  ResetSlotsKernel<<<GridFor(num_slots), kThreads, 0, stream>>>(
      num_slots, initial_lambda, active, lambda, valid, iterations, num_accepted);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace ransac_internal
}  // namespace cunls
