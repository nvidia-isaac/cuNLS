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
#include "cunls/common/log.h"
#include "cunls/minimizer/problem_kernels.h"

namespace cunls {
namespace problem_internal {

namespace {

constexpr int kBlockSize = 256;

inline int GridFor(size_t count) { return static_cast<int>((count + kBlockSize - 1) / kBlockSize); }

/** Records the first error: the thread that wins the CAS on `code` writes the details. */
__device__ void Report(ValidationError *error, int code, int batch, long long entry) {
  if (atomicCAS(&error->code, 0, code) == 0) {
    error->batch = batch;
    error->entry = entry;
  }
}

__global__ void ExpandIndicesKernel(const int *__restrict__ indices, size_t count, SlotTable slots,
                                    float **__restrict__ out) {
  const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= count) {
    return;
  }
  const int b = static_cast<int>(t % slots.num_slots);
  out[t] =
      const_cast<float *>(slots.base[b]) + static_cast<ptrdiff_t>(indices[t]) * slots.ambient[b];
}

__global__ void CheckPointersKernel(float *const *__restrict__ pointers, size_t count, int batch,
                                    SlotTangents slots, const StateRange *__restrict__ ranges,
                                    int num_ranges, int *__restrict__ used,
                                    ValidationError *error) {
  const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= count) {
    return;
  }
  const float *p = pointers[t];
  for (int r = 0; r < num_ranges; ++r) {
    const StateRange &range = ranges[r];
    const ptrdiff_t diff = p - range.base;
    if (diff < 0 || diff >= static_cast<ptrdiff_t>(range.num_blocks) * range.ambient ||
        diff % range.ambient != 0) {
      continue;
    }
    if (range.tangent != slots.tangent[t % slots.num_slots]) {
      Report(error, 2, batch, static_cast<long long>(t));
      return;
    }
    used[range.used_offset + diff / range.ambient] = 1;
    return;
  }
  Report(error, 1, batch, static_cast<long long>(t));
}

__global__ void CheckUsedKernel(const int *__restrict__ used, int batch, StateRange range,
                                ValidationError *error) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < range.num_blocks && used[range.used_offset + k] == 0) {
    Report(error, 3, batch, k);
  }
}

__global__ void CheckConstantIdsKernel(const int *__restrict__ ids, size_t count, int num_blocks,
                                       int batch, ValidationError *error) {
  const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < count && (ids[k] < 0 || ids[k] >= num_blocks)) {
    Report(error, 4, batch, static_cast<long long>(k));
  }
}

}  // namespace

void LaunchExpandIndices(cudaStream_t stream, const int *indices, size_t count,
                         const SlotTable &slots, float **out) {
  if (count == 0) {
    return;
  }
  ExpandIndicesKernel<<<GridFor(count), kBlockSize, 0, stream>>>(indices, count, slots, out);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchCheckPointers(cudaStream_t stream, float *const *pointers, size_t count, int batch,
                         const SlotTangents &slots, const StateRange *ranges, int num_ranges,
                         int *used, ValidationError *error) {
  if (count == 0) {
    return;
  }
  CheckPointersKernel<<<GridFor(count), kBlockSize, 0, stream>>>(pointers, count, batch, slots,
                                                                 ranges, num_ranges, used, error);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchCheckUsed(cudaStream_t stream, const int *used, int batch, const StateRange &range,
                     ValidationError *error) {
  if (range.num_blocks == 0) {
    return;
  }
  CheckUsedKernel<<<GridFor(range.num_blocks), kBlockSize, 0, stream>>>(used, batch, range, error);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

void LaunchCheckConstantIds(cudaStream_t stream, const int *ids, size_t count, int num_blocks,
                            int batch, ValidationError *error) {
  if (count == 0 || ids == nullptr) {
    return;
  }
  CheckConstantIdsKernel<<<GridFor(count), kBlockSize, 0, stream>>>(ids, count, num_blocks, batch,
                                                                    error);
  THROW_ON_CUDA_ERROR(cudaGetLastError());
}

}  // namespace problem_internal
}  // namespace cunls
