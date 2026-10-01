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

#pragma once

/**
 * @file problem_kernels.h
 * @brief Kernels behind Problem's device connectivity: expanding index tables
 * into pointer tables, and GPU validation. Internal (not in cunls.h).
 */

#include <cuda_runtime.h>

#include <cstddef>

namespace cunls {
namespace problem_internal {

/** @brief Most state slots a factor may have (state batches per index table). */
constexpr int kMaxSlots = 8;

/** @brief Base address and ambient size of the state batch of each state slot. */
struct SlotTable {
  const float *base[kMaxSlots] = {};
  int ambient[kMaxSlots] = {};
  int num_slots = 0;
};

/**
 * @brief out[t] = base[b] + indices[t] * ambient[b] for t < count, b = t % num_slots.
 * Out-of-range indices produce out-of-range pointers, which validation reports.
 */
void LaunchExpandIndices(cudaStream_t stream, const int *indices, size_t count,
                         const SlotTable &slots, float **out);

/** @brief Active range of one state batch, for validation. */
struct StateRange {
  const float *base = nullptr;
  int num_blocks = 0;  ///< Active states.
  int ambient = 0;
  int tangent = 0;
  int used_offset = 0;  ///< Offset of the batch's flags in the `used` array.
};

/** @brief First validation failure found on the device (code 0 = none). */
struct ValidationError {
  int code = 0;         ///< 1 dangling pointer, 2 tangent mismatch, 3 unconstrained state,
                        ///< 4 constant id out of range.
  int batch = 0;        ///< Residual batch (codes 1, 2) or state batch (codes 3, 4).
  long long entry = 0;  ///< Table entry (1, 2), state (3) or constant-id slot (4).
};

/** @brief Tangent size of each state slot of one residual batch. */
struct SlotTangents {
  int tangent[kMaxSlots] = {};
  int num_slots = 0;
};

/**
 * @brief Checks `count` pointers of residual batch `batch`: each must point at
 * the start of an active state of a state batch whose tangent size matches the
 * slot's; marks used[range.used_offset + state] = 1.
 */
void LaunchCheckPointers(cudaStream_t stream, float *const *pointers, size_t count, int batch,
                         const SlotTangents &slots, const StateRange *ranges, int num_ranges,
                         int *used, ValidationError *error);

/** @brief Reports the first active state of `range` that no factor reads. */
void LaunchCheckUsed(cudaStream_t stream, const int *used, int batch, const StateRange &range,
                     ValidationError *error);

/** @brief Checks that the first `count` constant ids are in [0, num_blocks). */
void LaunchCheckConstantIds(cudaStream_t stream, const int *ids, size_t count, int num_blocks,
                            int batch, ValidationError *error);

}  // namespace problem_internal
}  // namespace cunls
