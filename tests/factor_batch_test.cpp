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

/** @file factor_batch_test.cpp
 *  @brief Tests for factor batch creation and basic properties.
 */

#include <gtest/gtest.h>

#include <cuda/std/array>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {

/** @brief Verifies that PriorVectorFactorBatch and VectorStateBatch report
 * correct sizes. */
TEST(FactorBatchTest, Simple) {
  constexpr size_t num_vectors = 100;

  std::vector<Vector<2>> host_vectors;
  host_vectors.reserve(num_vectors);
  for (size_t i = 0; i < num_vectors; i++) {
    float x = static_cast<float>(i);
    host_vectors.push_back({x, x + 1});
  }

  DeviceVector<Vector<2>> vectors(host_vectors);
  const float *data_ptr = reinterpret_cast<const float *>(vectors.data());
  VectorStateBatch<2> vector_states(data_ptr, num_vectors);

  std::vector<Vector<2>> observations_host;
  for (size_t i = 0; i < num_vectors - 1; i++) {
    float dummy_data = static_cast<float>(num_vectors - i);
    observations_host.push_back({dummy_data, dummy_data});
  }
  DeviceVector<Vector<2>> observations_device(observations_host);

  PriorVectorFactorBatch<2> factor_batch(observations_device.data(),
                                         observations_host.size());

  ASSERT_EQ(factor_batch.NumFactors(), num_vectors - 1);
  ASSERT_EQ(vector_states.NumStateBlocks(), num_vectors);
}

} // namespace cunls
