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

#include <cstddef>
#include <vector>

#include "cunls/factor/factor_batch.h"
#include "cunls/state/articulation_topology.h"

namespace cunls {

/**
 * @brief Base of the kinematic factors: every factor of the batch reads one
 * instance of an ArticulationStateBatch built on `Topology()` (state size =
 * the topology's DOF count) and only its cache (link poses, motion columns),
 * never running kinematics itself. Holds what all kinematic factors share:
 * the topology and the residual size.
 *
 * The Jacobian of a factor is ResidualsSize() x DofCount(), row-major; only
 * the columns of DOFs that move the links the factor reads are nonzero.
 */
class ArticulationFactorBatch : public FactorBatch {
 public:
  /** @brief The kinematic tree of the state the factors read. */
  const ArticulationTopology &Topology() const { return topology_; }

  /** @brief Residual rows per factor. */
  size_t ResidualsSize() const final { return residual_size_; }

  /** @brief One state: an articulation instance (tangent = the topology's DOF count). */
  std::vector<size_t> StateSizes() const final {
    return {static_cast<size_t>(topology_.DofCount())};
  }

 protected:
  /**
   * @param topology Kinematic tree; must outlive the batch.
   * @param residual_size Rows per factor.
   * @param capacity Factors the batch's buffers hold (active count starts at 0).
   */
  ArticulationFactorBatch(const ArticulationTopology &topology, size_t residual_size,
                          size_t capacity)
      : FactorBatch(capacity), topology_(topology), residual_size_(residual_size) {}

 private:
  const ArticulationTopology &topology_;
  size_t residual_size_;
};

}  // namespace cunls
