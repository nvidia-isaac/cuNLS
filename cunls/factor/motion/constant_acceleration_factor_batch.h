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

#include "cunls/common/manifold.h"
#include "cunls/factor/motion/constant_acceleration_se2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_se3_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so2_factor_batch.h"
#include "cunls/factor/motion/constant_acceleration_so3_factor_batch.h"

namespace cunls {

/**
 * @brief Manifold-generic facade for constant-acceleration motion-prior
 * factors.
 *
 * See `ConstantVelocityFactorBatch<Manifold>` — identical mechanism and the
 * same "explicit template argument only" design choice (the `dt_ptr`
 * argument carries no manifold-specific type information to deduce from).
 *
 * @code
 * cunls::ConstantAccelerationFactorBatch<cunls::manifold::SE3> factor(dt_ptr, num_factors);
 * @endcode
 */
template <class Manifold>
class ConstantAccelerationFactorBatch {
  static_assert(sizeof(Manifold) == 0,
                "ConstantAccelerationFactorBatch<Manifold>: no constant-acceleration factor "
                "implementation for this manifold. Supported: "
                "manifold::{SO2,SO3,SE2,SE3}.");
};

template <>
class ConstantAccelerationFactorBatch<manifold::SE3> : public ConstantAccelerationSE3FactorBatch {
 public:
  using ConstantAccelerationSE3FactorBatch::ConstantAccelerationSE3FactorBatch;
};

template <>
class ConstantAccelerationFactorBatch<manifold::SO3> : public ConstantAccelerationSO3FactorBatch {
 public:
  using ConstantAccelerationSO3FactorBatch::ConstantAccelerationSO3FactorBatch;
};

template <>
class ConstantAccelerationFactorBatch<manifold::SE2> : public ConstantAccelerationSE2FactorBatch {
 public:
  using ConstantAccelerationSE2FactorBatch::ConstantAccelerationSE2FactorBatch;
};

template <>
class ConstantAccelerationFactorBatch<manifold::SO2> : public ConstantAccelerationSO2FactorBatch {
 public:
  using ConstantAccelerationSO2FactorBatch::ConstantAccelerationSO2FactorBatch;
};

}  // namespace cunls
