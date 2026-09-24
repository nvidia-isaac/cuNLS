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
#include "cunls/factor/motion/constant_velocity_se2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_se3_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so2_factor_batch.h"
#include "cunls/factor/motion/constant_velocity_so3_factor_batch.h"

namespace cunls {

/**
 * @brief Manifold-generic facade for constant-velocity motion-prior
 * factors.
 *
 * Same zero-cost specialization mechanism as `BetweenFactorBatch<Manifold>`
 * (see that class's docs), but **always requires the manifold as an
 * explicit template argument** — `ConstantVelocityXxxFactorBatch`'s
 * constructor is `(const float* dt_ptr, size_t num_factors)` for every
 * manifold, so unlike `BetweenFactorBatch`/`PriorFactorBatch` there is no
 * manifold-specific argument type to deduce from (`dt_ptr` carries no
 * information distinguishing SE(3) from SO(2)) — this is a deliberate
 * design choice, not a limitation to work around later.
 *
 * @code
 * cunls::ConstantVelocityFactorBatch<cunls::manifold::SE3> factor(dt_ptr, num_factors);
 * @endcode
 */
template <class Manifold>
class ConstantVelocityFactorBatch {
  static_assert(sizeof(Manifold) == 0,
                "ConstantVelocityFactorBatch<Manifold>: no constant-velocity factor "
                "implementation for this manifold. Supported: "
                "manifold::{SO2,SO3,SE2,SE3}.");
};

template <>
class ConstantVelocityFactorBatch<manifold::SE3> : public ConstantVelocitySE3FactorBatch {
 public:
  using ConstantVelocitySE3FactorBatch::ConstantVelocitySE3FactorBatch;
};

template <>
class ConstantVelocityFactorBatch<manifold::SO3> : public ConstantVelocitySO3FactorBatch {
 public:
  using ConstantVelocitySO3FactorBatch::ConstantVelocitySO3FactorBatch;
};

template <>
class ConstantVelocityFactorBatch<manifold::SE2> : public ConstantVelocitySE2FactorBatch {
 public:
  using ConstantVelocitySE2FactorBatch::ConstantVelocitySE2FactorBatch;
};

template <>
class ConstantVelocityFactorBatch<manifold::SO2> : public ConstantVelocitySO2FactorBatch {
 public:
  using ConstantVelocitySO2FactorBatch::ConstantVelocitySO2FactorBatch;
};

}  // namespace cunls
