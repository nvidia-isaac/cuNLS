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
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se2_prior_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/similarity2_prior_factor_batch.h"
#include "cunls/factor/prior/similarity3_prior_factor_batch.h"
#include "cunls/factor/prior/sl4_prior_factor_batch.h"
#include "cunls/factor/prior/so2_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"

namespace cunls {

/**
 * @brief Manifold-generic facade for prior factors.
 *
 * Same construction as `BetweenFactorBatch<Manifold>` (see that class's
 * docs), for
 * `XxxPriorFactorBatch` instead of `XxxBetweenFactorBatch`. `Manifold` is
 * deduced from the observations pointer's type:
 *
 * @code
 * cunls::PriorFactorBatch prior(observations_ptr, num_factors);  // manifold deduced
 * @endcode
 */
template <class Manifold>
class PriorFactorBatch {
  static_assert(sizeof(Manifold) == 0,
                "PriorFactorBatch<Manifold>: no prior-factor implementation for this "
                "manifold. Supported: manifold::{SO2,SO3,SE2,SE3,Similarity2,Similarity3,SL4,"
                "Vector<Dim>}.");
};

template <>
class PriorFactorBatch<manifold::SE3> : public SE3PriorFactorBatch {
 public:
  using SE3PriorFactorBatch::SE3PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::SO3> : public SO3PriorFactorBatch {
 public:
  using SO3PriorFactorBatch::SO3PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::SE2> : public SE2PriorFactorBatch {
 public:
  using SE2PriorFactorBatch::SE2PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::SO2> : public SO2PriorFactorBatch {
 public:
  using SO2PriorFactorBatch::SO2PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::Similarity2> : public Similarity2PriorFactorBatch {
 public:
  using Similarity2PriorFactorBatch::Similarity2PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::Similarity3> : public Similarity3PriorFactorBatch {
 public:
  using Similarity3PriorFactorBatch::Similarity3PriorFactorBatch;
};

template <>
class PriorFactorBatch<manifold::SL4> : public SL4PriorFactorBatch {
 public:
  using SL4PriorFactorBatch::SL4PriorFactorBatch;
};

template <int Dim>
class PriorFactorBatch<manifold::Vector<Dim>> : public PriorVectorFactorBatch<Dim> {
 public:
  using PriorVectorFactorBatch<Dim>::PriorVectorFactorBatch;
};

// Deduction guides: see between_factor_batch.h for the mechanism.
PriorFactorBatch(const SE3Transform *, size_t)->PriorFactorBatch<manifold::SE3>;
PriorFactorBatch(const SO3Rotation *, size_t)->PriorFactorBatch<manifold::SO3>;
PriorFactorBatch(const SE2Transform *, size_t)->PriorFactorBatch<manifold::SE2>;
PriorFactorBatch(const SO2Rotation *, size_t)->PriorFactorBatch<manifold::SO2>;
PriorFactorBatch(const Similarity2Transform *, size_t)->PriorFactorBatch<manifold::Similarity2>;
PriorFactorBatch(const Similarity3Transform *, size_t)->PriorFactorBatch<manifold::Similarity3>;
PriorFactorBatch(const SL4Transform *, size_t)->PriorFactorBatch<manifold::SL4>;
// No deduction guide for manifold::Vector<Dim> — see the equivalent note in
// between_factor_batch.h. Use `PriorFactorBatch<manifold::Vector<Dim>>(...)`
// explicitly.

}  // namespace cunls
