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
#include "cunls/factor/between/se2_between_factor_batch.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/similarity2_between_factor_batch.h"
#include "cunls/factor/between/similarity3_between_factor_batch.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"

namespace cunls {

/**
 * @brief Manifold-generic facade for between-pose factors.
 *
 * `BetweenFactorBatch<Manifold>` is a zero-cost compile-time alias for the
 * existing, hand-optimized `XxxBetweenFactorBatch` implementation for that
 * manifold — each specialization below adds no data members and no
 * additional virtual dispatch, so `sizeof(BetweenFactorBatch<manifold::SE3>)
 * == sizeof(SE3BetweenFactorBatch)` and the two are interchangeable
 * everywhere a `FactorBatch*` is expected (`Problem::AddFactorBatch`
 * included).
 *
 * `Manifold` is rarely written explicitly: the deduction guides below let
 * class template argument deduction (CTAD) pick it from the deltas
 * pointer's own type, since `cunls/common/types.h`'s per-manifold ambient
 * types are now distinct C++ types rather than aliases sharing `Matrix<N>`:
 *
 * @code
 * cunls::BetweenFactorBatch between(deltas_ptr, num_factors);  // manifold deduced
 * @endcode
 *
 * The explicit-template-argument spelling
 * (`BetweenFactorBatch<manifold::SE3>(deltas_ptr, num_factors)`) remains
 * available and is required in generic code parameterized over `Manifold`.
 */
template <class Manifold>
class BetweenFactorBatch {
  static_assert(sizeof(Manifold) == 0,
                "BetweenFactorBatch<Manifold>: no between-factor implementation for this "
                "manifold. Supported: manifold::{SO2,SO3,SE2,SE3,Similarity2,Similarity3,SL4,"
                "Vector<Dim>}.");
};

template <>
class BetweenFactorBatch<manifold::SE3> : public SE3BetweenFactorBatch {
 public:
  using SE3BetweenFactorBatch::SE3BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::SO3> : public SO3BetweenFactorBatch {
 public:
  using SO3BetweenFactorBatch::SO3BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::SE2> : public SE2BetweenFactorBatch {
 public:
  using SE2BetweenFactorBatch::SE2BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::SO2> : public SO2BetweenFactorBatch {
 public:
  using SO2BetweenFactorBatch::SO2BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::Similarity2> : public Similarity2BetweenFactorBatch {
 public:
  using Similarity2BetweenFactorBatch::Similarity2BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::Similarity3> : public Similarity3BetweenFactorBatch {
 public:
  using Similarity3BetweenFactorBatch::Similarity3BetweenFactorBatch;
};

template <>
class BetweenFactorBatch<manifold::SL4> : public SL4BetweenFactorBatch {
 public:
  using SL4BetweenFactorBatch::SL4BetweenFactorBatch;
};

template <int Dim>
class BetweenFactorBatch<manifold::Vector<Dim>> : public VectorBetweenFactorBatch<Dim> {
 public:
  using VectorBetweenFactorBatch<Dim>::VectorBetweenFactorBatch;
};

// Deduction guides: each keys off the corresponding backend constructor's
// own (now manifold-distinct) argument types, so `BetweenFactorBatch(ptr,
// n)` alone resolves to the right specialization with no tag needed.
BetweenFactorBatch(const SE3Transform *, size_t)->BetweenFactorBatch<manifold::SE3>;
BetweenFactorBatch(const SO3Rotation *, size_t)->BetweenFactorBatch<manifold::SO3>;
BetweenFactorBatch(const SE2Transform *, size_t)->BetweenFactorBatch<manifold::SE2>;
BetweenFactorBatch(const SO2Rotation *, size_t)->BetweenFactorBatch<manifold::SO2>;
BetweenFactorBatch(const Similarity2Transform *, size_t)->BetweenFactorBatch<manifold::Similarity2>;
BetweenFactorBatch(cuBLASHandle &, const Similarity3Transform *, size_t)
    ->BetweenFactorBatch<manifold::Similarity3>;
BetweenFactorBatch(const SL4Transform *, size_t)->BetweenFactorBatch<manifold::SL4>;
// No deduction guide for manifold::Vector<Dim>: cunls::Vector<Dim>'s `int
// Dim` doesn't deduction-match cuda::std::array's `size_t` extent
// parameter (a real C++ rule, not a workaround-able limitation — deduction
// requires an exact non-type-parameter type match, no int/size_t
// coercion). This isn't a loss in practice: distinct `Dim`s were already
// distinct types, so VectorBetweenFactorBatch<Dim> was never actually
// ambiguous and didn't need CTAD to begin with — use
// `BetweenFactorBatch<manifold::Vector<Dim>>(deltas_ptr, num_factors)`
// explicitly.

}  // namespace cunls
