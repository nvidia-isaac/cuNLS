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

namespace cunls::manifold {

/**
 * @brief Empty tag types identifying a Lie group / manifold at compile
 * time.
 *
 * Used as the template parameter of the manifold facades (`BetweenFactorBatch<Manifold>`,
 * `PriorFactorBatch<Manifold>`, `ConstantVelocityFactorBatch<Manifold>`,
 * `ConstantAccelerationFactorBatch<Manifold>`). These carry no data and are
 * never constructed at runtime for `BetweenFactorBatch`/`PriorFactorBatch`
 * (the manifold is deduced from the distinct pointer types in
 * `cunls/common/types.h` instead, via the deduction guides in
 * `cunls/factor/between/between_factor_batch.h`/`cunls/factor/prior/prior_factor_batch.h`);
 * for `ConstantVelocityFactorBatch`/`ConstantAccelerationFactorBatch`, whose
 * arguments carry no manifold-specific type information to deduce from,
 * the tag is supplied explicitly as a template argument, e.g.
 * `ConstantVelocityFactorBatch<manifold::SE3>`.
 */
struct SO2 {};
struct SO3 {};
struct SE2 {};
struct SE3 {};
struct Similarity2 {};
struct Similarity3 {};
struct SL4 {};

/** @brief Tag for the generic Euclidean-vector manifold `R^Dim`. */
template <int Dim>
struct Vector {};

}  // namespace cunls::manifold
