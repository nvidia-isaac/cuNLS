/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <type_traits>
#include <utility>

#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Helper struct for SFINAE-based type checking.
 *
 * Used to detect if a type T derives from SizedFactorBatch.
 */
struct DerivedFromAnySizedFactorBatchHelper {
  template <int Dim, int... StateBlockSizes>
  static std::true_type test(
      const SizedFactorBatch<Dim, StateBlockSizes...>*);

  /// Fallback overload for types that don't derive from SizedFactorBatch
  static std::false_type test(...);
};

/**
 * @brief Type trait that checks if T derives from SizedFactorBatch.
 *
 * Uses SFINAE to determine if a type is derived from any instantiation
 * of SizedFactorBatch. C++17 compatible alternative to concepts.
 */
template <class T>
struct IsDerivedFromAnySizedFactorBatch
    : decltype(DerivedFromAnySizedFactorBatchHelper::test(
          std::declval<T*>())){};

}  // namespace cunls
