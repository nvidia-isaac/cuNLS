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
#include "factor_batch.h"

namespace cunls {

/**
 * @brief Compile-time sized base class for batched factors.
 *
 * This CRTP-style base class fixes the residual dimension and state block
 * sizes at compile time via template arguments, enabling compile-time
 * optimizations in derived classes (e.g., fixed-size matrix operations).
 *
 * @tparam kResidualSize Dimension of the residual vector for each factor.
 * @tparam kStateBlockSizes Variadic list of state block dimensions.
 *         Each value specifies the number of scalar components in that block.
 *
 * @see FactorBatch for the polymorphic base interface.
 */
template <int kResidualSize, int... kStateBlockSizes>
class SizedFactorBatch : public FactorBatch {
 public:
  /**
   * @brief Same SizedFactorBatch specialization as this base (for wrapper types).
   *
   * Enables patterns like ``class Wrapper<T> : public T::sized_layout`` so
   * wrappers share the compile-time residual and state-block layout of T.
   */
  using sized_layout = SizedFactorBatch<kResidualSize, kStateBlockSizes...>;

  /**
   * @brief Returns the compile-time residual dimension.
   * @return kResidualSize
   */
  size_t ResidualsSize() const final { return kResidualSize; };

  /**
   * @brief Returns the compile-time state block sizes.
   * @return Vector containing {kStateBlockSizes...}.
   */
  std::vector<size_t> StateBlockSizes() const final {
    return {kStateBlockSizes...};
  };

  /** @brief Compile-time constant for the residual dimension. */
  static constexpr size_t residual_size_ = kResidualSize;
};
}  // namespace cunls
