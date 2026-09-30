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
#include <cuda_runtime.h>

#include <vector>

namespace cunls {

/**
 * @brief Abstract base class for batched factor evaluation on GPU.
 *
 * A FactorBatch represents a collection of identical factors that
 * can be evaluated in parallel on the GPU. Each factor in the batch
 * computes residuals and optionally Jacobians from a set of state blocks.
 *
 * Subclasses must implement the pure virtual methods to define the residual
 * size, state block sizes, number of factors, and the evaluation logic.
 *
 * @see SizedFactorBatch for a convenience base that fixes residual and
 *      state block sizes at compile time.
 */
class FactorBatch {
 public:
  /**
   * @brief Evaluates residuals and optionally Jacobians for all factors
   *        in the batch.
   *
   * @param residuals Output device pointer for residuals. Must have space for
   *                  NumFactors() * ResidualsSize() floats.
   * @param jacobians Output device pointer for Jacobians, or nullptr to skip
   *                  Jacobian computation.
   * @param state_pointers Device pointer to an array of state block pointers.
   *                       Each factor consumes StateBlockSizes().size()
   *                       consecutive entries.
   * @param stream CUDA stream for asynchronous execution.
   * @return true if evaluation succeeded, false otherwise.
   */
  virtual bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                        cudaStream_t stream) const = 0;

  /**
   * @brief Evaluates arbitrary (factor, state set) items in one call.
   *
   * Item t evaluates the measurement of factor `factor_ids[t]` against the
   * state blocks `state_pointers[t * StateBlockSizes().size() + b]`. Outputs
   * are laid out exactly as by Evaluate(), per item: residuals +
   * t * ResidualsSize(), jacobians + t * ResidualsSize() * (sum of block
   * sizes). With `factor_ids == nullptr`, item t uses factor t % NumFactors(),
   * i.e. the batch evaluated against num_items / NumFactors() state sets.
   *
   * Optional. The default returns false ("not supported") and callers fall
   * back to one Evaluate() per state set. Built-in factor batches implement
   * it; the RANSAC minimizers use it to evaluate every hypothesis in one launch.
   *
   * @param residuals Output, num_items * ResidualsSize() floats.
   * @param jacobians Output or nullptr.
   * @param state_pointers Device array of num_items * StateBlockSizes().size() pointers.
   * @param factor_ids Device array of num_items factor indices, or nullptr.
   * @param num_items Number of items.
   * @param stream CUDA stream for asynchronous execution.
   * @return true if evaluated, false if not supported (nothing was written).
   */
  virtual bool EvaluateIndexed(float *residuals, float *jacobians,
                               float const *const *state_pointers, const int *factor_ids,
                               size_t num_items, cudaStream_t stream) const {
    (void)residuals;
    (void)jacobians;
    (void)state_pointers;
    (void)factor_ids;
    (void)num_items;
    (void)stream;
    return false;
  }

  /** @brief Virtual destructor for safe polymorphic deletion. */
  virtual ~FactorBatch() = default;

  /**
   * @brief Returns the dimension of the residual vector for each factor.
   * @return Number of residual components per factor.
   */
  virtual size_t ResidualsSize() const = 0;

  /**
   * @brief Returns the sizes of all state blocks consumed by each factor.
   * @return Vector where element i is the dimension of state block i.
   */
  virtual std::vector<size_t> StateBlockSizes() const = 0;

  /**
   * @brief Returns the number of factors in this batch.
   * @return Number of factors to be evaluated in parallel.
   */
  virtual size_t NumFactors() const = 0;
};

}  // namespace cunls
