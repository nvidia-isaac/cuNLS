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

#include <stdexcept>
#include <string>
#include <vector>

namespace cunls {

/**
 * @brief Abstract base class for batched factor evaluation on GPU.
 *
 * A FactorBatch represents a collection of identical factors that
 * can be evaluated in parallel on the GPU. Each factor in the batch
 * computes residuals and optionally Jacobians from a set of states.
 *
 * Subclasses must implement the pure virtual methods to define the residual
 * size, state sizes, number of factors, and the evaluation logic.
 *
 * @see SizedFactorBatch for a convenience base that fixes residual and
 *      state sizes at compile time.
 */
class FactorBatch {
 public:
  /**
   * @brief Evaluates residuals and, optionally, Jacobians for a list of items.
   *
   * <b>Terms</b>
   *
   * - N = NumActiveFactors(): number of factors (measurements) in the batch.
   * - B = StateSizes().size(): states read by one factor.
   * - m = ResidualsSize(): residual dimension of one factor.
   * - J = sum of StateSizes(): Jacobian columns of one factor.
   * - **Item**: one factor evaluated at one set of B states. The call
   *   evaluates n items, t = 0 .. n-1. Item t reads factor f(t)'s measurement
   *   and its own B state pointers, and writes its own output rows.
   *
   * With the default arguments, item t is simply factor t (n = N, f(t) = t).
   * This is what the regular minimizers use. The last two arguments let one
   * call evaluate the same factors at many state sets: the RANSAC minimizers
   * evaluate every hypothesis this way.
   *
   * <b>Parameters</b>
   *
   * @param residuals [out] Device array of n * m floats. Item t writes
   *        `residuals[t * m + r]` for r in [0, m).
   * @param jacobians [out] Device array of n * m * J floats, or nullptr when
   *        only residuals are needed. Item t writes a row-major m x J block
   *        starting at `jacobians[t * m * J]`: element (r, c) is
   *        `jacobians[(t * m + r) * J + c]`. Columns follow the states
   *        in order (state 0 first), each state contributing its tangent size.
   * @param state_pointers [in] Device array of n * B device pointers. Item t
   *        reads state b from `state_pointers[t * B + b]`. Different
   *        items may point to the same state (e.g. every factor of a PnP batch
   *        points to the one camera pose).
   * @param stream CUDA stream on which all work is enqueued. The call may
   *        return before the work completes.
   * @param factor_ids [in] Which factor each item evaluates:
   *        - nullptr (default): f(t) = t % N. With n = k * N this evaluates the
   *          whole batch k times in a row: copy c is items [c * N, (c + 1) * N).
   *        - otherwise a device array of n indices in [0, N) with
   *          f(t) = factor_ids[t]. Any order, repeats allowed.
   * @param num_factor_ids Number of items n. 0 (default) means n = N. When
   *        factor_ids is given, it is the length of that array.
   * @return true on success, false on failure.
   *
   * <b>Examples</b> (a batch of N = 3 factors, B = 1 state, m = 2).
   * `ptrs[t]` is the state pointer of item t (Pk: state set P's state for
   * factor k), and `res rows` is the range of `residuals` that item t writes.
   *
   * @verbatim
   *   1. Plain evaluation: Evaluate(res, jac, ptrs, stream)        n = 3
   *        item t          0     1     2
   *        factor f(t)     0     1     2
   *        ptrs[t]         x0    x1    x2
   *        res rows        [0,2) [2,4) [4,6)
   *
   *   2. Whole batch at two state sets P and Q:
   *      Evaluate(res, jac, ptrs, stream, nullptr, 6)              n = 6
   *        item t          0     1     2     3     4     5
   *        factor f(t)     0     1     2     0     1     2      (t % 3)
   *        ptrs[t]         P0    P1    P2    Q0    Q1    Q2
   *        res rows        [0,2) [2,4) [4,6) [6,8) [8,10) [10,12)
   *
   *   3. Chosen factors: ids = {2, 0, 2, 1} (device array)
   *      Evaluate(res, jac, ptrs, stream, ids, 4)                  n = 4
   *        item t          0     1     2     3
   *        factor f(t)     2     0     2     1
   *        ptrs[t]         P     P     Q     Q
   *        res rows        [0,2) [2,4) [4,6) [6,8)
   * @endverbatim
   *
   * <b>Implementing it</b>: launch one thread per item, and index
   * measurements by f(t) but everything else (state pointers, outputs) by t:
   *
   * @code
   * __global__ void MyKernel(const Measurement *meas, const int *factor_ids, int N, int n,
   *                          float const *const *state_pointers, float *res, float *jac) {
   *   const int t = blockIdx.x * blockDim.x + threadIdx.x;
   *   if (t >= n) return;
   *   const int f = factor_ids != nullptr ? factor_ids[t] : t % N;  // measurement index
   *   const float *x = state_pointers[t * B + 0];                   // this item's state
   *   res[t * m + 0] = Residual(meas[f], x);                        // this item's row
   *   if (jac != nullptr) { ... jac[(t * m + r) * J + c] ... }
   * }
   * // In Evaluate: n = num_factor_ids == 0 ? N : num_factor_ids; launch n threads.
   * @endcode
   *
   * <b>Requirements</b>
   *
   * - Item t must produce exactly what a plain evaluation produces for factor
   *   f(t) at item t's states. The built-in batches are bitwise equal.
   * - Size any internal per-factor scratch for n items, not N.
   * - Do not assume n == N or f(t) == t: the RANSAC minimizers rely on both
   *   parameters, and the regular minimizers pass the defaults.
   */
  virtual bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                        cudaStream_t stream, const int *factor_ids = nullptr,
                        size_t num_factor_ids = 0) const = 0;

  /** @brief Virtual destructor for safe polymorphic deletion. */
  virtual ~FactorBatch() = default;

  /**
   * @brief Returns the dimension of the residual vector for each factor.
   * @return Number of residual components per factor.
   */
  virtual size_t ResidualsSize() const = 0;

  /**
   * @brief Returns the sizes of all states consumed by each factor.
   * @return Vector where element i is the dimension of state i.
   */
  virtual std::vector<size_t> StateSizes() const = 0;

  /**
   * @brief Number of active factors n: the first n measurements of the batch's
   * buffers are evaluated (see Evaluate's item contract for n == 0 calls).
   *
   * Built-in batches store it in the base (FactorBatch(capacity) / SetNumActiveFactors).
   * Custom batches may override it instead; they then cannot be resized with
   * SetNumActiveFactors.
   */
  virtual size_t NumActiveFactors() const { return num_active_factors_; }

  /**
   * @brief Number of factors the batch's buffers hold: the capacity passed to
   * the constructor. Constant for the lifetime of the batch; SetNumActiveFactors
   * accepts any value up to it. Custom batches pass it to the base constructor
   * (FactorBatch(capacity) / SizedFactorBatch(capacity)). A batch constructed
   * without one (that overrides NumActiveFactors() instead) has a fixed size:
   * its capacity is NumActiveFactors(), like StateBatch::Capacity().
   */
  virtual size_t Capacity() const { return capacity_ != 0 ? capacity_ : NumActiveFactors(); }

  /**
   * @brief Sets the number of active factors, for buffers that are allocated
   * once and rewritten in place between solves.
   *
   * A host-only assignment: no allocation, no device work. Takes effect at the
   * next Evaluate / Minimize, which then read the first `num_active_factors`
   * measurements. Must not be called while a minimization that uses this batch
   * is running.
   *
   * @param num_active_factors Active count, at most Capacity().
   * @throws std::invalid_argument if num_active_factors > Capacity().
   * @throws std::logic_error if the batch overrides NumActiveFactors() and so cannot
   *         be resized.
   */
  virtual void SetNumActiveFactors(size_t num_active_factors) {
    if (num_active_factors > Capacity()) {
      throw std::invalid_argument("SetNumActiveFactors(" + std::to_string(num_active_factors) +
                                  ") exceeds the capacity of " + std::to_string(Capacity()));
    }
    num_active_factors_ = num_active_factors;
    if (NumActiveFactors() != num_active_factors) {
      throw std::logic_error(
          "SetNumActiveFactors: this factor batch overrides NumActiveFactors() and cannot be "
          "resized");
    }
  }

 protected:
  /** @brief Batch without a capacity: subclasses that override NumActiveFactors(). */
  FactorBatch() = default;

  /**
   * @brief Batch whose buffers hold `capacity` factors. The active count starts
   * at 0: call SetNumActiveFactors(n) before evaluating or solving.
   */
  explicit FactorBatch(size_t capacity) : capacity_(capacity), num_active_factors_(0) {}

 private:
  size_t capacity_ = 0;            ///< Factors the buffers hold.
  size_t num_active_factors_ = 0;  ///< Active factors.
};

}  // namespace cunls
