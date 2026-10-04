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

#include "cunls/common/types.h"
#include "cunls/factor/factor_batch.h"

namespace cunls {

/** @brief Kind of a constraint row: `c(x) = 0` or `c(x) <= 0`. */
enum class ConstraintKind {
  kEquality = 0,    ///< c(x) = 0
  kInequality = 1,  ///< c(x) <= 0
};

/**
 * @brief A factor batch whose rows are constraints, solved by the augmented
 * Lagrangian (AL) method of AugmentedLagrangianMinimizer.
 *
 * Each factor has m = ResidualsSize() constraint rows c(x) (scaled by Scale()).
 * Evaluate() does not return c itself but the AL residuals of the current
 * multipliers (one per row) and penalty ρ (one per factor):
 *
 * @verbatim
 *   equality:    r = √ρ (c + λ/ρ)                J_r = √ρ ∂c/∂x
 *   inequality:  r = √ρ max(0, c + μ/ρ)          J_r = √ρ ∂c/∂x on active rows, 0 otherwise
 * @endverbatim
 *
 * so that ½‖r‖² is, up to a constant, the AL term of the rows and any
 * least-squares minimizer minimizes the AL function of the problem. The
 * multipliers start at 0 and the penalties at kDefaultPenalty: without
 * AugmentedLagrangianMinimizer (which updates both) a constraint batch acts as a
 * quadratic penalty.
 *
 * Subclasses provide the unscaled constraint values and Jacobians through
 * EvaluateConstraintRows() (same layout and item contract as
 * FactorBatch::Evaluate) and call AllocateMultipliers() once their sizes are
 * known. ConstraintFactorBatch turns any factor batch into constraints;
 * BoundFactorBatch is a built-in inequality constraint.
 */
class ConstraintFactorBatchBase : public FactorBatch {
 public:
  /** @brief Penalty ρ of every factor until it is set otherwise. */
  static constexpr float kDefaultPenalty = 10.f;

  /** @brief Equality or inequality. */
  ConstraintKind Kind() const { return kind_; }

  /** @brief Scale s applied to every row: the constraint is s * c(x). */
  float Scale() const { return scale_; }

  /**
   * @brief AL residuals and Jacobians (see the class description). Same
   * contract as FactorBatch::Evaluate.
   */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final;

  /**
   * @brief Scaled constraint values s * c(x) and, when `jacobians` is not
   * null, their Jacobians s * ∂c/∂x. Same layout and item contract as
   * FactorBatch::Evaluate.
   */
  bool EvaluateConstraint(float *values, float *jacobians, float const *const *state_pointers,
                          cudaStream_t stream, const int *factor_ids = nullptr,
                          size_t num_factor_ids = 0) const;

  /**
   * @brief Device array of Capacity() * ResidualsSize() multipliers: row r
   * of factor f is entry `f * ResidualsSize() + r` (λ for equalities, μ >= 0
   * for inequalities).
   */
  float *Multipliers() { return multipliers_.data(); }
  const float *Multipliers() const { return multipliers_.data(); }

  /** @brief Device array of Capacity() penalties ρ > 0, one per factor. */
  float *Penalties() { return penalties_.data(); }
  const float *Penalties() const { return penalties_.data(); }

  /** @brief Sets every multiplier to 0 (asynchronous on `stream`). */
  void ResetMultipliers(cudaStream_t stream);

  /** @brief Sets the penalty of every factor to `penalty` (asynchronous on `stream`). */
  void SetPenalty(float penalty, cudaStream_t stream);

 protected:
  /** @brief Batch whose size is managed by the subclass (it overrides NumActiveFactors()). */
  ConstraintFactorBatchBase(ConstraintKind kind, float scale);

  /** @brief Batch whose buffers hold `capacity` factors; see FactorBatch(size_t). */
  ConstraintFactorBatchBase(ConstraintKind kind, float scale, size_t capacity);

  /**
   * @brief Unscaled constraint values c(x) and, when `jacobians` is not null,
   * ∂c/∂x. Same contract as FactorBatch::Evaluate.
   */
  virtual bool EvaluateConstraintRows(float *values, float *jacobians,
                                      float const *const *state_pointers, cudaStream_t stream,
                                      const int *factor_ids, size_t num_factor_ids) const = 0;

  /**
   * @brief Allocates the multipliers (zero) and penalties (kDefaultPenalty)
   * for Capacity() factors. Subclass constructors call it once their
   * Capacity() and ResidualsSize() are valid.
   */
  void AllocateMultipliers();

 private:
  ConstraintKind kind_;
  float scale_;
  dvector<float> multipliers_;
  dvector<float> penalties_;
};

/**
 * @brief Turns any factor batch into constraint rows: every residual row of
 * the wrapped batch becomes `scale * r(x) = 0` (equality) or
 * `scale * r(x) <= 0` (inequality).
 *
 * Like the other wrappers it forwards the sizes to the wrapped batch; unlike
 * WeightedFactorBatch it does not own it (it is non-owning, so that any batch,
 * including custom and Python batches, can be wrapped). The wrapped batch must
 * outlive this one.
 *
 * Example: "the last pose equals the goal" is a prior factor wrapped as an
 * equality; "keep 0.3 m from the obstacle" is a signed-distance factor wrapped
 * as an inequality.
 */
class ConstraintFactorBatch : public ConstraintFactorBatchBase {
 public:
  /**
   * @param inner Factor batch whose residuals are the constraint rows (not owned).
   * @param kind Equality or inequality.
   * @param scale Positive row scale, so that the constraint tolerance means the
   *        same in meters, radians or newtons.
   * @throws std::invalid_argument if inner is null or scale is not positive.
   */
  ConstraintFactorBatch(FactorBatch *inner, ConstraintKind kind, float scale = 1.f);

  size_t ResidualsSize() const override { return inner_->ResidualsSize(); }
  std::vector<size_t> StateSizes() const override { return inner_->StateSizes(); }
  size_t NumActiveFactors() const override { return inner_->NumActiveFactors(); }
  size_t Capacity() const override { return inner_->Capacity(); }
  void SetNumActiveFactors(size_t num_active_factors) override {
    inner_->SetNumActiveFactors(num_active_factors);
  }

  /** @brief The wrapped batch. */
  FactorBatch *Inner() const { return inner_; }

 protected:
  bool EvaluateConstraintRows(float *values, float *jacobians, float const *const *state_pointers,
                              cudaStream_t stream, const int *factor_ids,
                              size_t num_factor_ids) const override {
    return inner_->Evaluate(values, jacobians, state_pointers, stream, factor_ids, num_factor_ids);
  }

 private:
  FactorBatch *inner_;
};

/**
 * @brief Applies the AL transform in place (see ConstraintFactorBatchBase):
 * `values` holds the unscaled constraint values of n items and `jacobians`
 * (nullable) their Jacobians with `jacobian_pitch` columns.
 */
void ApplyAugmentedLagrangian(ConstraintKind kind, float scale, const float *multipliers,
                              const float *penalties, float *values, float *jacobians, size_t rows,
                              size_t jacobian_pitch, size_t num_items, const int *factor_ids,
                              size_t num_factors, cudaStream_t stream);

/** @brief values[i] *= scale and, when non-null, jacobians[i] *= scale. */
void ScaleConstraintRows(float scale, float *values, size_t num_values, float *jacobians,
                         size_t num_jacobian_values, cudaStream_t stream);

}  // namespace cunls
