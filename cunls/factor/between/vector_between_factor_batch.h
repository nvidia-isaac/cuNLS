/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Evaluates num_items vector between items; item t reads delta
 * factor_ids[t] (or t % num_factors when factor_ids is null).
 */
void LaunchVectorBetweenFactorKernel(const float *deltas, float const *const *state_pointers,
                                     float *residuals, float *jacobians, int dim, int num_items,
                                     const int *factor_ids, int num_factors, cudaStream_t stream);

/**
 * @brief Euclidean between factor: residual = left - right - delta.
 *
 * Jacobian blocks are [I | -I] with respect to (left, right) tangent vectors.
 */
template <int Dim>
class VectorBetweenFactorBatch : public SizedFactorBatch<Dim, Dim, Dim> {
  using Base = SizedFactorBatch<Dim, Dim, Dim>;
  using VectorType = Vector<Dim>;

 public:
  VectorBetweenFactorBatch(const VectorType *deltas_ptr, size_t capacity)
      : SizedFactorBatch<Dim, Dim, Dim>(capacity), deltas_ptr_(deltas_ptr) {}

  /** @brief Evaluates residuals and Jacobians; follows FactorBatch::Evaluate's item contract. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override {
    const size_t num_items = num_factor_ids == 0 ? this->NumFactors() : num_factor_ids;
    if (num_items == 0 || this->NumFactors() == 0) {
      return true;
    }
    LaunchVectorBetweenFactorKernel(reinterpret_cast<const float *>(deltas_ptr_), state_pointers,
                                    residuals, jacobians, Dim, static_cast<int>(num_items),
                                    factor_ids, static_cast<int>(this->NumFactors()), stream);
    return true;
  }

 private:
  VectorBetweenFactorBatch() = default;
  const VectorType *deltas_ptr_;
};

}  // namespace cunls
