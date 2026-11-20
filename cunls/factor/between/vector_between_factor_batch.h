/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <vector>

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
 * @brief Between factor on `dim`-dimensional vector states, the size set at run time (the
 * Python VectorBetweenFactorBatch; VectorBetweenFactorBatch<Dim> fixes it at compile time):
 * residual = left - right - delta, Jacobians +I and -I.
 *
 * @note `deltas` is device memory of capacity * dim floats, valid for the batch's lifetime.
 */
class DynamicVectorBetweenFactorBatch : public FactorBatch {
 public:
  /**
   * @param dim Components per vector (>= 1).
   * @param deltas Device array of capacity * dim floats.
   * @param capacity Factors the buffer holds; the active count starts at 0.
   * @throws std::invalid_argument if dim < 1.
   */
  DynamicVectorBetweenFactorBatch(int dim, const float *deltas, size_t capacity)
      : FactorBatch(capacity),
        dim_(CheckedVectorDim("VectorBetweenFactorBatch", dim)),
        deltas_(deltas) {}

  /** @brief Components per vector. */
  int Dimension() const { return static_cast<int>(dim_); }
  size_t ResidualsSize() const final { return dim_; }
  std::vector<size_t> StateSizes() const final { return {dim_, dim_}; }

  /** @brief Evaluates residuals and Jacobians; follows FactorBatch::Evaluate's item contract. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
    if (num_items == 0 || NumActiveFactors() == 0) return true;
    LaunchVectorBetweenFactorKernel(deltas_, state_pointers, residuals, jacobians,
                                    static_cast<int>(dim_), static_cast<int>(num_items), factor_ids,
                                    static_cast<int>(NumActiveFactors()), stream);
    return true;
  }

 private:
  size_t dim_;           ///< Components per vector.
  const float *deltas_;  ///< Device deltas, capacity * dim floats.
};

/**
 * @brief DynamicVectorBetweenFactorBatch of Dim components, the size fixed at compile time.
 *
 * @tparam Dim Dimension of each vector.
 */
template <int Dim>
class VectorBetweenFactorBatch : public DynamicVectorBetweenFactorBatch {
 public:
  /** @brief The compile-time layout (residual Dim, two states of Dim). */
  using sized_layout = SizedFactorBatch<Dim, Dim, Dim>;
  /** @brief Residual rows per factor. */
  static constexpr size_t residual_size_ = Dim;

  /** @brief Deltas: device array of capacity vectors. */
  VectorBetweenFactorBatch(const Vector<Dim> *deltas_ptr, size_t capacity)
      : DynamicVectorBetweenFactorBatch(Dim, reinterpret_cast<const float *>(deltas_ptr),
                                        capacity) {}
};

}  // namespace cunls
