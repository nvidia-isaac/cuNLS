/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

void LaunchVectorBetweenFactorKernel(const float* deltas,
                                     float const* const* state_pointers,
                                     float* residuals, float* jacobians, int dim,
                                     int num_factors, cudaStream_t stream);

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
  VectorBetweenFactorBatch(const VectorType* deltas_ptr, size_t num_factors)
      : deltas_ptr_(deltas_ptr), num_factors_(num_factors) {}

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final {
    LaunchVectorBetweenFactorKernel(
        reinterpret_cast<const float*>(deltas_ptr_), state_pointers, residuals,
        jacobians, Dim, static_cast<int>(num_factors_), stream);
    return true;
  }

  size_t NumFactors() const final { return num_factors_; }

 private:
  VectorBetweenFactorBatch() = default;
  const VectorType* deltas_ptr_;
  size_t num_factors_;
};

}  // namespace cunls
