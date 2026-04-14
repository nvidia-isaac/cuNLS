/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for SO(3) between constraints (no cuBLAS handle).
 *
 * residual = Log( Delta * R_left^{-1} * R_right )  (3-vector).
 *
 * Jacobians follow the SE(3) between pattern with SO(3) adjoint Ad(R_delta) = R_delta.
 */
class SO3BetweenFactorBatch : public SizedFactorBatch<3, 3, 3> {
  using Base = SizedFactorBatch<3, 3, 3>;

 public:
  SO3BetweenFactorBatch(const Matrix<3>* pose_deltas_ptr, size_t num_factors);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  SO3BetweenFactorBatch() = default;
  void ComputeDeltaAdjoints(cudaStream_t stream);

  const Matrix<3>* pose_deltas_ptr_;
  size_t num_factors_;
  DeviceVector<Matrix<3>> delta_adjoints_;
  mutable DeviceVector<Matrix<3>> poses_left_;
  mutable DeviceVector<Matrix<3>> poses_right_;
  mutable DeviceVector<Matrix<3>> poses_left_inverse_;
};

}  // namespace cunls
