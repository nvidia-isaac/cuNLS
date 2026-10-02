/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for Sim(3) between constraints.
 *
 * residual = Log(Delta * T_left^{-1} * T_right) (7-vector).
 *
 * Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)
 *   H_right =  J_r^{-1}(r)
 */
class Similarity3BetweenFactorBatch : public SizedFactorBatch<7, 7, 7> {
  using Base = SizedFactorBatch<7, 7, 7>;

 public:
  Similarity3BetweenFactorBatch(const Similarity3Transform *pose_deltas_ptr, size_t capacity);

  /** @brief Evaluates residuals and Jacobians; follows FactorBatch::Evaluate's item contract. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  Similarity3BetweenFactorBatch() = default;

  const Matrix<4> *pose_deltas_ptr_;
  mutable DeviceVector<Matrix<4>> poses_left_;
  mutable DeviceVector<Matrix<4>> poses_right_;
  mutable DeviceVector<Matrix<4>> poses_left_inverse_;
  mutable DeviceVector<float> jacobian_temp_;
};

}  // namespace cunls
