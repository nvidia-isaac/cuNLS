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
 * @brief Batch factor for Sim(2) between constraints (no cuBLAS handle).
 *
 * residual = Log(Delta * T_left^{-1} * T_right) (4-vector).
 *
 * Jacobians:
 *   H_left  = -J_l^{-1}(r) * Ad(Delta)
 *   H_right =  J_r^{-1}(r)
 */
class Similarity2BetweenFactorBatch : public SizedFactorBatch<4, 4, 4> {
  using Base = SizedFactorBatch<4, 4, 4>;

 public:
  Similarity2BetweenFactorBatch(const Similarity2Transform *pose_deltas_ptr, size_t capacity);

  /** @brief Evaluates residuals and Jacobians; follows FactorBatch::Evaluate's item contract. */
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  Similarity2BetweenFactorBatch() = default;

  const Matrix<3> *pose_deltas_ptr_;
  mutable DeviceVector<Matrix<3>> poses_left_;
  mutable DeviceVector<Matrix<3>> poses_right_;
  mutable DeviceVector<Matrix<3>> poses_left_inverse_;
};

}  // namespace cunls
