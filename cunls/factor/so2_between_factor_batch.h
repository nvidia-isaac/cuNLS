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
 * @brief Batch factor for SO(2) between constraints (no cuBLAS handle).
 *
 * residual = Log(R_delta^T * R_left^T * R_right) (scalar angle).
 * Since SO(2) is abelian, Jacobians are exact: H_left = -1, H_right = 1.
 */
class SO2BetweenFactorBatch : public SizedFactorBatch<1, 1, 1> {
  using Base = SizedFactorBatch<1, 1, 1>;

public:
  SO2BetweenFactorBatch(const Matrix<2> *pose_deltas_ptr, size_t num_factors);

  bool Evaluate(float *residuals, float *jacobians,
                float const *const *state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

private:
  SO2BetweenFactorBatch() = default;

  const Matrix<2> *pose_deltas_ptr_;
  size_t num_factors_;
  mutable DeviceVector<Matrix<2>> poses_left_;
  mutable DeviceVector<Matrix<2>> poses_right_;
  mutable DeviceVector<Matrix<2>> poses_left_inverse_;
};

} // namespace cunls
