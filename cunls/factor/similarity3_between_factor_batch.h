/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/cublas_helper.h"
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
  Similarity3BetweenFactorBatch(cuBLASHandle& cublas_handle,
                                const Matrix<4>* pose_deltas_ptr,
                                size_t num_factors);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  Similarity3BetweenFactorBatch() = default;

  void ComputeDeltaAdjoints(cudaStream_t stream);

  const Matrix<4>* pose_deltas_ptr_;
  size_t num_factors_;
  cuBLASHandle& cublas_handle_;
  mutable DeviceVector<Matrix<4>> poses_left_;
  mutable DeviceVector<Matrix<4>> poses_right_;
  mutable DeviceVector<Matrix<4>> poses_left_inverse_;
  DeviceVector<float> delta_adjoints_;
  mutable DeviceVector<float> jacobian_temp_;
};

}  // namespace cunls
