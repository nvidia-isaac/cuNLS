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
 * @brief Batch factor for SL(4) between constraints.
 *
 * residual = Vee(Log(Delta * T_left^{-1} * T_right))  (15-vector).
 *
 * Jacobians use Ad(Delta) for the left pose and identity for the right (approximate
 * inverse left/right Jacobians of Exp, consistent with small-residual linearization).
 */
class SL4BetweenFactorBatch : public SizedFactorBatch<15, 15, 15> {
  using Base = SizedFactorBatch<15, 15, 15>;

 public:
  SL4BetweenFactorBatch(cuBLASHandle& cublas_handle,
                          const SL4Transform* pose_deltas_ptr, size_t num_factors);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  SL4BetweenFactorBatch() = default;
  void ComputeDeltaAdjoints(cudaStream_t stream);

  const SL4Transform* pose_deltas_ptr_;
  size_t num_factors_;
  DeviceVector<float> delta_adjoints_;
  cuBLASHandle& cublas_handle_;
  mutable DeviceVector<SL4Transform> poses_left_;
  mutable DeviceVector<SL4Transform> poses_right_;
  mutable DeviceVector<SL4Transform> poses_left_inverse_;
};

}  // namespace cunls
