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
 * @brief Batch factor for SL(4) prior constraints.
 *
 * residual = Vee(Log(T_target^{-1} * T_current))  (15-vector).
 *
 * Jacobian w.r.t. right perturbation is approximated as identity (15x15),
 * matching the small-residual linearization used when exact SL(4) right
 * Jacobians are omitted.
 */
class SL4PriorFactorBatch : public SizedFactorBatch<15, 15> {
  using Base = SizedFactorBatch<15, 15>;

 public:
  SL4PriorFactorBatch(cuBLASHandle& cublas_handle,
                      const SL4Transform* observations_ptr, size_t num_factors);

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  size_t NumFactors() const final { return num_factors_; }

 private:
  SL4PriorFactorBatch() = default;

  const SL4Transform* observations_ptr_;
  size_t num_factors_;
  DeviceVector<SL4Transform> observations_inverse_;
  cuBLASHandle& cublas_handle_;
  mutable DeviceVector<SL4Transform> transforms_current_;
  mutable DeviceVector<SL4Transform> transforms_error_;
};

}  // namespace cunls
