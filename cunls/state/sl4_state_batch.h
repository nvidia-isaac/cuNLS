/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/types.h"
#include "cunls/state/sized_state_batch.h"

namespace cunls {

/**
 * @brief Batch processing for SL(4) Lie group state blocks (projective special
 * linear).
 *
 * Tangent space dimension is 15 (Lie algebra sl(4)); ambient storage is a 4x4
 * row-major matrix (16 floats). The Plus operation is
 *   T_new = T * Exp(Hat(xi)),
 * with Exp the matrix exponential and projection to det = 1. *
 */
class SL4StateBatch : public SizedStateBatch<16, 15> {
 public:
  using Base = SizedStateBatch<16, 15>;

  SL4StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                size_t num_blocks)
      : Base(device_ptr, num_blocks), cublas_handle_(cublas_handle) {}

  SL4StateBatch(cuBLASHandle& cublas_handle, const float* device_ptr,
                size_t num_blocks, const int* device_constant_state_ids,
                size_t num_const_state_blocks)
      : Base(device_ptr, num_blocks, device_constant_state_ids,
             num_const_state_blocks),
        cublas_handle_(cublas_handle) {}

  void Plus(const float* x, const float* delta, float* x_plus_delta,
            cudaStream_t stream) override;

 private:
  cuBLASHandle& cublas_handle_;
  mutable dvector<SL4Transform> delta_transforms_;
  mutable dvector<float> twists_;
};

}  // namespace cunls
