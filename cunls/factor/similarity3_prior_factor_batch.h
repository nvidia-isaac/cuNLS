/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_runtime.h>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/sized_factor_batch.h"

namespace cunls {

/**
 * @brief Batch factor for Sim(3) prior constraints.
 *
 * Computes the residual between a current Sim(3) transform and a target:
 *   residual = Log(T_target^{-1} * T_current)
 *
 * The Jacobian with respect to a right-perturbation delta is:
 *   J = J_r^{-1}(residual)
 * where J_r^{-1} is the inverse right Jacobian of Sim(3).
 *
 * The factor has:
 * - 7 residuals (7D tangent vector [w, u, lambda])
 * - 1 state block with tangent dimension 7 (transform stored as 4x4 matrix)
 *
 * @note The observations_ptr must point to GPU device memory containing target
 *       transformation matrices and remain valid for the lifetime of this object.
 *       Memory layout: [T0: 16 floats][T1: 16 floats]...[TN-1: 16 floats]
 *       Each Sim(3) matrix is stored row-major as:
 *       [R00 R01 R02 tx, R10 R11 R12 ty, R20 R21 R22 tz, 0 0 0 1/s]
 */
class Similarity3PriorFactorBatch : public SizedFactorBatch<7, 7> {
  using Base = SizedFactorBatch<7, 7>;

 public:
  /**
   * @brief Constructs a batch of Sim(3) prior factors.
   *
   * Pre-computes T_target^{-1} for all targets during construction.
   *
   * @param observations_ptr Pointer to GPU device memory containing target transforms.
   *                         Must point to at least num_factors * 16 floats.
   * @param num_factors Number of factors in the batch.
   */
  Similarity3PriorFactorBatch(const Matrix<4>* observations_ptr,
                                    size_t num_factors);

  /**
   * @brief Evaluates Sim(3) prior residuals and optionally Jacobians.
   *
   * @param residuals Output residuals (7 floats per factor, device pointer).
   * @param jacobians Output Jacobians (7x7 floats per factor, device pointer).
   *                  Can be nullptr to skip Jacobian computation.
   * @param state_pointers Device pointer to state block pointers.
   * @param stream CUDA stream for asynchronous execution.
   * @return true on success.
   */
  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Returns the number of factors in the batch.
   * @return Number of factors.
   */
  size_t NumFactors() const final { return num_factors_; }

 private:
  Similarity3PriorFactorBatch() = default;

  const Matrix<4>* observations_ptr_;
  size_t num_factors_;
  DeviceVector<Matrix<4>> observations_inverse_;
  mutable DeviceVector<Matrix<4>> transforms_error_;
};

}  // namespace cunls
