/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
 * @brief Batch factor for SE(3) between constraints.
 *
 * Computes the residual between two SE(3) poses: residual = Log( Delta @
 * T_left^{-1} @ T_right). This represents the relative transformation between
 * two poses in the Lie algebra se(3).
 *
 * The factor has:
 * - 6 residuals (6D twist vector)
 * - 2 state blocks, each with 6 state components (SE(3) transform stored as 4x4
 * matrix)
 *
 * The Jacobians are computed with respect to both state blocks using the
 * left and right Jacobians of SE(3).
 *
 * @note The pose_deltas pointer must point to GPU device memory and remain
 *       valid for the lifetime of this object. The memory layout is:
 *       [delta0: 16 floats][delta1: 16 floats]...[deltaN-1: 16 floats]
 */
class SE3BetweenFactorBatch : public SizedFactorBatch<6, 6, 6> {
  using Base = SizedFactorBatch<6, 6, 6>;

public:
  /**
   * @brief Constructs a batch of SE(3) between factors.
   *
   * @param pose_deltas_ptr Pointer to GPU device memory containing pose deltas.
   *                        Must point to at least num_factors * 16 floats of
   * allocated memory. Each delta represents the constraint Delta = T_right^{-1}
   * * T_left for some true transforms T_left and T_right.
   * @param num_factors Number of factors in the batch.
   */
  SE3BetweenFactorBatch(const SE3Transform *pose_deltas_ptr,
                        size_t num_factors);

  /**
   * @brief Evaluates the factor and optionally computes Jacobians.
   *
   * Computes residuals = Log(T_left^{-1} * T_right) for each factor in the
   * batch. If jacobians is not nullptr, also computes the Jacobians with
   * respect to both state blocks.
   *
   * @param residuals Output residuals (6 floats per factor, device pointer)
   * @param jacobians Output Jacobians (12x6 floats per factor, device pointer).
   *                  Can be nullptr if Jacobians are not needed.
   * @param state_pointers Array of state block pointers (device pointer to
   * device pointers)
   * @param stream CUDA stream for asynchronous execution
   * @return true if evaluation succeeded, false otherwise
   */
  bool Evaluate(float *residuals, float *jacobians,
                float const *const *state_pointers,
                cudaStream_t stream) const final;

  /**
   * @brief Returns the number of factors in the batch.
   *
   * @return Number of factors
   */
  size_t NumFactors() const final { return num_factors_; }

private:
  /// Private default constructor to prevent default construction
  SE3BetweenFactorBatch() = default;

  /**
   * @brief Computes the SE(3) adjoints of the pose deltas.
   *
   * The adjoint matrices are used in Jacobian computation. This is called
   * during construction to precompute values needed for efficient evaluation.
   *
   * @param stream CUDA stream for asynchronous execution
   */
  void ComputeDeltaAdjoints(cudaStream_t stream);

  /// Pointer to user-managed device memory containing pose deltas.
  const SE3Transform *pose_deltas_ptr_;

  /// Number of factors in the batch.
  size_t num_factors_;

  /// SE3 adjoints of the pose deltas (precomputed once at construction)
  DeviceVector<Matrix<6>> delta_adjoints_;

  /// Scratch buffer reused for error = Delta * T_left^{-1} * T_right
  mutable DeviceVector<SE3Transform> poses_left_inverse_;
};

} // namespace cunls
