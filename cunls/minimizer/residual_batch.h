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

#include "cunls/factor/factor_batch.h"
#include "cunls/robustifier/loss_function_batch.h"
namespace cunls {

/**
 * @brief Wraps a factor batch with an optional loss function for evaluation.
 *
 * ResidualBatch combines a FactorBatch (which computes residuals and
 * Jacobians) with an optional LossFunctionBatch (for robust estimation).
 * During evaluation, it computes residuals, applies the loss function to
 * scale residuals and Jacobians, and optionally computes per-residual costs.
 */
class ResidualBatch {
 public:
  /**
   * @brief Constructs a residual batch from a factor batch and loss function.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param loss_function Pointer to the loss function batch, or nullptr for
   *                      trivial (identity) loss (not owned).
   */
  ResidualBatch(FactorBatch* factor_batch,
                LossFunctionBatch* loss_function);

  /**
   * @brief Evaluates residuals, Jacobians, and/or cost for the given state pointers.
   *
   * Computes the residuals from the factor batch, applies the loss function
   * (if present) to scale residuals and Jacobians for robust estimation, and
   * optionally extracts per-residual cost values.
   *
   * @param[out] cost Output per-residual cost values (0.5 * rho(||r||^2)),
   *                  or nullptr if cost is not needed.
   * @param[out] residuals Output residual vector (scaled by loss function).
   *                       Must not be nullptr.
   * @param[out] jacobians Output Jacobian values (scaled by loss function),
   *                       or nullptr if Jacobians are not needed.
   * @param state_pointers Device pointer array to state blocks for each
   *                       factor instance.
   * @param stream CUDA stream for asynchronous GPU operations.
   * @return True on success.
   */
  bool Evaluate(float* cost, float* residuals, float* jacobians,
                float const* const* state_pointers, cudaStream_t stream) const;

  /**
   * @brief Gets the factor batch.
   *
   * @return Pointer to the associated factor batch.
   */
  FactorBatch* GetFactorBatch() const { return factor_batch_; }

  /**
   * @brief Gets the loss function batch.
   *
   * @return Pointer to the associated loss function batch, or nullptr if none.
   */
  LossFunctionBatch* GetLossFunction() const { return loss_function_; }

 private:
  FactorBatch* factor_batch_ = nullptr;  ///< Factor batch.
  LossFunctionBatch* loss_function_ = nullptr;   ///< Optional loss function batch.
};
}  // namespace cunls
