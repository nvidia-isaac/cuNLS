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

#include <cstddef>

#include <cuda_runtime.h>

#include "cunls/factor/factor_batch.h"
#include "cunls/robustifier/loss_function_batch.h"

namespace cunls {

/**
 * @brief Returns the byte size of device scratch needed for one ResidualBatch::Evaluate call.
 *
 * Layout: `num_residuals` consecutive `float` squared norms, then padding to `alignof(float3)`,
 * then `num_residuals` consecutive `float3` values (rho triplets). Caller must provide a device
 * buffer of at least this many bytes, interpreted as `float*` to the first element for the API.
 *
 * @param num_residuals Number of factors (same as `FactorBatch::NumFactors()` for this batch).
 */
inline size_t ResidualBatchWorkspaceSizeBytes(size_t num_residuals) {
  const size_t sq_bytes = num_residuals * sizeof(float);
  const size_t align = alignof(float3);
  const size_t rho_offset = (sq_bytes + align - 1u) / align * align;
  return rho_offset + num_residuals * sizeof(float3);
}

/**
 * @brief Same scratch as `ResidualBatchWorkspaceSizeBytes`, expressed in `float` elements (rounded up).
 *
 * Use this when sub-allocating inside a larger `float` arena (e.g. GaussNewton `buffer_`).
 */
inline size_t ResidualBatchWorkspaceNumFloats(size_t num_residuals) {
  return (ResidualBatchWorkspaceSizeBytes(num_residuals) + sizeof(float) - 1u) /
         sizeof(float);
}

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
  ResidualBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function);

  /**
   * @brief Evaluates residuals, Jacobians, and/or cost for the given state pointers.
   *
   * @param stream CUDA stream used for all kernels launched by this call (factor, loss, scaling).
   * @param workspace Device pointer to scratch memory on the **same stream's device**. Must not
   *                alias `residuals`, `jacobians`, or `cost` ranges used by this call. Minimum size:
   *                `ResidualBatchWorkspaceSizeBytes(NumFactors())` bytes (or
   *                `ResidualBatchWorkspaceNumFloats(NumFactors())` floats in a `float` arena); layout:
   *                the first
   *                `NumFactors()` floats hold per-factor squared L2 norms, then aligned `float3`
   *                rho values written by the loss batch.
   * @param residuals Device array of length `NumFactors() * ResidualsSize()`, row-major packed
   *                  residual vectors (one block of length `ResidualsSize()` per factor). Written by
   *                  the factor batch, then scaled in place when a non-trivial loss is set. Must not
   *                  be `nullptr`.
   * @param state_pointers Device array of `float*` with length matching `FactorBatch::Evaluate`
   *                       requirements (one pointer per factor instance into state blocks). Ownership
   *                       of pointed-to memory is unchanged. Must not be `nullptr`.
   * @param cost Optional. If non-null, device array of length `NumFactors()` filled with
   *             `0.5 * rho(||r_i||^2).x` per factor after the loss is applied (trivial loss: `0.5*s`).
   * @param jacobians Optional. If non-null, Jacobian values for this batch in the same layout as
   *                  `FactorBatch::Evaluate`: row-major with dimensions
   *                  `(NumFactors() * ResidualsSize())` rows and `sum(StateBlockSizes())` columns
   *                  (one row block of height `ResidualsSize()` per factor). Written by the factor
   *                  batch then scaled when a non-trivial loss is set.
   * @return True on success.
   */
  bool Evaluate(cudaStream_t stream, float* workspace, float* residuals,
                float const* const* state_pointers, float* cost,
                float* jacobians) const;

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
  FactorBatch* factor_batch_ = nullptr;         ///< Factor batch.
  LossFunctionBatch* loss_function_ = nullptr;  ///< Optional loss function batch.
};
}  // namespace cunls
