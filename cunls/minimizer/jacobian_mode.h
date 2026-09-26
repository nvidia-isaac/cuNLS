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

namespace cunls {

/**
 * @brief Selects how a residual batch's Jacobian is obtained.
 *
 * `kAnalytic` (default) uses the FactorBatch's own hand-derived Evaluate()
 * Jacobian output. `kNumeric` instead derives the Jacobian via finite
 * differences on the manifold tangent space of each referenced state block,
 * requiring only that the factor batch support residual-only evaluation
 * (`jacobians == nullptr`), which every FactorBatch implementation must
 * already do.
 */
enum class JacobianMode { kAnalytic, kNumeric };

/**
 * @brief Tuning knobs for numeric (finite-difference) Jacobian computation.
 */
struct NumericDiffOptions {
  /** @brief Finite-difference scheme. */
  enum class Method {
    /** @brief One-sided: (f(x+eps) - f(x)) / eps. Cheaper, less accurate. */
    kForward,
    /** @brief Two-sided: (f(x+eps) - f(x-eps)) / (2*eps). Default. */
    kCentral,
  };

  /** @brief Finite-difference scheme. Default: central. */
  Method method = Method::kCentral;

  /**
   * @brief Per-tangent-coordinate perturbation step size.
   *
   * Phase 1 uses this as a fixed scalar step (not scaled by the current
   * state magnitude); see `numeric_diff_jacobian.h` for rationale.
   * Default: 1e-4.
   */
  float relative_step_size = 1e-4f;
};

}  // namespace cunls
