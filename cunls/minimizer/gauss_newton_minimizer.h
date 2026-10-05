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

#include "cunls/minimizer/minimizer.h"

namespace cunls {

/**
 * @brief Gauss-Newton: solves the undamped normal equations and takes every
 *        step that lowers the cost.
 *
 * Fast near the solution and on well-conditioned problems. A step that does
 * not lower the cost is rejected (with line search enabled it is shortened
 * first), and the subproblem stops: Gauss-Newton has no way to change the
 * next step. Prefer LevenbergMarquardtMinimizer far from the solution or when
 * H is close to singular.
 *
 * Per subproblem:
 *   - a step is rejected if it does not lower the cost;
 *   - the subproblem has converged if ‖δ‖² < state_tolerance, the trial cost
 *     is below cost_tolerance, or the step does not lower the cost.
 *
 * See Minimizer for the iteration and MinimizerOptions for the settings.
 */
class GaussNewtonMinimizer : public Minimizer {
 public:
  /** @brief Constructs the minimizer; see MinimizerOptions. */
  explicit GaussNewtonMinimizer(const MinimizerOptions &options = MinimizerOptions());

 private:
  /** @brief Leaves the Gauss-Newton system unchanged. */
  void UpdateSystem(cudaStream_t, size_t, NormalEquations &, const ProblemPartition &) override {}

  /** @brief Rejects steps that do not lower the cost; see the class comment. */
  void ClassifySteps(cudaStream_t stream, const NormalEquations &system, const dvector<float> &step,
                     ProblemPartition &partition) override;
};

}  // namespace cunls
