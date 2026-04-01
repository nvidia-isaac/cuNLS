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

#include <unordered_map>
#include <vector>

#include "cunls/factor/factor_batch.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/state/state_batch.h"
#include "cunls/robustifier/loss_function_batch.h"

namespace cunls {

/**
 * @brief Defines a nonlinear least-squares optimization problem.
 *
 * A Problem aggregates factor batches, state batches, and
 * the mapping between them. It is the primary interface for specifying an
 * optimization problem to be solved by GaussNewtonMinimizer or
 * LevenbergMarquardtMinimizer.
 *
 * Factor batches define residual computations, state batches
 * hold the optimization variables, and state pointers connect each factor
 * instance to its corresponding state blocks on the GPU.
 */
class Problem {
 public:
  /**
   * @brief Adds a factor batch without a loss function.
   *
   * Registers a batch of factors with the problem. Each factor
   * in the batch operates on states identified by the given pointers.
   * A trivial (identity) loss function is used.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param state_pointers Host-side list of device pointers to state blocks for each
   *                      factor instance, flattened in row-major
   *                      order: [cf0_state0, cf0_state1, ..., cfN_stateM].
   *                      The problem stores a copy on the host.
   */
  void AddFactorBatch(FactorBatch* factor_batch,
                            const std::vector<float*>& state_pointers);

  /**
   * @brief Adds a factor batch with a robust loss function.
   *
   * Registers a batch of factors together with a loss function for
   * robust estimation. The loss function modifies the cost and Jacobian to
   * reduce sensitivity to outliers.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param loss_function_batch Pointer to the loss function batch (not owned).
   * @param state_pointers Host-side list of device pointers to state blocks for each
   *                      factor instance, flattened in row-major
   *                      order: [cf0_state0, cf0_state1, ..., cfN_stateM].
   *                      The problem stores a copy on the host.
   */
  void AddFactorBatch(FactorBatch* factor_batch,
                            LossFunctionBatch* loss_function_batch,
                            const std::vector<float*>& state_pointers);

  /**
   * @brief Adds a state batch to the problem.
   *
   * Registers a batch of state blocks as optimization variables.
   * Every state block referenced by factors must belong to
   * a registered state batch.
   *
   * @param state_batch Pointer to the state batch (not owned).
   */
  void AddStateBatch(StateBatch* state_batch);

  /**
   * @brief Validates the problem structure.
   *
   * Checks that all inputs are valid (no null pointers, matching sizes) and
   * that the factor graph is properly connected (every state is
   * constrained by at least one factor, every factor references
   * valid states).
   *
   * @return True if the problem is well-formed, false otherwise.
   */
  bool CheckConsistency() const;

  /**
   * @brief Gets the residual batches.
   *
   * @return Const reference to the vector of residual batches.
   */
  const std::vector<ResidualBatch>& GetResidualBatches() const;

  /**
   * @brief Gets the state batches.
   *
   * @return Const reference to the vector of state batch pointers.
   */
  const std::vector<StateBatch*>& GetStateBatches() const;

  /**
   * @brief Gets the state pointer arrays.
   *
   * Each element corresponds to one residual batch and contains a host
   * std::vector of device pointers (float*) mapping factor instances to
   * their state blocks.
   *
   * @return Const reference to the vector of per-batch host pointer lists.
   */
  const std::vector<std::vector<float*>>& GetStatePointers() const;

 private:
  /**
   * @brief Validates that all inputs are non-null and sizes are consistent.
   *
   * @return True if all inputs are valid, false otherwise.
   */
  bool CheckForValidInputs() const;

  /**
   * @brief Validates that the factor graph is properly connected.
   *
   * Ensures every factor references existing state blocks and every
   * state block is constrained by at least one factor.
   *
   * @return True if the graph is connected, false otherwise.
   */
  bool CheckGraphConnectivity() const;

 private:
  std::vector<ResidualBatch> residual_batches_;         ///< Registered residual batches.
  std::vector<StateBatch*> state_batches_;  ///< Registered state batches.
  std::vector<std::vector<float*>> state_pointers_; ///< Host copies of per-residual-batch state pointer lists.
};

}  // namespace cunls
