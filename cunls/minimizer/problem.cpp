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

#include "cunls/minimizer/problem.h"

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
namespace cunls {

/**
 * @brief Adds a factor batch without a loss function.
 *
 * Creates a ResidualBatch with a trivial (identity) loss function and stores
 * the device state pointers for later use during evaluation.
 *
 * @param factor_batch Pointer to the factor batch.
 * @param state_pointers Device pointers to state blocks.
 */
void Problem::AddFactorBatch(FactorBatch *factor_batch,
                             const std::vector<float *> &state_pointers) {
  state_pointers_.emplace_back(state_pointers);
  residual_batches_.emplace_back(factor_batch, nullptr);
}

/**
 * @brief Adds a factor batch with a robust loss function.
 *
 * Creates a ResidualBatch pairing the factor batch with the given loss
 * function for robust estimation, and stores the device state pointers.
 *
 * @param factor_batch Pointer to the factor batch.
 * @param loss_function_batch Pointer to the loss function batch.
 * @param state_pointers Device pointers to state blocks.
 */
void Problem::AddFactorBatch(FactorBatch *factor_batch,
                             LossFunctionBatch *loss_function_batch,
                             const std::vector<float *> &state_pointers) {
  state_pointers_.emplace_back(state_pointers);
  residual_batches_.emplace_back(factor_batch, loss_function_batch);
}

/**
 * @brief Adds a state batch to the problem.
 *
 * @param state_batch Pointer to the state batch.
 */
void Problem::AddStateBatch(StateBatch *state_batch) {
  state_batches_.push_back(state_batch);
}

/**
 * @brief Validates that all inputs are non-null and sizes are consistent.
 *
 * Checks that state batch and factor batch pointers are not null,
 * and that the number of state pointers matches the expected count for
 * each factor batch.
 *
 * @return True if all inputs are valid, false otherwise.
 */
bool Problem::CheckForValidInputs() const {
  // Check that provided state batch pointers are valid
  for (const auto &param : state_batches_) {
    if (param == nullptr) {
      LogError("State batch is nullptr.");
      return false;
    }
  }

  // Check that provided factor batch pointers are valid
  for (const auto &rg : residual_batches_) {
    if (rg.GetFactorBatch() == nullptr) {
      LogError("Factor batch is nullptr.");
      return false;
    }
  }

  if (residual_batches_.size() != state_pointers_.size()) {
    LogError("State pointers must be of same size as residual batches.");
    return false;
  }

  for (size_t i = 0; i < residual_batches_.size(); i++) {
    const auto &rb = residual_batches_[i];
    const auto &param_ptr = state_pointers_[i];
    const auto &factor_batch = rb.GetFactorBatch();

    size_t total_num_state_pointers =
        factor_batch->NumFactors() * factor_batch->StateBlockSizes().size();

    if (total_num_state_pointers != param_ptr.size()) {
      LogError(
          "Number of state pointers must match the number of state pointers in "
          "factor batch.");
      return false;
    }
  }
  return true;
}

/**
 * @brief Validates that the factor graph is properly connected.
 *
 * Builds a lookup table of all state block device pointers, then verifies:
 * 1. No two state batches share the same device pointer.
 * 2. Every factor references an existing state block.
 * 3. Every state block is constrained by at least one factor.
 *
 * @return True if the graph is connected and well-formed, false otherwise.
 */
bool Problem::CheckGraphConnectivity() const {
  // Create a lookup table for state block pointers.
  // For each pointer it contains bolean states whether the pointer has been
  // visited.
  std::unordered_map<float *, bool> visited;
  for (const auto &param_batch : state_batches_) {
    for (size_t i = 0; i < param_batch->NumStateBlocks(); ++i) {
      float *p = param_batch->StateBlockDevicePtr(i);
      if (visited.find(p) != visited.end()) {
        // Same pointer in different state blocks.
        LogError("Same pointer to a state in different state batches.");
        return false;
      }
      visited.insert({p, false});
    }
  }

  // Check whether factors are connected to valid state blocks.
  for (const auto &host_param_ptrs : state_pointers_) {
    for (const auto p : host_param_ptrs) {
      auto it = visited.find(p);
      if (it == visited.end()) {
        // Pointer to non-existing state block.
        LogError("Cost function refers to state that does not exist.");
        return false;
      }
      it->second = true;
    }
  }

  // Check that all the state blocks are constrained by at least one cost
  // function.
  for (const auto &[p, state] : visited) {
    if (!state) {
      // State block is not constrained by any factor.
      LogError("State block is not constrained by any factor.");
      return false;
    }
  }
  return true;
}

/**
 * @brief Validates the complete problem structure.
 *
 * Runs input validation and graph-connectivity checks.
 *
 * @return True if the problem is well-formed, false otherwise.
 */
bool Problem::CheckConsistency() const {
  if (!CheckForValidInputs()) {
    return false;
  }
  if (!CheckGraphConnectivity()) {
    return false;
  }
  return true;
}

/** @brief Gets the residual batches. */
const std::vector<ResidualBatch> &Problem::GetResidualBatches() const {
  return residual_batches_;
}

/** @brief Gets the state batches. */
const std::vector<StateBatch *> &Problem::GetStateBatches() const {
  return state_batches_;
}

/** @brief Gets the per-residual-batch state pointer arrays. */
const std::vector<std::vector<float *>> &Problem::GetStatePointers() const {
  return state_pointers_;
}

} // namespace cunls
