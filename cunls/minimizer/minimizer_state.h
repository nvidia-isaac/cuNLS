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

#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/state_batch.h"

namespace cunls {

/**
 * @brief Snapshot of state values for optimization iterations.
 *
 * This class maintains a copy of state values from a Problem, allowing
 * the optimizer to work with minimizer states without modifying the original
 * problem until convergence. It also manages state pointers that map
 * residual batches to their corresponding state blocks.
 *
 * The class stores:
 * - states_: One device vector per state batch containing all state values
 *            flattened into a single vector.
 * - state_pointers_: One device vector per residual batch containing pointers
 *                   to state blocks, remapped to point into the copied state storage.
 * - problem_state_ptrs_device_: Device copy of host problem pointer lists for
 *   Jacobian structure and remap kernels.
 */
class MinimizerState {
 public:
  MinimizerState() = default;

  /**
   * @brief Constructs a minimizer state from a problem.
   *
   * Creates a snapshot of all state values and sets up state pointers
   * for residual batch evaluation.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The problem to create a state snapshot from.
   */
  MinimizerState(cudaStream_t stream, const Problem& problem) {
    Create(stream, problem);
  }

  /**
   * @brief Refreshes storage from the problem (realloc only when capacity is insufficient).
   */
  void Recreate(cudaStream_t stream, const Problem& problem) {
    Create(stream, problem);
  }

  /**
   * @brief Copies state values from another state.
   *
   * Updates this state's values with values from another state's
   * state vectors.
   *
   * @param stream CUDA stream for GPU operations.
   * @param other Source state to copy from.
   */
  void Copy(cudaStream_t stream, const std::vector<dvector<float>>& other);

  /**
   * @brief Gets the state value vectors.
   *
   * @return Reference to vector of state value vectors (one per batch).
   */
  std::vector<dvector<float>>& GetStates() { return states_; }

  /**
   * @brief Gets the state value vectors (const version).
   *
   * @return Const reference to vector of state value vectors.
   */
  const std::vector<dvector<float>>& GetStates() const {
    return states_;
  }

  /**
   * @brief Gets the state pointer vectors.
   *
   * Returns pointers remapped to point into the copied state storage.
   * Used by residual batches to access state blocks during evaluation.
   *
   * @return Reference to vector of state pointer vectors (one per residual
   * batch).
   */
  std::vector<dvector<float*>>& GetStatePointers() {
    return state_pointers_;
  }

  /**
   * @brief Gets the state pointer vectors (const version).
   *
   * @return Const reference to vector of state pointer vectors.
   */
  const std::vector<dvector<float*>>& GetStatePointers() const {
    return state_pointers_;
  }

  /**
   * @brief Builds the triplet (COO) Jacobian sparsity structure on the GPU.
   *
   * Definition (implementation) in jacobian_ops.cu.
   *
   * @param stream CUDA stream for GPU operations.
   * @param problem The optimization problem.
   * @param[out] structure Output row and column index arrays.
   */
  void BuildTripletSparseStructure(cudaStream_t stream, const Problem& problem,
                                   TripletSparseStructure& structure);

 private:
  /**
   * @brief Creates minimizer state from a problem.
   *
   * Allocates storage, copies state values, and remaps pointers.
   */
  void Create(cudaStream_t stream, const Problem& problem);

  /**
   * @brief Allocates state storage vectors.
   *
   * Creates one device vector per state batch with appropriate size.
   */
  void CreateStates(const Problem& problem);

  /**
   * @brief Allocates state pointer vectors.
   *
   * Creates one device vector per residual batch for storing state pointers.
   */
  void CreateStatePointers(const Problem& problem);

  /**
   * @brief Copies problem state pointer lists from host to problem_state_ptrs_device_.
   */
  void CopyProblemStatePointersFromHost(const Problem& problem);

  /**
   * @brief State value storage.
   *
   * One device vector per state batch, containing all state values
   * flattened into a single contiguous vector.
   */
  std::vector<dvector<float>> states_;

  /**
   * @brief State pointer storage.
   *
   * One device vector per residual batch, containing pointers to state
   * blocks. These pointers are remapped to point into states_ storage
   * rather than the original problem's state storage.
   */
  std::vector<dvector<float*>> state_pointers_;

  /// Device copy of problem.GetStatePointers() for Jacobian FillColIds and remap.
  std::vector<dvector<float*>> problem_state_ptrs_device_;
};

/**
 * @brief Copies minimizer state back to a problem.
 *
 * Updates the problem's state values with values from the minimizer state.
 * This is typically called at the end of optimization to commit the final
 * state values.
 *
 * @param stream CUDA stream for GPU operations.
 * @param state Source minimizer state.
 * @param[out] problem Destination problem to update.
 */
void Copy(cudaStream_t stream, const MinimizerState& state, Problem& problem);
}  // namespace cunls
