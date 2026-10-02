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

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/factor_batch.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/robustifier/loss_function_batch.h"
#include "cunls/state/state_batch.h"

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
 * hold the optimization variables, and the *connectivity* tells which
 * states each factor reads.
 *
 * Connectivity comes in three forms, chosen per factor batch at registration:
 *  - a host list of state pointers (copied once into a device table);
 *  - a user-owned device table of state pointers;
 *  - a user-owned device table of state indices, with the state batch of
 *    every state slot fixed at registration.
 * Device tables are bound once and read at every minimization, so the user
 * can rewrite them in place between solves (ordered before Minimize on the
 * GPU), together with measurements, states and the sizes set through
 * FactorBatch::SetNumActiveFactors / StateBatch::SetNumActiveStates. Only the first
 * NumActiveFactors() x B entries of a table are read (B = StateSizes().size()).
 * See docs/design/reusable_buffers.md.
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
   * @param state_pointers Host-side list of device pointers to states for
   * each factor instance, flattened in row-major order: [cf0_state0,
   * cf0_state1, ..., cfN_stateM]. The problem stores a copy on the host.
   * @param jacobian_mode_override Optional per-group override of the
   * minimizer's global `MinimizerOptions::jacobian_mode`. When set, this
   * factor batch always uses the given mode regardless of the minimizer's
   * default; when `std::nullopt` (default), the minimizer's global default
   * applies. See `JacobianModeFor`.
   */
  void AddFactorBatch(FactorBatch *factor_batch, const std::vector<float *> &state_pointers,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /**
   * @brief Adds a factor batch with a robust loss function.
   *
   * Registers a batch of factors together with a loss function for
   * robust estimation. The loss function modifies the cost and Jacobian to
   * reduce sensitivity to outliers.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param loss_function_batch Pointer to the loss function batch (not owned).
   * @param state_pointers Host-side list of device pointers to states for
   * each factor instance, flattened in row-major order: [cf0_state0,
   * cf0_state1, ..., cfN_stateM]. The problem stores a copy on the host.
   * @param jacobian_mode_override Optional per-group override of the
   * minimizer's global `MinimizerOptions::jacobian_mode`; see the other
   * `AddFactorBatch` overload.
   */
  void AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                      const std::vector<float *> &state_pointers,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /**
   * @brief Adds a factor batch whose connectivity is a user-owned device table
   * of state pointers.
   *
   * Entry `f * B + b` (B = StateSizes().size()) points at the state
   * that factor f reads in its state slot b. The table must hold
   * `Capacity() * B` entries and stay valid for the problem's lifetime; its
   * contents may be rewritten between solves.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param loss_function_batch Loss function batch, or nullptr for none (not owned).
   * @param device_state_pointers Device array of `Capacity() * B` pointers (not owned).
   * @param jacobian_mode_override See the host-list overload.
   */
  void AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                      float *const *device_state_pointers,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /**
   * @brief Adds a factor batch whose connectivity is a user-owned device table
   * of state indices.
   *
   * State slot b of every factor reads from `slot_state_batches[b]` (one entry
   * per slot, B in total); factor f reads state `device_state_indices[f * B + b]`
   * of it. Indices must be below that batch's NumActiveStates(). The table must
   * hold `Capacity() * B` ints and stay valid for the problem's lifetime; its
   * contents may be rewritten between solves.
   *
   * @param factor_batch Pointer to the factor batch (not owned).
   * @param loss_function_batch Loss function batch, or nullptr for none (not owned).
   * @param slot_state_batches State batch of each state slot (registered with
   *        AddStateBatch, before or after this call).
   * @param device_state_indices Device array of `Capacity() * B` ints (not owned).
   * @param jacobian_mode_override See the host-list overload.
   */
  void AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                      const std::vector<StateBatch *> &slot_state_batches,
                      const int *device_state_indices,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /** @brief Device pointer table without a loss function; see the overload above. */
  void AddFactorBatch(FactorBatch *factor_batch, float *const *device_state_pointers,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /** @brief Device index table without a loss function; see the overload above. */
  void AddFactorBatch(FactorBatch *factor_batch,
                      const std::vector<StateBatch *> &slot_state_batches,
                      const int *device_state_indices,
                      std::optional<JacobianMode> jacobian_mode_override = std::nullopt);

  /**
   * @brief Replaces the host-list connectivity of a residual batch registered
   * with a host list (synchronous copy into its device table).
   *
   * @param residual_batch_index Index into GetResidualBatches().
   * @param state_pointers `NumActiveFactors() * B` state pointers (at most
   *        `Capacity() * B`).
   */
  void SetStatePointers(size_t residual_batch_index, const std::vector<float *> &state_pointers);

  /**
   * @brief Adds a state batch to the problem.
   *
   * Registers a batch of states as optimization variables.
   * Every state referenced by factors must belong to
   * a registered state batch.
   *
   * @param state_batch Pointer to the state batch (not owned).
   */
  void AddStateBatch(StateBatch *state_batch);

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
   * @brief Quick host-only check of the active sizes, run by every minimizer at
   * the start of Minimize.
   *
   * Loops over batches only (no device work, no synchronization). Checks that
   * every count is within its capacity, that every connectivity table covers
   * the active factors, and that at least one factor is active (factor and
   * state batches start with zero active elements). Warns about residual
   * batches with no active factors.
   *
   * @throws std::invalid_argument with an actionable message on failure.
   */
  void CheckSizes() const;

  /**
   * @brief GPU validation of every active connection, for problems whose
   * connectivity is rewritten on the device.
   *
   * Checks that every pointer of the first `NumActiveFactors() * B` entries lies in an
   * active state of a registered state batch with the slot's tangent size (or
   * every index is below the slot batch's NumActiveStates()), that every active
   * constant id is below its batch's NumActiveStates(), and that every active,
   * non-constant state is read by at least one factor. One kernel per
   * table and one readback; the first failure is logged.
   *
   * @param stream CUDA stream; synchronized before returning.
   * @return True if the problem is well-formed.
   */
  bool Validate(cudaStream_t stream) const;

  /**
   * @brief Brings every residual batch's device pointer table up to date for
   * a solve: index tables are expanded into pointers (one kernel each, on
   * `stream`). Called by the minimizers at the start of every Minimize.
   */
  void PrepareStatePointers(cudaStream_t stream) const;

  /**
   * @brief Device table of `NumActiveFactors() * B` state pointers of a
   * residual batch. Valid after PrepareStatePointers() on the same stream.
   */
  float *const *DeviceStatePointers(size_t residual_batch_index) const;

  /** @brief True if the residual batch's connectivity is a user-owned device table. */
  bool HasDeviceConnectivity(size_t residual_batch_index) const;

  /** @brief Number of active state pointers of a residual batch: NumActiveFactors() * B. */
  size_t NumStatePointers(size_t residual_batch_index) const;

  /**
   * @brief Host view of the active state pointers of one residual batch
   * (`NumStatePointers()` entries).
   *
   * Host-list batches: the stored list itself when it holds exactly the active
   * pointers (no copy), else its active prefix. Device tables: expanded and
   * downloaded, which synchronizes the device. The reference stays valid until
   * the next call for the same batch or a change to the problem.
   */
  const std::vector<float *> &HostStatePointers(size_t residual_batch_index) const;

  /**
   * @brief Gets the residual batches.
   *
   * @return Const reference to the vector of residual batches.
   */
  const std::vector<ResidualBatch> &GetResidualBatches() const;

  /**
   * @brief Gets the state batches.
   *
   * @return Const reference to the vector of state batch pointers.
   */
  const std::vector<StateBatch *> &GetStateBatches() const;

  /**
   * @brief Host copies of the active connectivity of every residual batch.
   *
   * Element i holds the first `NumActiveFactors() * B` state pointers of
   * residual batch i. Host-list batches are returned as stored; device tables
   * are expanded and downloaded first, which synchronizes the device (call it
   * only where a host copy is really needed).
   *
   * @return Const reference to the per-batch host pointer lists.
   */
  const std::vector<std::vector<float *>> &GetStatePointers() const;

  /**
   * @brief Resolves the effective Jacobian mode for a residual batch.
   *
   * Returns the per-group override registered via `AddFactorBatch` if one
   * was given, otherwise `global_default` (typically
   * `MinimizerOptions::jacobian_mode`).
   *
   * @param residual_batch_index Index into `GetResidualBatches()`.
   * @param global_default Minimizer-wide default mode.
   * @return The effective JacobianMode for this residual batch.
   */
  JacobianMode JacobianModeFor(size_t residual_batch_index, JacobianMode global_default) const;

 private:
  /**
   * @brief Validates that all inputs are non-null and sizes are consistent.
   *
   * @return True if all inputs are valid, false otherwise.
   */
  bool CheckForValidInputs() const;

  /** @brief The error CheckSizes() reports, or an empty string. */
  std::string SizeError() const;

  /**
   * @brief Validates that the factor graph is properly connected.
   *
   * Ensures every factor references existing states and every
   * state is constrained by at least one factor.
   *
   * @return True if the graph is connected, false otherwise.
   */
  bool CheckGraphConnectivity() const;

  /** @brief How one residual batch's connectivity is stored. */
  struct Connectivity {
    enum class Kind { kHostList, kDevicePointers, kDeviceIndices };
    Kind kind = Kind::kHostList;
    float *const *user_pointers = nullptr;   ///< kDevicePointers: user table.
    const int *user_indices = nullptr;       ///< kDeviceIndices: user table.
    std::vector<float *> host;               ///< kHostList: the user's list.
    std::vector<StateBatch *> slot_batches;  ///< kDeviceIndices: state batch per state slot.
    /// Library-owned device pointer table: the copied host list (kHostList) or
    /// the expanded index table (kDeviceIndices).
    mutable dvector<float *> table;
  };

  /** Registers a residual batch with its connectivity. */
  void Register(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                Connectivity connectivity, std::optional<JacobianMode> jacobian_mode_override);
  /** Expands the index table of residual batch i into its pointer table. */
  void ExpandIndices(size_t residual_batch_index, cudaStream_t stream) const;

 private:
  std::vector<ResidualBatch> residual_batches_;  ///< Registered residual batches.
  std::vector<StateBatch *> state_batches_;      ///< Registered state batches.
  std::vector<Connectivity> connectivity_;       ///< Index-aligned with residual_batches_.
  /// Host pointer lists returned by GetStatePointers(): the user's lists for
  /// host-list batches, downloads of the device tables otherwise.
  mutable std::vector<std::vector<float *>> state_pointers_;
  std::vector<std::optional<JacobianMode>>
      jacobian_mode_overrides_;  ///< Per-residual-batch JacobianMode
                                 ///< override, index-aligned with
                                 ///< residual_batches_.
};

}  // namespace cunls
