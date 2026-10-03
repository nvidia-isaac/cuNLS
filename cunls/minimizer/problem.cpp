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

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/minimizer/problem_kernels.h"

namespace cunls {

namespace {

size_t NumSlots(const FactorBatch &factor_batch) { return factor_batch.StateSizes().size(); }

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void Problem::Register(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                       Connectivity connectivity,
                       std::optional<JacobianMode> jacobian_mode_override) {
  residual_batches_.emplace_back(factor_batch, loss_function_batch);
  connectivity_.push_back(std::move(connectivity));
  jacobian_mode_overrides_.emplace_back(jacobian_mode_override);
  state_pointers_.emplace_back();
}

void Problem::AddFactorBatch(FactorBatch *factor_batch, const std::vector<float *> &state_pointers,
                             std::optional<JacobianMode> jacobian_mode_override) {
  AddFactorBatch(factor_batch, nullptr, state_pointers, jacobian_mode_override);
}

void Problem::AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                             const std::vector<float *> &state_pointers,
                             std::optional<JacobianMode> jacobian_mode_override) {
  Connectivity c;
  c.kind = Connectivity::Kind::kHostList;
  c.host = state_pointers;
  // The device table holds the whole capacity, so a later SetStatePointers
  // after SetNumActiveFactors never reallocates.
  const size_t capacity =
      factor_batch == nullptr ? 0 : factor_batch->Capacity() * NumSlots(*factor_batch);
  c.table.resize(std::max(capacity, state_pointers.size()));
  if (!state_pointers.empty()) {
    c.table.CopyFromHost(state_pointers.data(), state_pointers.size());
  }
  Register(factor_batch, loss_function_batch, std::move(c), jacobian_mode_override);
}

void Problem::AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                             float *const *device_state_pointers,
                             std::optional<JacobianMode> jacobian_mode_override) {
  Connectivity c;
  c.kind = Connectivity::Kind::kDevicePointers;
  c.user_pointers = device_state_pointers;
  Register(factor_batch, loss_function_batch, std::move(c), jacobian_mode_override);
}

void Problem::AddFactorBatch(FactorBatch *factor_batch, LossFunctionBatch *loss_function_batch,
                             const std::vector<StateBatch *> &slot_state_batches,
                             const int *device_state_indices,
                             std::optional<JacobianMode> jacobian_mode_override) {
  Connectivity c;
  c.kind = Connectivity::Kind::kDeviceIndices;
  c.user_indices = device_state_indices;
  c.slot_batches = slot_state_batches;
  if (factor_batch != nullptr) {
    c.table.resize(factor_batch->Capacity() * NumSlots(*factor_batch));
  }
  Register(factor_batch, loss_function_batch, std::move(c), jacobian_mode_override);
}

void Problem::AddFactorBatch(FactorBatch *factor_batch, float *const *device_state_pointers,
                             std::optional<JacobianMode> jacobian_mode_override) {
  AddFactorBatch(factor_batch, nullptr, device_state_pointers, jacobian_mode_override);
}

void Problem::AddFactorBatch(FactorBatch *factor_batch,
                             const std::vector<StateBatch *> &slot_state_batches,
                             const int *device_state_indices,
                             std::optional<JacobianMode> jacobian_mode_override) {
  AddFactorBatch(factor_batch, nullptr, slot_state_batches, device_state_indices,
                 jacobian_mode_override);
}

void Problem::SetStatePointers(size_t residual_batch_index,
                               const std::vector<float *> &state_pointers) {
  if (residual_batch_index >= connectivity_.size()) {
    throw std::out_of_range("Problem::SetStatePointers: residual batch index out of range");
  }
  Connectivity &c = connectivity_[residual_batch_index];
  if (c.kind != Connectivity::Kind::kHostList) {
    throw std::logic_error("Problem::SetStatePointers: residual batch " +
                           std::to_string(residual_batch_index) +
                           " was registered with a device table; rewrite that table instead");
  }
  c.host = state_pointers;
  c.table.resize(std::max(c.table.size(), state_pointers.size()));
  if (!state_pointers.empty()) {
    c.table.CopyFromHost(state_pointers.data(), state_pointers.size());
  }
}

void Problem::AddStateBatch(StateBatch *state_batch) { state_batches_.push_back(state_batch); }

// ---------------------------------------------------------------------------
// Device tables
// ---------------------------------------------------------------------------

size_t Problem::NumStatePointers(size_t residual_batch_index) const {
  const FactorBatch *f = residual_batches_[residual_batch_index].GetFactorBatch();
  return f->NumActiveFactors() * NumSlots(*f);
}

bool Problem::HasDeviceConnectivity(size_t residual_batch_index) const {
  return connectivity_[residual_batch_index].kind != Connectivity::Kind::kHostList;
}

void Problem::ExpandIndices(size_t residual_batch_index, cudaStream_t stream) const {
  const Connectivity &c = connectivity_[residual_batch_index];
  problem_internal::SlotTable slots;
  slots.num_slots = static_cast<int>(c.slot_batches.size());
  for (size_t b = 0; b < c.slot_batches.size(); ++b) {
    slots.base[b] = c.slot_batches[b]->StateDevicePtr(0);
    slots.ambient[b] = static_cast<int>(c.slot_batches[b]->AmbientSize());
  }
  problem_internal::LaunchExpandIndices(
      stream, c.user_indices, NumStatePointers(residual_batch_index), slots, c.table.data());
}

void Problem::PrepareStatePointers(cudaStream_t stream) const {
  for (size_t i = 0; i < connectivity_.size(); ++i) {
    if (connectivity_[i].kind == Connectivity::Kind::kDeviceIndices) {
      ExpandIndices(i, stream);
    }
  }
}

float *const *Problem::DeviceStatePointers(size_t residual_batch_index) const {
  const Connectivity &c = connectivity_[residual_batch_index];
  return c.kind == Connectivity::Kind::kDevicePointers ? c.user_pointers : c.table.data();
}

const std::vector<float *> &Problem::HostStatePointers(size_t residual_batch_index) const {
  const Connectivity &c = connectivity_[residual_batch_index];
  const size_t n = NumStatePointers(residual_batch_index);
  std::vector<float *> &view = state_pointers_[residual_batch_index];
  if (c.kind == Connectivity::Kind::kHostList) {
    if (c.host.size() == n) {
      return c.host;  // the common case: no copy
    }
    view.assign(c.host.begin(), c.host.begin() + std::min(n, c.host.size()));
    return view;
  }
  // Device tables: order the download after any pending writes to the table on
  // other streams, then expand (index tables) and copy.
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
  if (c.kind == Connectivity::Kind::kDeviceIndices) {
    ExpandIndices(residual_batch_index, nullptr);
  }
  view.resize(n);
  if (n > 0) {
    THROW_ON_CUDA_ERROR(cudaMemcpy(view.data(), DeviceStatePointers(residual_batch_index),
                                   n * sizeof(float *), cudaMemcpyDeviceToHost));
  }
  return view;
}

const std::vector<std::vector<float *>> &Problem::GetStatePointers() const {
  for (size_t i = 0; i < connectivity_.size(); ++i) {
    const std::vector<float *> &view = HostStatePointers(i);
    if (&view != &state_pointers_[i]) {
      state_pointers_[i] = view;
    }
  }
  return state_pointers_;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * @brief Host-only check of the inputs and active sizes; returns an error
 * message, or an empty string when everything is consistent.
 *
 * Loops over batches only (never over factors or states), so it is cheap
 * enough to run at the start of every minimization.
 */
std::string Problem::SizeError() const {
  std::ostringstream err;
  for (size_t j = 0; j < state_batches_.size(); ++j) {
    const StateBatch *sb = state_batches_[j];
    if (sb == nullptr) {
      return "State batch " + std::to_string(j) + " is nullptr.";
    }
    if (sb->NumActiveStates() > sb->Capacity()) {
      err << "State batch " << j << ": NumActiveStates() = " << sb->NumActiveStates()
          << " exceeds Capacity() = " << sb->Capacity() << ".";
      return err.str();
    }
    if (sb->NumConstStates() > sb->ConstCapacity() ||
        sb->NumConstStates() > sb->NumActiveStates()) {
      err << "State batch " << j << ": " << sb->NumConstStates() << " constant states for "
          << sb->NumActiveStates() << " active states (constant-id "
          << "capacity " << sb->ConstCapacity() << ").";
      return err.str();
    }
  }
  size_t active_factors = 0;
  for (size_t i = 0; i < residual_batches_.size(); i++) {
    const FactorBatch *factor_batch = residual_batches_[i].GetFactorBatch();
    if (factor_batch == nullptr) {
      return "Factor batch of residual batch " + std::to_string(i) + " is nullptr.";
    }
    if (factor_batch->NumActiveFactors() > factor_batch->Capacity()) {
      err << "Residual batch " << i << ": NumActiveFactors() = " << factor_batch->NumActiveFactors()
          << " exceeds Capacity() = " << factor_batch->Capacity()
          << "; custom factor batches must pass their capacity to the base constructor "
             "(SizedFactorBatch(capacity)).";
      return err.str();
    }
    active_factors += factor_batch->NumActiveFactors();
    const Connectivity &c = connectivity_[i];
    const size_t needed = NumStatePointers(i);
    switch (c.kind) {
      case Connectivity::Kind::kHostList:
        if (c.host.size() < needed) {
          err << "Residual batch " << i << ": " << c.host.size()
              << " state pointers for NumActiveFactors() x StateSizes().size() = " << needed
              << "; pass a list of that size (Problem::SetStatePointers after "
                 "SetNumActiveFactors).";
          return err.str();
        }
        break;
      case Connectivity::Kind::kDevicePointers:
        if (c.user_pointers == nullptr && needed > 0) {
          return "Residual batch " + std::to_string(i) + ": device state-pointer table is nullptr.";
        }
        break;
      case Connectivity::Kind::kDeviceIndices: {
        if (c.user_indices == nullptr && needed > 0) {
          return "Residual batch " + std::to_string(i) + ": device state-index table is nullptr.";
        }
        const auto sizes = factor_batch->StateSizes();
        if (c.slot_batches.size() != sizes.size() ||
            sizes.size() > static_cast<size_t>(problem_internal::kMaxSlots)) {
          err << "Residual batch " << i << ": " << c.slot_batches.size()
              << " slot state batches for " << sizes.size() << " state slots (at most "
              << problem_internal::kMaxSlots << ").";
          return err.str();
        }
        for (size_t b = 0; b < sizes.size(); ++b) {
          StateBatch *sb = c.slot_batches[b];
          if (sb == nullptr ||
              std::find(state_batches_.begin(), state_batches_.end(), sb) == state_batches_.end()) {
            err << "Residual batch " << i << ": slot " << b
                << " names a state batch that is not registered.";
            return err.str();
          }
          if (sb->TangentSize() != sizes[b]) {
            err << "Residual batch " << i << ": slot " << b << " has size " << sizes[b]
                << " but its state batch has tangent " << sb->TangentSize() << ".";
            return err.str();
          }
        }
        break;
      }
    }
  }
  if (!residual_batches_.empty() && active_factors == 0) {
    return "No residual batch has active factors: factor and state batches start with zero "
           "active elements; call SetNumActiveFactors(n) and SetNumActiveStates(n, num_const) "
           "before "
           "solving.";
  }
  return "";
}

bool Problem::CheckForValidInputs() const {
  const std::string error = SizeError();
  if (!error.empty()) {
    LogError("{}", error);
    return false;
  }
  return true;
}

void Problem::SetProblemPartition(size_t num_problems,
                                  const std::vector<const int *> &state_problem_ids) {
  if (num_problems <= 1) {
    num_problems_ = 0;
    state_problem_ids_.clear();
    return;
  }
  if (state_problem_ids.size() != state_batches_.size()) {
    throw std::invalid_argument(
        "Problem::SetProblemPartition: expected one id array per state "
        "batch (" +
        std::to_string(state_batches_.size()) + "), got " +
        std::to_string(state_problem_ids.size()));
  }
  for (const int *ids : state_problem_ids) {
    if (ids == nullptr) {
      throw std::invalid_argument("Problem::SetProblemPartition: null id array");
    }
  }
  num_problems_ = num_problems;
  state_problem_ids_ = state_problem_ids;
}

void Problem::CheckSizes() const {
  const std::string error = SizeError();
  if (!error.empty()) {
    LogError("{}", error);
    throw std::invalid_argument(error);
  }
  for (size_t i = 0; i < residual_batches_.size(); ++i) {
    if (residual_batches_[i].GetFactorBatch()->NumActiveFactors() == 0) {
      LogWarning("Residual batch {} has no active factors (SetNumActiveFactors not called?).", i);
    }
  }
}

/**
 * @brief Host check of the graph for problems whose connectivity is all host
 * lists: no two state batches share a state, every factor references an
 * existing state, and every state is constrained by a factor.
 */
bool Problem::CheckGraphConnectivity() const {
  std::unordered_map<float *, bool> visited;
  for (const auto &state_batch : state_batches_) {
    for (size_t i = 0; i < state_batch->NumActiveStates(); ++i) {
      float *p = state_batch->StateDevicePtr(i);
      if (visited.find(p) != visited.end()) {
        LogError("Same pointer to a state in different state batches.");
        return false;
      }
      visited.insert({p, false});
    }
  }
  for (size_t i = 0; i < connectivity_.size(); ++i) {
    const auto &host = connectivity_[i].host;
    const size_t n = NumStatePointers(i);
    for (size_t k = 0; k < n; ++k) {
      auto it = visited.find(host[k]);
      if (it == visited.end()) {
        LogError("Cost function refers to state that does not exist.");
        return false;
      }
      it->second = true;
    }
  }
  for (const auto &[p, constrained] : visited) {
    if (!constrained) {
      LogError("State is not constrained by any factor.");
      return false;
    }
  }
  return true;
}

bool Problem::Validate(cudaStream_t stream) const {
  using problem_internal::StateRange;
  using problem_internal::ValidationError;
  if (!CheckForValidInputs()) {
    return false;
  }
  // State batches must not overlap (checked on the host from their ranges).
  std::vector<StateRange> ranges(state_batches_.size());
  int total_blocks = 0;
  for (size_t j = 0; j < state_batches_.size(); ++j) {
    const StateBatch *sb = state_batches_[j];
    ranges[j] = {sb->StateDevicePtr(0), static_cast<int>(sb->NumActiveStates()),
                 static_cast<int>(sb->AmbientSize()), static_cast<int>(sb->TangentSize()),
                 total_blocks};
    total_blocks += ranges[j].num_blocks;
  }
  for (size_t a = 0; a < ranges.size(); ++a) {
    for (size_t b = a + 1; b < ranges.size(); ++b) {
      const float *a0 = ranges[a].base, *a1 = a0 + ranges[a].num_blocks * ranges[a].ambient;
      const float *b0 = ranges[b].base, *b1 = b0 + ranges[b].num_blocks * ranges[b].ambient;
      if (ranges[a].num_blocks > 0 && ranges[b].num_blocks > 0 && a0 < b1 && b0 < a1) {
        LogError("State batches {} and {} overlap.", a, b);
        return false;
      }
    }
  }

  PrepareStatePointers(stream);
  dvector<StateRange> d_ranges(ranges);
  dvector<int> used(static_cast<size_t>(std::max(total_blocks, 1)));
  dvector<ValidationError> error(1);
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(used.data(), 0, used.size() * sizeof(int), stream));
  THROW_ON_CUDA_ERROR(cudaMemsetAsync(error.data(), 0, sizeof(ValidationError), stream));
  for (size_t i = 0; i < residual_batches_.size(); ++i) {
    const auto sizes = residual_batches_[i].GetFactorBatch()->StateSizes();
    problem_internal::SlotTangents slots;
    slots.num_slots = static_cast<int>(std::min<size_t>(sizes.size(), problem_internal::kMaxSlots));
    for (int b = 0; b < slots.num_slots; ++b) slots.tangent[b] = static_cast<int>(sizes[b]);
    if (slots.num_slots == 0) {
      continue;
    }
    problem_internal::LaunchCheckPointers(
        stream, DeviceStatePointers(i), NumStatePointers(i), static_cast<int>(i), slots,
        d_ranges.data(), static_cast<int>(ranges.size()), used.data(), error.data());
  }
  for (size_t j = 0; j < state_batches_.size(); ++j) {
    problem_internal::LaunchCheckUsed(stream, used.data(), static_cast<int>(j), ranges[j],
                                      error.data());
    problem_internal::LaunchCheckConstantIds(
        stream, state_batches_[j]->ConstStateIds(), state_batches_[j]->NumConstStates(),
        ranges[j].num_blocks, static_cast<int>(j), error.data());
  }
  ValidationError result;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&result, error.data(), sizeof(result), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  switch (result.code) {
    case 0:
      return true;
    case 1:
      LogError(
          "Residual batch {}: state pointer {} does not point at an active state of a "
          "registered state batch.",
          result.batch, result.entry);
      break;
    case 2:
      LogError(
          "Residual batch {}: state pointer {} points into a state batch whose tangent size "
          "differs from the state slot's.",
          result.batch, result.entry);
      break;
    case 3:
      LogError("State batch {}: state {} is not constrained by any factor.", result.batch,
               result.entry);
      break;
    case 4:
      LogError("State batch {}: constant id #{} is not below NumActiveStates().", result.batch,
               result.entry);
      break;
    default:
      LogError("Problem validation failed (code {}).", result.code);
  }
  return false;
}

/**
 * @brief Validates the complete problem structure.
 *
 * Host lists are checked on the host as before. As soon as one residual batch
 * uses a device table, the whole problem is checked on the GPU (Validate).
 */
bool Problem::CheckConsistency() const {
  if (!CheckForValidInputs()) {
    return false;
  }
  const bool any_device =
      std::any_of(connectivity_.begin(), connectivity_.end(),
                  [](const Connectivity &c) { return c.kind != Connectivity::Kind::kHostList; });
  if (any_device) {
    CudaStream stream;
    return Validate(stream.GetStream());
  }
  return CheckGraphConnectivity();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const std::vector<ResidualBatch> &Problem::GetResidualBatches() const { return residual_batches_; }

const std::vector<StateBatch *> &Problem::GetStateBatches() const { return state_batches_; }

JacobianMode Problem::JacobianModeFor(size_t residual_batch_index,
                                      JacobianMode global_default) const {
  if (residual_batch_index < jacobian_mode_overrides_.size() &&
      jacobian_mode_overrides_[residual_batch_index].has_value()) {
    return *jacobian_mode_overrides_[residual_batch_index];
  }
  return global_default;
}

}  // namespace cunls
