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

#include "py_factor_wrappers.h"

#include <numeric>
#include <stdexcept>
#include <string>

#include "bindings.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"

PyFactorBatch::PyFactorBatch(size_t res_size, std::vector<size_t> state_sizes, size_t capacity)
    : FactorBatch(capacity), residual_size_(res_size), state_sizes_(std::move(state_sizes)) {}

bool PyFactorBatch::Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                             cudaStream_t stream, const int *factor_ids,
                             size_t num_factor_ids) const {
  nb::gil_scoped_acquire gil;
  nb::object self_obj = nb::find(this);
  const size_t num_items = num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids;
  nb::object result = self_obj.attr("evaluate")(
      reinterpret_cast<uintptr_t>(residuals), reinterpret_cast<uintptr_t>(jacobians),
      reinterpret_cast<uintptr_t>(state_pointers), reinterpret_cast<uintptr_t>(stream),
      reinterpret_cast<uintptr_t>(factor_ids), num_items);
  return nb::cast<bool>(result);
}

size_t PyFactorBatch::ResidualsSize() const { return residual_size_; }

std::vector<size_t> PyFactorBatch::StateSizes() const { return state_sizes_; }

PyWrappedFactorBatch::PyWrappedFactorBatch(cunls::FactorBatch *inner, const char *wrapper)
    : inner_(inner) {
  if (dynamic_cast<const cunls::ConstraintFactorBatchBase *>(inner) != nullptr) {
    throw std::invalid_argument(std::string(wrapper) +
                                ": cannot wrap a constraint factor batch (constraints are hard; "
                                "scale them with the constraint's own scale instead)");
  }
}

size_t PyWrappedFactorBatch::ResidualsSize() const { return inner_->ResidualsSize(); }

size_t PyWrappedFactorBatch::NumActiveFactors() const { return inner_->NumActiveFactors(); }

size_t PyWrappedFactorBatch::Capacity() const { return inner_->Capacity(); }

void PyWrappedFactorBatch::SetNumActiveFactors(size_t num_active_factors) {
  inner_->SetNumActiveFactors(num_active_factors);
}

std::vector<size_t> PyWrappedFactorBatch::StateSizes() const { return inner_->StateSizes(); }

PyInformationFactorBatch::PyInformationFactorBatch(cunls::FactorBatch *inner,
                                                   const float *sqrt_information_matrices_ptr)
    : PyWrappedFactorBatch(inner, "InformationFactorBatch"),
      sqrt_info_ptr_(sqrt_information_matrices_ptr) {}

bool PyInformationFactorBatch::Evaluate(float *residuals, float *jacobians,
                                        float const *const *state_pointers, cudaStream_t stream,
                                        const int *factor_ids, size_t num_factor_ids) const {
  return cunls::EvaluateWithSqrtInformation(*inner_, sqrt_info_ptr_, residuals, jacobians,
                                            state_pointers, stream, factor_ids, num_factor_ids);
}

PyWeightedFactorBatch::PyWeightedFactorBatch(cunls::FactorBatch *inner, float weight)
    : PyWrappedFactorBatch(inner, "WeightedFactorBatch"),
      uniform_weight_(weight),
      per_factor_weights_(nullptr) {}

PyWeightedFactorBatch::PyWeightedFactorBatch(cunls::FactorBatch *inner,
                                             const float *per_factor_weights)
    : PyWrappedFactorBatch(inner, "WeightedFactorBatch"),
      uniform_weight_(0.0f),
      per_factor_weights_(per_factor_weights) {
  if (per_factor_weights_ == nullptr) {
    throw std::invalid_argument("WeightedFactorBatch: per_factor_weights must not be null");
  }
}

bool PyWeightedFactorBatch::Evaluate(float *residuals, float *jacobians,
                                     float const *const *state_pointers, cudaStream_t stream,
                                     const int *factor_ids, size_t num_factor_ids) const {
  return cunls::EvaluateWeighted(*inner_, uniform_weight_, per_factor_weights_, residuals,
                                 jacobians, state_pointers, stream, factor_ids, num_factor_ids);
}
