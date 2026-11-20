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

#include <vector>

#include "cunls/factor/factor_batch.h"

// Python trampoline for cunls::FactorBatch.
//
// When the C++ optimizer calls Evaluate(), this class:
//   1. Re-acquires the GIL (released by the minimizer wrapper).
//   2. Looks up the Python object that wraps *this* via nb::find().
//   3. Forwards the raw pointers as plain ints, plus the resolved item count,
//      to Python's evaluate(residuals, jacobians, state_pointers, stream,
//      factor_ids, num_items) (see FactorBatch::Evaluate for the item contract).
//
// All GPU pointer arguments are passed as uintptr_t so that the Python side
// can wrap them with Warp arrays without any C++ <-> Python type
// conversion overhead.
class PyFactorBatch : public cunls::FactorBatch {
 public:
  size_t residual_size_;
  std::vector<size_t> state_sizes_;

  PyFactorBatch(size_t res_size, std::vector<size_t> state_sizes, size_t capacity);

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

  size_t ResidualsSize() const override;
  std::vector<size_t> StateSizes() const override;
};

// Base of the polymorphic Python wrappers around another FactorBatch: holds
// the wrapped batch (not owned) and forwards the sizes to it (residual size,
// state sizes, active count and capacity).
class PyWrappedFactorBatch : public cunls::FactorBatch {
 public:
  size_t ResidualsSize() const override;
  size_t NumActiveFactors() const override;
  size_t Capacity() const override;                              // of the wrapped batch
  void SetNumActiveFactors(size_t num_active_factors) override;  // forwarded to the wrapped batch
  std::vector<size_t> StateSizes() const override;

 protected:
  // Throws std::invalid_argument if `inner` is a constraint batch: a wrapped
  // constraint would be hidden from AugmentedLagrangianMinimizer (it finds
  // constraints by type) and solved as an ordinary residual. `wrapper` names
  // the Python class in the message.
  PyWrappedFactorBatch(cunls::FactorBatch *inner, const char *wrapper);

  cunls::FactorBatch *inner_;
};

// Polymorphic wrapper that applies sqrt-information matrices to any
// FactorBatch. Unlike the C++ template InformationFactorBatch<T>, this operates
// on the FactorBatch* interface so Python users never need to name
// specializations.
class PyInformationFactorBatch : public PyWrappedFactorBatch {
 public:
  PyInformationFactorBatch(cunls::FactorBatch *inner, const float *sqrt_information_matrices_ptr);

  cunls::FactorBatch *Inner() const { return inner_; }  // the wrapped batch (not owned)

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  const float *sqrt_info_ptr_;
};

// Polymorphic wrapper that applies scalar weights to any FactorBatch.
// Supports uniform (single float) and per-factor (device array) weights.
class PyWeightedFactorBatch : public PyWrappedFactorBatch {
 public:
  PyWeightedFactorBatch(cunls::FactorBatch *inner, float weight);

  PyWeightedFactorBatch(cunls::FactorBatch *inner, const float *per_factor_weights);

  cunls::FactorBatch *Inner() const { return inner_; }  // the wrapped batch (not owned)

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const override;

 private:
  float uniform_weight_;
  const float *per_factor_weights_;
};
