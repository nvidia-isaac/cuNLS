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

#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/factor/factor_batch.h"

// Python trampoline for cunls::FactorBatch.
//
// When the C++ optimizer calls Evaluate(), this class:
//   1. Re-acquires the GIL (released by the minimizer wrapper).
//   2. Looks up the Python object that wraps *this* via nb::find().
//   3. Forwards the four raw pointers as plain ints to Python's evaluate().
//
// All GPU pointer arguments are passed as uintptr_t so that the Python side
// can wrap them with CuPy or Warp arrays without any C++ <-> Python type
// conversion overhead.
class PyFactorBatch : public cunls::FactorBatch {
 public:
    size_t residual_size_;
    std::vector<size_t> state_block_sizes_;
    size_t num_factors_;

    PyFactorBatch(size_t res_size, std::vector<size_t> block_sizes, size_t num);

    bool Evaluate(float* residuals, float* jacobians,
                  float const* const* state_pointers,
                  cudaStream_t stream) const override;

    size_t ResidualsSize() const override;
    std::vector<size_t> StateBlockSizes() const override;
    size_t NumFactors() const override;
};

// Polymorphic wrapper that applies sqrt-information matrices to any FactorBatch.
// Unlike the C++ template InformationFactorBatch<T>, this operates on the
// FactorBatch* interface so Python users never need to name specializations.
class PyInformationFactorBatch : public cunls::FactorBatch {
 public:
  PyInformationFactorBatch(cunls::cuBLASHandle& cublas_handle,
                           cunls::FactorBatch* inner,
                           const float* sqrt_information_matrices_ptr);

  size_t ResidualsSize() const override;
  size_t NumFactors() const override;
  std::vector<size_t> StateBlockSizes() const override;

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const override;

 private:
  cunls::cuBLASHandle& cublas_handle_;
  cunls::FactorBatch* inner_;
  const float* sqrt_info_ptr_;
};

// Polymorphic wrapper that applies scalar weights to any FactorBatch.
// Supports uniform (single float) and per-factor (device array) weights.
class PyWeightedFactorBatch : public cunls::FactorBatch {
 public:
  PyWeightedFactorBatch(cunls::FactorBatch* inner, float weight);

  PyWeightedFactorBatch(cunls::FactorBatch* inner,
                        const float* per_factor_weights);

  size_t ResidualsSize() const override;
  size_t NumFactors() const override;
  std::vector<size_t> StateBlockSizes() const override;

  bool Evaluate(float* residuals, float* jacobians,
                float const* const* state_pointers,
                cudaStream_t stream) const override;

 private:
  cunls::FactorBatch* inner_;
  float uniform_weight_;
  const float* per_factor_weights_;
};
