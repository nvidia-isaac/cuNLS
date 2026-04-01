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

#include "py_factor_wrappers.h"

#include <numeric>
#include <stdexcept>

#include "bindings.h"

#include "cunls/factor/information_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"

PyFactorBatch::PyFactorBatch(size_t res_size, std::vector<size_t> block_sizes,
                             size_t num)
    : residual_size_(res_size),
      state_block_sizes_(std::move(block_sizes)),
      num_factors_(num) {}

bool PyFactorBatch::Evaluate(float* residuals, float* jacobians,
                             float const* const* state_pointers,
                             cudaStream_t stream) const {
  nb::gil_scoped_acquire gil;
  nb::object self_obj = nb::find(this);
  nb::object result = self_obj.attr("evaluate")(
      reinterpret_cast<uintptr_t>(residuals),
      reinterpret_cast<uintptr_t>(jacobians),
      reinterpret_cast<uintptr_t>(state_pointers),
      reinterpret_cast<uintptr_t>(stream));
  return nb::cast<bool>(result);
}

size_t PyFactorBatch::ResidualsSize() const { return residual_size_; }

std::vector<size_t> PyFactorBatch::StateBlockSizes() const {
  return state_block_sizes_;
}

size_t PyFactorBatch::NumFactors() const { return num_factors_; }

PyInformationFactorBatch::PyInformationFactorBatch(
    cunls::cuBLASHandle& cublas_handle, cunls::FactorBatch* inner,
    const float* sqrt_information_matrices_ptr)
    : cublas_handle_(cublas_handle),
      inner_(inner),
      sqrt_info_ptr_(sqrt_information_matrices_ptr) {}

size_t PyInformationFactorBatch::ResidualsSize() const {
  return inner_->ResidualsSize();
}

size_t PyInformationFactorBatch::NumFactors() const {
  return inner_->NumFactors();
}

std::vector<size_t> PyInformationFactorBatch::StateBlockSizes() const {
  return inner_->StateBlockSizes();
}

bool PyInformationFactorBatch::Evaluate(
    float* residuals, float* jacobians, float const* const* state_pointers,
    cudaStream_t stream) const {
  inner_->Evaluate(residuals, jacobians, state_pointers, stream);

  auto handle = cublas_handle_.GetHandle(stream);
  const size_t rsize = inner_->ResidualsSize();
  const size_t nf = inner_->NumFactors();

  cunls::ApplyInformationToResiduals(handle, sqrt_info_ptr_, residuals, rsize,
                                     nf);
  if (jacobians == nullptr) return true;

  auto sbs = inner_->StateBlockSizes();
  const size_t jpitch = std::accumulate(sbs.begin(), sbs.end(), size_t{0});
  cunls::ApplyInformationToJacobians(handle, sqrt_info_ptr_, jacobians, rsize,
                                     jpitch, nf);
  return true;
}

PyWeightedFactorBatch::PyWeightedFactorBatch(cunls::FactorBatch* inner,
                                             float weight)
    : inner_(inner), uniform_weight_(weight), per_factor_weights_(nullptr) {}

PyWeightedFactorBatch::PyWeightedFactorBatch(cunls::FactorBatch* inner,
                                             const float* per_factor_weights)
    : inner_(inner),
      uniform_weight_(0.0f),
      per_factor_weights_(per_factor_weights) {
  if (per_factor_weights_ == nullptr) {
    throw std::invalid_argument(
        "WeightedFactorBatch: per_factor_weights must not be null");
  }
}

size_t PyWeightedFactorBatch::ResidualsSize() const {
  return inner_->ResidualsSize();
}

size_t PyWeightedFactorBatch::NumFactors() const {
  return inner_->NumFactors();
}

std::vector<size_t> PyWeightedFactorBatch::StateBlockSizes() const {
  return inner_->StateBlockSizes();
}

bool PyWeightedFactorBatch::Evaluate(float* residuals, float* jacobians,
                                     float const* const* state_pointers,
                                     cudaStream_t stream) const {
  inner_->Evaluate(residuals, jacobians, state_pointers, stream);

  const size_t rsize = inner_->ResidualsSize();
  const size_t nf = inner_->NumFactors();

  if (per_factor_weights_ != nullptr) {
    cunls::ApplyPerFactorWeightToResiduals(per_factor_weights_, residuals,
                                           rsize, nf, stream);
  } else {
    cunls::ApplyUniformWeightToResiduals(uniform_weight_, residuals,
                                         nf * rsize, stream);
  }

  if (jacobians == nullptr) return true;

  auto sbs = inner_->StateBlockSizes();
  const size_t jpitch = std::accumulate(sbs.begin(), sbs.end(), size_t{0});

  if (per_factor_weights_ != nullptr) {
    cunls::ApplyPerFactorWeightToJacobians(per_factor_weights_, jacobians,
                                           rsize, jpitch, nf, stream);
  } else {
    cunls::ApplyUniformWeightToJacobians(uniform_weight_, jacobians,
                                         nf * rsize * jpitch, stream);
  }
  return true;
}
