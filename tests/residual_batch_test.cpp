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

/**
 * @file residual_batch_test.cpp
 * @brief Unit tests for ResidualBatch evaluation with Huber loss robustification.
 *
 * Tests residual computation, cost evaluation, and Jacobian scaling using
 * PriorVectorFactorBatch with HuberLossFunctionBatch across multiple vector sizes.
 */

#include "cunls/minimizer/residual_batch.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/**
 * @brief Computes the residual scaling factor for robustified residuals.
 *
 * @param sq_norm Squared norm of the residual vector.
 * @param rho Huber loss rho values (rho, rho', rho'').
 * @return Scaling factor to apply to raw residuals.
 */
float residual_scaling(float sq_norm, const float3& rho) {
  assert(sq_norm >= 0);
  float sqrt_rho1_ = sqrt(rho.y);

  if ((sq_norm == 0.0) || (rho.z <= 0.0)) {
    return sqrt_rho1_;
  }
  assert(rho.y > 0);

  const float D = 1.0 + 2.0 * sq_norm * rho.z / rho.y;

  const float alpha = 1.0 - sqrt(D);
  return sqrt_rho1_ / (1 - alpha);
}

/**
 * @brief Computes the alpha factor for Jacobian scaling under robust loss.
 *
 * @param sq_norm Squared norm of the residual vector.
 * @param rho Huber loss rho values (rho, rho', rho'').
 * @return Alpha scaling factor for Jacobian correction.
 */
float jacobian_scaling_alpha(float sq_norm, const float3& rho) {
  assert(sq_norm >= 0.0);
  if ((sq_norm == 0.0) || (rho.z <= 0.0)) {
    return 0.0;
  }

  assert(rho.y > 0.0);

  const float D = 1.0 + 2.0 * sq_norm * rho.z / rho.y;

  const float alpha = 1.0 - sqrt(D);
  return alpha / sq_norm;
}
}  // namespace

/**
 * @brief Typed test fixture for ResidualBatch tests across multiple vector sizes.
 *
 * Creates prior factors with known parameter vectors and observations,
 * then validates residual scaling, cost computation, and Jacobian correction
 * under Huber loss robustification.
 */
template <class VectorSize>
class ResidualBatchTest : public ::testing::Test {
 public:
  static constexpr int kDim = VectorSize::size;
  using VectorType = Vector<kDim>;
  using StateData = test_utils::VectorStateData<kDim>;
  using FactorData = test_utils::PriorFactorData<kDim>;

  void SetUp() override {
    const int residual_size = num_vectors_ * VectorSize::size;
    residuals_.resize(residual_size);
    jacobians_.resize(residual_size * VectorSize::size);
    cost_.resize(num_vectors_);
    robust_workspace_.resize(
        ResidualBatchWorkspaceNumFloats(num_vectors_));

    state_vectors.resize(num_vectors_);
    observations.resize(num_vectors_);

    for (size_t i = 0; i < num_vectors_; i++) {
      float x = static_cast<float>(i);
      state_vectors[i].fill(x);
      observations[i].fill(1.f);

      float residual = x - 1;
      float squared_error = residual * residual * VectorSize::size;

      gt_residuals_.push_back(residual);
      squared_error_.push_back(squared_error);
      rho_.push_back(test_utils::HuberLossCPU(squared_error, huber_delta));
    }
  }

  const size_t num_vectors_ = 100;
  DeviceVector<float> residuals_;
  DeviceVector<float> jacobians_;
  DeviceVector<float> cost_;
  dvector<float> robust_workspace_;

  std::vector<VectorType> state_vectors, observations;

  std::vector<float> gt_residuals_;
  std::vector<float> squared_error_;
  std::vector<float3> rho_;

  const float huber_delta = 5.f;

  profiler::Domain profiler_domain_{"ResidualBatchTest"};
};

typedef ::testing::Types<test_utils::Size<1>, test_utils::Size<2>,
                         test_utils::Size<3>, test_utils::Size<4>,
                         test_utils::Size<50>>
    VectorSizes;
TYPED_TEST_CASE(ResidualBatchTest, VectorSizes);

/** @brief Tests that robustified residuals match the expected Huber-scaled values. */
TYPED_TEST(ResidualBatchTest, Residuals) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Residuals");
  CudaStream stream;

  typename TestFixture::StateData state_data(this->state_vectors);
  auto& states = state_data.get();
  auto state_pointers = test_utils::CollectStatePointersDevice(states);
  typename TestFixture::FactorData factor_data(this->observations);
  auto& factor_batch = factor_data.get();

  float* residuals_ptr = this->residuals_.data();
  float** state_ptrs = state_pointers.data();

  HuberLossFunctionBatch loss(this->huber_delta);

  ResidualBatch residual_batch(&factor_batch, &loss);
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Residuals");
    residual_batch.Evaluate(stream.GetStream(), this->robust_workspace_.data(),
                              residuals_ptr, state_ptrs, nullptr, nullptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy residuals from device to host for verification
  std::vector<float> host_residuals(this->residuals_.size());
  this->residuals_.CopyToHost(host_residuals.data(), host_residuals.size());

  const int size = TypeParam::size;

  for (size_t i = 0; i < this->num_vectors_; i++) {
    const float residual = this->gt_residuals_[i];
    const float squared_err = this->squared_error_[i];
    const float3& rho = this->rho_[i];

    float res_scaled = residual * residual_scaling(squared_err, rho);

    for (size_t j = 0; j < size; j++) {
      ASSERT_NEAR(host_residuals[size * i + j], res_scaled, 1e-5);
    }
  }
}

/** @brief Tests that per-element cost values equal 0.5 * rho(squared_error). */
TYPED_TEST(ResidualBatchTest, Cost) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Cost");
  CudaStream stream;

  typename TestFixture::StateData state_data(this->state_vectors);
  auto& states = state_data.get();
  auto state_pointers = test_utils::CollectStatePointersDevice(states);
  typename TestFixture::FactorData factor_data(this->observations);
  auto& factor_batch = factor_data.get();

  float* residuals_ptr = this->residuals_.data();
  float* cost_ptr = this->cost_.data();
  float** state_ptrs = state_pointers.data();

  HuberLossFunctionBatch loss(this->huber_delta);

  ResidualBatch residual_batch(&factor_batch, &loss);
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Cost");
    residual_batch.Evaluate(stream.GetStream(), this->robust_workspace_.data(),
                              residuals_ptr, state_ptrs, cost_ptr, nullptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy cost from device to host for verification
  std::vector<float> host_cost(this->cost_.size());
  this->cost_.CopyToHost(host_cost.data(), host_cost.size());

  for (size_t i = 0; i < this->num_vectors_; i++) {
    const float3& rho = this->rho_[i];

    float expected_cost = 0.5f * rho.x;
    // Use relaxed tolerance: GPU squared-error is from warp-reduced sum of r*r,
    // so can differ from host residual*residual*dim due to float order (worse for large dim).
    ASSERT_NEAR(host_cost[i], expected_cost, 2e-4f);
  }
}

/** @brief Tests that robustified Jacobians match the expected Huber-corrected values. */
TYPED_TEST(ResidualBatchTest, Jacobians) {
  auto test_range = this->profiler_domain_.CreateDomainRange("Jacobian");
  CudaStream stream;

  typename TestFixture::StateData state_data(this->state_vectors);
  auto& states = state_data.get();
  auto state_pointers = test_utils::CollectStatePointersDevice(states);
  typename TestFixture::FactorData factor_data(this->observations);
  auto& factor_batch = factor_data.get();

  float* residuals_ptr = this->residuals_.data();
  float* jacobians_ptr = this->jacobians_.data();
  float** state_ptrs = state_pointers.data();

  HuberLossFunctionBatch loss(this->huber_delta);

  ResidualBatch residual_batch(&factor_batch, &loss);
  {
    auto range = this->profiler_domain_.CreateDomainRange("Evaluate Jacobians");
    residual_batch.Evaluate(stream.GetStream(), this->robust_workspace_.data(),
                              residuals_ptr, state_ptrs, nullptr, jacobians_ptr);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy jacobians from device to host for verification
  std::vector<float> host_jacobians(this->jacobians_.size());
  this->jacobians_.CopyToHost(host_jacobians.data(), host_jacobians.size());

  const int size = TypeParam::size;

  for (size_t i = 0; i < this->num_vectors_; i++) {
    const float residual = this->gt_residuals_[i];
    const float squared_err = this->squared_error_[i];
    const float3& rho = this->rho_[i];

    float alpha = jacobian_scaling_alpha(squared_err, rho);
    float sqrt_rho1 = sqrt(rho.y);

    std::vector<std::vector<float>> gt_jacobian(this->num_vectors_);
    {
      // Calculate the ground truth robustified jacobian,
      // Refer to http://ceres-solver.org/nnls_modeling.html#lossfunction

      float jac_value = -sqrt_rho1 * alpha * residual * residual;
      for (auto& x : gt_jacobian) {
        x.assign(size * size, jac_value);

        for (size_t k = 0; k < size; k++) {
          x[k * size + k] += sqrt_rho1;  // Adjust the diagonal
        }
      }
    }

    for (size_t k = 0; k < size; k++) {
      for (size_t j = 0; j < size; j++) {
        ASSERT_NEAR(host_jacobians[(size * i + k) * size + j],
                    gt_jacobian[i][k * size + j], 1e-5);
      }
    }
  }
}
}  // namespace cunls
