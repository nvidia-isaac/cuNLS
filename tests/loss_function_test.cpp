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

/** @file loss_function_test.cpp
 *  @brief Tests for robust loss functions (trivial, Huber, SoftLOne, Cauchy,
 *         Arctan, Tolerant, Tukey) evaluated on GPU.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for loss function evaluation on GPU.
 *
 * Generates random squared errors on device and provides buffers for rho output.
 */
class LossTest : public ::testing::Test {
 protected:
  /** @brief Initializes device vectors with random squared error values. */
  void SetUp() override {
    squared_errors.resize(num_errors);
    rho.resize(num_errors);
    std::uniform_real_distribution<float> dist(0, 100);

    // Generate host data first, then copy to device
    std::vector<float> host_squared_errors(num_errors);
    for (size_t i = 0; i < num_errors; i++) {
      host_squared_errors[i] = dist(generator);
    }
    squared_errors.CopyFromHost(host_squared_errors.data(), num_errors);
  }

  const size_t num_errors = 1000;
  std::mt19937 generator;
  DeviceVector<float> squared_errors;
  DeviceVector<float3> rho;

  profiler::Domain profiler_domain_{"LossTest"};
};

/** @brief Verifies that the trivial loss returns identity-like rho values (rho=[s, 1, 0]). */
TEST_F(LossTest, TrivialLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("TrivialLoss");
  CudaStream stream;

  TrivialLossFunctionBatch loss;
  auto s_ptr = this->squared_errors.data();
  auto rho_ptr = this->rho.data();

  {
    auto range = this->profiler_domain_.CreateDomainRange(
        "TrivialLossFunctionBatch::Evaluate");
    loss.Evaluate(s_ptr, rho_ptr, num_errors, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for comparison
  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    ASSERT_NEAR(host_rho[i].x, host_squared_errors[i], 1e-5);
    ASSERT_NEAR(host_rho[i].y, 1.f, 1e-5);
    ASSERT_NEAR(host_rho[i].z, 0, 1e-5);
  }
}

/** @brief Verifies that the Huber loss GPU output matches the CPU reference for a random delta. */
TEST_F(LossTest, HuberLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("HuberLoss");
  CudaStream stream;

  std::uniform_real_distribution<float> dist(0.5, 20);
  float delta = dist(this->generator);

  HuberLossFunctionBatch loss(delta);
  auto s_ptr = this->squared_errors.data();
  auto rho_ptr = this->rho.data();

  {
    auto range = this->profiler_domain_.CreateDomainRange(
        "HuberLossFunctionBatch::Evaluate");
    loss.Evaluate(s_ptr, rho_ptr, num_errors, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Copy results back to host for comparison
  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt_rho = test_utils::HuberLossCPU(host_squared_errors[i], delta);
    ASSERT_NEAR(host_rho[i].x, gt_rho.x, 1e-5);
    ASSERT_NEAR(host_rho[i].y, gt_rho.y, 1e-5);
    ASSERT_NEAR(host_rho[i].z, gt_rho.z, 1e-5);
  }
}

/** @brief Verifies SoftLOne loss GPU output matches CPU reference. */
TEST_F(LossTest, SoftLOneLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("SoftLOneLoss");
  CudaStream stream;

  std::uniform_real_distribution<float> dist_b(0.1f, 2.0f);
  std::uniform_real_distribution<float> dist_c(0.01f, 1.0f);
  float b = dist_b(this->generator);
  float c = dist_c(this->generator);

  SoftLOneLossFunctionBatch loss(b, c);
  loss.Evaluate(this->squared_errors.data(), this->rho.data(), num_errors,
                stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt = test_utils::SoftLOneLossCPU(host_squared_errors[i], b, c);
    ASSERT_NEAR(host_rho[i].x, gt.x, 1e-4f);
    ASSERT_NEAR(host_rho[i].y, gt.y, 1e-4f);
    ASSERT_NEAR(host_rho[i].z, gt.z, 1e-4f);
  }
}

/** @brief Verifies Cauchy loss GPU output matches CPU reference. */
TEST_F(LossTest, CauchyLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("CauchyLoss");
  CudaStream stream;

  std::uniform_real_distribution<float> dist_b(0.1f, 2.0f);
  std::uniform_real_distribution<float> dist_c(0.01f, 1.0f);
  float b = dist_b(this->generator);
  float c = dist_c(this->generator);

  CauchyLossFunctionBatch loss(b, c);
  loss.Evaluate(this->squared_errors.data(), this->rho.data(), num_errors,
                stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt = test_utils::CauchyLossCPU(host_squared_errors[i], b, c);
    ASSERT_NEAR(host_rho[i].x, gt.x, 1e-4f);
    ASSERT_NEAR(host_rho[i].y, gt.y, 1e-4f);
    ASSERT_NEAR(host_rho[i].z, gt.z, 1e-4f);
  }
}

/** @brief Verifies Arctan loss GPU output matches CPU reference. */
TEST_F(LossTest, ArctanLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ArctanLoss");
  CudaStream stream;

  std::uniform_real_distribution<float> dist_a(0.5f, 5.0f);
  std::uniform_real_distribution<float> dist_b(0.01f, 0.5f);
  float a = dist_a(this->generator);
  float b = dist_b(this->generator);

  ArctanLossFunctionBatch loss(a, b);
  loss.Evaluate(this->squared_errors.data(), this->rho.data(), num_errors,
                stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt = test_utils::ArctanLossCPU(host_squared_errors[i], a, b);
    ASSERT_NEAR(host_rho[i].x, gt.x, 1e-4f);
    ASSERT_NEAR(host_rho[i].y, gt.y, 1e-4f);
    ASSERT_NEAR(host_rho[i].z, gt.z, 1e-4f);
  }
}

/** @brief Verifies Tolerant loss GPU output matches CPU reference. */
TEST_F(LossTest, TolerantLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("TolerantLoss");
  CudaStream stream;

  float a = 1.0f;
  float b = 0.5f;
  float c = b * std::log(1.0f + std::exp(-a / b));

  TolerantLossFunctionBatch loss(a, b);
  loss.Evaluate(this->squared_errors.data(), this->rho.data(), num_errors,
                stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt = test_utils::TolerantLossCPU(host_squared_errors[i], a, b, c);
    ASSERT_NEAR(host_rho[i].x, gt.x, 1e-4f);
    ASSERT_NEAR(host_rho[i].y, gt.y, 1e-4f);
    ASSERT_NEAR(host_rho[i].z, gt.z, 1e-4f);
  }
}

/** @brief Verifies Tukey loss GPU output matches CPU reference. */
TEST_F(LossTest, TukeyLoss) {
  auto test_range = this->profiler_domain_.CreateDomainRange("TukeyLoss");
  CudaStream stream;

  std::uniform_real_distribution<float> dist_a(2.0f, 15.0f);
  float a = dist_a(this->generator);
  float a_squared = a * a;

  TukeyLossFunctionBatch loss(a);
  loss.Evaluate(this->squared_errors.data(), this->rho.data(), num_errors,
                stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> host_squared_errors(num_errors);
  std::vector<float3> host_rho(num_errors);
  this->squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
  this->rho.CopyToHost(host_rho.data(), num_errors);

  for (size_t i = 0; i < this->num_errors; i++) {
    auto gt = test_utils::TukeyLossCPU(host_squared_errors[i], a_squared);
    ASSERT_NEAR(host_rho[i].x, gt.x, 1e-4f);
    ASSERT_NEAR(host_rho[i].y, gt.y, 1e-4f);
    ASSERT_NEAR(host_rho[i].z, gt.z, 1e-4f);
  }
}
}  // namespace cunls
