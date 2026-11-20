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

/** @file loss_function_test.cpp
 *  @brief Tests for robust loss functions (trivial, Huber, SoftLOne, Cauchy,
 *         Arctan, Tolerant, Tukey, Scaled) evaluated on GPU.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/scaled_loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Test fixture for loss function evaluation on GPU.
 *
 * Generates random squared errors on device and checks a loss's output
 * against a CPU reference.
 */
class LossTest : public ::testing::Test {
 protected:
  /** @brief Initializes device vectors with random squared error values. */
  void SetUp() override {
    squared_errors.resize(num_errors);
    rho.resize(num_errors);
    std::uniform_real_distribution<float> dist(0, 100);
    std::vector<float> host_squared_errors(num_errors);
    for (size_t i = 0; i < num_errors; i++) {
      host_squared_errors[i] = dist(generator);
    }
    squared_errors.CopyFromHost(host_squared_errors.data(), num_errors);
  }

  /** @brief Evaluates @p loss on the squared errors and checks every output against
   * @p reference (squared error -> (rho, rho', rho'')) within @p tolerance. */
  template <typename Reference>
  void ExpectMatches(const LossFunctionBatch &loss, Reference reference, float tolerance) {
    CudaStream stream;
    loss.Evaluate(squared_errors.data(), rho.data(), num_errors, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    std::vector<float> host_squared_errors(num_errors);
    std::vector<float3> host_rho(num_errors);
    squared_errors.CopyToHost(host_squared_errors.data(), num_errors);
    rho.CopyToHost(host_rho.data(), num_errors);
    for (size_t i = 0; i < num_errors; i++) {
      const float3 expected = reference(host_squared_errors[i]);
      ASSERT_NEAR(host_rho[i].x, expected.x, tolerance) << "s = " << host_squared_errors[i];
      ASSERT_NEAR(host_rho[i].y, expected.y, tolerance) << "s = " << host_squared_errors[i];
      ASSERT_NEAR(host_rho[i].z, expected.z, tolerance) << "s = " << host_squared_errors[i];
    }
  }

  const size_t num_errors = 1000;
  std::mt19937 generator;
  DeviceVector<float> squared_errors;
  DeviceVector<float3> rho;
};

/** @brief The trivial loss returns identity-like rho values (rho=[s, 1, 0]). */
TEST_F(LossTest, TrivialLoss) {
  ExpectMatches(
      TrivialLossFunctionBatch(),
      [](float s) {
        return float3{s, 1.f, 0.f};
      },
      1e-5f);
}

/** @brief The Huber loss matches the CPU reference for a random delta. */
TEST_F(LossTest, HuberLoss) {
  std::uniform_real_distribution<float> dist(0.5, 20);
  const float delta = dist(generator);
  ExpectMatches(
      HuberLossFunctionBatch(delta), [&](float s) { return test_utils::HuberLossCPU(s, delta); },
      1e-5f);
}

/** @brief The SoftLOne loss matches the CPU reference. */
TEST_F(LossTest, SoftLOneLoss) {
  std::uniform_real_distribution<float> dist_b(0.1f, 2.0f);
  std::uniform_real_distribution<float> dist_c(0.01f, 1.0f);
  const float b = dist_b(generator);
  const float c = dist_c(generator);
  ExpectMatches(
      SoftLOneLossFunctionBatch(b, c),
      [&](float s) { return test_utils::SoftLOneLossCPU(s, b, c); }, 1e-4f);
}

/** @brief The Cauchy loss matches the CPU reference. */
TEST_F(LossTest, CauchyLoss) {
  std::uniform_real_distribution<float> dist_b(0.1f, 2.0f);
  std::uniform_real_distribution<float> dist_c(0.01f, 1.0f);
  const float b = dist_b(generator);
  const float c = dist_c(generator);
  ExpectMatches(
      CauchyLossFunctionBatch(b, c), [&](float s) { return test_utils::CauchyLossCPU(s, b, c); },
      1e-4f);
}

/** @brief The Arctan loss matches the CPU reference. */
TEST_F(LossTest, ArctanLoss) {
  std::uniform_real_distribution<float> dist_a(0.5f, 5.0f);
  std::uniform_real_distribution<float> dist_b(0.01f, 0.5f);
  const float a = dist_a(generator);
  const float b = dist_b(generator);
  ExpectMatches(
      ArctanLossFunctionBatch(a, b), [&](float s) { return test_utils::ArctanLossCPU(s, a, b); },
      1e-4f);
}

/** @brief The Tolerant loss matches the CPU reference. */
TEST_F(LossTest, TolerantLoss) {
  const float a = 1.0f;
  const float b = 0.5f;
  const float c = b * std::log(1.0f + std::exp(-a / b));
  ExpectMatches(
      TolerantLossFunctionBatch(a, b),
      [&](float s) { return test_utils::TolerantLossCPU(s, a, b, c); }, 1e-4f);
}

/** @brief The Tukey loss matches the CPU reference. */
TEST_F(LossTest, TukeyLoss) {
  std::uniform_real_distribution<float> dist_a(2.0f, 15.0f);
  const float a = dist_a(generator);
  ExpectMatches(
      TukeyLossFunctionBatch(a), [&](float s) { return test_utils::TukeyLossCPU(s, a * a); },
      1e-4f);
}

/** @brief ScaledLossFunctionBatch wrapping Huber matches a * HuberCPU; a <= 0 is rejected. */
TEST_F(LossTest, ScaledHuberLoss) {
  std::uniform_real_distribution<float> dist_a(0.1f, 10.0f);
  std::uniform_real_distribution<float> dist_delta(0.5f, 20.0f);
  const float scale = dist_a(generator);
  const float delta = dist_delta(generator);
  const HuberLossFunctionBatch huber(delta);
  ExpectMatches(
      ScaledLossFunctionBatch(huber, scale),
      [&](float s) { return test_utils::ScaledLossCPU(scale, test_utils::HuberLossCPU(s, delta)); },
      1e-4f);
  EXPECT_THROW(ScaledLossFunctionBatch(huber, 0.f), std::invalid_argument);
}
}  // namespace cunls
