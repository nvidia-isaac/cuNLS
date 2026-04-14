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

/** @file utils.cpp
 *  @brief Implementations of common test utilities.
 */

#include "tests/utils.h"

#include <cmath>
#include <limits>

namespace cunls {
namespace test_utils {

void CreateCSRSparseMatrix(const std::vector<int> &row_ptr,
                           const std::vector<int> &col_idx,
                           const std::vector<float> &values,
                           CSRSparseMatrix &matrix) {
  size_t rows = row_ptr.size() - 1;
  size_t num_nonzeros = values.size();

  // Allocate device memory for the sparse matrix arrays
  matrix.row_offsets.resize(row_ptr.size());
  matrix.col_ids.resize(col_idx.size());
  matrix.values.resize(values.size());

  // Copy data from host vectors to device vectors
  matrix.row_offsets.CopyFromHost(row_ptr.data(), row_ptr.size());
  matrix.col_ids.CopyFromHost(col_idx.data(), col_idx.size());
  matrix.values.CopyFromHost(values.data(), values.size());
}

void GenerateRandomVector(size_t size, std::vector<float> &values) {
  unsigned int fixed_seed = 0;
  std::mt19937 rng(fixed_seed);
  std::uniform_real_distribution<float> val_dist(0.1, 1.0);
  values.clear();
  values.resize(size);

  for (size_t i = 0; i < size; ++i) {
    values[i] = val_dist(rng);
  }
}

float3 HuberLossCPU(float sq_error, float delta) {
  float3 rho;
  float delta_squared = delta * delta;
  if (sq_error > delta_squared) {
    const float r = sqrt(sq_error);
    rho.x = 2.0f * delta * r - delta_squared;
    rho.y = std::max(std::numeric_limits<float>::min(), delta / r);
    rho.z = -rho.y / (2.0f * sq_error);
  } else {
    rho = {sq_error, 1.0f, 0};
  }
  return rho;
}

float3 SoftLOneLossCPU(float s, float b, float c) {
  const float sum = 1.0f + s * c;
  const float tmp = std::sqrt(sum);
  float3 rho;
  rho.x = 2.0f * b * (tmp - 1.0f);
  rho.y = std::max(std::numeric_limits<float>::min(), 1.0f / tmp);
  rho.z = -(c * rho.y) / (2.0f * sum);
  return rho;
}

float3 CauchyLossCPU(float s, float b, float c) {
  const float sum = 1.0f + s * c;
  const float inv = 1.0f / sum;
  float3 rho;
  rho.x = b * std::log(sum);
  rho.y = std::max(std::numeric_limits<float>::min(), inv);
  rho.z = -c * (inv * inv);
  return rho;
}

float3 ArctanLossCPU(float s, float a, float b) {
  const float sum = 1.0f + s * s * b;
  const float inv = 1.0f / sum;
  float3 rho;
  rho.x = a * std::atan(s / a);
  rho.y = std::max(std::numeric_limits<float>::min(), inv);
  rho.z = -2.0f * s * b * (inv * inv);
  return rho;
}

constexpr float kTolerantLog2Pow53 = 36.7f;

float3 TolerantLossCPU(float s, float a, float b, float c) {
  const float x = (s - a) / b;
  float3 rho;
  if (x > kTolerantLog2Pow53) {
    rho.x = s - a - c;
    rho.y = 1.0f;
    rho.z = 0.0f;
  } else {
    const float e_x = std::exp(x);
    const float one_plus_e_x = 1.0f + e_x;
    rho.x = b * std::log(one_plus_e_x) - c;
    rho.y = std::max(std::numeric_limits<float>::min(), e_x / one_plus_e_x);
    rho.z = 0.5f / (b * (1.0f + std::cosh(x)));
  }
  return rho;
}

float3 TukeyLossCPU(float s, float a_squared) {
  float3 rho;
  if (s <= a_squared) {
    const float value = 1.0f - s / a_squared;
    const float value_sq = value * value;
    rho.x = (a_squared / 3.0f) * (1.0f - value_sq * value);
    rho.y = value_sq;
    rho.z = -2.0f / a_squared * value;
  } else {
    rho.x = a_squared / 3.0f;
    rho.y = 0.0f;
    rho.z = 0.0f;
  }
  return rho;
}

float3 ScaledLossCPU(float a, const float3 &inner_rho) {
  return {a * inner_rho.x, a * inner_rho.y, a * inner_rho.z};
}

} // namespace test_utils
} // namespace cunls
