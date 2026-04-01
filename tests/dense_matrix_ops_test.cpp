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

/** @file dense_matrix_ops_test.cpp
 *  @brief Tests for dense matrix operations, including matrix square root computation.
 */

#include "cunls/math/dense_matrix_ops.h"

#include <cublas_v2.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "tests/utils.h"

namespace cunls {

/**
 * @brief Generates a random SPD matrix using eigenvalue decomposition.
 *
 * Creates A = Q * D * Q^T where Q is a random orthogonal matrix
 * and D is a diagonal matrix with positive eigenvalues.
 *
 * @param matrix Output matrix (row-major, size x size)
 * @param size Matrix size
 * @param rng Random number generator
 */
void GenerateRandomSPDMatrix(float* matrix, size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> eig_dist(1.f, 10.0f);

  // Generate random eigenvalues (positive for SPD)
  hvector<float> eigenvalues(size);
  for (size_t i = 0; i < size; ++i) {
    eigenvalues[i] = eig_dist(rng);
  }

  // Generate a random orthogonal matrix Q using Gram-Schmidt
  hvector<float> Q(size * size);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  // Generate random matrix
  for (size_t i = 0; i < size * size; ++i) {
    Q[i] = normal_dist(rng);
  }

  // Gram-Schmidt orthogonalization
  for (size_t i = 0; i < size; ++i) {
    // Normalize column i
    float norm = 0.0f;
    for (size_t j = 0; j < size; ++j) {
      norm += Q[j * size + i] * Q[j * size + i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
      for (size_t j = 0; j < size; ++j) {
        Q[j * size + i] /= norm;
      }
    }

    // Subtract projections onto previous columns
    for (size_t k = 0; k < i; ++k) {
      float dot = 0.0f;
      for (size_t j = 0; j < size; ++j) {
        dot += Q[j * size + i] * Q[j * size + k];
      }
      for (size_t j = 0; j < size; ++j) {
        Q[j * size + i] -= dot * Q[j * size + k];
      }
    }

    // Renormalize
    norm = 0.0f;
    for (size_t j = 0; j < size; ++j) {
      norm += Q[j * size + i] * Q[j * size + i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
      for (size_t j = 0; j < size; ++j) {
        Q[j * size + i] /= norm;
      }
    }
  }

  // Compute A = Q * D * Q^T (row-major)
  // First compute D * Q^T (which is just scaling rows of Q^T by eigenvalues)
  hvector<float> DQT(size * size);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      DQT[i * size + j] = eigenvalues[i] * Q[j * size + i];
    }
  }

  // Then compute Q * (D * Q^T)
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < size; ++k) {
        sum += Q[i * size + k] * DQT[k * size + j];
      }
      matrix[i * size + j] = sum;
    }
  }
}

/**
 * @brief Test fixture for matrix square root computation on GPU.
 *
 * Generates random SPD matrices and verifies that (A^0.5)^2 == A.
 *
 * @tparam MatrixSize Compile-time matrix dimension.
 */
template <class MatrixSize>
class ComputeSqrtMatrixTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Initialize random number generator with fixed seed for reproducibility
    std::mt19937 rng(42);

    matrix_size_ = MatrixSize::value;

    // Generate random SPD matrices on host
    matrices_host_.resize(num_matrices_ * matrix_size_ * matrix_size_);
    for (size_t i = 0; i < num_matrices_; ++i) {
      GenerateRandomSPDMatrix(&matrices_host_[i * matrix_size_ * matrix_size_],
                              matrix_size_, rng);
    }

    // Copy to device
    matrices_device_ = DeviceVector<float>(matrices_host_);
  }

 protected:
  const size_t num_matrices_ = 10000;
  dvector<float> matrices_device_;
  hvector<float> matrices_host_;
  size_t matrix_size_;
  profiler::Domain profiler_domain_{"ComputeSqrtMatrixTest"};
};

/// Test with different matrix sizes (2x2, 3x3, and 6x6)
using MatrixSizes = ::testing::Types<test_utils::SizeT<2>, test_utils::SizeT<3>,
                                     test_utils::SizeT<6>>;

TYPED_TEST_SUITE(ComputeSqrtMatrixTest, MatrixSizes);

/** @brief Verifies that squaring the matrix square root recovers the original SPD matrix. */
TYPED_TEST(ComputeSqrtMatrixTest, PowerHalf) {
  auto test_range = this->profiler_domain_.CreateDomainRange("PowerHalf");
  constexpr float tolerance = 1e-3f;

  CudaStream stream;
  size_t size = this->matrix_size_;

  // Get device pointer
  float* matrices_ptr = this->matrices_device_.data();

  // Make a copy for verification (device-to-device copy)
  dvector<float> matrices_copy(this->matrices_device_.size());
  THROW_ON_CUDA_ERROR(cudaMemcpy(matrices_copy.data(), this->matrices_device_.data(),
                                  this->matrices_device_.size() * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
  float* matrices_copy_ptr = matrices_copy.data();

  {  // Compute matrix square root on GPU
    auto range = this->profiler_domain_.CreateDomainRange("ComputeSqrtMatrix");
    cuBLASHandle cublas_handle;
    ComputeSqrtMatrix(cublas_handle, stream.GetStream(), matrices_ptr, size, size,
                      this->num_matrices_);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify: (A^0.5)^2 should equal A
  // Compute (A^0.5)^2 using matrix multiplication
  cuBLASHandle cublas_handle;
  auto handle = static_cast<cublasHandle_t>(cublas_handle.GetHandle(stream.GetStream()));
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  dvector<float> squared_result(this->num_matrices_ * size * size);
  float* squared_result_ptr = squared_result.data();

  // Compute squared_result = result * result (matrix multiplication)
  size_t stride = size * size;
  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, matrices_ptr,
      size, stride, matrices_ptr, size, stride, &beta, squared_result_ptr, size,
      stride, this->num_matrices_));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Copy squared result to host
  std::vector<float> squared_host(squared_result.size());
  squared_result.CopyToHost(squared_host.data(), squared_host.size());

  // Compare squared result with original matrix
  for (size_t mat_idx = 0; mat_idx < this->num_matrices_; ++mat_idx) {
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        size_t idx = mat_idx * size * size + i * size + j;
        ASSERT_NEAR(squared_host[idx], this->matrices_host_[idx], tolerance);
      }
    }
  }
}

}  // namespace cunls
