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

/**
 * @file sparse_matrix_test.cpp
 * @brief Unit tests for sparse matrix operations (diagonal, SpMV, weighted
 * squared step norms).
 *
 * Tests GPU implementations of diagonal extraction and weighted squared step
 * norms against CPU reference implementations.
 */

#include "cunls/minimizer/sparse_matrix.h"

#include <gtest/gtest.h>

#include <random>
#include <set>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/cusparse_helper.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/device_reduction.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/**
 * @brief Runs an async reduction to completion and returns the scalar.
 *
 * The minimizers only ever use the async forms, so the tests do too; this wraps
 * the device-side result the way Levenberg-Marquardt reads it.
 */
template <typename AsyncFn>
float RunAsyncReduction(cudaStream_t stream, size_t length, AsyncFn &&enqueue) {
  dvector<float> scratch(ReducePartialCount(length) + 1);
  float *d_out = scratch.data();
  float *d_partials = d_out + 1;
  enqueue(d_out, d_partials);
  float result = 0.f;
  THROW_ON_CUDA_ERROR(
      cudaMemcpyAsync(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost, stream));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
  return result;
}

/**
 * @brief Computes y = A * x on the CPU where A is in CSR format.
 *
 * @param row_ptr CSR row offsets.
 * @param col_idx CSR column indices.
 * @param values CSR values.
 * @param x Input vector.
 * @param[out] y Output result vector.
 */
void MultiplyCSRMatrixByVector(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                               const std::vector<float> &values, const std::vector<float> &x,
                               std::vector<float> &y) {
  const size_t num_rows = row_ptr.size() - 1;
  y.assign(num_rows, 0.f);
  for (size_t row = 0; row < num_rows; ++row) {
    float sum = 0.f;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
      sum += values[k] * x[col_idx[k]];
    }
    y[row] = sum;
  }
}

/**
 * @brief Extracts diagonal elements from a CSR matrix on the CPU.
 *
 * @param row_ptr CSR row offsets.
 * @param col_idx CSR column indices.
 * @param values CSR values.
 * @param diagonal Output diagonal vector.
 */
void ExtractDiagonalCPU(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                        const std::vector<float> &values, std::vector<float> &diagonal) {
  size_t rows = row_ptr.size() - 1;

  diagonal.clear();
  diagonal.resize(rows, 0);

  // For each row, search for the diagonal element
  for (int i = 0; i < rows; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      if (col_idx[j] == i) {
        diagonal[i] = values[j];
        break;  // Found diagonal element for this row
      }
    }
  }
}

}  // namespace

/**
 * @brief Test fixture for sparse matrix operations.
 *
 * Sets up a random sparse matrix in CSR format
 * for use across multiple test cases.
 */
class SparseMatrixTest : public ::testing::Test {
 public:
  /**
   * @brief Generates a random sparse matrix directly in CSR format.
   *
   * Roughly 10% of each row is non-zero and the diagonal is always present, so
   * the diagonal-extraction and damping tests have something to find.  Column
   * indices are sorted within a row, which every consumer relies on.
   *
   * @param rng Random number generator.
   * @param rows Number of rows (and columns) in the square matrix.
   */
  void GenerateRandomCSRMatrix(std::mt19937 &rng, int rows) {
    const int cols = rows;
    std::uniform_real_distribution<float> value_dist(0.1f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    constexpr float kSparsity = 0.1f;

    csr_row_offsets.assign(1, 0);
    for (int row = 0; row < rows; ++row) {
      const size_t nnz_in_row = 1 + static_cast<size_t>(cols * kSparsity);
      std::set<int> row_cols{row};  // the diagonal is always present
      while (row_cols.size() < nnz_in_row) {
        row_cols.insert(col_dist(rng));
      }
      for (int col : row_cols) {
        csr_col_idx.push_back(col);
        csr_values.push_back(value_dist(rng));
      }
      csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
    }
  }

  /** @brief Generates the random CSR matrix shared by the tests. */
  void SetUp() override {
    std::mt19937 rng(0);
    GenerateRandomCSRMatrix(rng, matrix_size);
  }

  const size_t matrix_size = 1000;

  // CSR (Compressed Sparse Row) format data
  std::vector<int> csr_row_offsets;
  std::vector<int> csr_col_idx;
  std::vector<float> csr_values;

  // Temporary buffer for GPU operations
  dvector<uint8_t> buffer;

  profiler::Domain profiler_domain_{"SparseMatrixTest"};
};

/** @brief Verifies GPU diagonal extraction from a CSR matrix (FindDiagonalPositions,
 * then ExtractDiagonalAt, as NormalEquations does) matches the CPU reference. */
TEST_F(SparseMatrixTest, ExtractDiagonal) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ExtractDiagonal");
  // Prepare GPU matrix
  CSRSparseMatrix csr_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values, csr_matrix);

  // Compute expected result using CPU reference
  std::vector<float> diagonal;
  ExtractDiagonalCPU(csr_row_offsets, csr_col_idx, csr_values, diagonal);

  // Extract diagonal on GPU
  CudaStream stream;

  dvector<float> device_diagonal;
  {
    auto range = this->profiler_domain_.CreateDomainRange("ExtractDiagonal");
    dvector<int> positions;
    FindDiagonalPositions(stream.GetStream(), csr_matrix, positions);
    ExtractDiagonalAt(stream.GetStream(), csr_matrix, positions, device_diagonal);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify GPU result matches CPU reference
  hvector<float> device_diagonal_host(device_diagonal.size());
  device_diagonal.CopyToHost(device_diagonal_host.data(), device_diagonal_host.size());
  ASSERT_EQ(device_diagonal_host, diagonal);
}

/** @brief Verifies the GPU diagonally weighted squared step norm sum(steps[i]^2 *
 * weights[i]) (WeightedDotProductToDevice, as Levenberg-Marquardt uses it). */
TEST_F(SparseMatrixTest, ComputeWeightedSquaredStepFirst) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ComputeWeightedSquaredStepFirstTest");
  // Generate random weights and steps
  std::vector<float> weights;
  test_utils::GenerateRandomVector(matrix_size, weights);

  std::vector<float> steps;
  test_utils::GenerateRandomVector(matrix_size, steps);

  // Compute expected result on CPU: sum of steps[i]^2 * weights[i]
  float gt_value = 0;
  for (size_t i = 0; i < matrix_size; i++) {
    gt_value += steps[i] * weights[i] * steps[i];
  }

  // Compute on GPU
  dvector<float> dweights(weights);
  dvector<float> dsteps(steps);

  CudaStream stream;
  float result;
  {
    auto range = this->profiler_domain_.CreateDomainRange("ComputeWeightedSquaredStepFirst");
    result =
        RunAsyncReduction(stream.GetStream(), dsteps.size(), [&](float *d_out, float *d_partials) {
          WeightedDotProductToDevice(stream.GetStream(), dsteps.data(), dweights.data(),
                                     dsteps.data(), dsteps.size(), d_out, d_partials);
        });
  }

  // Verify GPU result matches CPU computation
  ASSERT_NEAR(result, gt_value, 1e-3);
}

/** @brief Verifies the GPU matrix-weighted squared step norm steps^T * A * steps
 * (CSR SpMV, then a dot product). */
TEST_F(SparseMatrixTest, ComputeWeightedSquaredStepSecond) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("ComputeWeightedSquaredStepSecondTest");
  // Generate random step vector
  std::vector<float> steps;
  test_utils::GenerateRandomVector(matrix_size, steps);

  // Compute expected result on CPU: steps^T * A * steps
  // First compute temp = A * steps
  std::vector<float> temp;
  MultiplyCSRMatrixByVector(csr_row_offsets, csr_col_idx, csr_values, steps, temp);
  // Then compute steps^T * temp
  float gt_value = 0;
  for (size_t i = 0; i < matrix_size; i++) {
    gt_value += steps[i] * temp[i];
  }
  gt_value /= matrix_size;

  // Prepare GPU matrix and vector
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values, input_matrix);

  dvector<float> dsteps(steps);

  // Compute on GPU
  CudaStream stream;
  cuSPARSEHandle cusparse_handle;
  auto handle = cusparse_handle.GetHandle(stream.GetStream());
  float result;
  {
    auto range = this->profiler_domain_.CreateDomainRange("ComputeWeightedSquaredStepSecond");
    const int n = static_cast<int>(matrix_size);
    dvector<float> spmv_scratch;
    MultiplyCSRByDenseVector(stream.GetStream(), handle, input_matrix, n, n,
                             static_cast<int>(input_matrix.NumNonZeros()), dsteps, spmv_scratch,
                             buffer);
    result =
        RunAsyncReduction(stream.GetStream(), dsteps.size(), [&](float *d_out, float *d_partials) {
          DotProductToDevice(stream.GetStream(), dsteps.data(), spmv_scratch.data(), dsteps.size(),
                             d_out, d_partials);
        });
  }

  result /= matrix_size;

  ASSERT_NEAR(result, gt_value, 1e-3);
}

/**
 * @brief A 0x0 system is a valid CSR matrix, and every op over it is a no-op.
 *
 * `row_offsets == {0}` is the well-formed encoding of an empty matrix -- one
 * more offset than rows, with no rows. Every one of these ops derives its grid
 * from the row count, so a zero row count must be recognized before launch
 * rather than turned into an empty grid.
 */
TEST(SparseMatrixEmptySystem, OperationsOnZeroRowMatrixAreNoOps) {
  CSRSparseMatrix empty;
  test_utils::CreateCSRSparseMatrix({0}, {}, {}, empty);
  ASSERT_EQ(empty.NumRows(), 0);

  CudaStream stream;
  dvector<int> positions;
  dvector<float> diagonal;
  FindDiagonalPositions(stream.GetStream(), empty, positions);
  ExtractDiagonalAt(stream.GetStream(), empty, positions, diagonal);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(positions.size(), 0u);
  EXPECT_EQ(diagonal.size(), 0u);
}

}  // namespace cunls
