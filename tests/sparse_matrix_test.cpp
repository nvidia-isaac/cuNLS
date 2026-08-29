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
 * @brief Unit tests for sparse matrix operations (COO-to-CSR, diagonal, copy,
 * SpMV, RHS).
 *
 * Tests GPU implementations of diagonal extraction,
 * scaled diagonal addition, matrix copy, weighted squared step norms, and
 * right-hand-side computation against CPU reference implementations.
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

/**
 * @brief Adds a scaled diagonal to a CSR matrix in place: A += scale * diag(d).
 *
 * @param row_ptr CSR row offsets.
 * @param col_idx CSR column indices.
 * @param values CSR values (modified in place).
 * @param scale Scalar multiplier for the diagonal.
 * @param diagonal Diagonal vector to add.
 */
void AddScaledDiagonalCPU(const std::vector<int> &row_ptr, const std::vector<int> &col_idx,
                          std::vector<float> &values, float scale,
                          const std::vector<float> &diagonal) {
  // For each row, find and update the diagonal element
  for (size_t i = 0; i < diagonal.size(); ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      if (col_idx[j] == i) {
        values[j] += scale * diagonal[i];
        break;  // Found and updated diagonal element for this row
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

/** @brief Verifies GPU diagonal extraction from a CSR matrix matches the CPU
 * reference. */
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
    ExtractDiagonal(stream.GetStream(), csr_matrix, device_diagonal);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify GPU result matches CPU reference
  hvector<float> device_diagonal_host(device_diagonal.size());
  device_diagonal.CopyToHost(device_diagonal_host.data(), device_diagonal_host.size());
  ASSERT_EQ(device_diagonal_host, diagonal);
}

/** @brief Verifies GPU AddScaledDiagonal (A + scale * diag(d)) matches the CPU
 * reference. */
TEST_F(SparseMatrixTest, AddScaledDiagonal) {
  auto test_range = this->profiler_domain_.CreateDomainRange("AddScaledDiagonalTest");
  // Prepare GPU matrix
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values, input_matrix);

  // Generate random diagonal and scale factor
  float scale = 10;
  std::vector<float> diagonal;
  test_utils::GenerateRandomVector(matrix_size, diagonal);
  dvector<float> device_diagonal(diagonal);

  // Compute expected result using CPU reference
  AddScaledDiagonalCPU(csr_row_offsets, csr_col_idx, csr_values, scale, diagonal);

  // Perform operation on GPU
  CSRSparseMatrix result_matrix;
  CudaStream stream;
  {
    auto range = this->profiler_domain_.CreateDomainRange("AddScaledDiagonal");
    AddScaledDiagonal(stream.GetStream(), scale, device_diagonal, input_matrix, result_matrix);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  constexpr float tolerance = 1e-3;

  // Verify structure is preserved
  hvector<int> result_row_offsets_host(result_matrix.row_offsets.size());
  result_matrix.row_offsets.CopyToHost(result_row_offsets_host.data(),
                                       result_row_offsets_host.size());
  hvector<int> result_col_ids_host(result_matrix.col_ids.size());
  result_matrix.col_ids.CopyToHost(result_col_ids_host.data(), result_col_ids_host.size());
  ASSERT_EQ(result_row_offsets_host, csr_row_offsets);
  ASSERT_EQ(result_col_ids_host, csr_col_idx);

  // Verify values match CPU reference (within tolerance)
  hvector<float> mat_values(result_matrix.values.size());
  result_matrix.values.CopyToHost(mat_values.data(), mat_values.size());
  ASSERT_EQ(mat_values.size(), csr_values.size());
  for (size_t i = 0; i < csr_values.size(); i++) {
    ASSERT_NEAR(mat_values[i], csr_values[i], tolerance);
  }
}

/** @brief Verifies GPU CSR matrix copy preserves structure and values. */
TEST_F(SparseMatrixTest, Copy) {
  auto test_range = this->profiler_domain_.CreateDomainRange("CopyTest");
  // Prepare source matrix on GPU
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values, input_matrix);

  // Copy matrix on GPU
  CSRSparseMatrix result_matrix;
  CudaStream stream;
  {
    auto range = this->profiler_domain_.CreateDomainRange("Copy");
    CopyCSRSparseMatrix(stream.GetStream(), input_matrix, result_matrix);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify structure is preserved
  hvector<int> result_row_offsets_host(result_matrix.row_offsets.size());
  result_matrix.row_offsets.CopyToHost(result_row_offsets_host.data(),
                                       result_row_offsets_host.size());
  hvector<int> result_col_ids_host(result_matrix.col_ids.size());
  result_matrix.col_ids.CopyToHost(result_col_ids_host.data(), result_col_ids_host.size());
  ASSERT_EQ(result_row_offsets_host, csr_row_offsets);
  ASSERT_EQ(result_col_ids_host, csr_col_idx);

  // Verify values match (within floating point tolerance)
  hvector<float> mat_values(result_matrix.values.size());
  result_matrix.values.CopyToHost(mat_values.data(), mat_values.size());
  ASSERT_EQ(mat_values.size(), csr_values.size());
  for (size_t i = 0; i < csr_values.size(); i++) {
    ASSERT_NEAR(mat_values[i], csr_values[i], 1e-3);
  }
}

/** @brief Verifies GPU vector-form weighted squared step norm: sum(steps[i]^2 *
 * weights[i]). */
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
          ComputeWeightedSquaredStepAsync(stream.GetStream(), dweights, dsteps, d_out, d_partials);
        });
  }

  // Verify GPU result matches CPU computation
  ASSERT_NEAR(result, gt_value, 1e-3);
}

/** @brief Verifies GPU matrix-form weighted squared step norm: steps^T * A *
 * steps. */
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
    int num_rows = 0, num_cols = 0, num_nonzeros = 0;
    ExtractMatrixMetadata(stream.GetStream(), input_matrix, num_rows, num_cols, num_nonzeros);
    dvector<float> spmv_scratch;
    result =
        RunAsyncReduction(stream.GetStream(), dsteps.size(), [&](float *d_out, float *d_partials) {
          ComputeWeightedSquaredStepAsync(stream.GetStream(), handle, input_matrix, num_rows,
                                          num_cols, num_nonzeros, dsteps, spmv_scratch, buffer,
                                          d_out, d_partials);
        });
  }

  result /= matrix_size;

  ASSERT_NEAR(result, gt_value, 1e-3);
}

/** @brief SHS scaling matches hand-derived 2x2 normal equations. */
TEST(SparseMatrixColumnScaling, SymmetricScaling2x2) {
  std::vector<int> csr_row_offsets = {0, 2, 4};
  std::vector<int> csr_col_idx = {0, 1, 0, 1};
  std::vector<float> csr_values = {4.f, 1.f, 1.f, 9.f};

  CSRSparseMatrix h;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values, h);

  std::vector<float> host_scale = {0.5f, 1.f / 3.f};
  dvector<float> scale(host_scale);

  CudaStream stream;
  ScaleSymmetricCSR(stream.GetStream(), h, scale);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  hvector<float> out(h.NumNonZeros());
  h.values.CopyToHost(out.data(), out.size());
  ASSERT_NEAR(out[0], 1.f, 1e-4f);
  ASSERT_NEAR(out[1], 1.f / 6.f, 1e-4f);
  ASSERT_NEAR(out[2], 1.f / 6.f, 1e-4f);
  ASSERT_NEAR(out[3], 1.f, 1e-4f);
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
  dvector<float> scale;
  dvector<float> diagonal;

  ScaleSymmetricCSR(stream.GetStream(), empty, scale);
  ExtractDiagonal(stream.GetStream(), empty, diagonal);
  EXPECT_EQ(diagonal.size(), 0u);

  // Damping still has to produce the (empty) output matrix, since callers read
  // `damped` afterwards regardless of size.
  CSRSparseMatrix damped;
  AddScaledDiagonal(stream.GetStream(), 1e-3f, diagonal, empty, damped);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  EXPECT_EQ(damped.NumRows(), 0);
  EXPECT_EQ(damped.NumNonZeros(), 0);
  ASSERT_EQ(damped.row_offsets.size(), 1u);
  hvector<int> offsets(1);
  damped.row_offsets.CopyToHost(offsets.data(), offsets.size());
  EXPECT_EQ(offsets[0], 0);
}

/** @brief Metadata of an empty-but-well-formed CSR is 0x0 with no nonzeros. */
TEST(SparseMatrixEmptySystem, MetadataOfZeroRowMatrixIsAllZero) {
  CSRSparseMatrix empty;
  test_utils::CreateCSRSparseMatrix({0}, {}, {}, empty);

  CudaStream stream;
  int num_rows = -1, num_cols = -1, num_nonzeros = -1;
  ExtractMatrixMetadata(stream.GetStream(), empty, num_rows, num_cols, num_nonzeros);

  EXPECT_EQ(num_rows, 0);
  EXPECT_EQ(num_cols, 0);
  EXPECT_EQ(num_nonzeros, 0);
}

}  // namespace cunls
