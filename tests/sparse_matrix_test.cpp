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
 * Tests GPU implementations of triplet-to-CSR conversion, diagonal extraction,
 * scaled diagonal addition, matrix copy, weighted squared step norms, and
 * right-hand-side computation against CPU reference implementations.
 */

#include "cunls/minimizer/sparse_matrix.h"

#include <gtest/gtest.h>

#include <random>
#include <unordered_set>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/cusparse_helper.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/** @brief Represents a sparse matrix entry in COO (Coordinate) format. */
struct Triplet {
  float value;
  int row_id;
  int col_id;
};

/**
 * @brief Converts a COO sparse matrix to CSR format on the CPU.
 *
 * Handles -1 sentinel column values as invalid/missing entries.
 *
 * @param coo_values COO non-zero values.
 * @param coo_col_indices COO column indices (-1 marks invalid entries).
 * @param coo_row_indices COO row indices.
 * @param csr_values Output CSR values.
 * @param csr_col_indices Output CSR column indices.
 * @param csr_row_pointers Output CSR row pointers.
 */
void ConvertTripletToCSRCPU(const std::vector<float> &coo_values,
                            const std::vector<int> &coo_col_indices,
                            const std::vector<int> &coo_row_indices,
                            std::vector<float> &csr_values,
                            std::vector<int> &csr_col_indices,
                            std::vector<int> &csr_row_pointers) {
  // Create triplet representation for easier sorting
  std::vector<Triplet> triplets;
  triplets.reserve(coo_values.size());
  for (int i = 0; i < coo_values.size(); i++) {
    triplets.push_back({coo_values[i], coo_row_indices[i], coo_col_indices[i]});
  }

  // Sort triplets, moving invalid entries (col_id == -1) to the end
  std::stable_sort(triplets.begin(), triplets.end(),
                   [](const Triplet &left, const Triplet &right) {
                     return right.col_id < 0;
                   });

  // Count the number of valid non-zero entries (excluding -1 column indices)
  size_t num_nonzeros =
      std::accumulate(coo_col_indices.begin(), coo_col_indices.end(), 0,
                      [](int acc, int val) { return acc + int(val != -1); });

  // Determine the number of rows in the matrix
  int num_rows =
      *std::max_element(coo_row_indices.begin(), coo_row_indices.end()) + 1;

  // Step 1: Count non-zeros per row
  std::vector<int> row_counts(num_rows, 0);
  for (size_t i = 0; i < num_nonzeros; i++) {
    const Triplet &triplet = triplets[i];
    row_counts[triplet.row_id]++;
  }

  // Step 2: Compute CSR row pointers (cumulative sum of row counts)
  // row_pointers[i] indicates where row i starts in the values array
  csr_row_pointers.resize(num_rows + 1);
  csr_row_pointers[0] = 0;
  for (int i = 0; i < num_rows; ++i) {
    csr_row_pointers[i + 1] = csr_row_pointers[i] + row_counts[i];
  }

  // Step 3: Populate CSR values and column indices arrays
  csr_values.resize(num_nonzeros);
  csr_col_indices.resize(num_nonzeros);

  // Track the current position for each row as we fill the arrays
  std::vector<int> current_row_positions = csr_row_pointers;
  for (size_t i = 0; i < num_nonzeros; i++) {
    const Triplet &triplet = triplets[i];
    int target_index = current_row_positions[triplet.row_id];
    csr_values[target_index] = triplet.value;
    csr_col_indices[target_index] = triplet.col_id;
    current_row_positions[triplet.row_id]++;
  }
}

/**
 * @brief Computes y = A * x on the CPU where A is in CSR format.
 *
 * @param row_ptr CSR row offsets.
 * @param col_idx CSR column indices.
 * @param values CSR values.
 * @param x Input vector.
 * @param y Output result vector.
 */
void MultiplyCSRMatrixByVector(const std::vector<int> &row_ptr,
                               const std::vector<int> &col_idx,
                               const std::vector<float> &values,
                               const std::vector<float> &x,
                               std::vector<float> &y) {
  int rows = row_ptr.size() - 1;
  y.assign(rows, 0.0);

  // For each row, iterate through its non-zero elements
  for (int i = 0; i < rows; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      y[i] += values[j] * x[col_idx[j]];
    }
  }
}

/**
 * @brief Computes y = -A^T * x on the CPU (negative transpose SpMV).
 *
 * @param row_ptr CSR row offsets.
 * @param col_idx CSR column indices.
 * @param values CSR values.
 * @param x Input vector.
 * @param y Output result vector.
 */
void ComputeRHSonCPU(const std::vector<int> &row_ptr,
                     const std::vector<int> &col_idx,
                     const std::vector<float> &values,
                     const std::vector<float> &x, std::vector<float> &y) {
  int rows = row_ptr.size() - 1;
  int cols = rows;

  y.assign(cols, 0.0); // A^T * x has size equal to number of columns in A

  // For transpose multiplication, iterate through rows but accumulate into
  // columns
  for (int i = 0; i < rows; ++i) {
    float xi = x[i];
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      int col = col_idx[j];
      y[col] -= values[j] * xi; // Note: negative sign
    }
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
void ExtractDiagonalCPU(const std::vector<int> &row_ptr,
                        const std::vector<int> &col_idx,
                        const std::vector<float> &values,
                        std::vector<float> &diagonal) {
  size_t rows = row_ptr.size() - 1;

  diagonal.clear();
  diagonal.resize(rows, 0);

  // For each row, search for the diagonal element
  for (int i = 0; i < rows; ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      if (col_idx[j] == i) {
        diagonal[i] = values[j];
        break; // Found diagonal element for this row
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
void AddScaledDiagonalCPU(const std::vector<int> &row_ptr,
                          const std::vector<int> &col_idx,
                          std::vector<float> &values, float scale,
                          const std::vector<float> &diagonal) {
  // For each row, find and update the diagonal element
  for (size_t i = 0; i < diagonal.size(); ++i) {
    for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
      if (col_idx[j] == i) {
        values[j] += scale * diagonal[i];
        break; // Found and updated diagonal element for this row
      }
    }
  }
}

} // namespace

/**
 * @brief Test fixture for sparse matrix operations.
 *
 * Sets up a random sparse matrix in both COO (triplet) and CSR formats
 * for use across multiple test cases.
 */
class SparseMatrixTest : public ::testing::Test {
public:
  /**
   * @brief Generates a random sparse matrix in COO (triplet) format.
   *
   * ~10% of entries are non-zero, diagonal is always included, and
   * 20% of column entries are set to -1 (invalid/sentinel).
   *
   * @param rng Random number generator.
   * @param rows Number of rows (and columns) in the square matrix.
   */
  void GenerateRandomTripletMatrix(std::mt19937 &rng, int rows) {
    int cols = rows;
    std::uniform_real_distribution<float> val_dist(0.1, 1.0);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);

    std::uniform_real_distribution<float> unit_distr(0, 1);

    constexpr float sparsity = 0.1;

    for (int i = 0; i < rows; ++i) {
      // Each row has at least the diagonal element plus ~10% of columns
      int nnz_in_row = 1 + static_cast<int>(cols * sparsity);
      std::unordered_set<int> used_cols{i}; // Start with diagonal

      // Generate random column indices for this row
      while (used_cols.size() < nnz_in_row) {
        int col = col_dist(rng);
        used_cols.insert(col);
      }

      // Add all non-zero entries for this row
      for (int col : used_cols) {
        triplet_row_idx.push_back(i);
        triplet_values.push_back(val_dist(rng));

        // 20% chance to mark column as invalid (-1)
        col = unit_distr(rng) > 0.8 ? -1 : col;
        triplet_col_idx.push_back(col);
      }
    }
  }

  /** @brief Generates a random matrix and converts to CSR format for tests. */
  void SetUp() override {
    unsigned int fixed_seed = 0;
    std::mt19937 gen(fixed_seed);

    GenerateRandomTripletMatrix(gen, matrix_size);

    // Convert from COO to CSR format for testing
    ConvertTripletToCSRCPU(triplet_values, triplet_col_idx, triplet_row_idx,
                           csr_values, csr_col_idx, csr_row_offsets);
  }

  const size_t matrix_size = 1000;

  // COO (Coordinate/Triplet) format data
  std::vector<int> triplet_row_idx;
  std::vector<int> triplet_col_idx;
  std::vector<float> triplet_values;

  // CSR (Compressed Sparse Row) format data
  std::vector<int> csr_row_offsets;
  std::vector<int> csr_col_idx;
  std::vector<float> csr_values;

  // Temporary buffer for GPU operations
  dvector<uint8_t> buffer;

  profiler::Domain profiler_domain_{"SparseMatrixTest"};
};

/** @brief Verifies GPU two-step COO-to-CSR conversion matches the CPU
 * reference. */
TEST_F(SparseMatrixTest, ConvertTripletToCSR) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("ConvertTripletToCSRTest");
  // Prepare GPU input in COO format
  size_t num_nonzeros = triplet_values.size();

  SparseJacobian sp_jacobian;
  sp_jacobian.structure.row_ids.resize(num_nonzeros);
  sp_jacobian.structure.col_ids.resize(num_nonzeros);
  sp_jacobian.values.resize(num_nonzeros);

  // Copy COO data to device memory
  sp_jacobian.structure.row_ids.CopyFromHost(triplet_row_idx.data(),
                                             triplet_row_idx.size());
  sp_jacobian.structure.col_ids.CopyFromHost(triplet_col_idx.data(),
                                             triplet_col_idx.size());
  sp_jacobian.values.CopyFromHost(triplet_values.data(), triplet_values.size());

  // Step 1: Convert structure to CSR and build mapping
  CudaStream stream;
  CSRSparseMatrix csr_matrix;
  dvector<int> mapping;
  cuSPARSEHandle cusparse_handle;
  auto handle = cusparse_handle.GetHandle(stream.GetStream());
  {
    auto range = this->profiler_domain_.CreateDomainRange(
        "ConvertTripletStructureToCSR");
    ConvertTripletStructureToCSR(stream.GetStream(), handle,
                                 sp_jacobian.structure, csr_matrix, mapping,
                                 this->buffer);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Step 2: Copy values using the mapping
  {
    auto range =
        this->profiler_domain_.CreateDomainRange("ConvertTripletToCSRValues");
    ConvertTripletToCSRValues(stream.GetStream(), sp_jacobian, mapping,
                              csr_matrix);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that GPU result matches CPU reference implementation
  hvector<float> csr_values_host(csr_matrix.values.size());
  csr_matrix.values.CopyToHost(csr_values_host.data(), csr_values_host.size());
  hvector<int> csr_row_offsets_host(csr_matrix.row_offsets.size());
  csr_matrix.row_offsets.CopyToHost(csr_row_offsets_host.data(),
                                    csr_row_offsets_host.size());
  hvector<int> csr_col_ids_host(csr_matrix.col_ids.size());
  csr_matrix.col_ids.CopyToHost(csr_col_ids_host.data(),
                                csr_col_ids_host.size());
  ASSERT_EQ(csr_values_host, csr_values);
  ASSERT_EQ(csr_row_offsets_host, csr_row_offsets);
  ASSERT_EQ(csr_col_ids_host, csr_col_idx);
}

/** @brief Verifies GPU diagonal extraction from a CSR matrix matches the CPU
 * reference. */
TEST_F(SparseMatrixTest, ExtractDiagonal) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ExtractDiagonal");
  // Prepare GPU matrix
  CSRSparseMatrix csr_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    csr_matrix);

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
  device_diagonal.CopyToHost(device_diagonal_host.data(),
                             device_diagonal_host.size());
  ASSERT_EQ(device_diagonal_host, diagonal);
}

/** @brief Verifies GPU AddScaledDiagonal (A + scale * diag(d)) matches the CPU
 * reference. */
TEST_F(SparseMatrixTest, AddScaledDiagonal) {
  auto test_range =
      this->profiler_domain_.CreateDomainRange("AddScaledDiagonalTest");
  // Prepare GPU matrix
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  // Generate random diagonal and scale factor
  float scale = 10;
  std::vector<float> diagonal;
  test_utils::GenerateRandomVector(matrix_size, diagonal);
  dvector<float> device_diagonal(diagonal);

  // Compute expected result using CPU reference
  AddScaledDiagonalCPU(csr_row_offsets, csr_col_idx, csr_values, scale,
                       diagonal);

  // Perform operation on GPU
  CSRSparseMatrix result_matrix;
  CudaStream stream;
  {
    auto range = this->profiler_domain_.CreateDomainRange("AddScaledDiagonal");
    AddScaledDiagonal(stream.GetStream(), scale, device_diagonal, input_matrix,
                      result_matrix);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  constexpr float tolerance = 1e-3;

  // Verify structure is preserved
  hvector<int> result_row_offsets_host(result_matrix.row_offsets.size());
  result_matrix.row_offsets.CopyToHost(result_row_offsets_host.data(),
                                       result_row_offsets_host.size());
  hvector<int> result_col_ids_host(result_matrix.col_ids.size());
  result_matrix.col_ids.CopyToHost(result_col_ids_host.data(),
                                   result_col_ids_host.size());
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
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

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
  result_matrix.col_ids.CopyToHost(result_col_ids_host.data(),
                                   result_col_ids_host.size());
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
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "ComputeWeightedSquaredStepFirstTest");
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
    auto range = this->profiler_domain_.CreateDomainRange(
        "ComputeWeightedSquaredStepFirst");
    result = ComputeWeightedSquaredStep(stream.GetStream(), dweights, dsteps,
                                        buffer);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify GPU result matches CPU computation
  ASSERT_NEAR(result, gt_value, 1e-3);
}

/** @brief Verifies GPU matrix-form weighted squared step norm: steps^T * A *
 * steps. */
TEST_F(SparseMatrixTest, ComputeWeightedSquaredStepSecond) {
  auto test_range = this->profiler_domain_.CreateDomainRange(
      "ComputeWeightedSquaredStepSecondTest");
  // Generate random step vector
  std::vector<float> steps;
  test_utils::GenerateRandomVector(matrix_size, steps);

  // Compute expected result on CPU: steps^T * A * steps
  // First compute temp = A * steps
  std::vector<float> temp;
  MultiplyCSRMatrixByVector(csr_row_offsets, csr_col_idx, csr_values, steps,
                            temp);
  // Then compute steps^T * temp
  float gt_value = 0;
  for (size_t i = 0; i < matrix_size; i++) {
    gt_value += steps[i] * temp[i];
  }
  gt_value /= matrix_size;

  // Prepare GPU matrix and vector
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  dvector<float> dsteps(steps);

  // Compute on GPU
  CudaStream stream;
  cuSPARSEHandle cusparse_handle;
  auto handle = cusparse_handle.GetHandle(stream.GetStream());
  float result;
  {
    auto range = this->profiler_domain_.CreateDomainRange(
        "ComputeWeightedSquaredStepSecond");
    result = ComputeWeightedSquaredStep(stream.GetStream(), handle,
                                        input_matrix, dsteps, buffer);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  result /= matrix_size;

  ASSERT_NEAR(result, gt_value, 1e-3);
}

/** @brief Verifies GPU RHS computation (y = -A^T * x) matches the CPU
 * reference. */
TEST_F(SparseMatrixTest, ComputeRHS) {
  auto test_range = this->profiler_domain_.CreateDomainRange("ComputeRHSTest");
  // Generate random input vector
  std::vector<float> steps;
  test_utils::GenerateRandomVector(matrix_size, steps);

  // Compute expected result using CPU reference
  std::vector<float> gt;
  ComputeRHSonCPU(csr_row_offsets, csr_col_idx, csr_values, steps, gt);

  // Prepare GPU matrix and vector
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  dvector<float> dsteps(steps);
  dvector<float> result;

  // Compute RHS on GPU
  CudaStream stream;
  cuSPARSEHandle cusparse_handle;
  auto handle = cusparse_handle.GetHandle(stream.GetStream());
  {
    auto range = this->profiler_domain_.CreateDomainRange("ComputeRHS");
    ComputeRHS(stream.GetStream(), handle, input_matrix, dsteps, result,
               buffer);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify GPU result matches CPU reference (within tolerance)
  hvector<float> hresult(result.size());
  result.CopyToHost(hresult.data(), hresult.size());
  ASSERT_EQ(hresult.size(), gt.size());
  for (size_t i = 0; i < gt.size(); i++) {
    ASSERT_NEAR(hresult[i], gt[i], 1e-3);
  }
}

/** @brief SHS scaling matches hand-derived 2x2 normal equations. */
TEST(SparseMatrixColumnScaling, SymmetricScaling2x2) {
  std::vector<int> csr_row_offsets = {0, 2, 4};
  std::vector<int> csr_col_idx = {0, 1, 0, 1};
  std::vector<float> csr_values = {4.f, 1.f, 1.f, 9.f};

  CSRSparseMatrix h;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    h);

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

} // namespace cunls
