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
 * @file sparse_matrix_multiplication_test.cpp
 * @brief Unit tests for GPU-based sparse matrix multiplication (A^T * A).
 *
 * Generates a random sparse CSR matrix, computes its Hessian (A^T * A) on
 * both CPU and GPU, and verifies that the GPU result matches the CPU reference.
 */

#include <gtest/gtest.h>

#include <random>
#include <unordered_set>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/sparse_matrix.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/**
 * @brief Generates a random square sparse matrix in CSR format.
 *
 * Each row has ~10% non-zero entries with a guaranteed diagonal element.
 *
 * @param rng Random number generator.
 * @param rows Number of rows (and columns) in the square matrix.
 * @param csr_values Output non-zero values.
 * @param csr_col_idx Output column indices.
 * @param csr_row_offsets Output row offset array.
 */
void GenerateRandomCSRMatrix(std::mt19937& rng, int rows,
                             std::vector<float>& csr_values,
                             std::vector<int>& csr_col_idx,
                             std::vector<int>& csr_row_offsets) {
  const int cols = rows;
  // Value distribution: random floats between 0.1 and 1.0
  std::uniform_real_distribution<float> val_dist(0.1, 1.0);
  // Column distribution: uniformly random column indices
  std::uniform_int_distribution<int> col_dist(0, cols - 1);

  // Sparsity level: approximately 10% of entries will be non-zero
  constexpr float sparsity = 0.1;

  // Clear output vectors to start fresh
  csr_row_offsets.clear();
  csr_col_idx.clear();
  csr_values.clear();

  // CSR format: row_offsets[i] indicates where row i starts in values/col_idx
  // arrays
  csr_row_offsets.push_back(0);

  for (int i = 0; i < rows; ++i) {
    // Calculate number of non-zero entries for this row
    int nnz_in_row = 1 + static_cast<int>(cols * sparsity);
    // Start with diagonal element to ensure matrix has full rank
    std::unordered_set<int> used_cols = {i};

    // Randomly select additional column indices (avoiding duplicates)
    while (used_cols.size() < nnz_in_row) {
      int col = col_dist(rng);
      used_cols.insert(col);
    }

    // Sort columns to maintain CSR format
    std::vector<int> sorted_cols(used_cols.begin(), used_cols.end());
    std::sort(sorted_cols.begin(), sorted_cols.end());

    // Add all non-zero entries for this row
    for (int col : sorted_cols) {
      csr_col_idx.push_back(col);
      csr_values.push_back(val_dist(rng));
    }

    // Record where the next row starts
    csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
  }
}

/**
 * @brief Computes A^T * A on the CPU as a reference implementation.
 *
 * For each row of A, accumulates the outer product of that row with itself.
 *
 * @param row_ptr CSR row offsets of A.
 * @param col_idx CSR column indices of A.
 * @param values CSR values of A.
 * @param AtA_row_ptr Output CSR row offsets of A^T * A.
 * @param AtA_col_idx Output CSR column indices of A^T * A.
 * @param AtA_values Output CSR values of A^T * A.
 */
void ComputeHessianCPU(const std::vector<int>& row_ptr,
                       const std::vector<int>& col_idx,
                       const std::vector<float>& values,
                       std::vector<int>& AtA_row_ptr,
                       std::vector<int>& AtA_col_idx,
                       std::vector<float>& AtA_values) {
  int rows = row_ptr.size() - 1;
  int cols = rows;  // Square matrix assumption

  // Accumulate entries row-wise for A^T * A
  // row_acc[i][j] will store the (i,j) entry of A^T * A
  std::vector<std::map<int, float>> row_acc(cols);  // A^T * A has 'cols' rows

  // For each row of A
  for (int i = 0; i < rows; ++i) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    // Compute outer product: row[i] * row[i]^T
    // This contributes to multiple entries of A^T * A
    for (int j = start; j < end; ++j) {
      int col_j = col_idx[j];   // Column index in A
      float val_j = values[j];  // Value A[i, col_j]

      for (int k = start; k < end; ++k) {
        int col_k = col_idx[k];   // Another column index in A
        float val_k = values[k];  // Value A[i, col_k]

        // Add A[i, col_j] * A[i, col_k] to (A^T * A)[col_j, col_k]
        row_acc[col_j][col_k] += val_j * val_k;
      }
    }
  }

  // Convert accumulated sparse rows to CSR format
  AtA_row_ptr.clear();
  AtA_col_idx.clear();
  AtA_values.clear();

  AtA_row_ptr.push_back(0);
  for (int i = 0; i < cols; ++i) {
    // Add all non-zero entries in row i of A^T * A
    for (const auto& [col, val] : row_acc[i]) {
      if (val != 0.0) {
        AtA_col_idx.push_back(col);
        AtA_values.push_back(val);
      }
    }
    // Record where the next row starts
    AtA_row_ptr.push_back(static_cast<int>(AtA_col_idx.size()));
  }
}

}  // namespace

/** @brief Verifies GPU A^T*A computation matches the CPU reference implementation. */
TEST(SparseMatrixMultiplicationTest, ComputeHessian) {
  profiler::ScopedRange range("ComputeHessianTest");
  // Use fixed seed for reproducibility
  unsigned int fixed_seed = 0;
  std::mt19937 gen(fixed_seed);

  // Test with a moderately large sparse matrix
  constexpr int matrix_size = 1000;

  // Generate random sparse matrix in CSR format
  std::vector<float> csr_values;
  std::vector<int> csr_col_idx;
  std::vector<int> csr_row_offsets;

  GenerateRandomCSRMatrix(gen, matrix_size, csr_values, csr_col_idx,
                          csr_row_offsets);

  // Compute reference result on CPU
  std::vector<int> AtA_row_ptr;
  std::vector<int> AtA_col_idx;
  std::vector<float> AtA_values;

  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, AtA_row_ptr,
                    AtA_col_idx, AtA_values);

  // Convert input matrix to device-compatible format
  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  // Compute A^T * A on GPU
  SparseMatrixMultiplication smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  {
    // Warm up
    smm.ComputeSquaredMatrix(stream.GetStream(), input_matrix, hessian);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  {
    profiler::ScopedRange range("ComputeSquaredMatrix");
    smm.ComputeSquaredMatrix(stream.GetStream(), input_matrix, hessian);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  // Verify that the sparsity structure matches exactly
  std::vector<int> hessian_row_offsets(hessian.row_offsets.size());
  hessian.row_offsets.CopyToHost(hessian_row_offsets.data(),
                                  hessian_row_offsets.size());
  ASSERT_EQ(hessian_row_offsets, AtA_row_ptr);

  std::vector<int> hessian_col_ids(hessian.col_ids.size());
  hessian.col_ids.CopyToHost(hessian_col_ids.data(), hessian_col_ids.size());
  ASSERT_EQ(hessian_col_ids, AtA_col_idx);

  // Verify that all non-zero values match within tolerance
  std::vector<float> hessian_values(hessian.values.size());
  hessian.values.CopyToHost(hessian_values.data(), hessian_values.size());
  ASSERT_EQ(hessian_values.size(), AtA_values.size());
  for (size_t i = 0; i < AtA_values.size(); i++) {
    ASSERT_NEAR(hessian_values[i], AtA_values[i], 1e-3);
  }
}

}  // namespace cunls
