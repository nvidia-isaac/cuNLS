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
 * @file sparse_linear_solver_test.cpp
 * @brief Unit tests for the cuDSS-based sparse linear solver.
 *
 * Generates a random symmetric sparse matrix A and vector b, solves Ax = b
 * on the GPU, then verifies correctness by checking A*x against b.
 */

#include "cunls/linear_solver/sparse_linear_solver.h"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"
#include "tests/utils.h"

namespace cunls {

namespace {

/**
 * @brief Generates a random sparse symmetric matrix in CSR format.
 *
 * Creates a symmetric sparse matrix with random values. The matrix has
 * diagonal elements and randomly placed off-diagonal elements (controlled
 * by sparsity parameter). Symmetry is enforced by setting A[i][j] = A[j][i].
 *
 * @param rng Random number generator for reproducibility.
 * @param rows Number of rows (and columns) in the matrix.
 * @param csr_values Output vector of non-zero values.
 * @param csr_col_idx Output vector of column indices.
 * @param csr_row_offsets Output vector of row offsets (size = rows + 1).
 */
void GenerateRandomSymmetricCSRMatrix(std::mt19937 &rng, int rows,
                                      std::vector<float> &csr_values,
                                      std::vector<int> &csr_col_idx,
                                      std::vector<int> &csr_row_offsets) {
  std::uniform_real_distribution<float> val_dist(-1, 1);
  std::uniform_real_distribution<float> prob_dist(0.0, 1.0);

  // Use a map-of-maps to store symmetric COO entries before converting to CSR
  std::unordered_map<int, std::unordered_map<int, float>> coo;
  float sparsity = 1e-3; // Probability of non-zero off-diagonal elements

  // Generate symmetric matrix entries
  for (int i = 0; i < rows; ++i) {
    // Always include diagonal elements for better conditioning
    coo[i][i] = 100;

    // Generate off-diagonal elements and ensure symmetry
    for (int j = i + 1; j < rows; ++j) {
      if (prob_dist(rng) > sparsity) {
        continue; // Skip this entry (sparse matrix)
      }
      float val = val_dist(rng);
      coo[i][j] = val; // Upper triangular
      coo[j][i] = val; // Lower triangular (ensure symmetry)
    }
  }

  // Convert COO format to CSR format
  csr_row_offsets.push_back(0);

  for (int i = 0; i < rows; ++i) {
    // Collect all entries in the current row
    std::vector<std::pair<int, float>> row_entries;
    for (const auto &[j, val] : coo[i]) {
      row_entries.emplace_back(j, val);
    }
    // Sort by column index (required for CSR format)
    std::sort(row_entries.begin(), row_entries.end());

    // Add sorted entries to CSR arrays
    for (const auto &[j, val] : row_entries) {
      csr_col_idx.push_back(j);
      csr_values.push_back(val);
    }
    csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
  }
}

/**
 * @brief Multiplies a symmetric CSR matrix by a vector: y = A * x.
 *
 * Performs sparse matrix-vector multiplication on the CPU for verification
 * purposes. Computes y = A * x where A is represented in CSR format.
 *
 * @param row_ptr Row offset array (size = num_rows + 1).
 * @param col_ind Column index array for non-zero elements.
 * @param values Value array for non-zero elements.
 * @param x Input vector to multiply.
 * @param y Output vector to store the result.
 */
void MultiplySymmetricCSRMatrixByVector(const std::vector<int> &row_ptr,
                                        const std::vector<int> &col_ind,
                                        const std::vector<float> &values,
                                        const std::vector<float> &x,
                                        std::vector<float> &y) {
  size_t size = row_ptr.size() - 1;
  y.assign(size, 0.0);

  // Standard CSR matrix-vector multiplication: y[i] = sum(A[i,j] * x[j])
  for (int i = 0; i < size; ++i) {
    for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
      int j = col_ind[idx];
      auto val = values[idx];

      y[i] += val * x[j];
    }
  }
}

} // namespace

/**
 * @brief Tests the sparse linear solver with a random symmetric matrix.
 *
 * This test generates a random symmetric sparse matrix A and a random
 * right-hand side vector b, solves the system Ax = b using cuDSSLinearSolver,
 * and verifies that the solution is correct by computing A * x and comparing
 * it with b. The test uses a tolerance of 1e-1 to account for numerical errors.
 */
TEST(SparseLinearSolverTest, Solve) {
  profiler::ScopedRange range("SolveTest");
  // Use fixed seed for reproducibility
  unsigned int fixed_seed = 0;
  std::mt19937 gen(fixed_seed);

  // Test with a moderately large sparse matrix
  constexpr int matrix_size = 10000;

  // Step 1: Generate a random symmetric sparse matrix in CSR format
  std::vector<float> csr_values;
  std::vector<int> csr_col_idx;
  std::vector<int> csr_row_offsets;

  GenerateRandomSymmetricCSRMatrix(gen, matrix_size, csr_values, csr_col_idx,
                                   csr_row_offsets);

  // Step 2: Convert host matrix to device-compatible format
  CSRSparseMatrix input_matrix;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  // Step 3: Generate random right-hand side vector b
  std::vector<float> rhs_cpu;
  test_utils::GenerateRandomVector(matrix_size, rhs_cpu);

  // Step 4: Copy RHS to device and allocate result vector
  dvector<float> rhs(rhs_cpu);
  dvector<float> result(matrix_size);

  // Step 5: Solve the linear system Ax = b on GPU
  CudaStream stream;
  cuDSSLinearSolverOptions cudss_solver_options = {
      cuDSSLinearSolverMode::SlowInitFastSolve,
      1,
      "",
  };
  {
    cuDSSLinearSolver solver(cudss_solver_options);
    {
      profiler::ScopedRange range("Warm up");
      solver.Initialize(stream.GetStream(), Problem(), input_matrix, rhs,
                        result);
      solver.Solve(stream.GetStream(), input_matrix, rhs, result);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    }

    {
      profiler::ScopedRange range("Solve");
      solver.Solve(stream.GetStream(), input_matrix, rhs, result);
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    }
  }

  // Step 6: Copy solution back to host
  std::vector<float> cpu_result(matrix_size);
  result.CopyToHost(cpu_result.data(), cpu_result.size());

  // Step 7: Verify solution by computing A * x and comparing with b
  std::vector<float> predicted_rhs;
  MultiplySymmetricCSRMatrixByVector(csr_row_offsets, csr_col_idx, csr_values,
                                     cpu_result, predicted_rhs);

  // Step 8: Check that A * x equals b within tolerance
  ASSERT_EQ(predicted_rhs.size(), rhs_cpu.size());
  float squared_error = 0;
  for (size_t i = 0; i < matrix_size; i++) {
    squared_error += powf(predicted_rhs[i] - rhs_cpu[i], 2);
  }

  ASSERT_NEAR(squared_error / matrix_size, 0, 1e-1);
}

namespace {

// Generates a block-SPD matrix: dense 6x6 diagonal blocks plus a sparse pattern
// of symmetric 6x6 off-diagonal couplings.  Mirrors the structure of a
// Hessian from an SE3 pose graph and is the natural fit for the block-Jacobi
// preconditioner.
void GenerateBlockSPDMatrix(std::mt19937 &gen, int num_blocks, int block_size,
                            float off_diag_strength,
                            std::vector<float> &csr_values,
                            std::vector<int> &csr_col_idx,
                            std::vector<int> &csr_row_offsets) {
  int n = num_blocks * block_size;
  std::uniform_real_distribution<float> off_dist(-off_diag_strength,
                                                 off_diag_strength);
  std::uniform_real_distribution<float> prob_dist(0.f, 1.f);
  // Dense per-row map for ordered CSR assembly.
  std::vector<std::map<int, float>> rows(n);

  // Strongly diagonal-dominant diagonal blocks for SPD guarantee.
  for (int b = 0; b < num_blocks; ++b) {
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        int gi = b * block_size + i;
        int gj = b * block_size + j;
        if (i == j) {
          rows[gi][gj] = 50.f;
        } else {
          float v = off_dist(gen) * 0.1f;
          rows[gi][gj] = (rows[gi].count(gj) ? rows[gi][gj] : 0.f) + v;
        }
      }
    }
  }
  // Add symmetric block off-diagonal couplings to a few neighbour blocks.
  for (int b = 0; b < num_blocks; ++b) {
    for (int nb = b + 1; nb < num_blocks; ++nb) {
      if (prob_dist(gen) > 5.f / num_blocks) {
        continue;
      }
      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          float v = off_dist(gen);
          int gi = b * block_size + i;
          int gj = nb * block_size + j;
          rows[gi][gj] = (rows[gi].count(gj) ? rows[gi][gj] : 0.f) + v;
          rows[gj][gi] = (rows[gj].count(gi) ? rows[gj][gi] : 0.f) + v;
        }
      }
    }
  }

  csr_row_offsets.clear();
  csr_col_idx.clear();
  csr_values.clear();
  csr_row_offsets.push_back(0);
  for (int i = 0; i < n; ++i) {
    for (const auto &[j, v] : rows[i]) {
      csr_col_idx.push_back(j);
      csr_values.push_back(v);
    }
    csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
  }
}

} // namespace

TEST(SparseLinearSolverTest, BlockSparsePCGSolve) {
  std::mt19937 gen(7);
  constexpr int num_blocks = 200;
  constexpr int block_size = 6;
  constexpr int matrix_size = num_blocks * block_size;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx;
  std::vector<int> csr_row_offsets;
  GenerateBlockSPDMatrix(gen, num_blocks, block_size, 1.0f, csr_values,
                         csr_col_idx, csr_row_offsets);

  CSRSparseMatrix mat;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    mat);

  std::vector<float> rhs_cpu;
  test_utils::GenerateRandomVector(matrix_size, rhs_cpu);
  dvector<float> rhs(rhs_cpu);
  dvector<float> result(matrix_size);

  CudaStream stream;
  BlockSparsePCGOptions opts;
  opts.block_size = block_size;
  opts.relative_tolerance = 1e-5f;
  opts.max_iterations = 500;
  BlockSparsePCGSolver solver(opts);
  ASSERT_TRUE(
      solver.Initialize(stream.GetStream(), Problem(), mat, rhs, result));
  ASSERT_TRUE(solver.Solve(stream.GetStream(), mat, rhs, result));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<float> x(matrix_size);
  result.CopyToHost(x.data(), matrix_size);

  std::vector<float> Ax;
  MultiplySymmetricCSRMatrixByVector(csr_row_offsets, csr_col_idx, csr_values,
                                     x, Ax);
  float sq_err = 0.f;
  float b_sq = 0.f;
  for (int i = 0; i < matrix_size; ++i) {
    float d = Ax[i] - rhs_cpu[i];
    sq_err += d * d;
    b_sq += rhs_cpu[i] * rhs_cpu[i];
  }
  float rel_err = std::sqrt(sq_err / std::max(b_sq, 1e-30f));
  EXPECT_LT(rel_err, 1e-3f) << "PCG residual too large (relative): " << rel_err;
  EXPECT_GT(solver.LastIterations(), 0);
  EXPECT_LE(solver.LastIterations(), opts.max_iterations);
}

} // namespace cunls
