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
 * @file fast_matrix_multiplier_test.cpp
 * @brief Unit tests for the custom GPU-based J^T * J computation.
 *
 * Generates random sparse CSR matrices, builds a matching Problem with
 * factor graph connectivity, computes the Hessian (A^T * A) on both CPU
 * and GPU using FastSparseMatrixMultiplier, and verifies the results match.
 */

#include "cunls/minimizer/fast_matrix_multiplier.h"

#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/profiler.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {

namespace {

// ============================================================================
// Mock factor batch for testing
// ============================================================================

class MockFactorBatch : public FactorBatch {
public:
  MockFactorBatch(size_t num_factors, size_t num_blocks_per_factor)
      : num_factors_(num_factors), sizes_(num_blocks_per_factor, 1) {}

  bool Evaluate(float *, float *, float const *const *,
                cudaStream_t) const override {
    return true;
  }
  size_t ResidualsSize() const override { return 1; }
  std::vector<size_t> StateBlockSizes() const override { return sizes_; }
  size_t NumFactors() const override { return num_factors_; }

private:
  size_t num_factors_;
  std::vector<size_t> sizes_;
};

// ============================================================================
// Test Problem builder
// ============================================================================

/**
 * Holds all resources needed to keep a test Problem alive.
 * The Problem references the state/factor batches via raw pointers,
 * so this struct ensures their lifetimes are properly managed.
 */
struct TestProblemData {
  dvector<float> state_data;
  dvector<int> const_ids;
  std::unique_ptr<VectorStateBatch<1>> state_batch;
  std::unique_ptr<MockFactorBatch> factor_batch;
  Problem problem;
};

/**
 * Builds a Problem whose factor graph structure matches the given CSR
 * Jacobian.  Each CSR row becomes one factor, each column becomes one
 * scalar state block (tangent_size = ambient_size = 1).  Rows with fewer
 * nonzeros than the maximum are padded with pointers to a dummy constant
 * state block that the structure-aware path skips.
 */
TestProblemData BuildProblemFromCSR(const std::vector<int> &csr_row_offsets,
                                    const std::vector<int> &csr_col_idx,
                                    int num_cols) {
  TestProblemData data;
  int num_rows = static_cast<int>(csr_row_offsets.size()) - 1;

  int max_nnz = 0;
  for (int r = 0; r < num_rows; r++) {
    max_nnz = std::max(max_nnz, csr_row_offsets[r + 1] - csr_row_offsets[r]);
  }

  int total_blocks = num_cols + 1;
  data.state_data.resize(total_blocks);

  std::vector<int> h_const_ids = {num_cols};
  data.const_ids = dvector<int>(h_const_ids);

  data.state_batch = std::make_unique<VectorStateBatch<1>>(
      data.state_data.data(), total_blocks, data.const_ids.data(),
      data.const_ids.size());

  data.factor_batch = std::make_unique<MockFactorBatch>(num_rows, max_nnz);

  float *dummy = data.state_batch->StateBlockDevicePtr(num_cols);
  std::vector<float *> state_ptrs;
  state_ptrs.reserve(static_cast<size_t>(num_rows) * max_nnz);
  for (int r = 0; r < num_rows; r++) {
    int start = csr_row_offsets[r];
    int end = csr_row_offsets[r + 1];
    for (int j = start; j < end; j++) {
      state_ptrs.push_back(
          data.state_batch->StateBlockDevicePtr(csr_col_idx[j]));
    }
    for (int j = end - start; j < max_nnz; j++) {
      state_ptrs.push_back(dummy);
    }
  }

  data.problem.AddStateBatch(data.state_batch.get());
  data.problem.AddFactorBatch(data.factor_batch.get(), state_ptrs);

  return data;
}

// ============================================================================
// CSR generation and CPU reference
// ============================================================================

void GenerateRandomCSRMatrix(std::mt19937 &rng, int rows, int cols,
                             float sparsity, std::vector<float> &csr_values,
                             std::vector<int> &csr_col_idx,
                             std::vector<int> &csr_row_offsets) {
  std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
  std::uniform_int_distribution<int> col_dist(0, cols - 1);

  csr_row_offsets.clear();
  csr_col_idx.clear();
  csr_values.clear();
  csr_row_offsets.push_back(0);

  for (int i = 0; i < rows; ++i) {
    int nnz_in_row = std::max(1, static_cast<int>(cols * sparsity));
    std::unordered_set<int> used_cols;
    used_cols.insert(col_dist(rng));

    while (static_cast<int>(used_cols.size()) < nnz_in_row) {
      used_cols.insert(col_dist(rng));
    }

    std::vector<int> sorted_cols(used_cols.begin(), used_cols.end());
    std::sort(sorted_cols.begin(), sorted_cols.end());

    for (int col : sorted_cols) {
      csr_col_idx.push_back(col);
      csr_values.push_back(val_dist(rng));
    }
    csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
  }
}

void ComputeHessianCPU(const std::vector<int> &row_ptr,
                       const std::vector<int> &col_idx,
                       const std::vector<float> &values, int num_cols,
                       std::vector<int> &AtA_row_ptr,
                       std::vector<int> &AtA_col_idx,
                       std::vector<float> &AtA_values) {
  int rows = row_ptr.size() - 1;

  std::vector<std::map<int, float>> row_acc(num_cols);

  for (int i = 0; i < rows; ++i) {
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    for (int j = start; j < end; ++j) {
      int col_j = col_idx[j];
      float val_j = values[j];

      for (int k = start; k < end; ++k) {
        int col_k = col_idx[k];
        float val_k = values[k];
        row_acc[col_j][col_k] += val_j * val_k;
      }
    }
  }

  AtA_row_ptr.clear();
  AtA_col_idx.clear();
  AtA_values.clear();
  AtA_row_ptr.push_back(0);

  for (int i = 0; i < num_cols; ++i) {
    for (const auto &[col, val] : row_acc[i]) {
      if (val != 0.0f) {
        AtA_col_idx.push_back(col);
        AtA_values.push_back(val);
      }
    }
    AtA_row_ptr.push_back(static_cast<int>(AtA_col_idx.size()));
  }
}

void VerifyHessian(const CSRSparseMatrix &hessian,
                   const std::vector<int> &ref_row_ptr,
                   const std::vector<int> &ref_col_idx,
                   const std::vector<float> &ref_values) {
  std::vector<int> hessian_row_offsets(hessian.row_offsets.size());
  hessian.row_offsets.CopyToHost(hessian_row_offsets.data(),
                                 hessian_row_offsets.size());
  ASSERT_EQ(hessian_row_offsets, ref_row_ptr);

  std::vector<int> hessian_col_ids(hessian.col_ids.size());
  hessian.col_ids.CopyToHost(hessian_col_ids.data(), hessian_col_ids.size());
  ASSERT_EQ(hessian_col_ids, ref_col_idx);

  std::vector<float> hessian_values(hessian.values.size());
  hessian.values.CopyToHost(hessian_values.data(), hessian_values.size());
  ASSERT_EQ(hessian_values.size(), ref_values.size());
  for (size_t i = 0; i < ref_values.size(); i++) {
    ASSERT_NEAR(hessian_values[i], ref_values[i], 1e-3)
        << "Mismatch at index " << i;
  }
}

} // namespace

TEST(FastSparseMatrixMultiplierTest, SquareMatrix) {
  profiler::ScopedRange range("SymmetricSquare_SquareMatrix");

  std::mt19937 gen(42);
  constexpr int size = 500;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, size, size, 0.05f, csr_values, csr_col_idx,
                          csr_row_offsets);

  std::vector<int> ref_row_ptr, ref_col_idx;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, size, ref_row_ptr,
                    ref_col_idx, ref_values);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, size);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, TallMatrix) {
  profiler::ScopedRange range("SymmetricSquare_TallMatrix");

  std::mt19937 gen(123);
  constexpr int rows = 2000;
  constexpr int cols = 200;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, rows, cols, 0.05f, csr_values, csr_col_idx,
                          csr_row_offsets);

  std::vector<int> ref_row_ptr, ref_col_idx;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, cols, ref_row_ptr,
                    ref_col_idx, ref_values);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, cols);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, SparseRowJacobian) {
  profiler::ScopedRange range("SymmetricSquare_SparseRow");

  std::mt19937 gen(7);
  constexpr int rows = 5000;
  constexpr int cols = 500;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, rows, cols, 0.02f, csr_values, csr_col_idx,
                          csr_row_offsets);

  std::vector<int> ref_row_ptr, ref_col_idx;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, cols, ref_row_ptr,
                    ref_col_idx, ref_values);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, cols);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, ValueReuseAfterInitialize) {
  profiler::ScopedRange range("SymmetricSquare_ValueReuse");

  std::mt19937 gen(99);
  constexpr int rows = 1000;
  constexpr int cols = 100;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, rows, cols, 0.1f, csr_values, csr_col_idx,
                          csr_row_offsets);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, cols);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Generate new values (same structure) and recompute without re-initializing.
  std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
  for (auto &v : csr_values)
    v = val_dist(gen);
  input_matrix.values = dvector<float>(csr_values);

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  std::vector<int> ref_row_ptr, ref_col_idx;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, cols, ref_row_ptr,
                    ref_col_idx, ref_values);

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, LargeSquareMatrix) {
  profiler::ScopedRange range("SymmetricSquare_LargeSquare");

  std::mt19937 gen(0);
  constexpr int size = 1000;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, size, size, 0.02f, csr_values, csr_col_idx,
                          csr_row_offsets);

  std::vector<int> ref_row_ptr, ref_col_idx;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, size, ref_row_ptr,
                    ref_col_idx, ref_values);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, size);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, SingleColumnPerRow) {
  profiler::ScopedRange range("SymmetricSquare_SingleColumn");

  constexpr int rows = 100;
  constexpr int cols = 50;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  csr_row_offsets.push_back(0);

  std::mt19937 gen(55);
  std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
  std::uniform_int_distribution<int> col_dist(0, cols - 1);

  for (int i = 0; i < rows; ++i) {
    csr_col_idx.push_back(col_dist(gen));
    csr_values.push_back(val_dist(gen));
    csr_row_offsets.push_back(static_cast<int>(csr_col_idx.size()));
  }

  std::vector<int> ref_row_ptr, ref_col_idx_ref;
  std::vector<float> ref_values;
  ComputeHessianCPU(csr_row_offsets, csr_col_idx, csr_values, cols, ref_row_ptr,
                    ref_col_idx_ref, ref_values);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, cols);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  VerifyHessian(hessian, ref_row_ptr, ref_col_idx_ref, ref_values);
}

TEST(FastSparseMatrixMultiplierTest, PerformanceBenchmark) {
  profiler::ScopedRange range("SymmetricSquare_Performance");

  std::mt19937 gen(42);
  constexpr int rows = 20000;
  constexpr int cols = 2000;
  constexpr float sparsity = 0.005f;

  std::vector<float> csr_values;
  std::vector<int> csr_col_idx, csr_row_offsets;
  GenerateRandomCSRMatrix(gen, rows, cols, sparsity, csr_values, csr_col_idx,
                          csr_row_offsets);

  CSRSparseMatrix input_matrix, hessian;
  test_utils::CreateCSRSparseMatrix(csr_row_offsets, csr_col_idx, csr_values,
                                    input_matrix);

  auto test_data = BuildProblemFromCSR(csr_row_offsets, csr_col_idx, cols);
  FastSparseMatrixMultiplier smm;
  CudaStream stream;

  smm.Initialize(stream.GetStream(), test_data.problem, input_matrix, hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Warmup
  smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem, input_matrix,
                           hessian);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  cudaEvent_t start, stop;
  THROW_ON_CUDA_ERROR(cudaEventCreate(&start));
  THROW_ON_CUDA_ERROR(cudaEventCreate(&stop));

  constexpr int kNumIterations = 50;
  THROW_ON_CUDA_ERROR(cudaEventRecord(start, stream.GetStream()));
  for (int i = 0; i < kNumIterations; i++) {
    smm.ComputeSquaredMatrix(stream.GetStream(), test_data.problem,
                             input_matrix, hessian);
  }
  THROW_ON_CUDA_ERROR(cudaEventRecord(stop, stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop));

  float ms;
  THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

  std::cout << "[Performance] ComputeSquaredMatrix: " << ms / kNumIterations
            << " ms/iteration (" << rows << "x" << cols
            << ", input_nnz=" << csr_values.size()
            << ", output_nnz=" << hessian.NumNonZeros() << ")" << std::endl;

  THROW_ON_CUDA_ERROR(cudaEventDestroy(start));
  THROW_ON_CUDA_ERROR(cudaEventDestroy(stop));
}

} // namespace cunls
