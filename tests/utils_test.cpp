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
 * @file utils_test.cpp
 * @brief Unit tests for utility functions (CSR matrix and vector file I/O).
 *
 * Tests DumpCSRSparseMatrixToFile and DumpVectorToFile for correct binary
 * serialization, append behavior, and empty-input handling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "cunls/common/types.h"
#include "cunls/common/utils.h"

namespace cunls {

namespace {

/**
 * @brief Creates a simple 3x3 CSR test matrix on the device.
 *
 * @param matrix Output CSRSparseMatrix populated with a known 3x3 matrix.
 */
void CreateTestCSRMatrix(CSRSparseMatrix& matrix) {
  // Create a simple 3x3 matrix:
  // [1.0  2.0  0.0]
  // [0.0  3.0  4.0]
  // [5.0  0.0  6.0]

  std::vector<int> row_offsets = {0, 2, 4, 6};
  std::vector<int> col_ids = {0, 1, 1, 2, 0, 2};
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  matrix.row_offsets.resize(row_offsets.size());
  matrix.col_ids.resize(col_ids.size());
  matrix.values.resize(values.size());


  matrix.row_offsets.CopyFromHost(row_offsets.data(), row_offsets.size());
  matrix.col_ids.CopyFromHost(col_ids.data(), col_ids.size());
  matrix.values.CopyFromHost(values.data(), values.size());
}

/**
 * @brief Reads a CSR sparse matrix from a binary file for verification.
 *
 * @param filename Path to the binary file.
 * @param row_offsets Output row offset array.
 * @param col_ids Output column index array.
 * @param values Output value array.
 * @param num_rows Output number of rows.
 * @param num_cols Output number of columns.
 * @param num_nonzeros Output number of non-zero elements.
 */
void ReadCSRSparseMatrixFromFile(const std::string& filename,
                                 std::vector<uint32_t>& row_offsets,
                                 std::vector<uint32_t>& col_ids,
                                 std::vector<float>& values, uint32_t& num_rows,
                                 uint32_t& num_cols, uint32_t& num_nonzeros) {
  std::ifstream file(filename, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "Failed to open file: " << filename;

  file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_cols), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_nonzeros), sizeof(uint32_t));

  row_offsets.resize(num_rows + 1);
  col_ids.resize(num_nonzeros);
  values.resize(num_nonzeros);

  file.read(reinterpret_cast<char*>(row_offsets.data()),
            (num_rows + 1) * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(col_ids.data()),
            num_nonzeros * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(values.data()),
            num_nonzeros * sizeof(float));

  ASSERT_TRUE(file.good()) << "Error reading file: " << filename;
}

/**
 * @brief Reads a vector from a binary file for verification.
 *
 * @param filename Path to the binary file.
 * @param values Output value vector.
 * @param size Output vector size.
 */
void ReadVectorFromFile(const std::string& filename, std::vector<float>& values,
                        uint32_t& size) {
  std::ifstream file(filename, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "Failed to open file: " << filename;

  file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
  values.resize(size);
  file.read(reinterpret_cast<char*>(values.data()), size * sizeof(float));

  ASSERT_TRUE(file.good()) << "Error reading file: " << filename;
}

}  // namespace

/** @brief Test fixture for utility function tests, managing temp file lifecycle. */
class UtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create temporary file names
    matrix_filename_ = "/tmp/test_matrix.bin";
    vector_filename_ = "/tmp/test_vector.bin";

    // Remove files if they exist
    std::remove(matrix_filename_.c_str());
    std::remove(vector_filename_.c_str());
  }

  void TearDown() override {
    // Clean up test files
    std::remove(matrix_filename_.c_str());
    std::remove(vector_filename_.c_str());
  }

  std::string matrix_filename_;
  std::string vector_filename_;
};

/** @brief Tests that DumpCSRSparseMatrixToFile writes the correct binary format. */
TEST_F(UtilsTest, DumpCSRSparseMatrixToFile) {
  // Create a test CSR matrix
  CSRSparseMatrix matrix;
  CreateTestCSRMatrix(matrix);

  // Dump matrix to file
  DumpCSRSparseMatrixToFile(matrix_filename_, matrix);

  // Read the file back and verify contents
  std::vector<uint32_t> row_offsets;
  std::vector<uint32_t> col_ids;
  std::vector<float> values;
  uint32_t num_rows, num_cols, num_nonzeros;

  ReadCSRSparseMatrixFromFile(matrix_filename_, row_offsets, col_ids, values,
                              num_rows, num_cols, num_nonzeros);

  // Verify dimensions
  EXPECT_EQ(num_rows, 3u);
  EXPECT_EQ(num_cols, 3u);
  EXPECT_EQ(num_nonzeros, 6u);

  // Verify row offsets
  EXPECT_EQ(row_offsets.size(), 4u);
  EXPECT_EQ(row_offsets[0], 0u);
  EXPECT_EQ(row_offsets[1], 2u);
  EXPECT_EQ(row_offsets[2], 4u);
  EXPECT_EQ(row_offsets[3], 6u);

  // Verify column IDs
  EXPECT_EQ(col_ids.size(), 6u);
  EXPECT_EQ(col_ids[0], 0u);
  EXPECT_EQ(col_ids[1], 1u);
  EXPECT_EQ(col_ids[2], 1u);
  EXPECT_EQ(col_ids[3], 2u);
  EXPECT_EQ(col_ids[4], 0u);
  EXPECT_EQ(col_ids[5], 2u);

  // Verify values
  EXPECT_EQ(values.size(), 6u);
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 2.0f);
  EXPECT_FLOAT_EQ(values[2], 3.0f);
  EXPECT_FLOAT_EQ(values[3], 4.0f);
  EXPECT_FLOAT_EQ(values[4], 5.0f);
  EXPECT_FLOAT_EQ(values[5], 6.0f);
}

/** @brief Tests that DumpCSRSparseMatrixToFile appends to an existing file. */
TEST_F(UtilsTest, DumpCSRSparseMatrixToFileAppends) {
  // Create and dump first matrix
  CSRSparseMatrix matrix1;
  CreateTestCSRMatrix(matrix1);
  DumpCSRSparseMatrixToFile(matrix_filename_, matrix1);

  // Create a second matrix (2x2 identity)
  CSRSparseMatrix matrix2;
  std::vector<int> row_offsets2 = {0, 1, 2};
  std::vector<int> col_ids2 = {0, 1};
  std::vector<float> values2 = {1.0f, 1.0f};

  matrix2.row_offsets.resize(row_offsets2.size());
  matrix2.col_ids.resize(col_ids2.size());
  matrix2.values.resize(values2.size());

  matrix2.row_offsets.CopyFromHost(row_offsets2.data(), row_offsets2.size());
  matrix2.col_ids.CopyFromHost(col_ids2.data(), col_ids2.size());
  matrix2.values.CopyFromHost(values2.data(), values2.size());

  // Dump second matrix (should append)
  DumpCSRSparseMatrixToFile(matrix_filename_, matrix2);

  // Read both matrices from file
  std::ifstream file(matrix_filename_, std::ios::binary);
  ASSERT_TRUE(file.is_open());

  // Read first matrix
  uint32_t num_rows1, num_cols1, num_nonzeros1;
  file.read(reinterpret_cast<char*>(&num_rows1), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_cols1), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_nonzeros1), sizeof(uint32_t));

  std::vector<uint32_t> row_offsets1(num_rows1 + 1);
  std::vector<uint32_t> col_ids1(num_nonzeros1);
  std::vector<float> values1(num_nonzeros1);
  file.read(reinterpret_cast<char*>(row_offsets1.data()),
            (num_rows1 + 1) * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(col_ids1.data()),
            num_nonzeros1 * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(values1.data()),
            num_nonzeros1 * sizeof(float));

  // Verify first matrix
  EXPECT_EQ(num_rows1, 3u);
  EXPECT_EQ(num_cols1, 3u);
  EXPECT_EQ(num_nonzeros1, 6u);

  // Read second matrix
  uint32_t num_rows2, num_cols2, num_nonzeros2;
  file.read(reinterpret_cast<char*>(&num_rows2), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_cols2), sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(&num_nonzeros2), sizeof(uint32_t));

  std::vector<uint32_t> row_offsets2_read(num_rows2 + 1);
  std::vector<uint32_t> col_ids2_read(num_nonzeros2);
  std::vector<float> values2_read(num_nonzeros2);
  file.read(reinterpret_cast<char*>(row_offsets2_read.data()),
            (num_rows2 + 1) * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(col_ids2_read.data()),
            num_nonzeros2 * sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(values2_read.data()),
            num_nonzeros2 * sizeof(float));

  // Verify second matrix
  EXPECT_EQ(num_rows2, 2u);
  EXPECT_EQ(num_cols2, 2u);
  EXPECT_EQ(num_nonzeros2, 2u);
  EXPECT_EQ(row_offsets2_read[0], 0u);
  EXPECT_EQ(row_offsets2_read[1], 1u);
  EXPECT_EQ(row_offsets2_read[2], 2u);
  EXPECT_EQ(col_ids2_read[0], 0u);
  EXPECT_EQ(col_ids2_read[1], 1u);
  EXPECT_FLOAT_EQ(values2_read[0], 1.0f);
  EXPECT_FLOAT_EQ(values2_read[1], 1.0f);
}

/** @brief Tests that DumpVectorToFile writes the correct binary format. */
TEST_F(UtilsTest, DumpVectorToFile) {
  // Create a test vector
  std::vector<float> host_vector = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
  dvector<float> device_vector(host_vector);

  // Dump vector to file
  DumpVectorToFile(vector_filename_, device_vector);

  // Read the file back and verify contents
  std::vector<float> read_values;
  uint32_t size;

  ReadVectorFromFile(vector_filename_, read_values, size);

  // Verify size
  EXPECT_EQ(size, 5u);
  EXPECT_EQ(read_values.size(), 5u);

  // Verify values
  for (size_t i = 0; i < host_vector.size(); ++i) {
    EXPECT_FLOAT_EQ(read_values[i], host_vector[i]);
  }
}

/** @brief Tests that DumpVectorToFile appends to an existing file. */
TEST_F(UtilsTest, DumpVectorToFileAppends) {
  // Create and dump first vector
  std::vector<float> host_vector1 = {1.0f, 2.0f, 3.0f};
  dvector<float> device_vector1(host_vector1);
  DumpVectorToFile(vector_filename_, device_vector1);

  // Create and dump second vector
  std::vector<float> host_vector2 = {4.0f, 5.0f};
  dvector<float> device_vector2(host_vector2);
  DumpVectorToFile(vector_filename_, device_vector2);

  // Read both vectors from file
  std::ifstream file(vector_filename_, std::ios::binary);
  ASSERT_TRUE(file.is_open());

  // Read first vector
  uint32_t size1;
  file.read(reinterpret_cast<char*>(&size1), sizeof(uint32_t));
  std::vector<float> values1(size1);
  file.read(reinterpret_cast<char*>(values1.data()), size1 * sizeof(float));

  // Verify first vector
  EXPECT_EQ(size1, 3u);
  EXPECT_FLOAT_EQ(values1[0], 1.0f);
  EXPECT_FLOAT_EQ(values1[1], 2.0f);
  EXPECT_FLOAT_EQ(values1[2], 3.0f);

  // Read second vector
  uint32_t size2;
  file.read(reinterpret_cast<char*>(&size2), sizeof(uint32_t));
  std::vector<float> values2(size2);
  file.read(reinterpret_cast<char*>(values2.data()), size2 * sizeof(float));

  // Verify second vector
  EXPECT_EQ(size2, 2u);
  EXPECT_FLOAT_EQ(values2[0], 4.0f);
  EXPECT_FLOAT_EQ(values2[1], 5.0f);
}

/** @brief Tests that DumpCSRSparseMatrixToFile handles an empty (0x0) matrix. */
TEST_F(UtilsTest, DumpCSRSparseMatrixToFileEmptyMatrix) {
  // Create an empty matrix (0x0)
  CSRSparseMatrix matrix;
  std::vector<int> host_row_offsets = {0};
  matrix.row_offsets = DeviceVector<int>(host_row_offsets);
  matrix.col_ids.resize(0);
  matrix.values.resize(0);

  // Dump matrix to file
  DumpCSRSparseMatrixToFile(matrix_filename_, matrix);

  // Read the file back
  std::vector<uint32_t> row_offsets;
  std::vector<uint32_t> col_ids;
  std::vector<float> values;
  uint32_t num_rows, num_cols, num_nonzeros;

  ReadCSRSparseMatrixFromFile(matrix_filename_, row_offsets, col_ids, values,
                              num_rows, num_cols, num_nonzeros);

  // Verify empty matrix
  EXPECT_EQ(num_rows, 0u);
  EXPECT_EQ(num_cols, 0u);
  EXPECT_EQ(num_nonzeros, 0u);
  EXPECT_EQ(row_offsets.size(), 1u);
  EXPECT_EQ(row_offsets[0], 0u);
}

/** @brief Tests that DumpVectorToFile handles an empty vector. */
TEST_F(UtilsTest, DumpVectorToFileEmptyVector) {
  // Create an empty vector
  dvector<float> device_vector(0);

  // Dump vector to file
  DumpVectorToFile(vector_filename_, device_vector);

  // Read the file back
  std::vector<float> read_values;
  uint32_t size;

  ReadVectorFromFile(vector_filename_, read_values, size);

  // Verify empty vector
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(read_values.size(), 0u);
}

}  // namespace cunls
