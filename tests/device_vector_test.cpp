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

/** @file device_vector_test.cpp
 *  @brief Tests for the DeviceVector CUDA device memory wrapper class, the
 *  CudaStream wrapper and the library handles bound to a stream (including a
 *  cuDSS solver used on more than one stream).
 */

#include "cunls/common/device_vector.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/cusolver_helper.h"
#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/sparse_linear_solver.h"
#include "cunls/minimizer/problem.h"
#include "tests/utils.h"

namespace cunls {

/** @brief Test fixture for DeviceVector tests; ensures CUDA device is
 * initialized. */
class DeviceVectorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure CUDA is initialized
    THROW_ON_CUDA_ERROR(cudaSetDevice(0));
  }
};

/** @brief Verifies that default constructor creates an empty vector with null
 * data pointer. */
TEST_F(DeviceVectorTest, DefaultConstructor) {
  DeviceVector<float> vec;

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_TRUE(vec.empty());
}

/** @brief Verifies that constructor with size allocates device memory
 * correctly. */
TEST_F(DeviceVectorTest, ConstructorWithSize) {
  const size_t num_elements = 100;
  DeviceVector<float> vec(num_elements);

  EXPECT_EQ(vec.size(), num_elements);
  EXPECT_EQ(vec.capacity(), num_elements);
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_FALSE(vec.empty());
}

/** @brief Verifies that constructor with zero size behaves like default
 * constructor. */
TEST_F(DeviceVectorTest, ConstructorWithZeroSize) {
  DeviceVector<float> vec(0);

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_TRUE(vec.empty());
}

/** @brief Verifies that constructing from std::vector copies data to device
 * correctly. */
TEST_F(DeviceVectorTest, ConstructorFromStdVector) {
  const size_t num_elements = 100;
  std::vector<float> host_data(num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i) * 1.5f;
  }

  DeviceVector<float> vec(host_data);

  EXPECT_EQ(vec.size(), num_elements);
  EXPECT_EQ(vec.capacity(), num_elements);
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_FALSE(vec.empty());

  // Verify data was copied correctly
  std::vector<float> result(num_elements);
  vec.CopyToHost(result.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that constructing from an empty std::vector creates an empty
 * DeviceVector. */
TEST_F(DeviceVectorTest, ConstructorFromEmptyStdVector) {
  std::vector<float> empty_host;

  DeviceVector<float> vec(empty_host);

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 0);
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_TRUE(vec.empty());
}

/** @brief Verifies asynchronous host-to-device and device-to-host copy with
 * pinned memory. */
TEST_F(DeviceVectorTest, AsyncCopyRoundTrip) {
  const size_t num_elements = 100;

  // Allocate pinned host memory for true async operation
  float *pinned_src = nullptr;
  float *pinned_dst = nullptr;
  THROW_ON_CUDA_ERROR(cudaMallocHost(&pinned_src, num_elements * sizeof(float)));
  THROW_ON_CUDA_ERROR(cudaMallocHost(&pinned_dst, num_elements * sizeof(float)));

  // Initialize source data
  for (size_t i = 0; i < num_elements; ++i) {
    pinned_src[i] = static_cast<float>(i) * 2.5f;
  }

  // Create device vector and stream
  DeviceVector<float> vec(num_elements);
  CudaStream stream(true);  // sync on destroy

  // Async copy to device
  vec.CopyFromHostAsync(pinned_src, num_elements, stream);

  // Async copy back to host
  vec.CopyToHostAsync(pinned_dst, num_elements, stream);

  // Synchronize to ensure operations complete
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // Verify data integrity
  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(pinned_dst[i], pinned_src[i]);
  }

  // Cleanup pinned memory
  THROW_ON_CUDA_ERROR(cudaFreeHost(pinned_src));
  THROW_ON_CUDA_ERROR(cudaFreeHost(pinned_dst));
}

/** @brief Verifies that CopyFromHost throws when num_elements exceeds capacity.
 */
TEST_F(DeviceVectorTest, CopyFromHostExceedsCapacity) {
  DeviceVector<float> vec(10);
  std::vector<float> host_data(20, 1.0f);

  EXPECT_THROW(vec.CopyFromHost(host_data.data(), 20), std::runtime_error);
}

/** @brief Verifies that CopyToHost throws when num_elements exceeds current
 * size. */
TEST_F(DeviceVectorTest, CopyToHostExceedsSize) {
  DeviceVector<float> vec(10);
  std::vector<float> host_data(10, 1.0f);
  vec.CopyFromHost(host_data.data(), 5);  // Only copy 5 elements

  std::vector<float> result(10);
  EXPECT_THROW(vec.CopyToHost(result.data(), 10), std::runtime_error);
}

/** @brief Verifies that resizing to a larger capacity preserves existing data.
 */
TEST_F(DeviceVectorTest, ResizeLargerPreservesData) {
  const size_t initial_size = 50;
  const size_t new_size = 100;

  std::vector<float> host_data(initial_size);
  for (size_t i = 0; i < initial_size; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> vec(initial_size);
  vec.CopyFromHost(host_data.data(), initial_size);

  // Resize to larger capacity with data preservation
  vec.resize(new_size, true);

  EXPECT_EQ(vec.size(), new_size);
  EXPECT_EQ(vec.capacity(), new_size);

  // Verify original data is preserved
  std::vector<float> result(initial_size);
  vec.CopyToHost(result.data(), initial_size);

  for (size_t i = 0; i < initial_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that resizing to a smaller size updates size but keeps
 * capacity unchanged. */
TEST_F(DeviceVectorTest, ResizeSmallerKeepsCapacity) {
  const size_t initial_size = 100;
  const size_t new_size = 50;

  DeviceVector<float> vec(initial_size);

  vec.resize(new_size);

  EXPECT_EQ(vec.size(), new_size);
  EXPECT_EQ(vec.capacity(), initial_size);  // Capacity unchanged
}

/** @brief Verifies that resizing without the preserve flag still allocates
 * correctly. */
TEST_F(DeviceVectorTest, ResizeWithoutPreservingData) {
  DeviceVector<float> vec(50);

  // Resize without preserving data should still work
  vec.resize(100, false);

  EXPECT_EQ(vec.size(), 100);
  EXPECT_EQ(vec.capacity(), 100);
}

/** @brief Verifies that reserve increases capacity without changing size, and
 * preserves data. */
TEST_F(DeviceVectorTest, ReserveIncreasesCapacity) {
  const size_t initial_size = 50;
  const size_t reserved_capacity = 100;

  std::vector<float> host_data(initial_size);
  for (size_t i = 0; i < initial_size; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> vec(initial_size);
  vec.CopyFromHost(host_data.data(), initial_size);

  vec.reserve(reserved_capacity);

  EXPECT_EQ(vec.size(), initial_size);  // Size unchanged
  EXPECT_EQ(vec.capacity(), reserved_capacity);

  // Verify data is preserved
  std::vector<float> result(initial_size);
  vec.CopyToHost(result.data(), initial_size);

  for (size_t i = 0; i < initial_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that reserving with a smaller capacity is a no-op. */
TEST_F(DeviceVectorTest, ReserveSmallerDoesNothing) {
  const size_t initial_capacity = 100;

  DeviceVector<float> vec(initial_capacity);
  float *original_data = vec.data();

  vec.reserve(50);

  EXPECT_EQ(vec.capacity(), initial_capacity);
  EXPECT_EQ(vec.data(), original_data);  // Pointer unchanged
}

/** @brief Verifies that clear sets size to 0 but retains allocated capacity. */
TEST_F(DeviceVectorTest, ClearKeepsCapacity) {
  const size_t initial_size = 100;

  DeviceVector<float> vec(initial_size);
  float *original_data = vec.data();

  vec.clear();

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), initial_size);
  EXPECT_EQ(vec.data(), original_data);
  EXPECT_TRUE(vec.empty());
}

/** @brief Verifies that shrink_to_fit reduces capacity to match current size
 * while preserving data. */
TEST_F(DeviceVectorTest, ShrinkToFitReducesCapacity) {
  const size_t initial_size = 100;
  const size_t reduced_size = 50;

  std::vector<float> host_data(reduced_size);
  for (size_t i = 0; i < reduced_size; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> vec(initial_size);
  vec.CopyFromHost(host_data.data(), reduced_size);

  vec.shrink_to_fit();

  EXPECT_EQ(vec.size(), reduced_size);
  EXPECT_EQ(vec.capacity(), reduced_size);

  // Verify data is preserved
  std::vector<float> result(reduced_size);
  vec.CopyToHost(result.data(), reduced_size);

  for (size_t i = 0; i < reduced_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that shrink_to_fit on an empty vector frees all device
 * memory. */
TEST_F(DeviceVectorTest, ShrinkToFitOnEmptyFreeMemory) {
  DeviceVector<float> vec(100);
  vec.clear();

  vec.shrink_to_fit();

  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 0);
  EXPECT_EQ(vec.data(), nullptr);
}

/** @brief Verifies that move constructor transfers ownership and leaves source
 * in empty state. */
TEST_F(DeviceVectorTest, MoveConstructor) {
  const size_t num_elements = 100;

  std::vector<float> host_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> original(host_data);
  float *original_ptr = original.data();

  // Move construct
  DeviceVector<float> moved(std::move(original));

  // Verify moved vector has correct state
  EXPECT_EQ(moved.size(), num_elements);
  EXPECT_EQ(moved.capacity(), num_elements);
  EXPECT_EQ(moved.data(), original_ptr);

  // Verify original is in valid empty state
  EXPECT_EQ(original.size(), 0);
  EXPECT_EQ(original.capacity(), 0);
  EXPECT_EQ(original.data(), nullptr);

  // Verify data is intact
  std::vector<float> result(num_elements);
  moved.CopyToHost(result.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that move assignment transfers ownership and leaves source
 * in empty state. */
TEST_F(DeviceVectorTest, MoveAssignment) {
  const size_t num_elements = 100;

  std::vector<float> host_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> original(host_data);
  float *original_ptr = original.data();

  // Create target vector with different data
  DeviceVector<float> target(50);

  // Move assign
  target = std::move(original);

  // Verify target vector has correct state
  EXPECT_EQ(target.size(), num_elements);
  EXPECT_EQ(target.capacity(), num_elements);
  EXPECT_EQ(target.data(), original_ptr);

  // Verify original is in valid empty state
  EXPECT_EQ(original.size(), 0);
  EXPECT_EQ(original.capacity(), 0);
  EXPECT_EQ(original.data(), nullptr);

  // Verify data is intact
  std::vector<float> result(num_elements);
  target.CopyToHost(result.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that self move assignment is safe and preserves vector
 * state. */
TEST_F(DeviceVectorTest, SelfMoveAssignment) {
  const size_t num_elements = 100;

  std::vector<float> host_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> vec(host_data);
  float *original_ptr = vec.data();

  // Self move assignment (use std::move to silence warning)
  vec = std::move(vec);

  // Vector should be unchanged
  EXPECT_EQ(vec.size(), num_elements);
  EXPECT_EQ(vec.capacity(), num_elements);
  EXPECT_EQ(vec.data(), original_ptr);

  // Data should be intact
  std::vector<float> result(num_elements);
  vec.CopyToHost(result.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

/** @brief Verifies that DeviceVector works correctly with a custom struct type.
 */
TEST_F(DeviceVectorTest, WorksWithStructType) {
  struct TestStruct {
    float x;
    float y;
    int id;
  };

  const size_t num_elements = 50;
  std::vector<TestStruct> host_data(num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    host_data[i] = {static_cast<float>(i), static_cast<float>(i * 2), static_cast<int>(i)};
  }

  DeviceVector<TestStruct> vec(host_data);

  std::vector<TestStruct> result(num_elements);
  vec.CopyToHost(result.data(), num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    EXPECT_FLOAT_EQ(result[i].x, host_data[i].x);
    EXPECT_FLOAT_EQ(result[i].y, host_data[i].y);
    EXPECT_EQ(result[i].id, host_data[i].id);
  }
}

/** @brief Verifies that data() returns consistent pointers for const and
 * non-const access. */
TEST_F(DeviceVectorTest, DataPointerAccessors) {
  DeviceVector<float> vec(100);

  float *non_const_ptr = vec.data();
  EXPECT_NE(non_const_ptr, nullptr);

  const DeviceVector<float> &const_vec = vec;
  const float *const_ptr = const_vec.data();
  EXPECT_EQ(const_ptr, non_const_ptr);
}

/** @brief Verifies that partial copy operations (less than capacity) work
 * correctly. */
TEST_F(DeviceVectorTest, PartialCopyOperations) {
  const size_t capacity = 100;
  const size_t copy_size = 50;

  std::vector<float> host_data(copy_size);
  for (size_t i = 0; i < copy_size; ++i) {
    host_data[i] = static_cast<float>(i);
  }

  DeviceVector<float> vec(capacity);
  vec.CopyFromHost(host_data.data(), copy_size);

  EXPECT_EQ(vec.size(), copy_size);
  EXPECT_EQ(vec.capacity(), capacity);

  // Copy back partial data
  const size_t read_size = 25;
  std::vector<float> result(read_size);
  vec.CopyToHost(result.data(), read_size);

  for (size_t i = 0; i < read_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }
}

// StreamBoundHandle is a base for the handle wrappers only: its destructor is protected, so an
// object is never destroyed through it.
static_assert(!std::is_destructible_v<StreamBoundHandle>);
static_assert(std::is_destructible_v<cuBLASHandle>);

// A library handle (StreamBoundHandle) rejects a null stream and is kept for its stream.
TEST(StreamBoundHandleTest, KeepsTheHandleOfItsStream) {
  CudaStream a;
  cuBLASHandle handle;
  EXPECT_THROW(handle.GetHandle(nullptr), std::invalid_argument);
  void *first = handle.GetHandle(a.GetStream());
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(handle.GetHandle(a.GetStream()), first);
}

// Another stream rebinds the handle (the library reports the new stream) instead of recreating
// it: the address stays, and so does a setting made on the handle (a new cuBLAS handle would
// start in host pointer mode, whatever address it gets).
TEST(StreamBoundHandleTest, RebindsTheHandleToAnotherStream) {
  CudaStream a;
  CudaStream b;
  cuBLASHandle cublas;
  cuSPARSEHandle cusparse;
  cuSolverHandle cusolver;
  auto blas = static_cast<cublasHandle_t>(cublas.GetHandle(a.GetStream()));
  THROW_ON_CUBLAS_ERROR(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));
  auto sparse = static_cast<cusparseHandle_t>(cusparse.GetHandle(a.GetStream()));
  auto solver = static_cast<cusolverDnHandle_t>(cusolver.GetHandle(a.GetStream()));
  for (cudaStream_t stream : {b.GetStream(), a.GetStream()}) {
    ASSERT_EQ(cublas.GetHandle(stream), blas);
    ASSERT_EQ(cusparse.GetHandle(stream), sparse);
    ASSERT_EQ(cusolver.GetHandle(stream), solver);
    cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
    THROW_ON_CUBLAS_ERROR(cublasGetPointerMode(blas, &mode));
    EXPECT_EQ(mode, CUBLAS_POINTER_MODE_DEVICE);
    cudaStream_t bound = nullptr;
    THROW_ON_CUBLAS_ERROR(cublasGetStream(blas, &bound));
    EXPECT_EQ(bound, stream);
    THROW_ON_CUSPARSE_ERROR(cusparseGetStream(sparse, &bound));
    EXPECT_EQ(bound, stream);
    THROW_ON_CUSOLVER_ERROR(cusolverDnGetStream(solver, &bound));
    EXPECT_EQ(bound, stream);
  }
}

#ifdef CUNLS_ENABLE_CUDSS
// A cuDSS solver initialized on stream A solves on stream B and again on A: its handle is
// rebound, so the analysis and the cuDSS data made with the handle stay valid, and teardown
// releases the data with the live handle (no cuNLS error is logged).
TEST(StreamBoundHandleTest, CudssSolverSolvesOnAnotherStream) {
  // Tridiagonal SPD matrix (4 on the diagonal, -1 beside it), a known solution and its rhs.
  constexpr int n = 500;
  std::vector<int> row_offsets{0};
  std::vector<int> col_ids;
  std::vector<float> values;
  std::vector<float> x_true(n);
  for (int i = 0; i < n; ++i) {
    x_true[i] = 2.f + std::sin(0.1f * i);
    for (int j = std::max(i - 1, 0); j <= std::min(i + 1, n - 1); ++j) {
      col_ids.push_back(j);
      values.push_back(i == j ? 4.f : -1.f);
    }
    row_offsets.push_back(static_cast<int>(col_ids.size()));
  }
  std::vector<float> b_host(n, 0.f);
  for (int i = 0; i < n; ++i) {
    for (int k = row_offsets[i]; k < row_offsets[i + 1]; ++k) {
      b_host[i] += values[k] * x_true[col_ids[k]];
    }
  }
  CSRSparseMatrix matrix;
  test_utils::CreateCSRSparseMatrix(row_offsets, col_ids, values, matrix);
  dvector<float> rhs(b_host);
  dvector<float> result(n);

  // The log callback is global: restore the default sinks and level even if a check fails.
  const char *level = std::getenv("CUNLS_LOG_LEVEL");
  struct RestoreLogger {
    Verbosity level;
    ~RestoreLogger() {
      SetLogCallback({});
      SetLoggerOptions(level);
    }
  } restore{level != nullptr ? ParseVerbosity(level) : Verbosity::Silent};
  std::vector<std::string> errors;
  SetLoggerOptions(Verbosity::Error);
  SetLogCallback([&errors](Verbosity, std::string_view msg) { errors.emplace_back(msg); });

  {
    CudaStream a;
    CudaStream b;
    SparseLinearSolverPtr solver =
        CreateSparseLinearSolver(SparseLinearSolverType::cuDSS, SparseLinearSolverConfig{});
    ASSERT_TRUE(solver->Initialize(a.GetStream(), Problem(), matrix, rhs, result));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(a.GetStream()));
    for (CudaStream *stream : {&b, &a}) {
      THROW_ON_CUDA_ERROR(
          cudaMemsetAsync(result.data(), 0, n * sizeof(float), stream->GetStream()));
      ASSERT_TRUE(solver->Solve(stream->GetStream(), matrix, rhs, result));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream->GetStream()));
      std::vector<float> x(n);
      result.CopyToHost(x.data(), n);
      for (int i = 0; i < n; ++i) ASSERT_NEAR(x[i], x_true[i], 1e-4f) << "row " << i;
    }
  }  // the solver is destroyed here, before its streams
  for (const std::string &e : errors) ADD_FAILURE() << e;
}
#endif  // CUNLS_ENABLE_CUDSS

// A CudaStream wrapping a stream created elsewhere runs work on it and leaves it alive.
TEST(CudaStreamTest, WrapsAnExternalStreamWithoutOwningIt) {
  cudaStream_t external;
  THROW_ON_CUDA_ERROR(cudaStreamCreate(&external));
  std::vector<float> host(256, 3.f);
  DeviceVector<float> d(host.size());
  {
    CudaStream wrapped(external);
    EXPECT_EQ(wrapped.GetStream(), external);
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(d.data(), host.data(), host.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, wrapped.GetStream()));
  }  // the wrapper is gone; the stream must still be valid
  EXPECT_EQ(cudaStreamSynchronize(external), cudaSuccess);
  std::vector<float> back(host.size());
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(back.data(), d.data(), back.size() * sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_EQ(back, host);
  EXPECT_EQ(cudaStreamDestroy(external), cudaSuccess);
}

}  // namespace cunls
