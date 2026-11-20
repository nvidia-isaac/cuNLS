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

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Converts a cuDSS status code to a human-readable error string.
 *
 * @param status The cuDSS status code to convert.
 * @return A string describing the status code.
 */
const char *cudssGetErrorString(int status);

/**
 * @brief Macro to check cuDSS status and throw an exception on error.
 *
 * If the status indicates an error, this macro will throw an exception with
 * a descriptive error message.
 */
#define THROW_ON_CUDSS_ERROR(status) CHECK_CUDA_ERROR(status, cudssGetErrorString, true)

/**
 * @brief Macro to check cuDSS status and log a warning on error.
 *
 * If the status indicates an error, this macro will log a warning but will
 * not throw an exception.
 */
#define WARN_ON_CUDSS_ERROR(status) CHECK_CUDA_ERROR(status, cudssGetErrorString, false)

/**
 * @brief Reusable device memory pool used by cuDSS callbacks.
 *
 * The pool retains freed allocations and reuses them for future requests.
 * A device allocation is reallocated only when the requested size exceeds
 * the retained capacity of an available block.
 */
class cuDSSDeviceMemPool {
 public:
  cuDSSDeviceMemPool() = default;

  cuDSSDeviceMemPool(const cuDSSDeviceMemPool &) = delete;
  cuDSSDeviceMemPool &operator=(const cuDSSDeviceMemPool &) = delete;

  /** @brief Releases all retained device allocations. */
  ~cuDSSDeviceMemPool();

  /**
   * @brief Allocates or reuses a device block for cuDSS.
   *
   * @param ptr Output pointer.
   * @param size Requested size in bytes.
   * @param stream CUDA stream (currently unused by this pool).
   * @return cudaSuccess on success, CUDA runtime error otherwise.
   */
  int Alloc(void **ptr, size_t size, cudaStream_t stream);

  /**
   * @brief Marks a previously allocated block as free for reuse.
   *
   * @param ptr Pointer previously returned by Alloc().
   * @param size Allocation size requested by cuDSS (unused).
   * @param stream CUDA stream (currently unused by this pool).
   * @return cudaSuccess on success, CUDA runtime error otherwise.
   */
  int Dealloc(void *ptr, size_t size, cudaStream_t stream);

 private:
  struct Block {
    void *ptr = nullptr;
    size_t capacity = 0;
    bool in_use = false;
  };

  std::vector<Block> blocks_;
  std::mutex mutex_;
};

/**
 * @brief Makes cuDSS allocate its device memory from @p pool for @p handle.
 *
 * Call it once per handle, after the handle is created and before its first
 * cuDSS phase; @p pool must outlive the handle (declare the pool before the
 * cuDSSHandle).
 *
 * @param handle Opaque cuDSS handle (cudssHandle_t); nothing happens when null.
 * @param pool The pool cuDSS allocates from.
 */
void SetcuDSSDeviceMemHandler(void *handle, cuDSSDeviceMemPool &pool);

/**
 * @brief The cuDSS handle of a CUDA stream (see StreamBoundHandle): created on
 * the first GetHandle(), rebound with cudssSetStream when the stream changes
 * (so a cuDSSData made with it stays valid), destroyed with the object. Each
 * instance owns its handle; non-copyable.
 */
class cuDSSHandle : public StreamBoundHandle {
 public:
  /** @brief No handle yet: the first GetHandle() creates a cudssHandle_t. */
  cuDSSHandle();
};

/**
 * @brief RAII wrapper for cuDSS matrix/vector descriptor.
 *
 * This class creates a cuDSS descriptor from either a CSR sparse matrix
 * or a dense vector and manages its lifecycle.
 */
class cuDSSDescription {
 public:
  /**
   * @brief Constructs a cuDSS matrix descriptor from a CSR sparse matrix.
   *
   * Creates a symmetric CSR matrix descriptor suitable for use with cuDSS
   * sparse solvers.
   *
   * @param matrix The CSR sparse matrix to wrap.
   */
  cuDSSDescription(const CSRSparseMatrix &matrix);

  /**
   * @brief Same, with @p matrix's structure and the given values (as many as
   * @p matrix has entries) instead of its own.
   */
  cuDSSDescription(const CSRSparseMatrix &matrix, const float *values);

  /**
   * @brief Constructs a cuDSS vector descriptor from a dense vector.
   *
   * @param vector The dense vector to wrap.
   */
  cuDSSDescription(const dvector<float> &vector);

  /** @brief Destructor that releases the cuDSS matrix/vector descriptor. */
  ~cuDSSDescription();

  /**
   * @brief Gets the underlying cuDSS matrix/vector descriptor.
   * @return An opaque pointer to the cuDSS matrix descriptor.
   */
  void *GetDescription() { return matrix_; }

 private:
  void *matrix_;  ///< The cuDSS matrix descriptor.
};

/**
 * @brief RAII wrapper for cuDSS configuration object.
 *
 * This class manages a cuDSS configuration object that controls solver
 * parameters and options.
 */
class cuDSSConfig {
 public:
  /** @brief Constructor that creates a cuDSS configuration object. */
  cuDSSConfig(int reordering_algorithm = 0, int nthreads = 1);

  /** @brief Destructor that releases the cuDSS configuration object. */
  ~cuDSSConfig();

  /**
   * @brief Gets the underlying cuDSS configuration object.
   * @return An opaque pointer to the cuDSS configuration object.
   */
  void *GetData() const { return config_; }

 private:
  void *config_ = nullptr;  ///< The cuDSS configuration handle.
};

/**
 * @brief RAII wrapper for cuDSS data object.
 *
 * This class manages a cuDSS data object that stores internal solver state
 * and working memory during the factorization and solve phases. The handle it
 * was made with must outlive it (declare the cuDSSData after the cuDSSHandle).
 */
class cuDSSData {
 public:
  cuDSSData() = default;

  /** @brief Destructor that releases the cuDSS data object. */
  ~cuDSSData();

  /**
   * @brief Gets or creates a cuDSS data object for the specified handle.
   *
   * If the data object is already associated with the given handle, returns it.
   * If the handle differs from the current one, the old data is destroyed
   * and a new one is created for the new handle.
   *
   * @param handle The cuDSS handle to associate with the data object.
   * @return The cuDSS data object associated with the handle.
   */
  void *GetData(void *handle);

 private:
  void *handle_ = nullptr;  ///< Associated cuDSS handle.
  void *data_ = nullptr;    ///< The cuDSS data handle.
};

}  // namespace cunls
