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

#include "cunls/common/cudss_helper.h"

#include <cassert>
#include <cstring>

#include <cudss.h>
#include "cunls/common/log.h"

namespace cunls {

/** @copydoc cudssGetErrorString */
const char* cudssGetErrorString(int status) {
  if (status == CUDSS_STATUS_SUCCESS) {
    return "CUDSS_STATUS_SUCCESS";
  } else if (status == CUDSS_STATUS_NOT_INITIALIZED) {
    return "CUDSS_STATUS_NOT_INITIALIZED";
  } else if (status == CUDSS_STATUS_ALLOC_FAILED) {
    return "CUDSS_STATUS_ALLOC_FAILED";
  } else if (status == CUDSS_STATUS_INVALID_VALUE) {
    return "CUDSS_STATUS_INVALID_VALUE";
  } else if (status == CUDSS_STATUS_NOT_SUPPORTED) {
    return "CUDSS_STATUS_NOT_SUPPORTED";
  } else if (status == CUDSS_STATUS_EXECUTION_FAILED) {
    return "CUDSS_STATUS_EXECUTION_FAILED";
  } else if (status == CUDSS_STATUS_INTERNAL_ERROR) {
    return "CUDSS_STATUS_INTERNAL_ERROR";
  }
  return "Unspecified cuDSS error";
}

cuDSSDeviceMemPool::~cuDSSDeviceMemPool() {
  for (auto& block : blocks_) {
    if (block.ptr != nullptr) {
      WARN_ON_CUDA_ERROR(cudaFree(block.ptr));
    }
  }
}

int cuDSSDeviceMemPool::Alloc(void** ptr, size_t size, cudaStream_t stream) {
  (void)stream;
  if (ptr == nullptr) {
    return cudaErrorInvalidValue;
  }

  if (size == 0) {
    *ptr = nullptr;
    return cudaSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Reuse any available block that already has enough capacity.
  for (auto& block : blocks_) {
    if (!block.in_use && block.capacity >= size) {
      block.in_use = true;
      *ptr = block.ptr;
      return cudaSuccess;
    }
  }

  // Grow the first available block when requested size does not fit.
  for (auto& block : blocks_) {
    if (!block.in_use) {
      if (block.ptr != nullptr) {
        auto free_status = cudaFreeAsync(block.ptr, stream);
        if (free_status != cudaSuccess) {
          return free_status;
        }
      }
      auto alloc_status = cudaMallocAsync(&block.ptr, size, stream);
      if (alloc_status != cudaSuccess) {
        block.ptr = nullptr;
        block.capacity = 0;
        return alloc_status;
      }
      block.capacity = size;
      block.in_use = true;
      *ptr = block.ptr;
      return cudaSuccess;
    }
  }

  // No free blocks are available, create a new pooled block.
  Block block;
  auto alloc_status = cudaMallocAsync(&block.ptr, size, stream);
  if (alloc_status != cudaSuccess) {
    return alloc_status;
  }
  block.capacity = size;
  block.in_use = true;
  *ptr = block.ptr;
  blocks_.push_back(block);
  return cudaSuccess;
}

int cuDSSDeviceMemPool::Dealloc(void* ptr, size_t size, cudaStream_t stream) {
  (void)size;
  (void)stream;
  if (ptr == nullptr) {
    return cudaSuccess;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& block : blocks_) {
    if (block.ptr == ptr) {
      block.in_use = false;
      return cudaSuccess;
    }
  }

  return cudaErrorInvalidDevicePointer;
}

int cuDSSDeviceMemPoolAlloc(void* ctx, void** ptr, size_t size,
                            cudaStream_t stream) {
  if (ctx == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }
  return reinterpret_cast<cuDSSDeviceMemPool*>(ctx)->Alloc(ptr, size, stream);
}

int cuDSSDeviceMemPoolDealloc(void* ctx, void* ptr, size_t size,
                              cudaStream_t stream) {
  if (ctx == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }
  return reinterpret_cast<cuDSSDeviceMemPool*>(ctx)->Dealloc(ptr, size, stream);
}

void SetcuDSSDeviceMemHandler(void* handle, cuDSSDeviceMemPool& pool,
                              const char* handler_name) {
  if (handle == nullptr) {
    return;
  }

  cudssDeviceMemHandler_t handler{};
  handler.ctx = reinterpret_cast<void*>(&pool);
  handler.device_alloc = cuDSSDeviceMemPoolAlloc;
  handler.device_free = cuDSSDeviceMemPoolDealloc;
  std::strncpy(handler.name,
               (handler_name == nullptr || handler_name[0] == '\0')
                   ? "cunls device pool"
                   : handler_name,
               CUDSS_ALLOCATOR_NAME_LEN);
  handler.name[CUDSS_ALLOCATOR_NAME_LEN - 1] = '\0';

  DetachcuDSSDeviceMemHandler(handle);

  THROW_ON_CUDSS_ERROR(cudssSetDeviceMemHandler(
      reinterpret_cast<cudssHandle_t>(handle), &handler));
}

void DetachcuDSSDeviceMemHandler(void* handle) {
  if (handle == nullptr) {
    return;
  }
  THROW_ON_CUDSS_ERROR(cudssSetDeviceMemHandler(
      reinterpret_cast<cudssHandle_t>(handle), nullptr));
}

/**
 * @brief Destructor that releases the cuDSS handle if initialized.
 *
 * Safely destroys the cuDSS handle, logging a warning if destruction fails
 * rather than throwing an exception.
 */
cuDSSHandle::~cuDSSHandle() {
  if (handle_ != nullptr) {
    WARN_ON_CUDSS_ERROR(cudssDestroy((cudssHandle_t)handle_));
  }
}

/** @copydoc cuDSSHandle::GetHandle */
void* cuDSSHandle::GetHandle(cudaStream_t stream) {
  // Validate input stream
  if (stream == nullptr) {
    const std::string msg = "cuDSSHandle received invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  // Return existing handle if already associated with the requested stream
  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  // Destroy old handle if switching to a different stream
  if (handle_ != nullptr) {
    auto h = reinterpret_cast<cudssHandle_t>(handle_);
    THROW_ON_CUDSS_ERROR(cudssDestroy(h));
  }

  // Create and initialize new handle for the requested stream
  stream_ = stream;

  cudssHandle_t h = nullptr;
  THROW_ON_CUDSS_ERROR(cudssCreate(&h));
  THROW_ON_CUDSS_ERROR(cudssSetStream(h, stream_));
  handle_ = reinterpret_cast<void*>(h);
  return handle_;
}

/**
 * @brief Constructs a cuDSS matrix descriptor from a CSR sparse matrix.
 *
 * Creates a symmetric CSR matrix descriptor suitable for use with cuDSS
 * sparse solvers. The matrix is assumed to be symmetric positive definite.
 *
 * @param symmetric_matrix The CSR sparse matrix to wrap. Must be symmetric.
 */
cuDSSDescription::cuDSSDescription(const CSRSparseMatrix& symmetric_matrix) {
  // Extract matrix dimensions from CSR format
  size_t matrix_size = symmetric_matrix.NumRows();
  size_t num_nonzeros = symmetric_matrix.NumNonZeros();

  // Get raw device pointers from thrust vectors
  // const_cast is required by cuDSS API but data is not modified
  auto rows_ptr = const_cast<int*>(symmetric_matrix.row_offsets.data());
  auto cols_ptr = const_cast<int*>(symmetric_matrix.col_ids.data());
  auto values_ptr = const_cast<float*>(symmetric_matrix.values.data());

  cudssMatrix_t mat = nullptr;
  // Create cuDSS CSR matrix descriptor
  // Parameters: symmetric matrix, full view, zero-based indexing
#ifdef CUDSS_NEW_API
  THROW_ON_CUDSS_ERROR(cudssMatrixCreateCsr(
      &mat, matrix_size, matrix_size, num_nonzeros, rows_ptr, NULL, cols_ptr,
      values_ptr, CUDA_R_32I, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_SYMMETRIC,
      CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
#else
  THROW_ON_CUDSS_ERROR(cudssMatrixCreateCsr(
      &mat, matrix_size, matrix_size, num_nonzeros, rows_ptr, NULL, cols_ptr,
      values_ptr, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_SYMMETRIC,
      CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
#endif

  matrix_ = reinterpret_cast<void*>(mat);
}

/**
 * @brief Constructs a cuDSS vector descriptor from a dense vector.
 *
 * Creates a dense matrix descriptor with dimensions (size x 1) to represent
 * a vector, suitable for use as a right-hand side or solution vector in
 * cuDSS operations.
 *
 * @param vector The dense vector to wrap.
 */
cuDSSDescription::cuDSSDescription(const dvector<float>& vector) {
  size_t size = vector.size();
  // Get raw device pointer from thrust vector
  // const_cast is required by cuDSS API but data is not modified
  auto ptr = const_cast<float*>(vector.data());
  // Create dense matrix descriptor with dimensions (size x 1) to represent a
  // vector

  cudssMatrix_t mat = nullptr;
  THROW_ON_CUDSS_ERROR(cudssMatrixCreateDn(&mat, size, 1, size, ptr, CUDA_R_32F,
                                           CUDSS_LAYOUT_COL_MAJOR));
  matrix_ = reinterpret_cast<void*>(mat);
}

/**
 * @brief Destructor that releases the cuDSS matrix/vector descriptor.
 *
 * Safely destroys the cuDSS descriptor, logging a warning if destruction
 * fails rather than throwing an exception.
 */
cuDSSDescription::~cuDSSDescription() {
  auto mat = reinterpret_cast<cudssMatrix_t>(matrix_);
  WARN_ON_CUDSS_ERROR(cudssMatrixDestroy(mat));
}

/**
 * @brief Constructor that creates a cuDSS configuration object.
 *
 * Creates a new cuDSS configuration and sets the reordering algorithm and
 * number of threads.
 *
 * @param reordering_algorithm The reordering algorithm to use (e.g.,
 *                             CUDSS_ALG_DEFAULT, CUDSS_ALG_1). Defaults to 0.
 * @param nthreads The number of threads to use. Defaults to 1.
 */
cuDSSConfig::cuDSSConfig(int reordering_algorithm, int nthreads) {
  cudssConfig_t cfg = nullptr;

  // Validate algorithm is within expected range
  if (reordering_algorithm < static_cast<int>(CUDSS_ALG_DEFAULT) ||
      reordering_algorithm > static_cast<int>(CUDSS_ALG_5)) {
    throw std::invalid_argument("Invalid reordering algorithm value");
  }

  THROW_ON_CUDSS_ERROR(cudssConfigCreate(&cfg));

  auto alg = (cudssAlgType_t)reordering_algorithm;

  THROW_ON_CUDSS_ERROR(cudssConfigSet(cfg, CUDSS_CONFIG_REORDERING_ALG, &alg,
                                      sizeof(cudssAlgType_t)));

  THROW_ON_CUDSS_ERROR(
      cudssConfigSet(cfg, CUDSS_CONFIG_HOST_NTHREADS, &nthreads, sizeof(int)));

  config_ = reinterpret_cast<void*>(cfg);
}

/**
 * @brief Destructor that releases the cuDSS configuration object.
 *
 * Safely destroys the cuDSS configuration, logging a warning if destruction
 * fails rather than throwing an exception.
 */
cuDSSConfig::~cuDSSConfig() {
  if (config_ != nullptr) {
    auto config = reinterpret_cast<cudssConfig_t>(config_);
    WARN_ON_CUDSS_ERROR(cudssConfigDestroy(config));
  }
}

/** @copydoc cuDSSData::GetData */
void* cuDSSData::GetData(void* handle) {
  assert(handle != nullptr);
  // Return existing data object if already associated with the requested handle
  if (handle == handle_ && data_ != nullptr) {
    return data_;
  }

  if (data_ != nullptr && handle_ != nullptr) {
    auto cudss_data = reinterpret_cast<cudssData_t>(data_);
    WARN_ON_CUDSS_ERROR(cudssDataDestroy((cudssHandle_t)handle_, cudss_data));
  }

  // Create and initialize new data object for the requested handle
  handle_ = handle;
  cudssData_t cudss_data = nullptr;
  THROW_ON_CUDSS_ERROR(cudssDataCreate((cudssHandle_t)handle_, &cudss_data));
  data_ = reinterpret_cast<void*>(cudss_data);
  return data_;
}

/**
 * @brief Destructor that releases the cuDSS data object.
 *
 * Safely destroys the cuDSS data object using its associated handle,
 * logging a warning if destruction fails rather than throwing an exception.
 */
cuDSSData::~cuDSSData() {
  if (data_ != nullptr && handle_ != nullptr) {
    auto data = reinterpret_cast<cudssData_t>(data_);
    WARN_ON_CUDSS_ERROR(cudssDataDestroy((cudssHandle_t)handle_, data));
  }
}

}  // namespace cunls