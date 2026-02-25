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

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Converts cuBLAS status code to human-readable error string.
 *
 * @param status cuBLAS status code to convert
 * @return C-string containing the error message
 */
inline const char* cublasGetErrorString(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return "CUBLAS_STATUS_SUCCESS";
  else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  else if (status == CUBLAS_STATUS_ALLOC_FAILED)
    return "CUBLAS_STATUS_ALLOC_FAILED";
  else if (status == CUBLAS_STATUS_INVALID_VALUE)
    return "CUBLAS_STATUS_INVALID_VALUE";
  else if (status == CUBLAS_STATUS_ARCH_MISMATCH)
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  else if (status == CUBLAS_STATUS_MAPPING_ERROR)
    return "CUBLAS_STATUS_MAPPING_ERROR";
  else if (status == CUBLAS_STATUS_EXECUTION_FAILED)
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  else if (status == CUBLAS_STATUS_INTERNAL_ERROR)
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  else if (status == CUBLAS_STATUS_NOT_SUPPORTED)
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  else if (status == CUBLAS_STATUS_LICENSE_ERROR)
    return "CUBLAS_STATUS_LICENSE_ERROR";
  else
    return "Unspecified cuBLAS error";
}

/**
 * @brief Macro to throw an exception on cuBLAS errors.
 *
 * If the cuBLAS status indicates an error, throws an exception with
 * a descriptive error message.
 */
#define THROW_ON_CUBLAS_ERROR(status) \
  CHECK_CUDA_ERROR(status, cublasGetErrorString, true)

/**
 * @brief Macro to log a warning on cuBLAS errors.
 *
 * If the cuBLAS status indicates an error, logs a warning message
 * but does not throw an exception.
 */
#define WARN_ON_CUBLAS_ERROR(status) \
  CHECK_CUDA_ERROR(status, cublasGetErrorString, false)

/**
 * @brief RAII wrapper for cuBLAS handle management.
 *
 * This class manages the lifecycle of a cuBLAS handle, automatically
 * creating and destroying it as needed. The handle is associated with
 * a specific CUDA stream and will be recreated if the stream changes.
 *
 * Thread-safe: Each instance manages its own handle independently.
 * Non-copyable: Prevents accidental handle duplication.
 */
class cuBLASHandle {
 public:
  cuBLASHandle() = default;

  cuBLASHandle(const cuBLASHandle&) = delete;
  cuBLASHandle& operator=(const cuBLASHandle&) = delete;

  /**
   * @brief Destructor that releases the cuBLAS handle if initialized.
   *
   * Automatically destroys the cuBLAS handle if one was created.
   * Uses WARN_ON_CUBLAS_ERROR to avoid throwing exceptions in destructor.
   */
  ~cuBLASHandle();

  /**
   * @brief Gets or creates a cuBLAS handle for the specified stream.
   *
   * Returns the existing handle if it's already associated with the
   * requested stream. Otherwise, destroys the old handle (if any) and
   * creates a new one for the specified stream.
   *
   * @param stream CUDA stream to associate the handle with (must not be
   * nullptr)
   * @return cuBLAS handle ready for use with the specified stream
   * @throws std::invalid_argument if stream is nullptr
   */
  cublasHandle_t GetHandle(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;    ///< Currently associated CUDA stream.
  cublasHandle_t handle_ = nullptr;  ///< The cuBLAS handle.
};

}  // namespace cunls
