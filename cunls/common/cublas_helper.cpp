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

#include <cublas_v2.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/log.h"

namespace cunls {

/**
 * @brief Destructor implementation for cuBLASHandle.
 *
 * Safely destroys the cuBLAS handle if one exists. Uses WARN_ON_CUBLAS_ERROR
 * instead of THROW_ON_CUBLAS_ERROR to avoid throwing exceptions in destructors,
 * which could lead to undefined behavior.
 */
cuBLASHandle::~cuBLASHandle() {
  if (handle_ != nullptr) {
    WARN_ON_CUBLAS_ERROR(cublasDestroy(handle_));
  }
}

/**
 * @brief Implementation of GetHandle method.
 *
 * Manages the lifecycle of the cuBLAS handle:
 * 1. Validates that the stream is not nullptr
 * 2. Returns existing handle if already associated with the requested stream
 * 3. Destroys old handle if switching to a different stream
 * 4. Creates and initializes new handle for the requested stream
 *
 * This lazy initialization approach avoids creating handles until they're
 * actually needed, and automatically handles stream changes.
 */
cublasHandle_t cuBLASHandle::GetHandle(cudaStream_t stream) {
  // Validate input stream
  if (stream == nullptr) {
    const std::string msg = "cuBLASHandle received invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  // Return existing handle if already associated with the requested stream
  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  // Destroy old handle if switching to a different stream
  if (handle_ != nullptr) {
    THROW_ON_CUBLAS_ERROR(cublasDestroy(handle_));
  }

  // Create and initialize new handle for the requested stream
  stream_ = stream;
  THROW_ON_CUBLAS_ERROR(cublasCreate(&handle_));
  THROW_ON_CUBLAS_ERROR(cublasSetStream(handle_, stream_));
  return handle_;
}

}  // namespace cunls
