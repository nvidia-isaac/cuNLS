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

const char* cublasGetErrorString(int status) {
  auto s = static_cast<cublasStatus_t>(status);
  if (s == CUBLAS_STATUS_SUCCESS)
    return "CUBLAS_STATUS_SUCCESS";
  else if (s == CUBLAS_STATUS_NOT_INITIALIZED)
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  else if (s == CUBLAS_STATUS_ALLOC_FAILED)
    return "CUBLAS_STATUS_ALLOC_FAILED";
  else if (s == CUBLAS_STATUS_INVALID_VALUE)
    return "CUBLAS_STATUS_INVALID_VALUE";
  else if (s == CUBLAS_STATUS_ARCH_MISMATCH)
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  else if (s == CUBLAS_STATUS_MAPPING_ERROR)
    return "CUBLAS_STATUS_MAPPING_ERROR";
  else if (s == CUBLAS_STATUS_EXECUTION_FAILED)
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  else if (s == CUBLAS_STATUS_INTERNAL_ERROR)
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  else if (s == CUBLAS_STATUS_NOT_SUPPORTED)
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  else if (s == CUBLAS_STATUS_LICENSE_ERROR)
    return "CUBLAS_STATUS_LICENSE_ERROR";
  else
    return "Unspecified cuBLAS error";
}

cuBLASHandle::~cuBLASHandle() {
  if (handle_ != nullptr) {
    WARN_ON_CUBLAS_ERROR(
        cublasDestroy(static_cast<cublasHandle_t>(handle_)));
  }
}

void* cuBLASHandle::GetHandle(cudaStream_t stream) {
  if (stream == nullptr) {
    const std::string msg = "cuBLASHandle received invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  if (handle_ != nullptr) {
    THROW_ON_CUBLAS_ERROR(
        cublasDestroy(static_cast<cublasHandle_t>(handle_)));
  }

  stream_ = stream;
  cublasHandle_t h = nullptr;
  THROW_ON_CUBLAS_ERROR(cublasCreate(&h));
  THROW_ON_CUBLAS_ERROR(cublasSetStream(h, stream_));
  handle_ = static_cast<void*>(h);
  return handle_;
}

}  // namespace cunls
