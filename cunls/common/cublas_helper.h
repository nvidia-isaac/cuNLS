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

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Converts cuBLAS status code to human-readable error string.
 *
 * @param status cuBLAS status code to convert
 * @return C-string containing the error message
 */
const char *cublasGetErrorString(int status);

/**
 * @brief Macro to throw an exception on cuBLAS errors.
 *
 * If the cuBLAS status indicates an error, throws an exception with
 * a descriptive error message.
 */
#define THROW_ON_CUBLAS_ERROR(status) CHECK_CUDA_ERROR(status, cublasGetErrorString, true)

/**
 * @brief Macro to log a warning on cuBLAS errors.
 *
 * If the cuBLAS status indicates an error, logs a warning message
 * but does not throw an exception.
 */
#define WARN_ON_CUBLAS_ERROR(status) CHECK_CUDA_ERROR(status, cublasGetErrorString, false)

/**
 * @brief The cuBLAS handle of a CUDA stream (see StreamBoundHandle): created on
 * the first GetHandle(), recreated when the stream changes, destroyed with the
 * object. Each instance owns its handle; non-copyable.
 */
class cuBLASHandle : public StreamBoundHandle {
 public:
  /** @brief No handle yet: GetHandle() creates a cublasHandle_t for its stream. */
  cuBLASHandle();
};

}  // namespace cunls
