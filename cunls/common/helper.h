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
#include <cuda_runtime.h>

#include <sstream>

#include "log.h"

#define LOCATION (__FILE__ ":" + std::to_string(__LINE__))

namespace cunls {

/**
 * @brief Generic CUDA error-checking macro.
 *
 * Evaluates the given status expression and, if non-zero, logs an error
 * message with file/line location. Depending on @p is_throw, either
 * throws a std::runtime_error or simply logs a warning.
 *
 * @param status       The CUDA/cuBLAS/cuSolver/cuSPARSE/cuDSS API call result.
 * @param to_string_fn Function that converts the status code to a C-string.
 * @param is_throw     Compile-time boolean: if true, throws on error; if false,
 *                     only logs.
 */
#define CHECK_CUDA_ERROR(status, to_string_fn, is_throw)                  \
  do {                                                                    \
    auto ret = (status);                                                  \
    if (ret != 0) {                                                       \
      std::stringstream msg;                                              \
      msg << "[CUDA] error " << (to_string_fn)(ret) << "(" << ret << ")"; \
      msg << " in " << LOCATION << std::endl;                             \
      LogError(msg.str());                                                \
      if constexpr (is_throw) {                                           \
        throw std::runtime_error(msg.str());                              \
      }                                                                   \
    }                                                                     \
  } while (0)

/**
 * @brief Throws std::runtime_error on CUDA runtime API errors.
 * @param status cudaError_t value returned by a CUDA runtime call.
 */
#define THROW_ON_CUDA_ERROR(status) \
  CHECK_CUDA_ERROR(status, cudaGetErrorString, true)

/**
 * @brief Logs a warning on CUDA runtime API errors without throwing.
 * @param status cudaError_t value returned by a CUDA runtime call.
 */
#define WARN_ON_CUDA_ERROR(status) \
  CHECK_CUDA_ERROR(status, cudaGetErrorString, false)
}  // namespace cunls
