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

#include "cunls/common/helper.h"
#include "cunls/common/log.h"

namespace cunls {

/**
 * @brief Converts a cuSolver status code to a human-readable error string.
 *
 * @param status The cuSolver status code to convert
 * @return C-string containing the error message
 */
const char* cusolverGetErrorString(int status);

/**
 * @brief Macro that throws an exception if cuSolver operation fails.
 *
 * Checks the cuSolver status and throws std::runtime_error with a descriptive
 * message if the status indicates an error.
 */
#define THROW_ON_CUSOLVER_ERROR(status) \
  CHECK_CUDA_ERROR(status, cusolverGetErrorString, true)

/**
 * @brief Macro that logs a warning if cuSolver operation fails.
 *
 * Checks the cuSolver status and logs a warning message if the status indicates
 * an error, but does not throw an exception.
 */
#define WARN_ON_CUSOLVER_ERROR(status) \
  CHECK_CUDA_ERROR(status, cusolverGetErrorString, false)

/**
 * @brief RAII wrapper for cuSolver handle management.
 *
 * Manages the lifecycle of a cuSolver handle. The handle is created
 * on construction and destroyed on destruction. The handle is associated with
 * a specific CUDA stream when GetHandle is called.
 */
class cuSolverHandle {
 public:
  /// Constructs a cuSolver handle (handle is created lazily on first GetHandle
  /// call)
  cuSolverHandle();
  /// Destroys the cuSolver handle
  ~cuSolverHandle();

  cuSolverHandle(const cuSolverHandle&) = delete;

  cuSolverHandle& operator=(const cuSolverHandle&) = delete;

  cuSolverHandle(cuSolverHandle&&) = delete;

  cuSolverHandle& operator=(cuSolverHandle&&) = delete;

  /**
   * @brief Gets or creates a cuSolver handle associated with the given stream.
   *
   * If the handle is already associated with the given stream, returns it.
   * Otherwise, destroys the old handle (if any) and creates a new one
   * associated with the stream.
   *
   * @param stream CUDA stream to associate the handle with (must not be
   * nullptr)
   * @return An opaque pointer to the cuSolver handle associated with the stream.
   * @throws std::invalid_argument if stream is nullptr
   */
  void* GetHandle(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;  ///< Currently associated CUDA stream
  void* handle_ = nullptr;         ///< cuSolver handle
};

/**
 * @brief RAII wrapper for cuSolver eigenvalue solver info object.
 *
 * Manages the lifecycle of an info object used for batched
 * symmetric eigenvalue decomposition.
 */
class cuSolverInfo {
 public:
  /// Constructs a cuSolver info object
  cuSolverInfo();
  /// Destroys the cuSolver info object
  ~cuSolverInfo();

  cuSolverInfo(const cuSolverInfo&) = delete;

  cuSolverInfo& operator=(const cuSolverInfo&) = delete;

  cuSolverInfo(cuSolverInfo&&) = delete;

  cuSolverInfo& operator=(cuSolverInfo&&) = delete;

  /**
   * @brief Gets the underlying info handle.
   *
   * @return An opaque pointer to the cuSolver info handle
   */
  void* GetInfo() const { return info_; }

 private:
  void* info_ = nullptr;  ///< cuSolver eigenvalue solver info handle
};

}  // namespace cunls
