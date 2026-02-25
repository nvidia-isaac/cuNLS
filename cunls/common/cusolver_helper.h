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
#include <cusolverDn.h>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"

namespace cunls {

/**
 * @brief Converts a cuSolver status code to a human-readable error string.
 *
 * @param status The cuSolver status code to convert
 * @return C-string containing the error message
 */
inline const char* cusolverGetErrorString(cusolverStatus_t status) {
  if (status == CUSOLVER_STATUS_SUCCESS) {
    return "CUSOLVER_STATUS_SUCCESS";
  } else if (status == CUSOLVER_STATUS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  } else if (status == CUSOLVER_STATUS_ALLOC_FAILED) {
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  } else if (status == CUSOLVER_STATUS_INVALID_VALUE) {
    return "CUSOLVER_STATUS_INVALID_VALUE";
  } else if (status == CUSOLVER_STATUS_ARCH_MISMATCH) {
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  } else if (status == CUSOLVER_STATUS_MAPPING_ERROR) {
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  } else if (status == CUSOLVER_STATUS_EXECUTION_FAILED) {
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  } else if (status == CUSOLVER_STATUS_INTERNAL_ERROR) {
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  } else if (status == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  } else if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_NOT_SUPPORTED";
  } else if (status == CUSOLVER_STATUS_ZERO_PIVOT) {
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  } else if (status == CUSOLVER_STATUS_INVALID_LICENSE) {
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  } else if (status == CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
  } else if (status == CUSOLVER_STATUS_IRS_PARAMS_INVALID) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
  } else if (status == CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
  } else if (status == CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
  } else if (status == CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
  } else if (status == CUSOLVER_STATUS_IRS_INTERNAL_ERROR) {
    return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
  } else if (status == CUSOLVER_STATUS_IRS_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
  } else if (status == CUSOLVER_STATUS_IRS_OUT_OF_RANGE) {
    return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
  } else if (status ==
             CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES) {
    return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
  } else if (status == CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
  } else if (status == CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED) {
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
  } else if (status == CUSOLVER_STATUS_IRS_MATRIX_SINGULAR) {
    return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
  } else if (status == CUSOLVER_STATUS_INVALID_WORKSPACE) {
    return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  }
  return "Unspecified cuSolver error";
}

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
 * Manages the lifecycle of a cusolverDnHandle_t handle. The handle is created
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
   * @return cuSolver handle associated with the stream
   * @throws std::invalid_argument if stream is nullptr
   */
  cusolverDnHandle_t GetHandle(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;        ///< Currently associated CUDA stream
  cusolverDnHandle_t handle_ = nullptr;  ///< cuSolver handle
};

/**
 * @brief RAII wrapper for cuSolver eigenvalue solver info object.
 *
 * Manages the lifecycle of a syevjInfo_t info object used for batched
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
   * @brief Gets the underlying syevjInfo_t handle.
   *
   * @return The cuSolver info handle
   */
  syevjInfo_t GetInfo() const { return info_; }

 private:
  syevjInfo_t info_ = nullptr;  ///< cuSolver eigenvalue solver info handle
};

}  // namespace cunls
