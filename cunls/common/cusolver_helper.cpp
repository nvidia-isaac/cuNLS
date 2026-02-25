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

#include "cunls/common/cusolver_helper.h"

#include "cunls/common/log.h"

namespace cunls {

/**
 * @brief Constructs a cuSolver handle.
 *
 * Note: The handle is created lazily. The actual handle creation happens
 * in GetHandle() when a stream is provided.
 */
cuSolverHandle::cuSolverHandle() {
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreate(&handle_));
}

/**
 * @brief Destroys the cuSolver handle.
 *
 * Uses WARN_ON_CUSOLVER_ERROR to avoid throwing exceptions in destructor.
 */
cuSolverHandle::~cuSolverHandle() {
  WARN_ON_CUSOLVER_ERROR(cusolverDnDestroy(handle_));
}

/**
 * @brief Gets or creates a cuSolver handle associated with the given stream.
 *
 * If the handle is already associated with the given stream, returns it.
 * Otherwise, destroys the old handle and creates a new one associated with
 * the stream.
 */
cusolverDnHandle_t cuSolverHandle::GetHandle(cudaStream_t stream) {
  if (stream == nullptr) {
    const std::string msg = "cuSolverHandle received invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  if (handle_ != nullptr) {
    THROW_ON_CUSOLVER_ERROR(cusolverDnDestroy(handle_));
  }

  stream_ = stream;
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreate(&handle_));
  THROW_ON_CUSOLVER_ERROR(cusolverDnSetStream(handle_, stream_));
  return handle_;
}

/**
 * @brief Constructs a cuSolver info object for eigenvalue decomposition.
 */
cuSolverInfo::cuSolverInfo() {
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreateSyevjInfo(&info_));
}

/**
 * @brief Destroys the cuSolver info object.
 *
 * Uses WARN_ON_CUSOLVER_ERROR to avoid throwing exceptions in destructor.
 */
cuSolverInfo::~cuSolverInfo() {
  WARN_ON_CUSOLVER_ERROR(cusolverDnDestroySyevjInfo(info_));
}

}  // namespace cunls