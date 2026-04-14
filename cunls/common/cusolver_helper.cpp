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

#include <cusolverDn.h>

#include "cunls/common/cusolver_helper.h"
#include "cunls/common/log.h"

namespace cunls {

const char *cusolverGetErrorString(int status) {
  auto s = static_cast<cusolverStatus_t>(status);
  if (s == CUSOLVER_STATUS_SUCCESS) {
    return "CUSOLVER_STATUS_SUCCESS";
  } else if (s == CUSOLVER_STATUS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
  } else if (s == CUSOLVER_STATUS_ALLOC_FAILED) {
    return "CUSOLVER_STATUS_ALLOC_FAILED";
  } else if (s == CUSOLVER_STATUS_INVALID_VALUE) {
    return "CUSOLVER_STATUS_INVALID_VALUE";
  } else if (s == CUSOLVER_STATUS_ARCH_MISMATCH) {
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
  } else if (s == CUSOLVER_STATUS_MAPPING_ERROR) {
    return "CUSOLVER_STATUS_MAPPING_ERROR";
  } else if (s == CUSOLVER_STATUS_EXECUTION_FAILED) {
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
  } else if (s == CUSOLVER_STATUS_INTERNAL_ERROR) {
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
  } else if (s == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  } else if (s == CUSOLVER_STATUS_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_NOT_SUPPORTED";
  } else if (s == CUSOLVER_STATUS_ZERO_PIVOT) {
    return "CUSOLVER_STATUS_ZERO_PIVOT";
  } else if (s == CUSOLVER_STATUS_INVALID_LICENSE) {
    return "CUSOLVER_STATUS_INVALID_LICENSE";
  } else if (s == CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
  } else if (s == CUSOLVER_STATUS_IRS_PARAMS_INVALID) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
  } else if (s == CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
  } else if (s == CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
  } else if (s == CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER) {
    return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
  } else if (s == CUSOLVER_STATUS_IRS_INTERNAL_ERROR) {
    return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
  } else if (s == CUSOLVER_STATUS_IRS_NOT_SUPPORTED) {
    return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
  } else if (s == CUSOLVER_STATUS_IRS_OUT_OF_RANGE) {
    return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
  } else if (s == CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES) {
    return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
  } else if (s == CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED) {
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
  } else if (s == CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED) {
    return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
  } else if (s == CUSOLVER_STATUS_IRS_MATRIX_SINGULAR) {
    return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
  } else if (s == CUSOLVER_STATUS_INVALID_WORKSPACE) {
    return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  }
  return "Unspecified cuSolver error";
}

cuSolverHandle::cuSolverHandle() {
  cusolverDnHandle_t h = nullptr;
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreate(&h));
  handle_ = static_cast<void *>(h);
}

cuSolverHandle::~cuSolverHandle() {
  WARN_ON_CUSOLVER_ERROR(
      cusolverDnDestroy(static_cast<cusolverDnHandle_t>(handle_)));
}

void *cuSolverHandle::GetHandle(cudaStream_t stream) {
  if (stream == nullptr) {
    const std::string msg = "cuSolverHandle received invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  if (handle_ != nullptr) {
    THROW_ON_CUSOLVER_ERROR(
        cusolverDnDestroy(static_cast<cusolverDnHandle_t>(handle_)));
  }

  stream_ = stream;
  cusolverDnHandle_t h = nullptr;
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreate(&h));
  THROW_ON_CUSOLVER_ERROR(cusolverDnSetStream(h, stream_));
  handle_ = static_cast<void *>(h);
  return handle_;
}

cuSolverInfo::cuSolverInfo() {
  syevjInfo_t info = nullptr;
  THROW_ON_CUSOLVER_ERROR(cusolverDnCreateSyevjInfo(&info));
  info_ = static_cast<void *>(info);
}

cuSolverInfo::~cuSolverInfo() {
  WARN_ON_CUSOLVER_ERROR(
      cusolverDnDestroySyevjInfo(static_cast<syevjInfo_t>(info_)));
}

} // namespace cunls
