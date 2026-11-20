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

#include "cunls/common/cublas_helper.h"

#include <cublas_v2.h>

namespace cunls {

const char *cublasGetErrorString(int status) {
  return cublasGetStatusString(static_cast<cublasStatus_t>(status));
}

namespace {

void *CreateCublasHandle(cudaStream_t stream) {
  cublasHandle_t handle = nullptr;
  THROW_ON_CUBLAS_ERROR(cublasCreate(&handle));
  THROW_ON_CUBLAS_ERROR(cublasSetStream(handle, stream));
  return handle;
}

void DestroyCublasHandle(void *handle, bool may_throw) {
  const cublasStatus_t status = cublasDestroy(static_cast<cublasHandle_t>(handle));
  if (may_throw) {
    THROW_ON_CUBLAS_ERROR(status);
  } else {
    WARN_ON_CUBLAS_ERROR(status);
  }
}

}  // namespace

cuBLASHandle::cuBLASHandle()
    : StreamBoundHandle("cuBLASHandle", &CreateCublasHandle, &DestroyCublasHandle) {}

}  // namespace cunls
