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

void *CreateCublasHandle() {
  cublasHandle_t handle = nullptr;
  THROW_ON_CUBLAS_ERROR(cublasCreate(&handle));
  return handle;
}

void SetCublasStream(void *handle, cudaStream_t stream) {
  THROW_ON_CUBLAS_ERROR(cublasSetStream(static_cast<cublasHandle_t>(handle), stream));
}

void DestroyCublasHandle(void *handle) {
  WARN_ON_CUBLAS_ERROR(cublasDestroy(static_cast<cublasHandle_t>(handle)));
}

}  // namespace

cuBLASHandle::cuBLASHandle()
    : StreamBoundHandle("cuBLASHandle", &CreateCublasHandle, &SetCublasStream,
                        &DestroyCublasHandle) {}

}  // namespace cunls
