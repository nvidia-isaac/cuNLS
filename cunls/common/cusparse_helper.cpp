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

#include <cuda_runtime.h>
#include <cusparse.h>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/log.h"

namespace cunls {

const char* cusparseGetErrorString(int status) {
  return ::cusparseGetErrorString(static_cast<cusparseStatus_t>(status));
}

/** @copydoc cuSPARSEHandle::~cuSPARSEHandle */
cuSPARSEHandle::~cuSPARSEHandle() {
  if (handle_ != nullptr) {
    WARN_ON_CUSPARSE_ERROR(
        cusparseDestroy(static_cast<cusparseHandle_t>(handle_)));
  }
}

/** @copydoc cuSPARSEHandle::GetHandle */
void* cuSPARSEHandle::GetHandle(cudaStream_t stream) {
  if (stream == nullptr) {
    const std::string msg = "cuSPARSEHandle recieved invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  if (handle_ != nullptr) {
    THROW_ON_CUSPARSE_ERROR(
        cusparseDestroy(static_cast<cusparseHandle_t>(handle_)));
  }

  stream_ = stream;
  cusparseHandle_t h = nullptr;
  THROW_ON_CUSPARSE_ERROR(cusparseCreate(&h));
  THROW_ON_CUSPARSE_ERROR(cusparseSetStream(h, stream_));
  handle_ = static_cast<void*>(h);
  return handle_;
}

/** @copydoc cuSPARSEMatrixDescription::operator=(cuSPARSEMatrixDescription&&) */
cuSPARSEMatrixDescription& cuSPARSEMatrixDescription::operator=(
    cuSPARSEMatrixDescription&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (description_) {
    WARN_ON_CUSPARSE_ERROR(
        cusparseDestroySpMat(static_cast<cusparseSpMatDescr_t>(description_)));
  }

  description_ = std::exchange(other.description_, nullptr);
  return *this;
}

/** @copydoc cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int,int,int,const CSRSparseMatrix&) */
cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(
    int num_rows, int num_cols, int num_nonzeros,
    const CSRSparseMatrix& matrix) {
  auto rows_ptr =
      const_cast<int*>(matrix.row_offsets.data());
  auto cols_ptr =
      const_cast<int*>(matrix.col_ids.data());
  auto values_ptr =
      const_cast<float*>(matrix.values.data());

  cusparseSpMatDescr_t descr = nullptr;
  THROW_ON_CUSPARSE_ERROR(cusparseCreateCsr(
      &descr, num_rows, num_cols, num_nonzeros, rows_ptr, cols_ptr,
      values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  description_ = static_cast<void*>(descr);
};

/** @copydoc cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int,int) */
cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int num_rows,
                                                     int num_cols) {
  cusparseSpMatDescr_t descr = nullptr;
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateCsr(&descr, num_rows, num_cols, 0, NULL, NULL, NULL,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  description_ = static_cast<void*>(descr);
};

/** @copydoc cuSPARSEMatrixDescription::~cuSPARSEMatrixDescription */
cuSPARSEMatrixDescription::~cuSPARSEMatrixDescription() {
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(
        cusparseDestroySpMat(static_cast<cusparseSpMatDescr_t>(description_)));
  }
}

/** @copydoc cuSPARSEMatrixDescription::UpdatePointers */
void cuSPARSEMatrixDescription::UpdatePointers(const CSRSparseMatrix& matrix) {
  auto rows_ptr =
      const_cast<int*>(matrix.row_offsets.data());
  auto cols_ptr =
      const_cast<int*>(matrix.col_ids.data());
  auto values_ptr =
      const_cast<float*>(matrix.values.data());

  THROW_ON_CUSPARSE_ERROR(cusparseCsrSetPointers(
      static_cast<cusparseSpMatDescr_t>(description_),
      rows_ptr, cols_ptr, values_ptr));
}

/** @copydoc cuSPARSEMatrixDescription::GetDescription */
void* cuSPARSEMatrixDescription::GetDescription() {
  return description_;
}

/** @copydoc cuSPARSEVectorDescription::cuSPARSEVectorDescription */
cuSPARSEVectorDescription::cuSPARSEVectorDescription(
    const dvector<float>& vec) {
  auto ptr = const_cast<float*>(vec.data());

  cusparseDnVecDescr_t descr = nullptr;
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&descr, vec.size(), ptr, CUDA_R_32F));
  description_ = static_cast<void*>(descr);
};

/** @copydoc cuSPARSEVectorDescription::~cuSPARSEVectorDescription */
cuSPARSEVectorDescription::~cuSPARSEVectorDescription() {
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(
        cusparseDestroyDnVec(
            static_cast<cusparseDnVecDescr_t>(description_)));
  }
}

/** @copydoc cuSPARSEVectorDescription::GetDescription */
void* cuSPARSEVectorDescription::GetDescription() {
  return description_;
}

}  // namespace cunls
