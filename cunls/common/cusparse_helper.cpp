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

/** @copydoc cuSPARSEHandle::~cuSPARSEHandle */
cuSPARSEHandle::~cuSPARSEHandle() {
  // Clean up the cuSPARSE handle if it was created.
  // Use WARN instead of THROW to avoid exceptions in destructor.
  if (handle_ != nullptr) {
    WARN_ON_CUSPARSE_ERROR(cusparseDestroy(handle_));
  }
}

/** @copydoc cuSPARSEHandle::GetHandle */
cusparseHandle_t cuSPARSEHandle::GetHandle(cudaStream_t stream) {
  if (stream == nullptr) {
    const std::string msg = "cuSPARSEHandle recieved invalid CUDA stream.";
    LogError(msg);
    throw std::invalid_argument(msg);
  }

  // Return existing handle if it's associated with the same stream
  if (stream == stream_ && handle_ != nullptr) {
    return handle_;
  }

  // Destroy existing handle if we need to create one for a different stream
  if (handle_ != nullptr) {
    THROW_ON_CUSPARSE_ERROR(cusparseDestroy(handle_));
  }

  // Create new handle and associate it with the requested stream
  stream_ = stream;
  THROW_ON_CUSPARSE_ERROR(cusparseCreate(&handle_));
  THROW_ON_CUSPARSE_ERROR(cusparseSetStream(handle_, stream_));
  return handle_;
}

/** @copydoc cuSPARSEMatrixDescription::operator=(cuSPARSEMatrixDescription&&) */
cuSPARSEMatrixDescription& cuSPARSEMatrixDescription::operator=(
    cuSPARSEMatrixDescription&& other) noexcept {
  // Protect against self-assignment
  if (this == &other) {
    return *this;
  }

  // Clean up existing descriptor before taking ownership of the new one
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(cusparseDestroySpMat(description_));
  }

  // Transfer ownership using std::exchange to ensure other's descriptor is
  // nulled
  description_ = std::exchange(other.description_, nullptr);
  return *this;
}

/** @copydoc cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int,int,int,const CSRSparseMatrix&) */
cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(
    int num_rows, int num_cols, int num_nonzeros,
    const CSRSparseMatrix& matrix) {
  // Extract raw pointers from device vectors
  // const_cast is needed because cuSPARSE API expects non-const pointers
  // even though it doesn't modify the data in read-only operations
  auto rows_ptr =
      const_cast<int*>(matrix.row_offsets.data());
  auto cols_ptr =
      const_cast<int*>(matrix.col_ids.data());
  auto values_ptr =
      const_cast<float*>(matrix.values.data());

  // Create CSR matrix descriptor with 32-bit indices, zero-based indexing, and
  // 32-bit floats
  THROW_ON_CUSPARSE_ERROR(cusparseCreateCsr(
      &description_, num_rows, num_cols, num_nonzeros, rows_ptr, cols_ptr,
      values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
};

/** @copydoc cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int,int) */
cuSPARSEMatrixDescription::cuSPARSEMatrixDescription(int num_rows,
                                                     int num_cols) {
  // Create empty CSR matrix descriptor (useful for output matrices)
  // NULL pointers are used since no data is provided initially
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateCsr(&description_, num_rows, num_cols, 0, NULL, NULL, NULL,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
};

/** @copydoc cuSPARSEMatrixDescription::~cuSPARSEMatrixDescription */
cuSPARSEMatrixDescription::~cuSPARSEMatrixDescription() {
  // Clean up cuSPARSE matrix descriptor
  // Use WARN instead of THROW to avoid exceptions in destructor
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(cusparseDestroySpMat(description_));
  }
}

/** @copydoc cuSPARSEMatrixDescription::UpdatePointers */
void cuSPARSEMatrixDescription::UpdatePointers(const CSRSparseMatrix& matrix) {
  // Update the descriptor to point to new matrix data
  // This is more efficient than recreating the descriptor when only data
  // changes
  auto rows_ptr =
      const_cast<int*>(matrix.row_offsets.data());
  auto cols_ptr =
      const_cast<int*>(matrix.col_ids.data());
  auto values_ptr =
      const_cast<float*>(matrix.values.data());

  THROW_ON_CUSPARSE_ERROR(
      cusparseCsrSetPointers(description_, rows_ptr, cols_ptr, values_ptr));
}

/** @copydoc cuSPARSEMatrixDescription::GetDescription */
cusparseSpMatDescr_t cuSPARSEMatrixDescription::GetDescription() {
  return description_;
}

/** @copydoc cuSPARSEVectorDescription::cuSPARSEVectorDescription */
cuSPARSEVectorDescription::cuSPARSEVectorDescription(
    const dvector<float>& vec) {
  // Extract raw pointer from device vector
  // const_cast is needed because cuSPARSE API expects non-const pointers
  auto ptr = const_cast<float*>(vec.data());

  // Create dense vector descriptor with 32-bit float data type
  THROW_ON_CUSPARSE_ERROR(
      cusparseCreateDnVec(&description_, vec.size(), ptr, CUDA_R_32F));
};

/** @copydoc cuSPARSEVectorDescription::~cuSPARSEVectorDescription */
cuSPARSEVectorDescription::~cuSPARSEVectorDescription() {
  // Clean up cuSPARSE vector descriptor
  // Use WARN instead of THROW to avoid exceptions in destructor
  if (description_) {
    WARN_ON_CUSPARSE_ERROR(cusparseDestroyDnVec(description_));
  }
}

/** @copydoc cuSPARSEVectorDescription::GetDescription */
cusparseDnVecDescr_t cuSPARSEVectorDescription::GetDescription() {
  return description_;
}

}  // namespace cunls
