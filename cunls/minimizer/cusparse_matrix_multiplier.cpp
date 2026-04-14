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

#include "cunls/minimizer/cusparse_matrix_multiplier.h"

#include <cusparse.h>

#include <cassert>

#include "cunls/common/cusparse_helper.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/minimizer/sparse_matrix.h"

namespace cunls {

constexpr cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;

cuSPARSESparseMatrixMultiplier::cuSPARSESparseMatrixMultiplier() {
  cusparseSpGEMMDescr_t descr = nullptr;
  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMM_createDescr(&descr));
  gemm_description_ = static_cast<void *>(descr);
}

cuSPARSESparseMatrixMultiplier::~cuSPARSESparseMatrixMultiplier() {
  WARN_ON_CUSPARSE_ERROR(cusparseSpGEMM_destroyDescr(
      static_cast<cusparseSpGEMMDescr_t>(gemm_description_)));
}

void cuSPARSESparseMatrixMultiplier::Transpose(cudaStream_t stream,
                                               const CSRSparseMatrix &matrix,
                                               CSRSparseMatrix &transposed) {
  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, matrix, num_rows, num_cols, num_nonzeros);

  if (transposed.values.size() != num_nonzeros) {
    transposed.values.resize(num_nonzeros);
  }

  if (transposed.col_ids.size() != num_nonzeros) {
    transposed.col_ids.resize(num_nonzeros);
  }

  if (transposed.row_offsets.size() != num_cols + 1) {
    transposed.row_offsets.resize(num_cols + 1);
  }

  auto cusparse_handle =
      static_cast<cusparseHandle_t>(handle_.GetHandle(stream));

  size_t bufferSize = 0;

  THROW_ON_CUSPARSE_ERROR(cusparseCsr2cscEx2_bufferSize(
      cusparse_handle, num_rows, num_cols, num_nonzeros, matrix.values.data(),
      matrix.row_offsets.data(), matrix.col_ids.data(),
      transposed.values.data(), transposed.row_offsets.data(),
      transposed.col_ids.data(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, &bufferSize));

  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

  if (buffer1.size() < bufferSize) {
    buffer1.resize(bufferSize);
  }

  THROW_ON_CUSPARSE_ERROR(cusparseCsr2cscEx2(
      cusparse_handle, num_rows, num_cols, num_nonzeros, matrix.values.data(),
      matrix.row_offsets.data(), matrix.col_ids.data(),
      transposed.values.data(), transposed.row_offsets.data(),
      transposed.col_ids.data(), CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer1.data()));
}

void cuSPARSESparseMatrixMultiplier::Initialize(cudaStream_t stream,
                                                const Problem & /*problem*/,
                                                const CSRSparseMatrix &input,
                                                CSRSparseMatrix &output) {
  Transpose(stream, input, temp_matrix_);

  auto handle = handle_.GetHandle(stream);

  int num_rows, num_cols, num_nonzeros;
  ExtractMatrixMetadata(stream, input, num_rows, num_cols, num_nonzeros);

  descrA_ = std::move(cuSPARSEMatrixDescription(num_cols, num_rows,
                                                num_nonzeros, temp_matrix_));
  descrB_ = std::move(
      cuSPARSEMatrixDescription(num_rows, num_cols, num_nonzeros, input));
  descrC_ = std::move(cuSPARSEMatrixDescription(num_cols, num_cols));

  EstimateWork(handle);
  ReuseNonzeros(handle);

  buffer1.clear();
  buffer2.clear();

  int64_t C_num_rows, C_num_cols, C_nnz;
  THROW_ON_CUSPARSE_ERROR(cusparseSpMatGetSize(
      static_cast<cusparseSpMatDescr_t>(descrC_.GetDescription()), &C_num_rows,
      &C_num_cols, &C_nnz));

  output.row_offsets.resize(C_num_rows + 1);
  output.col_ids.resize(C_nnz);
  output.values.resize(C_nnz, 0);

  descrC_.UpdatePointers(output);
  ReuseCopy(handle);

  buffer3.clear();
}

void cuSPARSESparseMatrixMultiplier::ComputeSquaredMatrix(
    cudaStream_t stream, const Problem & /*problem*/,
    const CSRSparseMatrix &input, CSRSparseMatrix &output) {
  Transpose(stream, input, temp_matrix_);

  auto handle = static_cast<cusparseHandle_t>(handle_.GetHandle(stream));

  constexpr float alpha = 1;
  constexpr float beta = 0;
  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_compute(
      handle, operation, operation, &alpha,
      static_cast<cusparseSpMatDescr_t>(descrA_.GetDescription()),
      static_cast<cusparseSpMatDescr_t>(descrB_.GetDescription()), &beta,
      static_cast<cusparseSpMatDescr_t>(descrC_.GetDescription()), CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT,
      static_cast<cusparseSpGEMMDescr_t>(gemm_description_)));
}

void cuSPARSESparseMatrixMultiplier::EstimateWork(void *handle) {
  auto h = static_cast<cusparseHandle_t>(handle);
  size_t bufferSize1 = 0;

  auto dA = static_cast<cusparseSpMatDescr_t>(descrA_.GetDescription());
  auto dB = static_cast<cusparseSpMatDescr_t>(descrB_.GetDescription());
  auto dC = static_cast<cusparseSpMatDescr_t>(descrC_.GetDescription());
  auto gemm = static_cast<cusparseSpGEMMDescr_t>(gemm_description_);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_workEstimation(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize1, NULL));

  buffer1.resize(bufferSize1);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_workEstimation(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize1, buffer1.data()));
}

void cuSPARSESparseMatrixMultiplier::ReuseNonzeros(void *handle) {
  auto h = static_cast<cusparseHandle_t>(handle);
  size_t bufferSize2 = 0;
  size_t bufferSize3 = 0;
  size_t bufferSize4 = 0;

  auto dA = static_cast<cusparseSpMatDescr_t>(descrA_.GetDescription());
  auto dB = static_cast<cusparseSpMatDescr_t>(descrB_.GetDescription());
  auto dC = static_cast<cusparseSpMatDescr_t>(descrC_.GetDescription());
  auto gemm = static_cast<cusparseSpGEMMDescr_t>(gemm_description_);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize2, NULL, &bufferSize3, NULL, &bufferSize4, NULL));

  buffer2.resize(bufferSize2);
  buffer3.resize(bufferSize3);
  buffer4.resize(bufferSize4);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_nnz(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize2, buffer2.data(), &bufferSize3, buffer3.data(), &bufferSize4,
      buffer4.data()));
}

void cuSPARSESparseMatrixMultiplier::ReuseCopy(void *handle) {
  auto h = static_cast<cusparseHandle_t>(handle);
  size_t bufferSize5 = 0;

  auto dA = static_cast<cusparseSpMatDescr_t>(descrA_.GetDescription());
  auto dB = static_cast<cusparseSpMatDescr_t>(descrB_.GetDescription());
  auto dC = static_cast<cusparseSpMatDescr_t>(descrC_.GetDescription());
  auto gemm = static_cast<cusparseSpGEMMDescr_t>(gemm_description_);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize5, NULL));

  buffer5.resize(bufferSize5);

  THROW_ON_CUSPARSE_ERROR(cusparseSpGEMMreuse_copy(
      h, operation, operation, dA, dB, dC, CUSPARSE_SPGEMM_DEFAULT, gemm,
      &bufferSize5, buffer5.data()));
}

} // namespace cunls
