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

#pragma once

#include <cuda_runtime.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"

namespace cunls {

/**
 * @brief Converts a cuSPARSE status code to a human-readable error string.
 *
 * @param status The cuSPARSE status code to convert.
 * @return A string describing the status code.
 */
const char *cusparseGetErrorString(int status);

/**
 * @brief Macro to check cuSPARSE status and throw an exception on error
 *
 * This macro wraps cuSPARSE API calls and throws an exception if the operation
 * fails. It uses the cusparseGetErrorString function to provide detailed error
 * messages.
 */
#define THROW_ON_CUSPARSE_ERROR(status) CHECK_CUDA_ERROR(status, cusparseGetErrorString, true)

/**
 * @brief Macro to check cuSPARSE status and issue a warning on error
 *
 * Similar to THROW_ON_CUSPARSE_ERROR but only issues a warning instead of
 * throwing. Useful for cleanup operations where exceptions should not be
 * thrown.
 */
#define WARN_ON_CUSPARSE_ERROR(status) CHECK_CUDA_ERROR(status, cusparseGetErrorString, false)

/**
 * @brief The cuSPARSE handle of a CUDA stream (see StreamBoundHandle): created
 * on the first GetHandle(), rebound with cusparseSetStream when the stream
 * changes, destroyed with the object. Each instance owns its handle;
 * non-copyable.
 */
class cuSPARSEHandle : public StreamBoundHandle {
 public:
  /** @brief No handle yet: the first GetHandle() creates a cusparseHandle_t. */
  cuSPARSEHandle();
};

/**
 * @class cuSPARSEMatrixDescription
 * @brief RAII wrapper for cuSPARSE sparse matrix descriptors
 *
 * This class encapsulates cuSPARSE sparse matrix descriptors and provides
 * automatic memory management. It supports CSR (Compressed Sparse Row) format
 * matrices and allows updating matrix data pointers without recreating the
 * descriptor.
 */
class cuSPARSEMatrixDescription {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an uninitialized matrix description. The descriptor will be null
   * until one of the parameterized constructors is used.
   */
  cuSPARSEMatrixDescription() = default;

  /**
   * @brief Constructor for matrices with existing data
   *
   * Creates a cuSPARSE matrix descriptor for a CSR sparse matrix with existing
   * data. The descriptor points to the matrix's row offsets, column indices,
   * and values.
   *
   * @param num_rows Number of rows in the matrix
   * @param num_cols Number of columns in the matrix
   * @param num_nonzeros Number of non-zero elements in the matrix
   * @param matrix The CSR sparse matrix containing the data pointers
   * @throws Runtime error if descriptor creation fails
   */
  cuSPARSEMatrixDescription(int num_rows, int num_cols, int num_nonzeros,
                            const CSRSparseMatrix &matrix);

  /**
   * @brief Move assignment operator
   *
   * Transfers ownership of the cuSPARSE descriptor from another object.
   * Properly destroys any existing descriptor before taking ownership.
   *
   * @param other The source object to move from
   * @return Reference to this object
   */
  cuSPARSEMatrixDescription &operator=(cuSPARSEMatrixDescription &&other) noexcept;

  /**
   * @brief Destructor
   *
   * Automatically destroys the cuSPARSE matrix descriptor if it exists,
   * ensuring proper resource cleanup.
   */
  ~cuSPARSEMatrixDescription();

  /**
   * @brief Update matrix data pointers
   *
   * Updates the descriptor to point to new matrix data without recreating
   * the descriptor. This is useful when matrix values change but structure
   * remains the same.
   *
   * @param matrix The CSR sparse matrix with updated data pointers
   * @throws Runtime error if pointer update fails
   */
  void UpdatePointers(const CSRSparseMatrix &matrix);

  /**
   * @brief Points the descriptor at @p matrix's structure and at @p values
   * (the same number of entries) instead of its own values.
   */
  void UpdatePointers(const CSRSparseMatrix &matrix, const float *values);

  /**
   * @brief Get the cuSPARSE matrix descriptor
   *
   * Returns the underlying cuSPARSE sparse matrix descriptor for use with
   * cuSPARSE API functions.
   *
   * @return An opaque pointer to the matrix descriptor.
   */
  void *GetDescription();

 private:
  void *description_ = nullptr;  ///< The cuSPARSE matrix descriptor
};

/**
 * @class cuSPARSEVectorDescription
 * @brief RAII wrapper for cuSPARSE dense vector descriptors
 *
 * This class encapsulates cuSPARSE dense vector descriptors and provides
 * automatic memory management. It's designed to work with device vectors
 * of float values and is commonly used in sparse matrix-vector operations.
 */
class cuSPARSEVectorDescription {
 public:
  /**
   * @brief Constructor from device vector
   *
   * Creates a cuSPARSE dense vector descriptor from a device vector of floats.
   * The descriptor references the vector's data pointer and size.
   *
   * @param vec The device vector containing float values
   * @throws Runtime error if descriptor creation fails
   */
  cuSPARSEVectorDescription(const dvector<float> &vec);

  /**
   * @brief Destructor
   *
   * Automatically destroys the cuSPARSE vector descriptor if it exists,
   * ensuring proper resource cleanup.
   */
  ~cuSPARSEVectorDescription();

  /**
   * @brief Get the cuSPARSE vector descriptor
   *
   * Returns the underlying cuSPARSE dense vector descriptor for use with
   * cuSPARSE API functions such as sparse matrix-vector multiplication.
   *
   * @return An opaque pointer to the vector descriptor.
   */
  void *GetDescription();

 private:
  void *description_ = nullptr;  ///< The cuSPARSE vector descriptor
};

}  // namespace cunls
