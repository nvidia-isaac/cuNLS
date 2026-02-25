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
#include <cusparse.h>

#include "cunls/common/helper.h"
#include "cunls/common/types.h"

/**
 * @brief Macro to check cuSPARSE status and throw an exception on error
 *
 * This macro wraps cuSPARSE API calls and throws an exception if the operation
 * fails. It uses the cusparseGetErrorString function to provide detailed error
 * messages.
 */
#define THROW_ON_CUSPARSE_ERROR(status) \
  CHECK_CUDA_ERROR(status, cusparseGetErrorString, true)

/**
 * @brief Macro to check cuSPARSE status and issue a warning on error
 *
 * Similar to THROW_ON_CUSPARSE_ERROR but only issues a warning instead of
 * throwing. Useful for cleanup operations where exceptions should not be
 * thrown.
 */
#define WARN_ON_CUSPARSE_ERROR(status) \
  CHECK_CUDA_ERROR(status, cusparseGetErrorString, false)

namespace cunls {

/**
 * @class cuSPARSEHandle
 * @brief RAII wrapper for cuSPARSE library handles
 *
 * This class manages the lifetime of cuSPARSE handles and automatically
 * associates them with CUDA streams. It ensures proper cleanup and provides
 * lazy initialization of handles only when needed. The handle is automatically
 * destroyed when the object goes out of scope or when a different stream is
 * requested.
 */
class cuSPARSEHandle {
 public:
  /**
   * @brief Default constructor
   *
   * Creates an uninitialized handle. The actual cuSPARSE handle creation is
   * deferred until GetHandle() is called for the first time.
   */
  cuSPARSEHandle() = default;

  /**
   * @brief Destructor
   *
   * Automatically destroys the cuSPARSE handle if it was created, ensuring
   * proper resource cleanup. Uses WARN_ON_CUSPARSE_ERROR to avoid throwing
   * exceptions from the destructor.
   */
  ~cuSPARSEHandle();

  /**
   * @brief Get or create a cuSPARSE handle for the specified stream
   *
   * Returns a cuSPARSE handle associated with the given CUDA stream. If a
   * handle already exists for the same stream, it is reused. If a different
   * stream is requested, the old handle is destroyed and a new one is created.
   *
   * @param stream The CUDA stream to associate with the cuSPARSE handle
   * @return cusparseHandle_t The cuSPARSE handle ready for use
   * @throws Runtime error if handle creation or stream association fails
   */
  cusparseHandle_t GetHandle(cudaStream_t stream);

 private:
  cudaStream_t stream_ = nullptr;      ///< Currently associated CUDA stream
  cusparseHandle_t handle_ = nullptr;  ///< The cuSPARSE handle object
};

/**
 * @class cuSPARSEMatrixDescription
 * @brief RAII wrapper for cuSPARSE sparse matrix descriptors
 *
 * This class encapsulates cuSPARSE sparse matrix descriptors
 * (cusparseSpMatDescr_t) and provides automatic memory management. It supports
 * CSR (Compressed Sparse Row) format matrices and allows updating matrix data
 * pointers without recreating the descriptor.
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
                            const CSRSparseMatrix& matrix);

  /**
   * @brief Constructor for empty matrices
   *
   * Creates a cuSPARSE matrix descriptor for an empty matrix with specified
   * dimensions but no data. Useful for output matrices in operations like SpMV.
   *
   * @param num_rows Number of rows in the matrix
   * @param num_cols Number of columns in the matrix
   * @throws Runtime error if descriptor creation fails
   */
  cuSPARSEMatrixDescription(int num_rows, int num_cols);

  /**
   * @brief Move assignment operator
   *
   * Transfers ownership of the cuSPARSE descriptor from another object.
   * Properly destroys any existing descriptor before taking ownership.
   *
   * @param other The source object to move from
   * @return Reference to this object
   */
  cuSPARSEMatrixDescription& operator=(
      cuSPARSEMatrixDescription&& other) noexcept;

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
  void UpdatePointers(const CSRSparseMatrix& matrix);

  /**
   * @brief Get the cuSPARSE matrix descriptor
   *
   * Returns the underlying cuSPARSE sparse matrix descriptor for use with
   * cuSPARSE API functions.
   *
   * @return cusparseSpMatDescr_t The matrix descriptor
   */
  cusparseSpMatDescr_t GetDescription();

 private:
  cusparseSpMatDescr_t description_ =
      nullptr;  ///< The cuSPARSE matrix descriptor
};

/**
 * @class cuSPARSEVectorDescription
 * @brief RAII wrapper for cuSPARSE dense vector descriptors
 *
 * This class encapsulates cuSPARSE dense vector descriptors
 * (cusparseDnVecDescr_t) and provides automatic memory management. It's
 * designed to work with device vectors of float values and is commonly used in
 * sparse matrix-vector operations.
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
  cuSPARSEVectorDescription(const dvector<float>& vec);

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
   * @return cusparseDnVecDescr_t The vector descriptor
   */
  cusparseDnVecDescr_t GetDescription();

 private:
  cusparseDnVecDescr_t description_ =
      nullptr;  ///< The cuSPARSE vector descriptor
};

}  // namespace cunls
