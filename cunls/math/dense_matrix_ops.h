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

#include "cunls/common/cublas_helper.h"

namespace cunls {

/**
 * @brief Computes the square root of a batch of symmetric positive definite
 * matrices.
 *
 * @param cublas_handle Reference to an externally-owned cuBLAS handle.
 * @param stream CUDA stream for asynchronous operations.
 * @param spd_matrix Input/output matrix buffer on device memory.
 * @param matrix_size Size (rows = cols) of each matrix.
 * @param pitch Leading dimension (row stride) of each matrix.
 * @param num_matrices Number of matrices in the batch.
 */
void ComputeSqrtMatrix(cuBLASHandle& cublas_handle, cudaStream_t stream,
                       float* spd_matrix, size_t matrix_size, size_t pitch,
                       size_t num_matrices);

}  // namespace cunls
