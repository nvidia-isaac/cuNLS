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

#include <cublas_v2.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/factor/information_factor_batch.h"

namespace cunls {

void ApplyInformationToResiduals(void *cublas_handle,
                                 const float *sqrt_information,
                                 float *residuals, size_t residual_size,
                                 size_t num_factors) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  const size_t stride = residual_size * residual_size;
  constexpr size_t inc = 1;

  THROW_ON_CUBLAS_ERROR(cublasSgemvStridedBatched(
      static_cast<cublasHandle_t>(cublas_handle), CUBLAS_OP_N, residual_size,
      residual_size, &alpha, sqrt_information, residual_size, stride, residuals,
      inc, residual_size, &beta, residuals, inc, residual_size, num_factors));
}

void ApplyInformationToJacobians(void *cublas_handle,
                                 const float *sqrt_information,
                                 float *jacobians, size_t residual_size,
                                 size_t jacobian_pitch, size_t num_factors) {
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  const size_t info_stride = residual_size * residual_size;
  const size_t jacobian_stride = jacobian_pitch * residual_size;

  THROW_ON_CUBLAS_ERROR(cublasSgemmStridedBatched(
      static_cast<cublasHandle_t>(cublas_handle), CUBLAS_OP_N, CUBLAS_OP_N,
      jacobian_pitch, residual_size, residual_size, &alpha, jacobians,
      jacobian_pitch, jacobian_stride, sqrt_information, residual_size,
      info_stride, &beta, jacobians, jacobian_pitch, jacobian_stride,
      num_factors));
}

} // namespace cunls
