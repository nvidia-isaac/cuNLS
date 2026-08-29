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
#include <cusolverDn.h>

#include "cunls/common/helper.h"
#include "cunls/common/log.h"
#include "cunls/linear_solver/dense_cholesky_solver.h"

namespace cunls {

bool DenseCholeskySolver::FactorizeAndSolve(cudaStream_t stream, int n, const dvector<float> &rhs,
                                            dvector<float> &result) {
  auto handle = static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));

  THROW_ON_CUSOLVER_ERROR(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, dense_matrix_.data(),
                                           n, workspace_.data(),
                                           static_cast<int>(workspace_.size()), dev_info_.data()));

  if (safety_checks_enabled_) {
    // Check devInfo from potrf before proceeding to potrs, since potrs would
    // overwrite it.  devInfo > 0 means the leading minor of order devInfo is
    // not positive-definite; devInfo < 0 means the devInfo-th parameter was
    // invalid.
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(), dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] != 0) {
      LogError(
          "Cholesky factorization failed (devInfo = {}). Matrix is likely "
          "not positive-definite.",
          dev_info_pinned_[0]);
      return false;
    }
  }

  // potrs solves in-place on the RHS buffer, so copy rhs -> result first.
  THROW_ON_CUDA_ERROR(cudaMemcpyAsync(result.data(), rhs.data(), n * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream));

  THROW_ON_CUSOLVER_ERROR(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1,
                                           dense_matrix_.data(), n, result.data(), n,
                                           dev_info_.data()));

  if (safety_checks_enabled_) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(dev_info_pinned_.data(), dev_info_.data(), sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (dev_info_pinned_[0] < 0) {
      LogError("cusolverDnSpotrs reported invalid parameter at index {}.", -dev_info_pinned_[0]);
      return false;
    }
    if (dev_info_pinned_[0] > 0) {
      LogError("cusolverDnSpotrs failed (devInfo = {}).", dev_info_pinned_[0]);
      return false;
    }
  }

  return true;
}

void DenseCholeskySolver::EnsureBuffersSize(cudaStream_t stream, size_t n) {
  if (dev_info_.size() != 1) {
    dev_info_.resize(1);
    dev_info_pinned_.resize(1);
  }

  if (n != last_n_ && n > 0) {
    auto handle = static_cast<cusolverDnHandle_t>(cusolver_handle_.GetHandle(stream));
    int lwork = 0;
    THROW_ON_CUSOLVER_ERROR(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
                                                        static_cast<int>(n), dense_matrix_.data(),
                                                        static_cast<int>(n), &lwork));
    workspace_.resize(static_cast<size_t>(lwork));
    last_n_ = n;
  }
}

}  // namespace cunls
