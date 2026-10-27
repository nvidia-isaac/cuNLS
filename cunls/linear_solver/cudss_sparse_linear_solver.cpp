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

#include "cunls/linear_solver/cudss_sparse_linear_solver.h"

#include <cudss.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include "cunls/common/cudss_dynamic.h"
#include "cunls/common/cudss_helper.h"
#include "cunls/common/log.h"
#include "cunls/common/profiler.h"

namespace cunls {

/** @copydoc cuDSSLinearSolver::cuDSSLinearSolver */
cuDSSLinearSolver::cuDSSLinearSolver(cuDSSLinearSolverOptions options)
    : options_(options), cudss_config_(options.nthreads, options.pivoting) {}

/** @copydoc cuDSSLinearSolver::Initialize */
bool cuDSSLinearSolver::Initialize(cudaStream_t stream, const Problem & /*problem*/,
                                   const CSRSparseMatrix &spd_matrix, const dvector<float> &rhs,
                                   dvector<float> &result) {
  size_t matrix_size = spd_matrix.NumRows();
  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size, rhs.size());
    return false;
  }

  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size, result.size());
    return false;
  }

  auto dss_handle = reinterpret_cast<cudssHandle_t>(cudss_handle_.GetHandle(stream));
  assert(dss_handle != nullptr && "Invalid cuDSS handle");
  auto cudss_data = reinterpret_cast<cudssData_t>(cudss_data_.GetData(dss_handle));
  assert(cudss_data != nullptr && "Invalid cuDSS data");

  if (options_.nthreads > 1 && !options_.threading_lib_path.empty()) {
    THROW_ON_CUDSS_ERROR(
        GetCudssApi().SetThreadingLayer(dss_handle, options_.threading_lib_path.c_str()));
  }
  if (pool_generation_ != cudss_handle_.Generation()) {
    SetcuDSSDeviceMemHandler(dss_handle, device_mem_pool_, "cuNLS cuDSS pool");
    pool_generation_ = cudss_handle_.Generation();
  }

  // Create cuDSS descriptors for the matrix and vectors
  cuDSSDescription m_desc(spd_matrix);
  cuDSSDescription rhs_desc(rhs);
  cuDSSDescription result_desc(result);

  auto desc_m = reinterpret_cast<cudssMatrix_t>(m_desc.GetDescription());
  auto desc_rhs = reinterpret_cast<cudssMatrix_t>(rhs_desc.GetDescription());
  auto desc_result = reinterpret_cast<cudssMatrix_t>(result_desc.GetDescription());

  auto cfg = reinterpret_cast<cudssConfig_t>(cudss_config_.GetData());

  // Perform symbolic analysis phase
  {
    profiler::ScopedRange range("cuDSS::analysis");
    THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_ANALYSIS, cfg, cudss_data,
                                               desc_m, desc_result, desc_rhs));
  }

  return true;
}

/** @copydoc cuDSSLinearSolver::Solve */
bool cuDSSLinearSolver::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                              const dvector<float> &rhs, dvector<float> &result) {
  size_t matrix_size = spd_matrix.NumRows();

  if (matrix_size != rhs.size()) {
    LogError("LHS size: {} does not match RHS size: {}", matrix_size, rhs.size());
    return false;
  }

  if (matrix_size != result.size()) {
    LogError("LHS size: {} does not match result size: {}", matrix_size, result.size());
    return false;
  }

  if (matrix_size == 0) {
    return true;
  }

  auto dss_handle = reinterpret_cast<cudssHandle_t>(cudss_handle_.GetHandle(stream));
  assert(dss_handle != nullptr && "Invalid cuDSS handle");
  if (pool_generation_ != cudss_handle_.Generation()) {
    SetcuDSSDeviceMemHandler(dss_handle, device_mem_pool_, "cuNLS cuDSS pool");
    pool_generation_ = cudss_handle_.Generation();
  }
  auto cudss_data = reinterpret_cast<cudssData_t>(cudss_data_.GetData(dss_handle));

  // Create cuDSS descriptors for the matrix and vectors
  cuDSSDescription m_desc(spd_matrix);
  cuDSSDescription rhs_desc(rhs);
  cuDSSDescription result_desc(result);

  auto desc_m = reinterpret_cast<cudssMatrix_t>(m_desc.GetDescription());
  auto desc_rhs = reinterpret_cast<cudssMatrix_t>(rhs_desc.GetDescription());
  auto desc_result = reinterpret_cast<cudssMatrix_t>(result_desc.GetDescription());

  cudssConfig_t cfg = reinterpret_cast<cudssConfig_t>(cudss_config_.GetData());

  // Perform factorization
  {
    profiler::ScopedRange range("cuDSS::factorization");
    THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_FACTORIZATION, cfg,
                                               cudss_data, desc_m, desc_result, desc_rhs));
  }
  // Solve the linear system
  {
    profiler::ScopedRange range("cuDSS::solve");
    THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_SOLVE, cfg, cudss_data,
                                               desc_m, desc_result, desc_rhs));
  }

  if (safety_checks_) {
    // cuDSS reports a failed factorization (e.g. a zero pivot) only through INFO, so read it back
    // rather than return a garbage solution.
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
    int info = 0;
    size_t written = 0;
    THROW_ON_CUDSS_ERROR(GetCudssApi().DataGet(dss_handle, cudss_data, CUDSS_DATA_INFO, &info,
                                               sizeof(info), &written));
    if (info != 0) {
      LogError("cuDSS factorization failed with INFO = {}", info);
      return false;
    }
  }

  return true;
}

}  // namespace cunls
