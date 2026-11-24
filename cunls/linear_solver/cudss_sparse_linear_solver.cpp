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

#include "cunls/common/cudss_dynamic.h"
#include "cunls/common/cudss_helper.h"
#include "cunls/common/log.h"

namespace cunls {
namespace {

/** @brief cuDSS's default reordering (CUDSS_ALG_DEFAULT before cuDSS 0.8). */
int DefaultOrdering() {
#ifdef CUNLS_CUDSS_NEW_API
  return static_cast<int>(CUDSS_REORDERING_ALG_DEFAULT);
#else
  return static_cast<int>(CUDSS_ALG_DEFAULT);
#endif
}

}  // namespace

/** @copydoc cuDSSLinearSolver::cuDSSLinearSolver */
cuDSSLinearSolver::cuDSSLinearSolver(cuDSSLinearSolverOptions options)
    : options_(options), cudss_config_(DefaultOrdering(), options.nthreads) {}

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

  auto dss_handle = static_cast<cudssHandle_t>(Handle(stream));
  auto cudss_data = reinterpret_cast<cudssData_t>(cudss_data_.GetData(dss_handle));
  assert(cudss_data != nullptr && "Invalid cuDSS data");

  if (options_.nthreads > 1 && !options_.threading_lib_path.empty()) {
    THROW_ON_CUDSS_ERROR(
        GetCudssApi().SetThreadingLayer(dss_handle, options_.threading_lib_path.c_str()));
  }

  // MatrixStorage(): values in the pattern's layout, starting as the given matrix's values.
  pattern_ = &spd_matrix;
  values_.resize(spd_matrix.values.size());
  if (values_.size() > 0) {
    THROW_ON_CUDA_ERROR(cudaMemcpyAsync(values_.data(), spd_matrix.values.data(),
                                        values_.size() * sizeof(float), cudaMemcpyDeviceToDevice,
                                        stream));
  }

  // Create cuDSS descriptors for the matrix and vectors
  cuDSSDescription m_desc(spd_matrix);
  cuDSSDescription rhs_desc(rhs);
  cuDSSDescription result_desc(result);

  auto desc_m = reinterpret_cast<cudssMatrix_t>(m_desc.GetDescription());
  auto desc_rhs = reinterpret_cast<cudssMatrix_t>(rhs_desc.GetDescription());
  auto desc_result = reinterpret_cast<cudssMatrix_t>(result_desc.GetDescription());

  auto cfg = reinterpret_cast<cudssConfig_t>(cudss_config_.GetData());

  // Symbolic analysis only: each Solve() factorizes the current values.
  THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_ANALYSIS, cfg, cudss_data,
                                             desc_m, desc_result, desc_rhs));
  return true;
}

/** @copydoc cuDSSLinearSolver::Solve */
bool cuDSSLinearSolver::Solve(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                              const dvector<float> &rhs, dvector<float> &result) {
  return SolveWithValues(stream, spd_matrix, spd_matrix.values.data(), rhs, result);
}

void *cuDSSLinearSolver::Handle(cudaStream_t stream) {
  void *handle = cudss_handle_.GetHandle(stream);
  // cudss_handle_ keeps one handle (rebound on a stream change): attach the pool once.
  if (!pool_attached_) {
    SetcuDSSDeviceMemHandler(handle, device_mem_pool_);
    pool_attached_ = true;
  }
  return handle;
}

SolverMatrixStorage cuDSSLinearSolver::MatrixStorage() { return {values_.data()}; }

bool cuDSSLinearSolver::SolveStoredMatrix(cudaStream_t stream, const dvector<float> &rhs,
                                          dvector<float> &result) {
  if (pattern_ == nullptr) {
    LogError("cuDSSLinearSolver::SolveStoredMatrix: not initialized. Fix: call Initialize().");
    return false;
  }
  return SolveWithValues(stream, *pattern_, values_.data(), rhs, result);
}

bool cuDSSLinearSolver::SolveWithValues(cudaStream_t stream, const CSRSparseMatrix &spd_matrix,
                                        const float *values, const dvector<float> &rhs,
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

  if (matrix_size == 0) {
    return true;
  }

  auto dss_handle = static_cast<cudssHandle_t>(Handle(stream));
  auto cudss_data = reinterpret_cast<cudssData_t>(cudss_data_.GetData(dss_handle));

  // Create cuDSS descriptors for the matrix and vectors
  cuDSSDescription m_desc(spd_matrix, values);
  cuDSSDescription rhs_desc(rhs);
  cuDSSDescription result_desc(result);

  auto desc_m = reinterpret_cast<cudssMatrix_t>(m_desc.GetDescription());
  auto desc_rhs = reinterpret_cast<cudssMatrix_t>(rhs_desc.GetDescription());
  auto desc_result = reinterpret_cast<cudssMatrix_t>(result_desc.GetDescription());

  cudssConfig_t cfg = reinterpret_cast<cudssConfig_t>(cudss_config_.GetData());

  // Factorize the current values (reusing the analysis of Initialize()).
  THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_FACTORIZATION, cfg, cudss_data,
                                             desc_m, desc_result, desc_rhs));
  // Solve the linear system
  THROW_ON_CUDSS_ERROR(GetCudssApi().Execute(dss_handle, CUDSS_PHASE_SOLVE, cfg, cudss_data, desc_m,
                                             desc_result, desc_rhs));

  return true;
}

}  // namespace cunls
