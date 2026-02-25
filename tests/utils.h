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

/** @file utils.h
 *  @brief Common test utilities shared across test fixtures.
 */

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace test_utils {

// ============================================================================
// Compile-time size wrappers for typed tests
// ============================================================================

/**
 * @brief Compile-time integer wrapper for typed test parameterization.
 *
 * Used with ::testing::Types to parameterize tests across vector dimensions.
 * @tparam Value Compile-time integer value.
 */
template <int Value>
struct Size {
  static constexpr int size = Value;
};

/**
 * @brief Compile-time size_t wrapper for typed test parameterization.
 *
 * Used for matrix dimensions and other size_t-valued compile-time parameters.
 * @tparam Value Compile-time size_t value.
 */
template <size_t Value>
struct SizeT {
  static constexpr size_t value = Value;
};

// ============================================================================
// Sparse matrix utilities
// ============================================================================

/**
 * @brief Converts host CSR data into a device-side CSRSparseMatrix object.
 *
 * Copies row offsets, column indices, and values from host vectors into
 * device-compatible vectors for GPU operations.
 *
 * @param row_ptr Row offset array (size = num_rows + 1).
 * @param col_idx Column index array for non-zero elements.
 * @param values Value array for non-zero elements.
 * @param matrix Output CSRSparseMatrix to populate with the data.
 */
void CreateCSRSparseMatrix(const std::vector<int>& row_ptr,
                           const std::vector<int>& col_idx,
                           const std::vector<float>& values,
                           CSRSparseMatrix& matrix);

/**
 * @brief Generates a random vector with values in [0.1, 1.0] using a fixed seed.
 *
 * @param size Number of elements to generate.
 * @param values Output vector populated with random values.
 */
void GenerateRandomVector(size_t size, std::vector<float>& values);

// ============================================================================
// Loss function utilities
// ============================================================================

/**
 * @brief CPU reference implementation of the Huber loss function.
 *
 * Computes rho(s) = (rho[0], rho[1], rho[2]) for a given squared error
 * and delta threshold.
 *
 * @param sq_error Squared residual error.
 * @param delta Huber loss threshold.
 * @return float3 containing rho value, first derivative, and second derivative.
 */
float3 HuberLossCPU(float sq_error, float delta);

/**
 * @brief CPU reference for SoftLOne loss (rho, rho', rho'').
 */
float3 SoftLOneLossCPU(float s, float b, float c);

/**
 * @brief CPU reference for Cauchy loss (rho, rho', rho'').
 */
float3 CauchyLossCPU(float s, float b, float c);

/**
 * @brief CPU reference for Arctan loss (rho, rho', rho'').
 */
float3 ArctanLossCPU(float s, float a, float b);

/**
 * @brief CPU reference for Tolerant loss (rho, rho', rho''). c = b*log(1+exp(-a/b)).
 */
float3 TolerantLossCPU(float s, float a, float b, float c);

/**
 * @brief CPU reference for Tukey loss (rho, rho', rho''). a_squared = a*a.
 */
float3 TukeyLossCPU(float s, float a_squared);

// ============================================================================
// Vector factory functions
// ============================================================================

/**
 * @brief Generates sequential host vectors where each vector is filled with its index.
 *
 * Creates count vectors of dimension Dim, where vector[i] is filled with float(i).
 *
 * @tparam Dim Dimension of each vector.
 * @param count Number of vectors to generate.
 * @return Vector of sequentially-filled vectors.
 */
template <int Dim>
std::vector<Vector<Dim>> MakeSequentialVectors(size_t count) {
  std::vector<Vector<Dim>> v(count);
  for (size_t i = 0; i < count; i++) {
    v[i].fill(static_cast<float>(i));
  }
  return v;
}

/**
 * @brief Generates zero-filled host vectors.
 *
 * @tparam Dim Dimension of each vector.
 * @param count Number of vectors to generate.
 * @return Vector of zero-filled vectors.
 */
template <int Dim>
std::vector<Vector<Dim>> MakeZeroVectors(size_t count) {
  std::vector<Vector<Dim>> v(count);
  for (size_t i = 0; i < count; i++) {
    v[i].fill(0);
  }
  return v;
}

/**
 * @brief Generates host vectors filled with a constant value.
 *
 * @tparam Dim Dimension of each vector.
 * @param count Number of vectors to generate.
 * @param value Constant fill value.
 * @return Vector of constant-filled vectors.
 */
template <int Dim>
std::vector<Vector<Dim>> MakeConstantVectors(size_t count, float value) {
  std::vector<Vector<Dim>> v(count);
  for (size_t i = 0; i < count; i++) {
    v[i].fill(value);
  }
  return v;
}

/**
 * @brief Generates a vector of sequential integer IDs [0, 1, ..., count-1].
 *
 * Useful for marking the first N parameter blocks as constant.
 *
 * @param count Number of IDs to generate.
 * @return Vector of integer IDs.
 */
inline std::vector<int> MakeSequentialIds(size_t count) {
  std::vector<int> ids(count);
  for (size_t i = 0; i < count; i++) {
    ids[i] = static_cast<int>(i);
  }
  return ids;
}

// ============================================================================
// VectorStateBatch helpers
// ============================================================================

/**
 * @brief Manages device memory for a VectorStateBatch with optional constant states.
 *
 * Bundles the device vector storage, constant state IDs, and the
 * VectorStateBatch instance together to ensure proper lifetime management.
 *
 * @tparam Dim Dimension of each vector state.
 */
template <int Dim>
struct VectorStateData {
  DeviceVector<Vector<Dim>> vectors;
  DeviceVector<int> const_ids;
  std::unique_ptr<VectorStateBatch<Dim>> batch;
  size_t num_vectors;

  /**
   * @brief Constructs from explicit host vectors and optional constant state IDs.
   *
   * @param host_vectors Host-side state vectors to copy to device.
   * @param const_state_ids IDs of state blocks to mark as constant (default: none).
   */
  VectorStateData(const std::vector<Vector<Dim>>& host_vectors,
                  const std::vector<int>& const_state_ids = {})
      : num_vectors(host_vectors.size()) {
    vectors = DeviceVector<Vector<Dim>>(host_vectors);
    const_ids = DeviceVector<int>(const_state_ids);
    const float* data_ptr = reinterpret_cast<const float*>(vectors.data());
    const int* const_ids_ptr =
        const_ids.empty() ? nullptr : const_ids.data();
    batch = std::make_unique<VectorStateBatch<Dim>>(
        data_ptr, num_vectors, const_ids_ptr, const_ids.size());
  }

  /** @brief Returns a reference to the managed VectorStateBatch. */
  VectorStateBatch<Dim>& get() { return *batch; }

  /** @brief Returns a pointer to the managed VectorStateBatch. */
  VectorStateBatch<Dim>* ptr() { return batch.get(); }
};

// ============================================================================
// PriorVectorFactorBatch helpers
// ============================================================================

/**
 * @brief Manages device observations and a PriorVectorFactorBatch together.
 *
 * Bundles the device observation storage and factor batch instance for
 * proper lifetime management.
 *
 * @tparam Dim Dimension of each observation vector.
 */
template <int Dim>
struct PriorFactorData {
  DeviceVector<Vector<Dim>> observations_device;
  std::unique_ptr<PriorVectorFactorBatch<Dim>> factor_batch;

  /**
   * @brief Constructs from explicit host observations.
   *
   * @param observations Host-side observation vectors to copy to device.
   */
  PriorFactorData(const std::vector<Vector<Dim>>& observations) {
    observations_device = DeviceVector<Vector<Dim>>(observations);
    factor_batch = std::make_unique<PriorVectorFactorBatch<Dim>>(
        observations_device.data(), observations.size());
  }

  /** @brief Returns a reference to the managed factor batch. */
  PriorVectorFactorBatch<Dim>& get() { return *factor_batch; }
};

// ============================================================================
// State pointer utilities
// ============================================================================

/**
 * @brief Collects per-block device pointers from a state batch into a host vector.
 *
 * @tparam StateBatchType State batch type (e.g. VectorStateBatch<Dim>).
 * @param state_batch State batch to extract pointers from.
 * @return Host vector of device pointers to each state block.
 */
template <typename StateBatchType>
std::vector<float*> CollectStatePointers(StateBatchType& state_batch) {
  std::vector<float*> ptrs;
  ptrs.reserve(state_batch.NumStateBlocks());
  for (size_t i = 0; i < state_batch.NumStateBlocks(); i++) {
    ptrs.push_back(state_batch.StateBlockDevicePtr(i));
  }
  return ptrs;
}

/**
 * @brief Collects per-block device pointers into a DeviceVector.
 *
 * @tparam StateBatchType State batch type (e.g. VectorStateBatch<Dim>).
 * @param state_batch State batch to extract pointers from.
 * @return DeviceVector of device pointers to each state block.
 */
template <typename StateBatchType>
DeviceVector<float*> CollectStatePointersDevice(StateBatchType& state_batch) {
  return DeviceVector<float*>(CollectStatePointers(state_batch));
}

// ============================================================================
// Copy helpers
// ============================================================================

/**
 * @brief Copies a VectorStateBatch's data from device to a host std::vector.
 *
 * @tparam Dim Dimension of each vector state block.
 * @param state_batch The state batch to copy from device.
 * @return Host vector of state values.
 */
template <int Dim>
std::vector<Vector<Dim>> CopyStateToHost(
    const VectorStateBatch<Dim>& state_batch) {
  auto ptr = state_batch.StateBlockDevicePtr(0);
  size_t num_blocks = state_batch.NumStateBlocks();
  auto vec_ptr = reinterpret_cast<const Vector<Dim>*>(ptr);
  std::vector<Vector<Dim>> out(num_blocks);
  THROW_ON_CUDA_ERROR(cudaMemcpy(out.data(), vec_ptr,
                                  num_blocks * sizeof(Vector<Dim>),
                                  cudaMemcpyDeviceToHost));
  return out;
}

}  // namespace test_utils
}  // namespace cunls
