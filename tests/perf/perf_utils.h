/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/math/sim_lie_math.h"
#include "cunls/math/sl_lie_math.h"
#include "cunls/math/so_se_lie_math.h"

namespace cunls {
namespace perf {

constexpr size_t kDefaultN = 5000;
constexpr size_t kLossN = 10000;
constexpr int kWarmupIterations = 3;
constexpr int kTimedIterations = 50;
constexpr uint32_t kSeed = 12345u;

class BenchmarkTimer {
 public:
  BenchmarkTimer() {
    THROW_ON_CUDA_ERROR(cudaEventCreate(&start_));
    THROW_ON_CUDA_ERROR(cudaEventCreate(&stop_));
  }

  ~BenchmarkTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  BenchmarkTimer(const BenchmarkTimer&) = delete;
  BenchmarkTimer& operator=(const BenchmarkTimer&) = delete;

  void Start(cudaStream_t stream) {
    THROW_ON_CUDA_ERROR(cudaEventRecord(start_, stream));
  }

  void Stop(cudaStream_t stream) {
    THROW_ON_CUDA_ERROR(cudaEventRecord(stop_, stream));
    THROW_ON_CUDA_ERROR(cudaEventSynchronize(stop_));
  }

  float ElapsedMs() const {
    float ms = 0.0f;
    THROW_ON_CUDA_ERROR(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

  static void PrintResult(const std::string& name, float total_ms,
                           int iterations, size_t n) {
    std::cout << "[Perf] " << name << ": " << total_ms / iterations
              << " ms/iter (N=" << n << ", iters=" << iterations << ")"
              << std::endl;
  }

 private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

// ============================================================================
// Random data generators
// ============================================================================

template <int Dim>
dvector<Vector<Dim>> GenerateRandomVectors(size_t n, float mag = 1.0f,
                                           uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-mag, mag);
  hvector<Vector<Dim>> h(n);
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < Dim; ++d) h[i][d] = dist(rng);
  }
  return dvector<Vector<Dim>>(h);
}

template <int Dim>
dvector<Matrix<Dim>> GenerateIdentityMatrices(size_t n) {
  hvector<Matrix<Dim>> h(n);
  for (size_t i = 0; i < n; ++i) {
    h[i].fill(0.0f);
    for (int d = 0; d < Dim; ++d) h[i][d * Dim + d] = 1.0f;
  }
  return dvector<Matrix<Dim>>(h);
}

inline dvector<SE3Transform> GenerateRandomSE3(size_t n, float rot_mag = 0.3f,
                                               float trans_mag = 1.0f,
                                               uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> rot_dist(-rot_mag, rot_mag);
  std::uniform_real_distribution<float> trans_dist(-trans_mag, trans_mag);
  hvector<Vector<6>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    twists[i][0] = rot_dist(rng);
    twists[i][1] = rot_dist(rng);
    twists[i][2] = rot_dist(rng);
    twists[i][3] = trans_dist(rng);
    twists[i][4] = trans_dist(rng);
    twists[i][5] = trans_dist(rng);
  }
  dvector<Vector<6>> twists_d(twists);
  dvector<SE3Transform> out(n);
  CudaStream stream;
  ComputeExpSE3(stream.GetStream(),
                reinterpret_cast<const float*>(twists_d.data()), 6, 4, 16, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<Matrix<3>> GenerateRandomSO3(size_t n, float rot_mag = 0.3f,
                                            uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-rot_mag, rot_mag);
  hvector<Vector<3>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    twists[i][0] = dist(rng);
    twists[i][1] = dist(rng);
    twists[i][2] = dist(rng);
  }
  dvector<Vector<3>> twists_d(twists);
  dvector<Matrix<3>> out(n);
  CudaStream stream;
  ComputeExpSO3(stream.GetStream(),
                reinterpret_cast<const float*>(twists_d.data()), 3, 3, 9, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<Matrix<2>> GenerateRandomSO2(size_t n, float mag = 0.5f,
                                            uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-mag, mag);
  hvector<float> angles(n);
  for (size_t i = 0; i < n; ++i) angles[i] = dist(rng);
  dvector<float> angles_d(angles);
  dvector<Matrix<2>> out(n);
  CudaStream stream;
  ComputeExpSO2(stream.GetStream(), angles_d.data(), 1, 4, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<Matrix<3>> GenerateRandomSE2(size_t n, float rot_mag = 0.3f,
                                            float trans_mag = 1.0f,
                                            uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> rdist(-rot_mag, rot_mag);
  std::uniform_real_distribution<float> tdist(-trans_mag, trans_mag);
  hvector<Vector<3>> tangent(n);
  for (size_t i = 0; i < n; ++i) {
    tangent[i][0] = tdist(rng);
    tangent[i][1] = tdist(rng);
    tangent[i][2] = rdist(rng);
  }
  dvector<Vector<3>> tangent_d(tangent);
  dvector<Matrix<3>> out(n);
  CudaStream stream;
  ComputeExpSE2(stream.GetStream(),
                reinterpret_cast<const float*>(tangent_d.data()), 3, 9, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<Matrix<3>> GenerateRandomSim2(size_t n, float mag = 0.3f,
                                             uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-mag, mag);
  hvector<Vector<4>> tangent(n);
  for (size_t i = 0; i < n; ++i) {
    tangent[i][0] = dist(rng);
    tangent[i][1] = dist(rng);
    tangent[i][2] = dist(rng);
    tangent[i][3] = dist(rng) * 0.1f;
  }
  dvector<Vector<4>> tangent_d(tangent);
  dvector<Matrix<3>> out(n);
  CudaStream stream;
  ComputeExpSim2(stream.GetStream(),
                 reinterpret_cast<const float*>(tangent_d.data()), 4, 9, n,
                 reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<Matrix<4>> GenerateRandomSim3(size_t n, float mag = 0.3f,
                                             uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-mag, mag);
  hvector<Vector<7>> tangent(n);
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 6; ++d) tangent[i][d] = dist(rng);
    tangent[i][6] = dist(rng) * 0.1f;
  }
  dvector<Vector<7>> tangent_d(tangent);
  dvector<Matrix<4>> out(n);
  CudaStream stream;
  ComputeExpSim3(stream.GetStream(),
                 reinterpret_cast<const float*>(tangent_d.data()), 7, 16, n,
                 reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<SL4Transform> GenerateRandomSL4(size_t n, float mag = 0.1f,
                                               uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-mag, mag);
  hvector<Vector<15>> twists(n);
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 15; ++d) twists[i][d] = dist(rng);
  }
  dvector<Vector<15>> twists_d(twists);
  dvector<SL4Transform> out(n);
  CudaStream stream;
  ComputeExpSL4(stream.GetStream(),
                reinterpret_cast<const float*>(twists_d.data()), 15, 4, 16, n,
                reinterpret_cast<float*>(out.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return out;
}

inline dvector<float> GenerateRandomFloats(size_t n, float lo = 0.0f,
                                           float hi = 100.0f,
                                           uint32_t seed = kSeed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  hvector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = dist(rng);
  return dvector<float>(h);
}

// ============================================================================
// State pointer helpers
// ============================================================================

template <typename StateBatchT>
dvector<const float*> MakeStatePointers(StateBatchT& batch, size_t n) {
  hvector<const float*> ptrs(n);
  size_t num_blocks = batch.NumStateBlocks();
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = reinterpret_cast<const float*>(
        batch.StateBlockDevicePtr(i % num_blocks));
  }
  return dvector<const float*>(ptrs);
}

template <typename StateBatchT>
dvector<const float*> MakePriorStatePointers(StateBatchT& batch) {
  size_t n = batch.NumStateBlocks();
  hvector<const float*> ptrs(n);
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = reinterpret_cast<const float*>(batch.StateBlockDevicePtr(i));
  }
  return dvector<const float*>(ptrs);
}

template <typename StateBatchT>
dvector<const float*> MakeBetweenStatePointers(StateBatchT& left_batch,
                                               StateBatchT& right_batch) {
  size_t n = left_batch.NumStateBlocks();
  hvector<const float*> ptrs(2 * n);
  for (size_t i = 0; i < n; ++i) {
    ptrs[2 * i] = reinterpret_cast<const float*>(
        left_batch.StateBlockDevicePtr(i));
    ptrs[2 * i + 1] = reinterpret_cast<const float*>(
        right_batch.StateBlockDevicePtr(i));
  }
  return dvector<const float*>(ptrs);
}

}  // namespace perf
}  // namespace cunls
