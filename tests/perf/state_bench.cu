/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/state_batch_ops.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/perf/perf_utils.h"

namespace cunls {
namespace {

constexpr size_t N = perf::kDefaultN;

// ============================================================================
// Vector state batches
// ============================================================================

template <int Dim>
void BenchVectorStatePlus(const std::string& name) {
  auto states = perf::GenerateRandomVectors<Dim>(N);
  auto deltas = perf::GenerateRandomVectors<Dim>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Vector<Dim>> output(N);

  VectorStateBatch<Dim> batch(reinterpret_cast<const float*>(states.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(states.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(states.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult(name, timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

TEST(StateBench, VectorStateBatch_Dim1) {
  BenchVectorStatePlus<1>("VectorStateBatch<1>::Plus");
}

TEST(StateBench, VectorStateBatch_Dim3) {
  BenchVectorStatePlus<3>("VectorStateBatch<3>::Plus");
}

TEST(StateBench, VectorStateBatch_Dim6) {
  BenchVectorStatePlus<6>("VectorStateBatch<6>::Plus");
}

// ============================================================================
// Lie group state batches (cuBLAS-dependent)
// ============================================================================

TEST(StateBench, SO2StateBatch) {
  auto rotations = perf::GenerateRandomSO2(N);
  auto deltas = perf::GenerateRandomVectors<1>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Matrix<2>> output(N);

  cuBLASHandle cublas;
  SO2StateBatch batch(cublas, reinterpret_cast<const float*>(rotations.data()),
                      N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(rotations.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(rotations.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("SO2StateBatch::Plus", timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

TEST(StateBench, SO3StateBatch) {
  auto rotations = perf::GenerateRandomSO3(N);
  auto deltas = perf::GenerateRandomVectors<3>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Matrix<3>> output(N);

  cuBLASHandle cublas;
  SO3StateBatch batch(cublas, reinterpret_cast<const float*>(rotations.data()),
                      N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(rotations.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(rotations.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("SO3StateBatch::Plus", timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

TEST(StateBench, SE2StateBatch) {
  auto transforms = perf::GenerateRandomSE2(N);
  auto deltas = perf::GenerateRandomVectors<3>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Matrix<3>> output(N);

  cuBLASHandle cublas;
  SE2StateBatch batch(cublas,
                      reinterpret_cast<const float*>(transforms.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("SE2StateBatch::Plus", timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

TEST(StateBench, SE3StateBatch) {
  auto transforms = perf::GenerateRandomSE3(N);
  auto deltas = perf::GenerateRandomVectors<6>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<SE3Transform> output(N);

  cuBLASHandle cublas;
  SE3StateBatch batch(cublas,
                      reinterpret_cast<const float*>(transforms.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("SE3StateBatch::Plus", timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

TEST(StateBench, Similarity2StateBatch) {
  auto transforms = perf::GenerateRandomSim2(N);
  auto deltas = perf::GenerateRandomVectors<4>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Matrix<3>> output(N);

  cuBLASHandle cublas;
  Similarity2StateBatch batch(
      cublas, reinterpret_cast<const float*>(transforms.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("Similarity2StateBatch::Plus",
                                    timer.ElapsedMs(), perf::kTimedIterations,
                                    N);
}

TEST(StateBench, Similarity3StateBatch) {
  auto transforms = perf::GenerateRandomSim3(N);
  auto deltas = perf::GenerateRandomVectors<7>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<Matrix<4>> output(N);

  cuBLASHandle cublas;
  Similarity3StateBatch batch(
      cublas, reinterpret_cast<const float*>(transforms.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("Similarity3StateBatch::Plus",
                                    timer.ElapsedMs(), perf::kTimedIterations,
                                    N);
}

TEST(StateBench, SL4StateBatch) {
  auto transforms = perf::GenerateRandomSL4(N);
  auto deltas = perf::GenerateRandomVectors<15>(N, 0.01f, perf::kSeed + 1);
  DeviceVector<SL4Transform> output(N);

  cuBLASHandle cublas;
  SL4StateBatch batch(cublas,
                      reinterpret_cast<const float*>(transforms.data()), N);

  CudaStream stream;
  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    batch.Plus(reinterpret_cast<const float*>(transforms.data()),
               reinterpret_cast<const float*>(deltas.data()),
               reinterpret_cast<float*>(output.data()), stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult("SL4StateBatch::Plus", timer.ElapsedMs(),
                                    perf::kTimedIterations, N);
}

// ============================================================================
// StateBatchOps (orchestration overhead with mixed batches)
// ============================================================================

TEST(StateBench, StateBatchOps_Mixed) {
  auto se3_data = perf::GenerateRandomSE3(N);
  auto vec_data = perf::GenerateRandomVectors<3>(N);

  cuBLASHandle cublas;
  SE3StateBatch se3_batch(cublas,
                          reinterpret_cast<const float*>(se3_data.data()), N);
  VectorStateBatch<3> vec_batch(
      reinterpret_cast<const float*>(vec_data.data()), N);

  std::vector<StateBatch*> batches = {&se3_batch, &vec_batch};

  CudaStream stream;
  StateBatchOps ops(stream.GetStream(), batches);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  size_t reduced = ops.NumReducedStates();
  auto delta = perf::GenerateRandomFloats(reduced, -0.01f, 0.01f);

  DeviceVector<SE3Transform> se3_out(N);
  DeviceVector<Vector<3>> vec_out(N);

  std::vector<const float*> x_ptrs = {
      reinterpret_cast<const float*>(se3_data.data()),
      reinterpret_cast<const float*>(vec_data.data())};
  std::vector<float*> out_ptrs = {
      reinterpret_cast<float*>(se3_out.data()),
      reinterpret_cast<float*>(vec_out.data())};

  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    ops.Plus(stream.GetStream(), x_ptrs, delta, out_ptrs);
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    ops.Plus(stream.GetStream(), x_ptrs, delta, out_ptrs);
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult(
      "StateBatchOps::Plus(SE3+Vec3)", timer.ElapsedMs(),
      perf::kTimedIterations, 2 * N);
}

}  // namespace
}  // namespace cunls
