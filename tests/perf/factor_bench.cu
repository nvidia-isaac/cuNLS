/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/point_to_plane_factor_batch.h"
#include "cunls/factor/point_to_point_factor_batch.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/factor/se2_between_factor_batch.h"
#include "cunls/factor/se2_prior_factor_batch.h"
#include "cunls/factor/se3_between_factor_batch.h"
#include "cunls/factor/se3_prior_factor_batch.h"
#include "cunls/factor/similarity2_between_factor_batch.h"
#include "cunls/factor/similarity2_prior_factor_batch.h"
#include "cunls/factor/similarity3_between_factor_batch.h"
#include "cunls/factor/similarity3_prior_factor_batch.h"
#include "cunls/factor/sl4_between_factor_batch.h"
#include "cunls/factor/sl4_prior_factor_batch.h"
#include "cunls/factor/so2_between_factor_batch.h"
#include "cunls/factor/so2_prior_factor_batch.h"
#include "cunls/factor/so3_between_factor_batch.h"
#include "cunls/factor/so3_prior_factor_batch.h"
#include "cunls/factor/symmetric_point_to_plane_factor_batch.h"
#include "cunls/factor/vector_between_factor_batch.h"
#include "cunls/factor/weighted_factor_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/perf/perf_utils.h"

namespace cunls {
namespace {

constexpr size_t N = perf::kDefaultN;

void RunFactorBench(const std::string& name, FactorBatch& factor,
                    const dvector<const float*>& state_ptrs,
                    size_t residual_size, size_t jacobian_pitch) {
  size_t num = factor.NumFactors();
  dvector<float> residuals(num * residual_size);
  dvector<float> jacobians(num * residual_size * jacobian_pitch);

  CudaStream stream;

  for (int i = 0; i < perf::kWarmupIterations; ++i) {
    factor.Evaluate(residuals.data(), jacobians.data(), state_ptrs.data(),
                    stream.GetStream());
  }
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  perf::BenchmarkTimer timer;
  timer.Start(stream.GetStream());
  for (int i = 0; i < perf::kTimedIterations; ++i) {
    factor.Evaluate(residuals.data(), jacobians.data(), state_ptrs.data(),
                    stream.GetStream());
  }
  timer.Stop(stream.GetStream());
  perf::BenchmarkTimer::PrintResult(name, timer.ElapsedMs(),
                                    perf::kTimedIterations, num);
}

// ============================================================================
// Vector prior / between factors
// ============================================================================

TEST(FactorBench, PriorVectorFactor_Dim3) {
  auto observations = perf::GenerateRandomVectors<3>(N);
  auto states = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 1);
  VectorStateBatch<3> batch(reinterpret_cast<const float*>(states.data()), N);
  PriorVectorFactorBatch<3> factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("PriorVectorFactor<3>", factor, ptrs, 3, 3);
}

TEST(FactorBench, PriorVectorFactor_Dim6) {
  auto observations = perf::GenerateRandomVectors<6>(N);
  auto states = perf::GenerateRandomVectors<6>(N, 1.0f, perf::kSeed + 1);
  VectorStateBatch<6> batch(reinterpret_cast<const float*>(states.data()), N);
  PriorVectorFactorBatch<6> factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("PriorVectorFactor<6>", factor, ptrs, 6, 6);
}

TEST(FactorBench, VectorBetweenFactor_Dim3) {
  auto deltas = perf::GenerateRandomVectors<3>(N, 0.1f);
  auto left = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 1);
  auto right = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 2);
  VectorStateBatch<3> left_batch(
      reinterpret_cast<const float*>(left.data()), N);
  VectorStateBatch<3> right_batch(
      reinterpret_cast<const float*>(right.data()), N);
  VectorBetweenFactorBatch<3> factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("VectorBetweenFactor<3>", factor, ptrs, 3, 6);
}

TEST(FactorBench, VectorBetweenFactor_Dim6) {
  auto deltas = perf::GenerateRandomVectors<6>(N, 0.1f);
  auto left = perf::GenerateRandomVectors<6>(N, 1.0f, perf::kSeed + 1);
  auto right = perf::GenerateRandomVectors<6>(N, 1.0f, perf::kSeed + 2);
  VectorStateBatch<6> left_batch(
      reinterpret_cast<const float*>(left.data()), N);
  VectorStateBatch<6> right_batch(
      reinterpret_cast<const float*>(right.data()), N);
  VectorBetweenFactorBatch<6> factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("VectorBetweenFactor<6>", factor, ptrs, 6, 12);
}

// ============================================================================
// SO(2) factors
// ============================================================================

TEST(FactorBench, SO2PriorFactor) {
  auto observations = perf::GenerateRandomSO2(N);
  auto states = perf::GenerateRandomSO2(N, 0.5f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SO2StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);
  SO2PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("SO2PriorFactor", factor, ptrs, 1, 1);
}

TEST(FactorBench, SO2BetweenFactor) {
  auto deltas = perf::GenerateRandomSO2(N, 0.1f);
  auto left = perf::GenerateRandomSO2(N, 0.5f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSO2(N, 0.5f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SO2StateBatch left_batch(cublas,
                           reinterpret_cast<const float*>(left.data()), N);
  SO2StateBatch right_batch(cublas,
                            reinterpret_cast<const float*>(right.data()), N);
  SO2BetweenFactorBatch factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("SO2BetweenFactor", factor, ptrs, 1, 2);
}

// ============================================================================
// SO(3) factors
// ============================================================================

TEST(FactorBench, SO3PriorFactor) {
  auto observations = perf::GenerateRandomSO3(N);
  auto states = perf::GenerateRandomSO3(N, 0.3f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SO3StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);
  SO3PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("SO3PriorFactor", factor, ptrs, 3, 3);
}

TEST(FactorBench, SO3BetweenFactor) {
  auto deltas = perf::GenerateRandomSO3(N, 0.1f);
  auto left = perf::GenerateRandomSO3(N, 0.3f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSO3(N, 0.3f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SO3StateBatch left_batch(cublas,
                           reinterpret_cast<const float*>(left.data()), N);
  SO3StateBatch right_batch(cublas,
                            reinterpret_cast<const float*>(right.data()), N);
  SO3BetweenFactorBatch factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("SO3BetweenFactor", factor, ptrs, 3, 6);
}

// ============================================================================
// SE(2) factors
// ============================================================================

TEST(FactorBench, SE2PriorFactor) {
  auto observations = perf::GenerateRandomSE2(N);
  auto states = perf::GenerateRandomSE2(N, 0.3f, 1.0f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SE2StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);
  SE2PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("SE2PriorFactor", factor, ptrs, 3, 3);
}

TEST(FactorBench, SE2BetweenFactor) {
  auto deltas = perf::GenerateRandomSE2(N, 0.1f, 0.1f);
  auto left = perf::GenerateRandomSE2(N, 0.3f, 1.0f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSE2(N, 0.3f, 1.0f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE2StateBatch left_batch(cublas,
                           reinterpret_cast<const float*>(left.data()), N);
  SE2StateBatch right_batch(cublas,
                            reinterpret_cast<const float*>(right.data()), N);
  SE2BetweenFactorBatch factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("SE2BetweenFactor", factor, ptrs, 3, 6);
}

// ============================================================================
// SE(3) factors
// ============================================================================

TEST(FactorBench, SE3PriorFactor) {
  auto observations = perf::GenerateRandomSE3(N);
  auto states = perf::GenerateRandomSE3(N, 0.3f, 1.0f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);
  SE3PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("SE3PriorFactor", factor, ptrs, 6, 6);
}

TEST(FactorBench, SE3BetweenFactor) {
  auto deltas = perf::GenerateRandomSE3(N, 0.1f, 0.1f);
  auto left = perf::GenerateRandomSE3(N, 0.3f, 1.0f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSE3(N, 0.3f, 1.0f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE3StateBatch left_batch(cublas,
                           reinterpret_cast<const float*>(left.data()), N);
  SE3StateBatch right_batch(cublas,
                            reinterpret_cast<const float*>(right.data()), N);
  SE3BetweenFactorBatch factor(cublas, deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("SE3BetweenFactor", factor, ptrs, 6, 12);
}

// ============================================================================
// Similarity2 factors
// ============================================================================

TEST(FactorBench, Similarity2PriorFactor) {
  auto observations = perf::GenerateRandomSim2(N);
  auto states = perf::GenerateRandomSim2(N, 0.3f, perf::kSeed + 1);
  cuBLASHandle cublas;
  Similarity2StateBatch batch(
      cublas, reinterpret_cast<const float*>(states.data()), N);
  Similarity2PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("Similarity2PriorFactor", factor, ptrs, 4, 4);
}

TEST(FactorBench, Similarity2BetweenFactor) {
  auto deltas = perf::GenerateRandomSim2(N, 0.1f);
  auto left = perf::GenerateRandomSim2(N, 0.3f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSim2(N, 0.3f, perf::kSeed + 2);
  cuBLASHandle cublas;
  Similarity2StateBatch left_batch(
      cublas, reinterpret_cast<const float*>(left.data()), N);
  Similarity2StateBatch right_batch(
      cublas, reinterpret_cast<const float*>(right.data()), N);
  Similarity2BetweenFactorBatch factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("Similarity2BetweenFactor", factor, ptrs, 4, 8);
}

// ============================================================================
// Similarity3 factors
// ============================================================================

TEST(FactorBench, Similarity3PriorFactor) {
  auto observations = perf::GenerateRandomSim3(N);
  auto states = perf::GenerateRandomSim3(N, 0.3f, perf::kSeed + 1);
  cuBLASHandle cublas;
  Similarity3StateBatch batch(
      cublas, reinterpret_cast<const float*>(states.data()), N);
  Similarity3PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("Similarity3PriorFactor", factor, ptrs, 7, 7);
}

TEST(FactorBench, Similarity3BetweenFactor) {
  auto deltas = perf::GenerateRandomSim3(N, 0.1f);
  auto left = perf::GenerateRandomSim3(N, 0.3f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSim3(N, 0.3f, perf::kSeed + 2);
  cuBLASHandle cublas;
  Similarity3StateBatch left_batch(
      cublas, reinterpret_cast<const float*>(left.data()), N);
  Similarity3StateBatch right_batch(
      cublas, reinterpret_cast<const float*>(right.data()), N);
  Similarity3BetweenFactorBatch factor(cublas, deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("Similarity3BetweenFactor", factor, ptrs, 7, 14);
}

// ============================================================================
// SL(4) factors
// ============================================================================

TEST(FactorBench, SL4PriorFactor) {
  auto observations = perf::GenerateRandomSL4(N);
  auto states = perf::GenerateRandomSL4(N, 0.1f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SL4StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);
  SL4PriorFactorBatch factor(observations.data(), N);
  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("SL4PriorFactor", factor, ptrs, 15, 15);
}

TEST(FactorBench, SL4BetweenFactor) {
  auto deltas = perf::GenerateRandomSL4(N, 0.05f);
  auto left = perf::GenerateRandomSL4(N, 0.1f, perf::kSeed + 1);
  auto right = perf::GenerateRandomSL4(N, 0.1f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SL4StateBatch left_batch(cublas,
                           reinterpret_cast<const float*>(left.data()), N);
  SL4StateBatch right_batch(cublas,
                            reinterpret_cast<const float*>(right.data()), N);
  SL4BetweenFactorBatch factor(deltas.data(), N);
  auto ptrs = perf::MakeBetweenStatePointers(left_batch, right_batch);
  RunFactorBench("SL4BetweenFactor", factor, ptrs, 15, 30);
}

// ============================================================================
// Point cloud factors (SE(3) state, geometric observations)
// ============================================================================

TEST(FactorBench, PointToPointFactor) {
  auto p_obs = perf::GenerateRandomVectors<3>(N);
  auto q_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 1);
  auto poses = perf::GenerateRandomSE3(N, 0.1f, 0.5f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(poses.data()), N);
  PointToPointFactorBatch factor(p_obs.data(), q_obs.data(), N);
  auto ptrs = perf::MakeStatePointers(batch, N);
  RunFactorBench("PointToPointFactor", factor, ptrs, 3, 6);
}

TEST(FactorBench, PointToPlaneFactor) {
  auto p_obs = perf::GenerateRandomVectors<3>(N);
  auto q_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 1);
  // Generate normals (unit-ish, but doesn't need to be exact for benchmarking)
  auto nq_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 3);
  auto poses = perf::GenerateRandomSE3(N, 0.1f, 0.5f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(poses.data()), N);
  PointToPlaneFactorBatch factor(p_obs.data(), q_obs.data(), nq_obs.data(), N);
  auto ptrs = perf::MakeStatePointers(batch, N);
  RunFactorBench("PointToPlaneFactor", factor, ptrs, 1, 6);
}

TEST(FactorBench, SymmetricPointToPlaneFactor) {
  auto p_obs = perf::GenerateRandomVectors<3>(N);
  auto q_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 1);
  auto np_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 3);
  auto nq_obs = perf::GenerateRandomVectors<3>(N, 1.0f, perf::kSeed + 4);
  auto poses = perf::GenerateRandomSE3(N, 0.1f, 0.5f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(poses.data()), N);
  SymmetricPointToPlaneFactorBatch factor(p_obs.data(), q_obs.data(),
                                          np_obs.data(), nq_obs.data(), N);
  auto ptrs = perf::MakeStatePointers(batch, N);
  RunFactorBench("SymmetricPointToPlaneFactor", factor, ptrs, 1, 6);
}

// ============================================================================
// PnP and Reprojection factors
// ============================================================================

TEST(FactorBench, PnPFactor) {
  auto observations = perf::GenerateRandomVectors<2>(N, 0.5f);
  auto points_world = perf::GenerateRandomVectors<3>(N, 2.0f, perf::kSeed + 1);

  // Single SE3 pose shared by all factors
  auto pose = perf::GenerateRandomSE3(1, 0.1f, 5.0f, perf::kSeed + 2);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(pose.data()), 1);

  PnPFactorBatch factor(observations.data(), points_world.data(), N);

  hvector<const float*> h_ptrs(N);
  const float* pose_ptr =
      reinterpret_cast<const float*>(batch.StateBlockDevicePtr(0));
  for (size_t i = 0; i < N; ++i) h_ptrs[i] = pose_ptr;
  dvector<const float*> ptrs(h_ptrs);

  RunFactorBench("PnPFactor", factor, ptrs, 2, 6);
}

TEST(FactorBench, ReprojectionFactor) {
  auto observations = perf::GenerateRandomVectors<2>(N, 0.5f);
  auto poses = perf::GenerateRandomSE3(N, 0.1f, 5.0f, perf::kSeed + 1);
  auto points = perf::GenerateRandomVectors<3>(N, 2.0f, perf::kSeed + 2);

  cuBLASHandle cublas;
  SE3StateBatch pose_batch(cublas,
                           reinterpret_cast<const float*>(poses.data()), N);
  VectorStateBatch<3> point_batch(
      reinterpret_cast<const float*>(points.data()), N);

  ReprojectionFactorBatch factor(observations.data(), N);

  hvector<const float*> h_ptrs(2 * N);
  for (size_t i = 0; i < N; ++i) {
    h_ptrs[2 * i] = reinterpret_cast<const float*>(
        pose_batch.StateBlockDevicePtr(i));
    h_ptrs[2 * i + 1] = reinterpret_cast<const float*>(
        point_batch.StateBlockDevicePtr(i));
  }
  dvector<const float*> ptrs(h_ptrs);

  RunFactorBench("ReprojectionFactor", factor, ptrs, 2, 9);
}

// ============================================================================
// Weighted factor (wrapper overhead)
// ============================================================================

TEST(FactorBench, WeightedSE3PriorFactor_Uniform) {
  auto observations = perf::GenerateRandomSE3(N);
  auto states = perf::GenerateRandomSE3(N, 0.3f, 1.0f, perf::kSeed + 1);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);

  WeightedFactorBatch<SE3PriorFactorBatch> factor(0.5f, observations.data(), N);

  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("WeightedSE3PriorFactor(uniform)", factor, ptrs, 6, 6);
}

TEST(FactorBench, WeightedSE3PriorFactor_PerFactor) {
  auto observations = perf::GenerateRandomSE3(N);
  auto states = perf::GenerateRandomSE3(N, 0.3f, 1.0f, perf::kSeed + 1);
  auto weights = perf::GenerateRandomFloats(N, 0.1f, 2.0f, perf::kSeed + 3);
  cuBLASHandle cublas;
  SE3StateBatch batch(cublas, reinterpret_cast<const float*>(states.data()), N);

  WeightedFactorBatch<SE3PriorFactorBatch> factor(
      weights.data(), N, observations.data(), N);

  auto ptrs = perf::MakePriorStatePointers(batch);
  RunFactorBench("WeightedSE3PriorFactor(per-factor)", factor, ptrs, 6, 6);
}

}  // namespace
}  // namespace cunls
