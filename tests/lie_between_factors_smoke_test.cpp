/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file lie_between_factors_smoke_test.cpp
 * @brief The SO(2) and SO(3) between factors share one residual convention
 *        (solved end to end). Evaluation of every between factor is covered by
 *        evaluate_items_prior_between_test.cpp and lie_jacobian_accuracy_test.cpp.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"

namespace cunls {

namespace {

/** Yaw of R_right after solving R_left = Rot(0.5) (constant), Delta = Rot(0.3). */
template <class Factor, class State, class Rotation, int D>
float SolvedYaw(const std::vector<float> &left, const std::vector<float> &delta,
                const std::vector<float> &right0) {
  std::vector<float> h(left);
  h.insert(h.end(), right0.begin(), right0.end());
  dvector<float> x(h), d(delta);
  dvector<int> const_ids(std::vector<int>{0});
  State states(x.data(), 2, const_ids.data(), 1);
  states.SetNumActiveStates(2, 1);
  Factor factor(reinterpret_cast<const Rotation *>(d.data()), 1);
  factor.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&factor, {states.StateDevicePtr(0), states.StateDevicePtr(1)});
  MinimizerOptions options;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
  GaussNewtonMinimizer minimizer(options);
  CudaStream stream;
  minimizer.Minimize(stream.GetStream(), problem);
  std::vector<float> out(2 * D * D);
  x.CopyToHost(out.data(), out.size());
  const float *r = out.data() + D * D;
  return std::atan2(r[D], r[0]);  // R[1][0], R[0][0]
}

std::vector<float> Rot2(float a) { return {std::cos(a), -std::sin(a), std::sin(a), std::cos(a)}; }
std::vector<float> Rot3z(float a) {
  return {std::cos(a), -std::sin(a), 0.f, std::sin(a), std::cos(a), 0.f, 0.f, 0.f, 1.f};
}

}  // namespace

// Both rotation between factors use residual = Log(R_left^T R_right Delta):
// zero at R_right = R_left Delta^T (yaw 0.5 - 0.3 = 0.2).
TEST(LieBetweenFactorsSmoke, SO2AndSO3ShareTheResidualConvention) {
  const float so2 = SolvedYaw<SO2BetweenFactorBatch, SO2StateBatch, SO2Rotation, 2>(
      Rot2(0.5f), Rot2(0.3f), Rot2(0.f));
  const float so3 = SolvedYaw<SO3BetweenFactorBatch, SO3StateBatch, SO3Rotation, 3>(
      Rot3z(0.5f), Rot3z(0.3f), Rot3z(0.f));
  EXPECT_NEAR(so2, 0.2f, 1e-5f);
  EXPECT_NEAR(so3, 0.2f, 1e-5f);
}

}  // namespace cunls
