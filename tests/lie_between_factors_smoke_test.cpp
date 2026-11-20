/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file lie_between_factors_smoke_test.cpp
 * @brief The SO(2) and SO(3) between factors share one residual convention
 *        (solved end to end), and the SL(4) between residual applies Delta on
 *        the left. Evaluation of every between factor is covered by
 *        evaluate_items_prior_between_test.cpp and lie_jacobian_accuracy_test.cpp.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/sl4_between_factor_batch.h"
#include "cunls/factor/between/so2_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/math/sl_lie_math.h"
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

/** A B of row-major 4x4 matrices, accumulated in double. */
std::vector<float> Mul4(const std::vector<float> &A, const std::vector<float> &B) {
  std::vector<float> C(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) sum += double(A[4 * i + k]) * B[4 * k + j];
      C[4 * i + j] = static_cast<float>(sum);
    }
  }
  return C;
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

// SL4BetweenFactorBatch's residual is Vee(Log(Delta * T_left^{-1} * T_right)),
// Delta on the left. T_left = Exp(dense tangent) exercises the 4x4 inverse;
// T_right = T_left M, so T_left^{-1} T_right = M = I + 0.125 e_01, which does not
// commute with Delta = diag(2, 0.5, 1, 1): Delta M has 0.25 at (0, 1), M Delta
// 0.0625. The residual must match the log of Delta M built on the host, not
// that of M Delta.
TEST(LieBetweenFactorsSmoke, SL4BetweenAppliesDeltaOnTheLeft) {
  CudaStream stream;
  std::vector<float> xi_left(15);
  for (int k = 0; k < 15; ++k) xi_left[k] = 0.25f * std::sin(1.7f * (k + 1));
  dvector<float> d_xi_left(xi_left), d_left(16);
  ComputeExpSL4(stream.GetStream(), d_xi_left.data(), 15, 4, 16, 1, d_left.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> left(16);
  d_left.CopyToHost(left.data(), left.size());

  std::vector<float> M(16, 0.f), delta(16, 0.f);
  for (int k = 0; k < 4; ++k) M[5 * k] = 1.f;
  M[1] = 0.125f;
  delta[0] = 2.f;
  delta[5] = 0.5f;
  delta[10] = delta[15] = 1.f;
  std::vector<float> states = left;
  const std::vector<float> right = Mul4(left, M);
  states.insert(states.end(), right.begin(), right.end());

  dvector<float> d_states(states), d_delta(delta), d_residual(15);
  dvector<float *> d_pointers(std::vector<float *>{d_states.data(), d_states.data() + 16});
  SL4BetweenFactorBatch factor(reinterpret_cast<const SL4Transform *>(d_delta.data()), 1);
  factor.SetNumActiveFactors(1);
  factor.Evaluate(d_residual.data(), nullptr, d_pointers.data(), stream.GetStream());

  // Logs of Delta M (the documented order) and M Delta (the reversed one).
  std::vector<float> products = Mul4(delta, M);
  const std::vector<float> reversed = Mul4(M, delta);
  products.insert(products.end(), reversed.begin(), reversed.end());
  dvector<float> d_products(products), d_logs(2 * 15);
  ComputeLogSL4(stream.GetStream(), d_products.data(), 4, 16, 15, 2, d_logs.data());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  std::vector<float> residual(15), logs(2 * 15);
  d_residual.CopyToHost(residual.data(), residual.size());
  d_logs.CopyToHost(logs.data(), logs.size());

  float to_expected = 0.f, to_reversed = 0.f;
  for (int k = 0; k < 15; ++k) {
    to_expected = std::max(to_expected, std::fabs(residual[k] - logs[k]));
    to_reversed = std::max(to_reversed, std::fabs(residual[k] - logs[15 + k]));
  }
  EXPECT_LT(to_expected, 1e-5f) << "residual != Log(Delta * T_left^{-1} * T_right)";
  EXPECT_GT(to_reversed, 0.05f) << "the two operand orders must give different residuals";
}

}  // namespace cunls
