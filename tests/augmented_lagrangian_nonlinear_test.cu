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

// AugmentedLagrangianMinimizer on nonlinear Hock-Schittkowski problems (curved
// constraints, Rosenbrock-like valleys at large penalties), with Gauss-Newton
// and Levenberg-Marquardt inner solvers.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/bound_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/sized_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/utils.h"

namespace cunls {
namespace {

// Rows of the HS1/HS2/HS6 functions of x = (x1, x2):
//   kind 0: 10 (x2 - x1²)    kind 1: 1 - x1    kind 2: both rows
__global__ void HsRowsKernel(int kind, float const *const *ptrs, float *res, float *jac, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const float *x = ptrs[t];
  const int rows = kind == 2 ? 2 : 1;
  int r = t * rows;
  if (kind != 1) {
    res[r] = 10.f * (x[1] - x[0] * x[0]);
    if (jac) {
      jac[2 * r] = -20.f * x[0];
      jac[2 * r + 1] = 10.f;
    }
    ++r;
  }
  if (kind != 0) {
    res[r] = 1.f - x[0];
    if (jac) {
      jac[2 * r] = -1.f;
      jac[2 * r + 1] = 0.f;
    }
  }
}

class HsRows : public FactorBatch {
 public:
  explicit HsRows(int kind) : FactorBatch(1), kind_(kind) {}
  size_t ResidualsSize() const override { return kind_ == 2 ? 2 : 1; }
  std::vector<size_t> StateSizes() const override { return {2}; }
  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *, size_t num_factor_ids) const override {
    const int n = static_cast<int>(num_factor_ids == 0 ? NumActiveFactors() : num_factor_ids);
    HsRowsKernel<<<1, 32, 0, stream>>>(kind_, state_pointers, residuals, jacobians, n);
    return cudaGetLastError() == cudaSuccess;
  }

 private:
  int kind_;
};

/** Augmented Lagrangian options whose inner solves run to convergence. */
AugmentedLagrangianMinimizerOptions Options() {
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options = test_utils::ConvergedSolveOptions();
  lm.relative_reduction_tolerance = 1e-12f;
  return test_utils::WithInnerSettings(lm);
}


// HS6: min (1 - x1)²  s.t.  10 (x2 - x1²) = 0, from (-1.2, 1); x* = (1, 1).
TEST(HockSchittkowski, Hs6NonlinearEquality) {
  CudaStream stream;
  dvector<float> x(std::vector<float>{-1.2f, 1.f});
  VectorStateBatch<2> states(x.data(), 1);
  states.SetNumActiveStates(1);
  HsRows objective(1), curve(0);
  objective.SetNumActiveFactors(1);
  curve.SetNumActiveFactors(1);
  ConstraintFactorBatch constraint(&curve, ConstraintKind::kEquality);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&objective, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&constraint, {states.StateDevicePtr(0)});
  const auto summary = AugmentedLagrangianMinimizer(Options()).Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  EXPECT_LE(summary.max_violation, 1e-4f);
  std::vector<float> result(2);
  x.CopyToHost(result.data(), 2);
  EXPECT_NEAR(result[0], 1.f, 1e-3f);
  EXPECT_NEAR(result[1], 1.f, 1e-3f);
}

// HS2: min 100 (x2 - x1²)² + (1 - x1)²  s.t.  x2 >= 1.5; x* = (1.2243707, 1.5)
// (started in its basin: from (-2, 1) a local method may stop at the other
// KKT point on the bound, x1 = -1.2210).
TEST(HockSchittkowski, Hs2ActiveBound) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  CudaStream stream;
  dvector<float> x(std::vector<float>{2.f, 1.f});
  dvector<float> lo(std::vector<float>{-kInf, 1.5f}), hi(std::vector<float>{kInf, kInf});
  VectorStateBatch<2> states(x.data(), 1);
  states.SetNumActiveStates(1);
  HsRows objective(2);
  objective.SetNumActiveFactors(1);
  BoundFactorBatch<2> bounds(lo.data(), hi.data(), 1);
  bounds.SetNumActiveFactors(1);
  Problem problem;
  problem.AddStateBatch(&states);
  problem.AddFactorBatch(&objective, {states.StateDevicePtr(0)});
  problem.AddFactorBatch(&bounds, {states.StateDevicePtr(0)});
  const auto summary = AugmentedLagrangianMinimizer(Options()).Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged);
  std::vector<float> result(2);
  x.CopyToHost(result.data(), 2);
  EXPECT_NEAR(result[0], 1.2243707f, 2e-4f);
  EXPECT_NEAR(result[1], 1.5f, 2e-4f);
}


}  // namespace
}  // namespace cunls
