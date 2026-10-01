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

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/manifold.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/prior_factor_batch.h"
#include "cunls/factor/sized_factor_batch.h"
#include "cunls/minimizer/jacobian_mode.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/vector_state_batch.h"
#include "utils/datasets.h"
#include "utils/report.h"
#include "utils/validation.h"

using cunls::dvector;
using cunls::LogError;
using cunls::Vector;

namespace {

// ---------------------------------------------------------------------------
// Custom factor kernel
// ---------------------------------------------------------------------------
// This kernel implements a tiny 1D "between" constraint:
//   residual_i = (x_{i+1} - x_i) - measurement_i
//
// One thread evaluates one *item*: a factor evaluated against its own set of
// state blocks (see FactorBatch::Evaluate). Regular minimizers pass
// factor_ids == nullptr and num_items == num_factors, so item idx is factor
// idx; the RANSAC minimizers evaluate many items per factor. For item idx:
//   measurement:  measurements[factor_ids ? factor_ids[idx] : idx % num_factors]
//   states:       state_pointers[2*idx + 0] -> x_i, state_pointers[2*idx + 1] -> x_{i+1}
//   outputs:      residuals[idx], jacobians[2*idx .. 2*idx + 1]
//
// Jacobian layout is row-major per item. Since residual dimension is 1 and
// state sizes are [1, 1], each item contributes two Jacobian values:
//   [dr/dx_i, dr/dx_{i+1}] = [-1, +1]
__global__ void ScalarDifferenceKernel(const float *measurements, const int *factor_ids,
                                       size_t num_factors, float const *const *state_pointers,
                                       float *residuals, float *jacobians, size_t num_items) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_items) {
    return;
  }

  const size_t factor = factor_ids != nullptr ? factor_ids[idx] : idx % num_factors;
  const float *left = state_pointers[idx * 2];
  const float *right = state_pointers[idx * 2 + 1];
  const float residual = (right[0] - left[0]) - measurements[factor];

  if (residuals != nullptr) {
    residuals[idx] = residual;
  }
  if (jacobians != nullptr) {
    jacobians[idx * 2] = -1.0f;
    jacobians[idx * 2 + 1] = 1.0f;
  }
}

// ---------------------------------------------------------------------------
// Custom FactorBatch implementation
// ---------------------------------------------------------------------------
// SizedFactorBatch<1, 1, 1> means:
// - residual size: 1
// - first state block tangent size: 1
// - second state block tangent size: 1
//
// The class only stores pointers to device memory (measurements) and launches
// the kernel in Evaluate(). cuNLS handles assembly and optimization using
// the residuals/Jacobians we provide here.
class ScalarDifferenceFactorBatch : public cunls::SizedFactorBatch<1, 1, 1> {
 public:
  ScalarDifferenceFactorBatch(const float *measurements, size_t num_factors)
      : measurements_(measurements), num_factors_(num_factors) {}

  bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_items = num_factor_ids == 0 ? num_factors_ : num_factor_ids;
    constexpr int kBlockSize = 256;
    const int grid_size = static_cast<int>((num_items + kBlockSize - 1) / kBlockSize);
    ScalarDifferenceKernel<<<grid_size, kBlockSize, 0, stream>>>(
        measurements_, factor_ids, num_factors_, state_pointers, residuals, jacobians, num_items);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
  }

  size_t NumFactors() const final { return num_factors_; }

 private:
  const float *measurements_;
  size_t num_factors_;
};

// ---------------------------------------------------------------------------
// Part 2: the same factor, but with only a residual implemented.
// ---------------------------------------------------------------------------
// Deriving a closed-form Jacobian by hand isn't always worth it -- for
// prototyping, or for factors whose residual is awkward to differentiate,
// cuNLS can compute the Jacobian for you via finite differences on the
// manifold tangent space of each referenced state block (see
// `cunls/minimizer/jacobian_mode.h`). All a factor has to do is support
// residual-only evaluation (`jacobians == nullptr`), which every FactorBatch
// must already do for cost-only evaluation.
//
// This kernel is a copy of ScalarDifferenceKernel with the Jacobian branch
// deleted entirely -- there is nothing else to write.
__global__ void ScalarDifferenceResidualOnlyKernel(const float *measurements, const int *factor_ids,
                                                   size_t num_factors,
                                                   float const *const *state_pointers,
                                                   float *residuals, size_t num_items) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_items) {
    return;
  }

  const size_t factor = factor_ids != nullptr ? factor_ids[idx] : idx % num_factors;
  const float *left = state_pointers[idx * 2];
  const float *right = state_pointers[idx * 2 + 1];
  if (residuals != nullptr) {
    residuals[idx] = (right[0] - left[0]) - measurements[factor];
  }
}

// Same SizedFactorBatch<1, 1, 1> shape as Part 1, but Evaluate() only ever
// writes residuals -- it doesn't even look at the `jacobians` argument.
// Registering this factor group with JacobianMode::kNumeric (see main()
// below) tells the minimizer to fill in the Jacobian itself by perturbing
// x_i/x_{i+1} with StateBatch::Plus and differencing the residual, instead
// of calling into a Jacobian code path that doesn't exist here.
class ScalarDifferenceResidualOnlyFactorBatch : public cunls::SizedFactorBatch<1, 1, 1> {
 public:
  ScalarDifferenceResidualOnlyFactorBatch(const float *measurements, size_t num_factors)
      : measurements_(measurements), num_factors_(num_factors) {}

  bool Evaluate(float *residuals, float * /*jacobians*/, float const *const *state_pointers,
                cudaStream_t stream, const int *factor_ids = nullptr,
                size_t num_factor_ids = 0) const final {
    const size_t num_items = num_factor_ids == 0 ? num_factors_ : num_factor_ids;
    constexpr int kBlockSize = 256;
    const int grid_size = static_cast<int>((num_items + kBlockSize - 1) / kBlockSize);
    ScalarDifferenceResidualOnlyKernel<<<grid_size, kBlockSize, 0, stream>>>(
        measurements_, factor_ids, num_factors_, state_pointers, residuals, num_items);
    THROW_ON_CUDA_ERROR(cudaGetLastError());
    return true;
  }

  size_t NumFactors() const final { return num_factors_; }

 private:
  const float *measurements_;
  size_t num_factors_;
};

// Solves a 1D chain x_0 -- x_1 -- ... -- x_{N-1} from its consecutive
// differences: N-1 custom difference factors plus a prior anchoring x_0.
// Part 1 registers the analytic-Jacobian factor; Part 2 the residual-only one,
// with JacobianMode::kNumeric so cuNLS differentiates it numerically.
int RunChainExample(const char *title, bool use_numeric_jacobian) {
  const size_t num_states = 256;
  const size_t num_diff_factors = num_states - 1;

  // 1. Synthetic data: ground truth, a noisy initial guess, exact differences.
  const examples::ScalarChainScene scene = examples::MakeScalarChainScene(num_states);

  // 2. Upload to the GPU. Without the anchor, adding a constant to every
  //    state leaves all differences unchanged (rank-deficient system).
  dvector<Vector<1>> states(scene.initial_states);
  dvector<float> differences(scene.differences);
  dvector<Vector<1>> anchor(std::vector<Vector<1>>{scene.gt_states[0]});

  // 3. One state batch with all scalar states.
  cunls::VectorStateBatch<1> state_batch(reinterpret_cast<const float *>(states.data()),
                                         num_states);

  // 4. Factors: the custom difference factor (one of the two classes above)
  //    reads [x_i, x_{i+1}]; the shipped prior reads x_0.
  ScalarDifferenceFactorBatch analytic_factor(differences.data(), num_diff_factors);
  ScalarDifferenceResidualOnlyFactorBatch residual_only_factor(differences.data(),
                                                               num_diff_factors);
  cunls::PriorFactorBatch<cunls::manifold::Vector<1>> anchor_factor(anchor.data(), 1);
  std::vector<float *> diff_pointers;
  for (size_t i = 0; i < num_diff_factors; ++i) {
    diff_pointers.push_back(state_batch.StateBlockDevicePtr(i));
    diff_pointers.push_back(state_batch.StateBlockDevicePtr(i + 1));
  }

  // 5. The problem. The per-group JacobianMode::kNumeric override makes cuNLS
  //    differentiate the residual-only factor; the prior stays analytic, so one
  //    Problem can mix both modes.
  cunls::Problem problem;
  problem.AddStateBatch(&state_batch);
  if (use_numeric_jacobian) {
    problem.AddFactorBatch(&residual_only_factor, diff_pointers, cunls::JacobianMode::kNumeric);
  } else {
    problem.AddFactorBatch(&analytic_factor, diff_pointers);
  }
  problem.AddFactorBatch(&anchor_factor, {state_batch.StateBlockDevicePtr(0)});
  if (!problem.CheckConsistency()) {
    std::cerr << "Problem consistency check failed\n";
    return 1;
  }

  // 6. Solve with Levenberg-Marquardt.
  cunls::LevenbergMarquardtMinimizerOptions options;
  options.base_options.max_num_iterations = 50;
  options.base_options.state_tolerance = 1e-8f;
  options.base_options.cost_tolerance = 1e-8f;
  options.initial_lambda = 1e-3f;
  cunls::LevenbergMarquardtMinimizer minimizer(options);
  cunls::CudaStream stream;
  const cunls::MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

  // 7. Read back and compare with the ground truth.
  std::vector<Vector<1>> final_states(num_states);
  states.CopyToHost(final_states.data(), num_states);
  const float mse_before = examples::ComputeVectorMSE(scene.initial_states, scene.gt_states);
  const float mse_after = examples::ComputeVectorMSE(final_states, scene.gt_states);
  examples::PrintTitle(title);
  examples::PrintSummary(summary);
  examples::PrintChange("State MSE", mse_before, mse_after);

  // Numeric Jacobians are float32 finite differences, so Part 2 reaches the
  // same optimum with a slightly looser cost tolerance.
  const float cost_tolerance = use_numeric_jacobian ? 5e-4f : 1e-5f;
  return examples::QualityExitCode(summary.final_cost <= cost_tolerance &&
                                   mse_after <= mse_before * 0.02f);
}

}  // namespace

int main() {
  try {
    // Part 1: a factor that implements both the residual and its analytic
    // Jacobian by hand -- the fastest option, worth the extra derivation
    // effort for factors that ship at scale.
    const int part1_status = RunChainExample("Part 1: analytic Jacobian", false);

    std::cout << "\n";

    // Part 2: the same factor family, but only the residual is implemented.
    // cuNLS fills in the Jacobian via finite differences -- the fastest way
    // to get a new factor working, at the cost of extra Jacobian-evaluation
    // time (several times slower than the analytic kernel above; see the
    // "Numeric (finite-difference) Jacobians" page in the docs for
    // benchmarks). Good for prototyping, or factors where a closed-form
    // derivative isn't worth deriving.
    const int part2_status = RunChainExample("Part 2: residual-only, numeric Jacobian", true);

    return (part1_status != 0) ? part1_status : part2_status;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return 3;
  }
}
