/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/prior_vector_factor_batch.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/perf/perf_utils.h"

namespace cunls {
namespace {

constexpr size_t kN = 10000;

class ResidualBatchBench : public ::testing::Test {
 protected:
  template <int Dim>
  void RunBenchmark(const std::string& name, LossFunctionBatch* loss,
                    bool compute_jacobians) {
    auto obs = perf::GenerateRandomVectors<Dim>(kN, 1.0f);
    auto state_data = perf::GenerateRandomVectors<Dim>(kN, 2.0f);
    VectorStateBatch<Dim> states(reinterpret_cast<const float*>(state_data.data()), kN);
    auto state_ptrs = perf::MakePriorStatePointers(states);
    PriorVectorFactorBatch<Dim> factor(obs.data(), kN);

    constexpr int residual_size = Dim;
    constexpr int num_cols = Dim;
    dvector<float> residuals(kN * residual_size);
    dvector<float> jacobians(compute_jacobians ? kN * residual_size * num_cols : 0);
    dvector<float> cost(kN);
    dvector<float> workspace(ResidualBatchWorkspaceNumFloats(kN));

    ResidualBatch rb(&factor, loss);
    CudaStream stream;

    float* jac_ptr = compute_jacobians ? jacobians.data() : nullptr;

    for (int i = 0; i < perf::kWarmupIterations; ++i) {
      rb.Evaluate(stream.GetStream(), workspace.data(), residuals.data(),
                  state_ptrs.data(), cost.data(), jac_ptr);
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    perf::BenchmarkTimer timer;
    timer.Start(stream.GetStream());
    for (int i = 0; i < perf::kTimedIterations; ++i) {
      rb.Evaluate(stream.GetStream(), workspace.data(), residuals.data(),
                  state_ptrs.data(), cost.data(), jac_ptr);
    }
    timer.Stop(stream.GetStream());
    perf::BenchmarkTimer::PrintResult(name, timer.ElapsedMs(),
                                      perf::kTimedIterations, kN);
  }
};

// ---- Trivial loss (no robustification) ----

TEST_F(ResidualBatchBench, TrivialLoss_Dim2_CostOnly) {
  RunBenchmark<2>("ResidualBatch::TrivialLoss_Dim2_CostOnly", nullptr, false);
}

TEST_F(ResidualBatchBench, TrivialLoss_Dim2_WithJacobians) {
  RunBenchmark<2>("ResidualBatch::TrivialLoss_Dim2_WithJacobians", nullptr, true);
}

TEST_F(ResidualBatchBench, TrivialLoss_Dim6_CostOnly) {
  RunBenchmark<6>("ResidualBatch::TrivialLoss_Dim6_CostOnly", nullptr, false);
}

TEST_F(ResidualBatchBench, TrivialLoss_Dim6_WithJacobians) {
  RunBenchmark<6>("ResidualBatch::TrivialLoss_Dim6_WithJacobians", nullptr, true);
}

TEST_F(ResidualBatchBench, TrivialLoss_Dim15_CostOnly) {
  RunBenchmark<15>("ResidualBatch::TrivialLoss_Dim15_CostOnly", nullptr, false);
}

TEST_F(ResidualBatchBench, TrivialLoss_Dim15_WithJacobians) {
  RunBenchmark<15>("ResidualBatch::TrivialLoss_Dim15_WithJacobians", nullptr, true);
}

// ---- Huber loss (robust path) ----

TEST_F(ResidualBatchBench, HuberLoss_Dim2_CostOnly) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<2>("ResidualBatch::HuberLoss_Dim2_CostOnly", &loss, false);
}

TEST_F(ResidualBatchBench, HuberLoss_Dim2_WithJacobians) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<2>("ResidualBatch::HuberLoss_Dim2_WithJacobians", &loss, true);
}

TEST_F(ResidualBatchBench, HuberLoss_Dim6_CostOnly) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<6>("ResidualBatch::HuberLoss_Dim6_CostOnly", &loss, false);
}

TEST_F(ResidualBatchBench, HuberLoss_Dim6_WithJacobians) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<6>("ResidualBatch::HuberLoss_Dim6_WithJacobians", &loss, true);
}

TEST_F(ResidualBatchBench, HuberLoss_Dim15_CostOnly) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<15>("ResidualBatch::HuberLoss_Dim15_CostOnly", &loss, false);
}

TEST_F(ResidualBatchBench, HuberLoss_Dim15_WithJacobians) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark<15>("ResidualBatch::HuberLoss_Dim15_WithJacobians", &loss, true);
}

}  // namespace
}  // namespace cunls
