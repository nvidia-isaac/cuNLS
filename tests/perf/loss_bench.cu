/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/robustifier/arctan_loss_function_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/robustifier/huber_loss_function_batch.h"
#include "cunls/robustifier/scaled_loss_function_batch.h"
#include "cunls/robustifier/soft_lone_loss_function_batch.h"
#include "cunls/robustifier/tolerant_loss_function_batch.h"
#include "cunls/robustifier/trivial_loss_function_batch.h"
#include "cunls/robustifier/tukey_loss_function_batch.h"
#include "tests/perf/perf_utils.h"

namespace cunls {
namespace {

class LossBench : public ::testing::Test {
 protected:
  void SetUp() override {
    squared_errors_ = perf::GenerateRandomFloats(N_, 0.0f, 100.0f);
    rho_.resize(N_);
  }

  void RunBenchmark(const std::string& name,
                    const LossFunctionBatch& loss) {
    CudaStream stream;
    for (int i = 0; i < perf::kWarmupIterations; ++i) {
      loss.Evaluate(squared_errors_.data(), rho_.data(), N_, stream.GetStream());
    }
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));

    perf::BenchmarkTimer timer;
    timer.Start(stream.GetStream());
    for (int i = 0; i < perf::kTimedIterations; ++i) {
      loss.Evaluate(squared_errors_.data(), rho_.data(), N_, stream.GetStream());
    }
    timer.Stop(stream.GetStream());
    perf::BenchmarkTimer::PrintResult(name, timer.ElapsedMs(),
                                      perf::kTimedIterations, N_);
  }

  static constexpr size_t N_ = perf::kLossN;
  DeviceVector<float> squared_errors_;
  DeviceVector<float3> rho_;
};

TEST_F(LossBench, TrivialLoss) {
  TrivialLossFunctionBatch loss;
  RunBenchmark("TrivialLoss::Evaluate", loss);
}

TEST_F(LossBench, HuberLoss) {
  HuberLossFunctionBatch loss(1.0f);
  RunBenchmark("HuberLoss::Evaluate", loss);
}

TEST_F(LossBench, CauchyLoss) {
  CauchyLossFunctionBatch loss(1.0f, 1.0f);
  RunBenchmark("CauchyLoss::Evaluate", loss);
}

TEST_F(LossBench, TukeyLoss) {
  TukeyLossFunctionBatch loss(4.685f);
  RunBenchmark("TukeyLoss::Evaluate", loss);
}

TEST_F(LossBench, ArctanLoss) {
  ArctanLossFunctionBatch loss(1.0f, 1.0f);
  RunBenchmark("ArctanLoss::Evaluate", loss);
}

TEST_F(LossBench, SoftLOneLoss) {
  SoftLOneLossFunctionBatch loss(1.0f, 1.0f);
  RunBenchmark("SoftLOneLoss::Evaluate", loss);
}

TEST_F(LossBench, TolerantLoss) {
  TolerantLossFunctionBatch loss(0.5f, 1.0f);
  RunBenchmark("TolerantLoss::Evaluate", loss);
}

TEST_F(LossBench, ScaledHuberLoss) {
  ScaledLossFunctionBatch<HuberLossFunctionBatch> loss(2.0f, 1.0f);
  RunBenchmark("ScaledHuberLoss::Evaluate", loss);
}

}  // namespace
}  // namespace cunls
