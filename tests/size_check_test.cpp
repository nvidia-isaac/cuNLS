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

// Every minimizer runs Problem::CheckSizes() at the start of Minimize: a quick
// host-only check that active sizes were set (batches start with none active)
// and fit their capacities.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/types.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

constexpr size_t kCapacity = 8;

/** A 1D prior problem built at capacity; sizes are left for each test to set. */
struct PriorProblem {
  // Finite contents: the solve tests read every active entry.
  dvector<Vector<1>> states{Vector<1>{0.f}, kCapacity}, targets{Vector<1>{1.f}, kCapacity};
  VectorStateBatch<1> state_batch{reinterpret_cast<const float *>(states.data()), kCapacity};
  PriorVectorFactorBatch<1> prior{targets.data(), kCapacity};
  Problem problem;

  PriorProblem() {
    std::vector<float *> ptrs;
    for (size_t i = 0; i < kCapacity; ++i) ptrs.push_back(state_batch.StateDevicePtr(i));
    problem.AddStateBatch(&state_batch);
    problem.AddFactorBatch(&prior, ptrs);
  }

  void SetSizes(size_t n) {
    state_batch.SetNumActiveStates(n);
    prior.SetNumActiveFactors(n);
  }
};

/** Expects `run` to throw std::invalid_argument whose message contains `what`. */
template <typename F>
void ExpectInvalid(F &&run, const std::string &what) {
  try {
    run();
    FAIL() << "expected std::invalid_argument containing '" << what << "'";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
  }
}

TEST(SizeCheck, EveryMinimizerRejectsUnsetSizes) {
  PriorProblem p;  // constructed, sizes never set: nothing active
  CudaStream stream;
  ExpectInvalid([&] { GaussNewtonMinimizer().Minimize(stream.GetStream(), p.problem); },
                "SetNumActiveFactors");
  ExpectInvalid([&] { LevenbergMarquardtMinimizer().Minimize(stream.GetStream(), p.problem); },
                "SetNumActiveFactors");
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.1f;
  ExpectInvalid([&] { RansacGaussNewtonMinimizer(o).Minimize(stream.GetStream(), p.problem); },
                "SetNumActiveFactors");
}

TEST(SizeCheck, SetSizesSolve) {
  PriorProblem p;
  p.SetSizes(5);
  CudaStream stream;
  EXPECT_NO_THROW(p.problem.CheckSizes());
  EXPECT_NO_THROW(LevenbergMarquardtMinimizer().Minimize(stream.GetStream(), p.problem));
  std::vector<Vector<1>> solved(kCapacity);
  p.states.CopyToHost(solved.data(), kCapacity);
  for (size_t i = 0; i < kCapacity; ++i) {
    EXPECT_NEAR(solved[i][0], i < 5 ? 1.f : 0.f, 1e-4f) << i;  // inactive states untouched
  }
}

TEST(SizeCheck, HostListMustCoverActiveFactors) {
  PriorProblem p;
  p.SetSizes(kCapacity);
  p.problem.SetStatePointers(0, {p.state_batch.StateDevicePtr(0)});  // one pointer for 8
  CudaStream stream;
  ExpectInvalid([&] { GaussNewtonMinimizer().Minimize(stream.GetStream(), p.problem); },
                "SetStatePointers");
}

/** A custom factor that reports more factors than the capacity it gave its base. */
class OverCapacityFactor : public SizedFactorBatch<1, 1> {
 public:
  OverCapacityFactor() : SizedFactorBatch(2) {}
  bool Evaluate(float *, float *, float const *const *, cudaStream_t, const int *,
                size_t) const override {
    return true;
  }
  size_t NumActiveFactors() const override { return 3; }
};

TEST(SizeCheck, NumActiveFactorsAboveCapacityIsRejected) {
  PriorProblem p;
  p.SetSizes(kCapacity);
  OverCapacityFactor bad;
  p.problem.AddFactorBatch(&bad, std::vector<float *>(3, p.state_batch.StateDevicePtr(0)));
  CudaStream stream;
  ExpectInvalid([&] { GaussNewtonMinimizer().Minimize(stream.GetStream(), p.problem); },
                "exceeds Capacity()");
}

}  // namespace
}  // namespace cunls
