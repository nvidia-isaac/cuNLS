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

// Reusable buffers (docs/design/reusable_buffers.md): factors, states and
// connectivity bound once at capacity, rewritten in place between solves.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/ransac_minimizer.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace {

template <typename T>
std::vector<T> ToHost(const dvector<T> &d, size_t n) {
  std::vector<T> h(n);
  if (n > 0) d.CopyToHost(h.data(), n);
  return h;
}

/** One frame of a linear 3D chain-with-loops problem; the solution is `gt`. */
struct Frame {
  std::vector<Vector<3>> gt, initial;
  std::vector<Vector<3>> deltas;  ///< gt[i] - gt[j] per edge [i, j].
  std::vector<int> edges;         ///< [i, j] per edge.
};

Frame MakeFrame(int num_states, int extra_edges, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> pos(-10.f, 10.f), noise(-0.5f, 0.5f);
  std::uniform_int_distribution<int> pick(0, num_states - 1);
  Frame f;
  f.gt.resize(num_states);
  f.initial.resize(num_states);
  for (int i = 0; i < num_states; ++i) {
    for (int k = 0; k < 3; ++k) {
      f.gt[i][k] = pos(rng);
      f.initial[i][k] = f.gt[i][k] + (i == 0 ? 0.f : noise(rng));
    }
  }
  // A random spanning chain (shuffled order) plus random extra edges.
  std::vector<int> order(num_states);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin() + 1, order.end(), rng);
  auto add = [&](int i, int j) {
    f.edges.push_back(i);
    f.edges.push_back(j);
    Vector<3> d;
    // VectorBetweenFactorBatch: r = x_left - x_right - delta.
    for (int k = 0; k < 3; ++k) d[k] = f.gt[i][k] - f.gt[j][k];
    f.deltas.push_back(d);
  };
  for (int i = 1; i < num_states; ++i) add(order[i - 1], order[i]);
  for (int e = 0; e < extra_edges; ++e) {
    int i = pick(rng), j = pick(rng);
    if (i != j) add(i, j);
  }
  return f;
}

float MaxError(const std::vector<Vector<3>> &a, const std::vector<Vector<3>> &b) {
  float e = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    for (int k = 0; k < 3; ++k) e = std::max(e, std::fabs(a[i][k] - b[i][k]));
  }
  return e;
}

LevenbergMarquardtMinimizerOptions SolverOptions() {
  LevenbergMarquardtMinimizerOptions o;
  o.base_options.max_num_iterations = 20;
  o.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseLDLT;
  return o;
}

/** Solves `frame` with a problem built from scratch at exactly its size. */
std::vector<Vector<3>> SolveFresh(const Frame &frame) {
  const size_t n = frame.gt.size(), m = frame.deltas.size();
  dvector<Vector<3>> states(frame.initial), deltas(frame.deltas),
      anchor(std::vector<Vector<3>>{frame.gt[0]});
  VectorStateBatch<3> state_batch(reinterpret_cast<const float *>(states.data()), n);
  state_batch.SetNumStateBlocks(state_batch.Capacity(), state_batch.ConstCapacity());
  VectorBetweenFactorBatch<3> between(deltas.data(), m);
  between.SetNumFactors(between.Capacity());
  PriorVectorFactorBatch<3> prior(anchor.data(), 1);
  prior.SetNumFactors(1);
  std::vector<float *> ptrs;
  for (int idx : frame.edges) ptrs.push_back(state_batch.StateBlockDevicePtr(idx));
  Problem problem;
  problem.AddStateBatch(&state_batch);
  problem.AddFactorBatch(&between, ptrs);
  problem.AddFactorBatch(&prior, {state_batch.StateBlockDevicePtr(0)});
  CudaStream stream;
  LevenbergMarquardtMinimizer(SolverOptions()).Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return ToHost(states, n);
}

/** Buffers bound once at capacity; each frame rewrites contents and sizes. */
struct BoundProblem {
  static constexpr int kCapStates = 300, kCapEdges = 900;
  enum class Form { kHostList, kDevicePointers, kDeviceIndices };

  Form form;
  dvector<Vector<3>> states{kCapStates}, deltas{kCapEdges};
  dvector<Vector<3>> anchor{1};
  dvector<int> indices{2 * kCapEdges}, anchor_index{1};
  dvector<float *> pointers{2 * kCapEdges};
  VectorStateBatch<3> state_batch{reinterpret_cast<const float *>(states.data()), kCapStates};
  VectorBetweenFactorBatch<3> between{deltas.data(), kCapEdges};
  PriorVectorFactorBatch<3> prior{anchor.data(), 1};
  Problem problem;
  LevenbergMarquardtMinimizer minimizer;
  CudaStream stream;

  explicit BoundProblem(Form f, const LevenbergMarquardtMinimizerOptions &options = SolverOptions())
      : form(f), minimizer(options) {
    THROW_ON_CUDA_ERROR(cudaMemset(anchor_index.data(), 0, sizeof(int)));
    prior.SetNumFactors(1);  // the anchor; states and edges are sized per frame
    problem.AddStateBatch(&state_batch);
    switch (form) {
      case Form::kHostList:
        problem.AddFactorBatch(
            &between, std::vector<float *>(2 * kCapEdges, state_batch.StateBlockDevicePtr(0)));
        break;
      case Form::kDevicePointers:
        problem.AddFactorBatch(&between, pointers.data());
        break;
      case Form::kDeviceIndices:
        problem.AddFactorBatch(&between, {&state_batch, &state_batch}, indices.data());
        break;
    }
    problem.AddFactorBatch(&prior, {&state_batch}, anchor_index.data());
  }

  std::vector<Vector<3>> Solve(const Frame &frame) {
    const size_t n = frame.gt.size(), m = frame.deltas.size();
    // 1. Rewrite contents in place (here from the host).
    states.CopyFromHost(frame.initial.data(), n);
    deltas.CopyFromHost(frame.deltas.data(), m);
    anchor.CopyFromHost(&frame.gt[0], 1);
    std::vector<float *> ptrs;
    for (int idx : frame.edges) ptrs.push_back(state_batch.StateBlockDevicePtr(idx));
    // 2. Sizes.
    state_batch.SetNumStateBlocks(n);
    between.SetNumFactors(m);
    // 3. Connectivity.
    switch (form) {
      case Form::kHostList:
        problem.SetStatePointers(0, ptrs);
        break;
      case Form::kDevicePointers:
        pointers.CopyFromHost(ptrs.data(), ptrs.size());
        break;
      case Form::kDeviceIndices:
        indices.CopyFromHost(frame.edges.data(), frame.edges.size());
        break;
    }
    EXPECT_TRUE(problem.Validate(stream.GetStream()));
    minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    return ToHost(states, n);
  }
};

class FrameLoopTest : public ::testing::TestWithParam<BoundProblem::Form> {};

TEST_P(FrameLoopTest, EveryFrameMatchesAFreshProblem) {
  BoundProblem bound(GetParam());
  // Sizes grow, shrink and grow again; connectivity is new every frame.
  const int sizes[][2] = {{120, 200}, {300, 600}, {40, 10}, {250, 400}, {300, 0}};
  uint32_t seed = 11;
  for (const auto &s : sizes) {
    const Frame frame = MakeFrame(s[0], s[1], seed++);
    const auto solved = bound.Solve(frame);
    const auto fresh = SolveFresh(frame);
    // Float32 solve of a long chain: a few 1e-3 off the truth (the 300-state
    // pure chain is the worst conditioned); the bound problem must match a
    // fresh one much more tightly.
    EXPECT_LT(MaxError(solved, frame.gt), 5e-2f) << "frame with " << s[0] << " states";
    EXPECT_LT(MaxError(solved, fresh), 1e-4f) << "frame with " << s[0] << " states";
  }
}

INSTANTIATE_TEST_SUITE_P(Forms, FrameLoopTest,
                         ::testing::Values(BoundProblem::Form::kHostList,
                                           BoundProblem::Form::kDevicePointers,
                                           BoundProblem::Form::kDeviceIndices));

TEST(DynamicProblem, ValidationCatchesBadConnectivity) {
  BoundProblem bound(BoundProblem::Form::kDeviceIndices);
  const Frame frame = MakeFrame(50, 20, 3);
  bound.Solve(frame);
  CudaStream stream;
  EXPECT_TRUE(bound.problem.Validate(stream.GetStream()));
  EXPECT_TRUE(bound.problem.CheckConsistency());

  // An index beyond the active states.
  int bad = 50;
  bound.indices.CopyFromHost(&bad, 1);
  EXPECT_FALSE(bound.problem.Validate(stream.GetStream()));
  EXPECT_FALSE(bound.problem.CheckConsistency());

  // States no factor reads: keep only the first between factor.
  bound.indices.CopyFromHost(frame.edges.data(), frame.edges.size());
  EXPECT_TRUE(bound.problem.Validate(stream.GetStream()));
  bound.between.SetNumFactors(1);
  EXPECT_FALSE(bound.problem.Validate(stream.GetStream()));
}

TEST(DynamicProblem, HostListMustCoverTheActiveFactors) {
  BoundProblem bound(BoundProblem::Form::kHostList);
  bound.problem.SetStatePointers(
      0, std::vector<float *>(2 * 10, bound.state_batch.StateBlockDevicePtr(0)));
  bound.between.SetNumFactors(11);
  EXPECT_FALSE(bound.problem.CheckConsistency());
  EXPECT_THROW(bound.problem.SetStatePointers(1, {}), std::logic_error);  // index-table batch
}

TEST(DynamicProblem, NumericJacobiansFollowConnectivityChanges) {
  // The same minimizer, numeric Jacobians, connectivity rewritten between
  // solves: every solve must match a fresh numeric solve.
  LevenbergMarquardtMinimizerOptions o = SolverOptions();
  o.base_options.jacobian_mode = JacobianMode::kNumeric;
  BoundProblem bound(BoundProblem::Form::kDeviceIndices, o);
  for (uint32_t seed : {21u, 22u, 23u}) {
    const Frame frame = MakeFrame(80, 40, seed);
    const auto solved = bound.Solve(frame);
    EXPECT_LT(MaxError(solved, frame.gt), 1e-2f) << "seed " << seed;
  }
}

TEST(DynamicProblem, RansacWithDeviceIndexTable) {
  // Per-frame PnP: one pose, matches rewritten in place, index table all zeros.
  constexpr int kCap = 2000;
  dvector<SE3Transform> pose(1);
  dvector<Vector<2>> obs(kCap);
  dvector<Vector<3>> pts(kCap);
  dvector<int> pose_index(kCap);
  THROW_ON_CUDA_ERROR(cudaMemset(pose_index.data(), 0, kCap * sizeof(int)));
  cuBLASHandle cublas;
  SE3StateBatch pose_state(cublas, reinterpret_cast<const float *>(pose.data()), 1);
  pose_state.SetNumStateBlocks(pose_state.Capacity(), pose_state.ConstCapacity());
  PnPFactorBatch pnp(obs.data(), pts.data(), kCap);
  pnp.SetNumFactors(pnp.Capacity());
  Problem problem;
  problem.AddStateBatch(&pose_state);
  problem.AddFactorBatch(&pnp, {&pose_state}, pose_index.data());
  RansacMinimizerOptions o;
  o.default_inlier_threshold = 0.01f;
  RansacGaussNewtonMinimizer ransac(o);
  CudaStream stream;
  for (auto [n, seed] : {std::pair<int, uint32_t>{1500, 5}, {600, 6}, {2000, 7}}) {
    const auto scene = ransac_test::MakePnPScene(n, 0.4, 2e-3, 0.05, seed);
    obs.CopyFromHost(scene.observations.data(), n);
    pts.CopyFromHost(scene.points_world.data(), n);
    pose.CopyFromHost(&scene.world_to_cam,
                      1);  // start at the truth: robustness is tested elsewhere
    pnp.SetNumFactors(n);
    const RansacSummary s = ransac.Minimize(stream.GetStream(), problem);
    EXPECT_EQ(ransac.InlierMaskSize(0), static_cast<size_t>(n));
    std::vector<uint8_t> mask(n);
    THROW_ON_CUDA_ERROR(cudaMemcpy(mask.data(), ransac.InlierMask(0), n, cudaMemcpyDeviceToHost));
    size_t accepted_outliers = 0;
    for (int i = 0; i < n; ++i) accepted_outliers += mask[i] && scene.is_outlier[i];
    EXPECT_EQ(accepted_outliers, 0u) << n << " matches";
    EXPECT_GT(s.inlier_ratio, 0.5f) << n << " matches";
  }
}

}  // namespace
}  // namespace cunls
