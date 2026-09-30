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

/**
 * @file indexed_evaluation_test.cpp
 * @brief StateBatch::PlusReplicated and FactorBatch::EvaluateIndexed must give
 * bitwise the same results as the plain Plus / Evaluate calls they replace.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "cunls/common/cublas_helper.h"
#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/pnp_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/residual_batch.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/state/se2_state_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/similarity2_state_batch.h"
#include "cunls/state/similarity3_state_batch.h"
#include "cunls/state/sl4_state_batch.h"
#include "cunls/state/so2_state_batch.h"
#include "cunls/state/so3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace {

template <typename T>
std::vector<T> ToHost(const dvector<T> &d) {
  std::vector<T> h(d.size());
  if (!h.empty()) d.CopyToHost(h.data(), h.size());
  return h;
}

template <typename T>
dvector<T> ToDevice(const std::vector<T> &h) {
  dvector<T> d(h.size());
  if (!h.empty()) d.CopyFromHost(h.data(), h.size());
  return d;
}

std::vector<float> RandomVector(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = dist(rng);
  return v;
}

// ============================================================================
// PlusReplicated
// ============================================================================

using StateFactory = std::function<std::unique_ptr<StateBatch>(float *, size_t)>;

struct StateCase {
  const char *name;
  int ambient;
  int tangent;
  bool matrix;  ///< Identity is a square matrix (else a zero vector).
  StateFactory make;
};

cuBLASHandle &Cublas() {
  static cuBLASHandle handle;
  return handle;
}

/** A state batch that does not override PlusReplicated (exercises the default). */
class PlainVectorStateBatch : public SizedStateBatch<3, 3> {
 public:
  using SizedStateBatch<3, 3>::SizedStateBatch;
  void Plus(const float *x, const float *delta, float *out, cudaStream_t stream) override {
    CalculateVectorPlus(x, delta, out, num_blocks_, 3, stream);
  }
};

template <typename T>
StateFactory LieFactory() {
  return [](float *p, size_t n) -> std::unique_ptr<StateBatch> {
    return std::make_unique<T>(Cublas(), p, n);
  };
}

std::vector<StateCase> StateCases() {
  return {
      {"SE3", 16, 6, true, LieFactory<SE3StateBatch>()},
      {"SE2", 9, 3, true, LieFactory<SE2StateBatch>()},
      {"SO3", 9, 3, true, LieFactory<SO3StateBatch>()},
      {"SO2", 4, 1, true, LieFactory<SO2StateBatch>()},
      {"Sim3", 16, 7, true, LieFactory<Similarity3StateBatch>()},
      {"Sim2", 9, 4, true, LieFactory<Similarity2StateBatch>()},
      {"SL4", 16, 15, true, LieFactory<SL4StateBatch>()},
      {"Vector5", 5, 5, false,
       [](float *p, size_t n) -> std::unique_ptr<StateBatch> {
         return std::make_unique<VectorStateBatch<5>>(p, n);
       }},
      {"CustomDefault", 3, 3, false,
       [](float *p, size_t n) -> std::unique_ptr<StateBatch> {
         return std::make_unique<PlainVectorStateBatch>(p, n);
       }},
  };
}

TEST(PlusReplicated, MatchesPlusPerReplicaBitwiseForEveryState) {
  const size_t blocks = 7;
  const size_t replicas = 5;
  CudaStream stream;
  for (const StateCase &c : StateCases()) {
    SCOPED_TRACE(c.name);
    const size_t states = replicas * blocks * c.ambient;
    const size_t tangents = replicas * blocks * c.tangent;
    // Valid manifold elements: identity (+) random delta.
    std::vector<float> identity(states, 0.f);
    if (c.matrix) {
      const int d = static_cast<int>(std::lround(std::sqrt(c.ambient)));
      for (size_t b = 0; b < replicas * blocks; ++b) {
        for (int i = 0; i < d; ++i) identity[b * c.ambient + i * d + i] = 1.f;
      }
    }
    auto base = ToDevice(identity);
    auto x = dvector<float>(states);
    auto delta0 = ToDevice(RandomVector(tangents, 0.3f, 1));
    auto all = c.make(base.data(), replicas * blocks);
    all->Plus(base.data(), delta0.data(), x.data(), stream.GetStream());

    auto delta = ToDevice(RandomVector(tangents, 0.2f, 2));
    dvector<float> looped(states), batched(states);
    auto batch = c.make(x.data(), blocks);
    const size_t s = blocks * c.ambient;
    const size_t t = blocks * c.tangent;
    for (size_t r = 0; r < replicas; ++r) {
      batch->Plus(x.data() + r * s, delta.data() + r * t, looped.data() + r * s,
                  stream.GetStream());
    }
    batch->PlusReplicated(x.data(), delta.data(), batched.data(), replicas, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const auto a = ToHost(looped);
    const auto b = ToHost(batched);
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
    EXPECT_EQ(batch->NumStateBlocks(), blocks);  // block count restored
  }
}

// ============================================================================
// EvaluateIndexed
// ============================================================================

/**
 * Checks EvaluateIndexed against Evaluate for `copies` state sets:
 * pointers_for_copy(k) returns the N * nb state pointers of copy k.
 */
void CheckIndexed(const FactorBatch &factor, int copies,
                  const std::function<std::vector<float *>(int)> &pointers_for_copy) {
  CudaStream stream;
  const int n_f = static_cast<int>(factor.NumFactors());
  const int m = static_cast<int>(factor.ResidualsSize());
  const auto sizes = factor.StateBlockSizes();
  const int nb = static_cast<int>(sizes.size());
  const int n = static_cast<int>(std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
  std::vector<float *> table;
  for (int k = 0; k < copies; ++k) {
    const auto p = pointers_for_copy(k);
    table.insert(table.end(), p.begin(), p.end());
  }
  auto d_table = ToDevice(table);
  const size_t items = static_cast<size_t>(copies) * n_f;

  // Reference: one Evaluate per copy.
  dvector<float> ref_r(items * m), ref_j(items * m * n);
  for (int k = 0; k < copies; ++k) {
    ASSERT_TRUE(factor.Evaluate(ref_r.data() + k * n_f * m, ref_j.data() + k * n_f * m * n,
                                d_table.data() + k * n_f * nb, stream.GetStream()));
  }
  // Replicated: factor_ids == nullptr.
  dvector<float> r(items * m), j(items * m * n);
  ASSERT_TRUE(factor.EvaluateIndexed(r.data(), j.data(), d_table.data(), nullptr, items,
                                     stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hr = ToHost(ref_r), hj = ToHost(ref_j);
  EXPECT_EQ(hr, ToHost(r));
  EXPECT_EQ(hj, ToHost(j));

  // Arbitrary items with repeats: (factor f, copy k) rows must equal the reference rows.
  std::mt19937 rng(7);
  const int num_items = 3 * n_f + 5;
  std::vector<int> ids(num_items), copy(num_items);
  std::vector<float *> item_table;
  for (int t = 0; t < num_items; ++t) {
    ids[t] = static_cast<int>(rng() % n_f);
    copy[t] = static_cast<int>(rng() % copies);
    for (int b = 0; b < nb; ++b) item_table.push_back(table[(copy[t] * n_f + ids[t]) * nb + b]);
  }
  auto d_ids = ToDevice(ids);
  auto d_items = ToDevice(item_table);
  dvector<float> ir(static_cast<size_t>(num_items) * m), ij(static_cast<size_t>(num_items) * m * n);
  ASSERT_TRUE(factor.EvaluateIndexed(ir.data(), ij.data(), d_items.data(), d_ids.data(), num_items,
                                     stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hir = ToHost(ir), hij = ToHost(ij);
  for (int t = 0; t < num_items; ++t) {
    const size_t row = static_cast<size_t>(copy[t]) * n_f + ids[t];
    ASSERT_EQ(0, std::memcmp(&hir[t * m], &hr[row * m], m * sizeof(float))) << "item " << t;
    ASSERT_EQ(0, std::memcmp(&hij[t * m * n], &hj[row * m * n], m * n * sizeof(float)))
        << "item " << t;
  }
}

/** K random SE3 poses in device memory. */
struct Poses {
  dvector<SE3Transform> d;
  explicit Poses(int k, uint32_t seed, double rot = 0.3, double trans = 1.0) {
    std::mt19937 rng(seed);
    std::vector<SE3Transform> h;
    for (int i = 0; i < k; ++i) {
      h.push_back(ransac_test::ExpSE3(ransac_test::RandomTwist(rng, rot, trans)));
    }
    d = ToDevice(h);
  }
  float *ptr(int i) { return reinterpret_cast<float *>(d.data() + i); }
};

TEST(EvaluateIndexed, PnPMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(60, 0.2, 1e-3, 2e-2, 11);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  PnPFactorBatch pnp(obs.data(), pts.data(), scene.observations.size());
  Poses poses(4, 12, 0.05, 0.1);
  CheckIndexed(pnp, 4, [&](int k) { return std::vector<float *>(60, poses.ptr(k)); });
}

TEST(EvaluateIndexed, PnPWithCameraFromRigMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(40, 0.0, 1e-3, 2e-2, 13);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  Poses rigs(40, 14, 0.02, 0.05);
  PnPFactorBatch pnp(obs.data(), rigs.d.data(), pts.data(), 40);
  Poses poses(3, 15, 0.05, 0.1);
  CheckIndexed(pnp, 3, [&](int k) { return std::vector<float *>(40, poses.ptr(k)); });
}

TEST(EvaluateIndexed, ReprojectionMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(50, 0.2, 1e-3, 2e-2, 16);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  ReprojectionFactorBatch reproj(obs.data(), 50);
  Poses poses(3, 17, 0.05, 0.1);
  CheckIndexed(reproj, 3, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < 50; ++i) {
      p.push_back(poses.ptr(k));
      p.push_back(reinterpret_cast<float *>(pts.data() + i));
    }
    return p;
  });
}

TEST(EvaluateIndexed, SE3PriorMatchesEvaluate) {
  Poses targets(20, 18);
  SE3PriorFactorBatch prior(targets.d.data(), 20);
  Poses poses(5, 19);
  CheckIndexed(prior, 5, [&](int k) { return std::vector<float *>(20, poses.ptr(k)); });
}

TEST(EvaluateIndexed, SE3BetweenMatchesEvaluate) {
  Poses deltas(25, 20);
  SE3BetweenFactorBatch between(deltas.d.data(), 25);
  Poses poses(6, 21);
  CheckIndexed(between, 3, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < 25; ++i) {
      p.push_back(poses.ptr(2 * k));
      p.push_back(poses.ptr(2 * k + 1));
    }
    return p;
  });
}

TEST(EvaluateIndexed, ResidualBatchWithLossMatchesEvaluate) {
  const int n_f = 80, copies = 4;
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(n_f, 0.4, 1e-3, 2e-2, 22);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  PnPFactorBatch pnp(obs.data(), pts.data(), n_f);
  CauchyLossFunctionBatch loss(1e-4f, 1e4f);
  ResidualBatch rb(&pnp, &loss);
  Poses poses(copies, 23, 0.05, 0.1);
  std::vector<float *> table;
  for (int k = 0; k < copies; ++k) table.insert(table.end(), n_f, poses.ptr(k));
  auto d_table = ToDevice(table);
  const size_t items = static_cast<size_t>(copies) * n_f;
  CudaStream stream;
  dvector<float> ws(ResidualBatchWorkspaceNumFloats(items));
  dvector<float> r1(items * 2), c1(items), j1(items * 12), r2(items * 2), c2(items), j2(items * 12);
  for (int k = 0; k < copies; ++k) {
    ASSERT_TRUE(rb.Evaluate(stream.GetStream(), ws.data(), r1.data() + k * n_f * 2,
                            d_table.data() + k * n_f, c1.data() + k * n_f,
                            j1.data() + k * n_f * 12));
  }
  ASSERT_TRUE(rb.EvaluateIndexed(stream.GetStream(), ws.data(), r2.data(), d_table.data(), nullptr,
                                 items, c2.data(), j2.data()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(ToHost(r1), ToHost(r2));
  EXPECT_EQ(ToHost(c1), ToHost(c2));
  EXPECT_EQ(ToHost(j1), ToHost(j2));
}

TEST(EvaluateIndexed, DefaultIsUnsupportedAndWritesNothing) {
  std::vector<float> a(4, 1.f), y(2, 0.f), x(2, 0.f);
  auto da = ToDevice(a), dy = ToDevice(y), dx = ToDevice(x);
  ransac_test::LinearRegressionFactorBatch<2> factor(da.data(), dy.data(), 2);
  ResidualBatch rb(&factor, nullptr);
  std::vector<float *> table(4, dx.data());
  auto d_table = ToDevice(table);
  auto out = ToDevice(std::vector<float>(4, 42.f));
  dvector<float> ws(ResidualBatchWorkspaceNumFloats(4));
  CudaStream stream;
  EXPECT_FALSE(factor.EvaluateIndexed(out.data(), nullptr, d_table.data(), nullptr, 4,
                                      stream.GetStream()));
  EXPECT_FALSE(rb.EvaluateIndexed(stream.GetStream(), ws.data(), out.data(), d_table.data(),
                                  nullptr, 4, nullptr, nullptr));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(ToHost(out), std::vector<float>(4, 42.f));
}

}  // namespace
}  // namespace cunls
