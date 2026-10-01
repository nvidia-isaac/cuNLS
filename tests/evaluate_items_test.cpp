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
 * @file evaluate_items_test.cpp
 * @brief Plus with replicas and Evaluate with items (num_items, factor_ids)
 * must give bitwise the same results as the corresponding plain calls.
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
#include "cunls/factor/information/information_factor_batch.h"
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
#include "tests/evaluate_items_check.h"
#include "tests/ransac_test_support.h"

namespace cunls {
namespace {

using evaluate_items_test::ToDevice;
using evaluate_items_test::ToHost;

std::vector<float> RandomVector(size_t n, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, scale);
  std::vector<float> v(n);
  for (float &x : v) x = dist(rng);
  return v;
}

// ============================================================================
// Plus with num_replicas
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

/** A user-style custom state batch implementing the replica contract. */
class CustomVectorStateBatch : public SizedStateBatch<3, 3> {
 public:
  using SizedStateBatch<3, 3>::SizedStateBatch;
  void Plus(const float *x, const float *delta, float *out, cudaStream_t stream,
            size_t num_replicas = 1) override {
    CalculateVectorPlus(x, delta, out, num_blocks_ * num_replicas, 3, stream);
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
      {"Custom", 3, 3, false,
       [](float *p, size_t n) -> std::unique_ptr<StateBatch> {
         return std::make_unique<CustomVectorStateBatch>(p, n);
       }},
  };
}

TEST(PlusReplicas, MatchesPlusPerReplicaBitwiseForEveryState) {
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
    all->SetNumStateBlocks(replicas * blocks);
    all->Plus(base.data(), delta0.data(), x.data(), stream.GetStream());

    auto delta = ToDevice(RandomVector(tangents, 0.2f, 2));
    dvector<float> looped(states), batched(states);
    auto batch = c.make(x.data(), blocks);
    batch->SetNumStateBlocks(blocks);
    const size_t s = blocks * c.ambient;
    const size_t t = blocks * c.tangent;
    for (size_t r = 0; r < replicas; ++r) {
      batch->Plus(x.data() + r * s, delta.data() + r * t, looped.data() + r * s,
                  stream.GetStream());
    }
    batch->Plus(x.data(), delta.data(), batched.data(), stream.GetStream(), replicas);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const auto a = ToHost(looped);
    const auto b = ToHost(batched);
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
    EXPECT_EQ(batch->NumStateBlocks(), blocks);  // block count restored
  }
}

// ============================================================================
// Evaluate with item parameters
// ============================================================================

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

TEST(EvaluateItems, PnPMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(60, 0.2, 1e-3, 2e-2, 11);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  PnPFactorBatch pnp(obs.data(), pts.data(), scene.observations.size());
  pnp.SetNumFactors(pnp.Capacity());
  Poses poses(4, 12, 0.05, 0.1);
  evaluate_items_test::CheckEvaluateItems(
      pnp, 4, [&](int k) { return std::vector<float *>(60, poses.ptr(k)); });
}

TEST(EvaluateItems, PnPWithCameraFromRigMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(40, 0.0, 1e-3, 2e-2, 13);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  Poses rigs(40, 14, 0.02, 0.05);
  PnPFactorBatch pnp(obs.data(), rigs.d.data(), pts.data(), 40);
  pnp.SetNumFactors(pnp.Capacity());
  Poses poses(3, 15, 0.05, 0.1);
  evaluate_items_test::CheckEvaluateItems(
      pnp, 3, [&](int k) { return std::vector<float *>(40, poses.ptr(k)); });
}

TEST(EvaluateItems, ReprojectionMatchesEvaluate) {
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(50, 0.2, 1e-3, 2e-2, 16);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  ReprojectionFactorBatch reproj(obs.data(), 50);
  reproj.SetNumFactors(reproj.Capacity());
  Poses poses(3, 17, 0.05, 0.1);
  evaluate_items_test::CheckEvaluateItems(reproj, 3, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < 50; ++i) {
      p.push_back(poses.ptr(k));
      p.push_back(reinterpret_cast<float *>(pts.data() + i));
    }
    return p;
  });
}

TEST(EvaluateItems, SE3PriorMatchesEvaluate) {
  Poses targets(20, 18);
  SE3PriorFactorBatch prior(targets.d.data(), 20);
  prior.SetNumFactors(prior.Capacity());
  Poses poses(5, 19);
  evaluate_items_test::CheckEvaluateItems(
      prior, 5, [&](int k) { return std::vector<float *>(20, poses.ptr(k)); });
}

TEST(EvaluateItems, SE3BetweenMatchesEvaluate) {
  Poses deltas(25, 20);
  SE3BetweenFactorBatch between(deltas.d.data(), 25);
  between.SetNumFactors(between.Capacity());
  Poses poses(6, 21);
  evaluate_items_test::CheckEvaluateItems(between, 3, [&](int k) {
    std::vector<float *> p;
    for (int i = 0; i < 25; ++i) {
      p.push_back(poses.ptr(2 * k));
      p.push_back(poses.ptr(2 * k + 1));
    }
    return p;
  });
}

TEST(EvaluateItems, ResidualBatchWithLossMatchesEvaluate) {
  const int n_f = 80, copies = 4;
  const ransac_test::PnPScene scene = ransac_test::MakePnPScene(n_f, 0.4, 1e-3, 2e-2, 22);
  auto obs = ToDevice(scene.observations);
  auto pts = ToDevice(scene.points_world);
  PnPFactorBatch pnp(obs.data(), pts.data(), n_f);
  pnp.SetNumFactors(pnp.Capacity());
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
  ASSERT_TRUE(rb.Evaluate(stream.GetStream(), ws.data(), r2.data(), d_table.data(), c2.data(),
                          j2.data(), nullptr, items));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(ToHost(r1), ToHost(r2));
  EXPECT_EQ(ToHost(c1), ToHost(c2));
  EXPECT_EQ(ToHost(j1), ToHost(j2));
}

// ============================================================================
// Sqrt-information item kernels
// ============================================================================

/**
 * Applies the sqrt-information item kernels to n items of size m (factor ids
 * with repeats) and compares with a double-precision CPU product. m = 3 takes
 * the shared-memory path; m = 100 (> 96) the direct path.
 */
void CheckInformationItems(int m, int pitch) {
  const int num_factors = 5;
  const int n = 13;
  std::mt19937 rng(m);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::vector<float> S(static_cast<size_t>(num_factors) * m * m), r(n * m), J(n * m * pitch);
  for (float &v : S) v = uni(rng);
  for (float &v : r) v = uni(rng);
  for (float &v : J) v = uni(rng);
  std::vector<int> ids(n);
  for (int &id : ids) id = static_cast<int>(rng() % num_factors);

  auto d_S = ToDevice(S);
  auto d_r = ToDevice(r);
  auto d_J = ToDevice(J);
  auto d_ids = ToDevice(ids);
  CudaStream stream;
  ApplyInformationToResidualItems(d_S.data(), d_r.data(), m, n, d_ids.data(), num_factors,
                                  stream.GetStream());
  ApplyInformationToJacobianItems(d_S.data(), d_J.data(), m, pitch, n, d_ids.data(), num_factors,
                                  stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hr = ToHost(d_r);
  const auto hJ = ToHost(d_J);

  for (int t = 0; t < n; ++t) {
    const float *St = S.data() + static_cast<size_t>(ids[t]) * m * m;
    for (int i = 0; i < m; ++i) {
      double res = 0.0;
      for (int k = 0; k < m; ++k) res += double(St[i * m + k]) * r[t * m + k];
      ASSERT_NEAR(hr[t * m + i], res, 1e-4 * m) << "m " << m << " item " << t;
      for (int c = 0; c < pitch; ++c) {
        double jac = 0.0;
        for (int k = 0; k < m; ++k) {
          jac += double(St[i * m + k]) * J[(static_cast<size_t>(t) * m + k) * pitch + c];
        }
        ASSERT_NEAR(hJ[(static_cast<size_t>(t) * m + i) * pitch + c], jac, 1e-4 * m)
            << "m " << m << " item " << t << " col " << c;
      }
    }
  }
}

TEST(InformationItems, SharedMemoryPathMatchesCpu) { CheckInformationItems(3, 4); }

TEST(InformationItems, LargeResidualSizeUsesDirectPathAndMatchesCpu) {
  CheckInformationItems(100, 3);
}

}  // namespace
}  // namespace cunls
