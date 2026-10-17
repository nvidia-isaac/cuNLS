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
 * @file schur_system_test.cpp
 * @brief LinearSystemType::SchurComplement against the assembled normal
 * equations on a small bundle adjustment: the same right-hand side, Hessian
 * diagonal, step (with column scaling, damping and masks) and step
 * curvature, the same minimization, and the setup checks.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/gauss_newton_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/minimizer/problem_partition.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"

namespace cunls {
namespace {

/** Row-major 3x4 [R | t] with R a rotation about z by `yaw`. */
SE3Transform Pose(float yaw, float x, float y, float z) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  return {c, -s, 0.f, x, s, c, 0.f, y, 0.f, 0.f, 1.f, z};
}

/** a * b of two rigid transforms (row-major 3x4). */
SE3Transform Compose(const SE3Transform &a, const SE3Transform &b) {
  SE3Transform out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      float v = c == 3 ? a[r * 4 + 3] : 0.f;
      for (int k = 0; k < 3; ++k) v += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = v;
    }
  }
  return out;
}

/** The inverse of a rigid transform. */
SE3Transform Inverse(const SE3Transform &t) {
  SE3Transform out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[r * 4 + c] = t[c * 4 + r];
    out[r * 4 + 3] = -(t[r] * t[3] + t[4 + r] * t[7] + t[8 + r] * t[11]);
  }
  return out;
}

/** world_from_rig of ring position k (rig z axis toward the center, y down). */
SE3Transform RingPose(int k, int num_poses, float radius) {
  const float a = 2.f * static_cast<float>(M_PI) * k / num_poses;
  return {-std::sin(a), 0.f,  -std::cos(a), radius * std::cos(a),  //
          std::cos(a),  0.f,  -std::sin(a), radius * std::sin(a),  //
          0.f,          -1.f, 0.f,          0.f};
}

/** Variants of SmallBundle. */
struct BundleOptions {
  /// Independent rings, declared as subproblems (Problem::SetProblemPartition).
  int rings = 1;
  /// Points per ring in a second, retained point batch (reprojection factors
  /// between retained states only: blocks of S of sizes 6 and 3).
  int retained_points = 0;
  /// Adds a landmark seen once, on the optical axis of an axis-aligned camera
  /// of a constant pose: its Jacobian has an exactly zero column, so its
  /// block of the Hessian is singular however it is damped.
  bool singular_landmark = false;
};

/**
 * Rings of cameras (world_from_rig, rig z axis toward the ring center, y
 * down) around point clouds, seen through two cameras of a stereo rig (two
 * reprojection batches, one with a Cauchy loss), plus SE3 between factors
 * along each ring (factors between retained states only). The first pose and
 * the first point of each ring are constant, so some reprojection factors
 * touch no free pose.
 */
class SmallBundle {
 public:
  static constexpr int kNumPoses = 12;    ///< Per ring.
  static constexpr int kNumPoints = 300;  ///< Per ring.
  static constexpr float kRadius = 6.f;
  static constexpr float kRingSpacing = 40.f;

  explicit SmallBundle(const BundleOptions &options = BundleOptions()) {
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.f, 1.f);
    std::uniform_real_distribution<float> cube(-2.f, 2.f);
    const int rings = options.rings;
    std::vector<SE3Transform> poses, ring;
    std::vector<Vector<3>> points, truth, retained, retained_truth;
    std::vector<int> pose_problem, point_problem, retained_problem, pose_const, point_const;
    for (int g = 0; g < rings; ++g) {
      for (int k = 0; k < kNumPoses; ++k) {
        ring.push_back(RingPose(k, kNumPoses, kRadius));
        ring.back()[3] += kRingSpacing * g;
        poses.push_back(ring.back());
        if (k > 0) {
          for (int r = 0; r < 3; ++r) poses.back()[r * 4 + 3] += 0.05f * noise(rng);
        }
        pose_problem.push_back(g);
      }
      pose_const.push_back(g * kNumPoses);
      point_const.push_back(g * kNumPoints);
      for (int i = 0; i < kNumPoints; ++i) {
        truth.push_back({kRingSpacing * g + cube(rng), cube(rng), cube(rng)});
        points.push_back({truth.back()[0] + 0.05f * noise(rng),
                          truth.back()[1] + 0.05f * noise(rng),
                          truth.back()[2] + 0.05f * noise(rng)});
        point_problem.push_back(g);
      }
      for (int i = 0; i < options.retained_points; ++i) {
        retained_truth.push_back({kRingSpacing * g + cube(rng), cube(rng), cube(rng)});
        retained.push_back({retained_truth.back()[0] + 0.05f * noise(rng),
                            retained_truth.back()[1] + 0.05f * noise(rng),
                            retained_truth.back()[2] + 0.05f * noise(rng)});
        retained_problem.push_back(g);
      }
    }
    if (options.singular_landmark) {
      // Ring pose 0 has an axis-aligned rotation, and camera 0 sits at rig
      // x = 0.1 (camera_from_rig translation -0.1): the rig point (0.1, 0, 3)
      // lies exactly on its optical axis.
      const SE3Transform &t = ring[0];
      const float pr[3] = {0.1f, 0.f, 3.f};
      Vector<3> p{};
      for (int r = 0; r < 3; ++r) {
        p[r] = t[r * 4 + 3];
        for (int c = 0; c < 3; ++c) p[r] += t[r * 4 + c] * pr[c];
      }
      truth.push_back(p);
      points.push_back(p);
      point_problem.push_back(0);
    }
    const int num_poses = static_cast<int>(poses.size());
    const int num_points = static_cast<int>(points.size());
    poses_ = dvector<SE3Transform>(poses);
    points_ = dvector<Vector<3>>(points);
    pose_const_ = dvector<int>(pose_const);
    point_const_ = dvector<int>(point_const);
    pose_batch_ = std::make_unique<SE3StateBatch>(reinterpret_cast<const float *>(poses_.data()),
                                                  num_poses, pose_const_.data(), rings);
    pose_batch_->SetNumActiveStates(num_poses, rings);
    point_batch_ = std::make_unique<VectorStateBatch<3>>(
        reinterpret_cast<const float *>(points_.data()), num_points, point_const_.data(), rings);
    point_batch_->SetNumActiveStates(num_points, rings);
    problem_.AddStateBatch(pose_batch_.get());
    problem_.AddStateBatch(point_batch_.get());
    if (!retained.empty()) {
      retained_ = dvector<Vector<3>>(retained);
      retained_batch_ = std::make_unique<VectorStateBatch<3>>(
          reinterpret_cast<const float *>(retained_.data()), retained.size());
      retained_batch_->SetNumActiveStates(retained.size());
      problem_.AddStateBatch(retained_batch_.get());
    }

    // Two cameras 0.2 m apart on the rig; observations of the true points from
    // the true rings (the initial poses are slightly off).
    auto observe = [&](const SE3Transform &camera_from_world, const Vector<3> &x, bool exact) {
      float pc[3];
      for (int r = 0; r < 3; ++r) {
        pc[r] = camera_from_world[r * 4 + 3];
        for (int c = 0; c < 3; ++c) pc[r] += camera_from_world[r * 4 + c] * x[c];
      }
      const float sigma = exact ? 0.f : 1e-3f;
      return Vector<2>{pc[0] / pc[2] + sigma * noise(rng), pc[1] / pc[2] + sigma * noise(rng)};
    };
    for (int camera = 0; camera < 3; ++camera) {
      // Batch 2: camera 0 seeing the retained points.
      if (camera == 2 && retained.empty()) break;
      const int lens = camera == 2 ? 0 : camera;
      const SE3Transform camera_from_rig = Pose(0.f, lens == 0 ? -0.1f : 0.1f, 0.f, 0.f);
      std::vector<Vector<2>> observations;
      std::vector<SE3Transform> extrinsics;
      std::vector<float *> states;
      for (int k = 0; k < num_poses; ++k) {
        const int g = k / kNumPoses;
        const SE3Transform camera_from_world = Compose(camera_from_rig, Inverse(ring[k]));
        const int count = camera == 2 ? options.retained_points : kNumPoints;
        for (int j = 0; j < count; ++j) {
          if ((j + k + camera) % 3 == 0) continue;  // not every camera sees every point
          if (camera == 2) {
            const int i = g * options.retained_points + j;
            observations.push_back(observe(camera_from_world, retained_truth[i], false));
            states.push_back(retained_batch_->StateDevicePtr(i));
          } else {
            const int i = g * kNumPoints + j;
            observations.push_back(observe(camera_from_world, truth[i], false));
            states.push_back(point_batch_->StateDevicePtr(i));
          }
          extrinsics.push_back(camera_from_rig);
          states.insert(states.end() - 1, pose_batch_->StateDevicePtr(k));
        }
        if (camera == 0 && k == 0 && options.singular_landmark) {
          observations.push_back(observe(camera_from_world, truth.back(), true));
          extrinsics.push_back(camera_from_rig);
          states.push_back(pose_batch_->StateDevicePtr(0));
          states.push_back(point_batch_->StateDevicePtr(num_points - 1));
        }
      }
      const size_t n = observations.size();
      observations_[camera] = dvector<Vector<2>>(observations);
      extrinsics_[camera] = dvector<SE3Transform>(extrinsics);
      reprojection_[camera] = std::make_unique<ReprojectionFactorBatch>(
          observations_[camera].data(), extrinsics_[camera].data(), n);
      reprojection_[camera]->SetNumActiveFactors(n);
      problem_.AddFactorBatch(reprojection_[camera].get(), camera == 1 ? &cauchy_ : nullptr,
                              states);
    }

    // Relative poses along each ring (the measured deltas of the true ring).
    std::vector<SE3Transform> deltas;
    std::vector<float *> between_states;
    for (int k = 0; k < num_poses; ++k) {
      const int next = (k / kNumPoses) * kNumPoses + (k + 1) % kNumPoses;
      // residual = Log(Δ T_k⁻¹ T_next): Δ = T_next⁻¹ T_k.
      deltas.push_back(Compose(Inverse(ring[next]), ring[k]));
      between_states.push_back(pose_batch_->StateDevicePtr(k));
      between_states.push_back(pose_batch_->StateDevicePtr(next));
    }
    deltas_ = dvector<SE3Transform>(deltas);
    between_ = std::make_unique<SE3BetweenFactorBatch>(deltas_.data(), deltas.size());
    between_->SetNumActiveFactors(deltas.size());
    problem_.AddFactorBatch(between_.get(), between_states);

    if (rings > 1) {
      pose_problem_ = dvector<int>(pose_problem);
      point_problem_ = dvector<int>(point_problem);
      std::vector<const int *> ids = {pose_problem_.data(), point_problem_.data()};
      if (!retained.empty()) {
        retained_problem_ = dvector<int>(retained_problem);
        ids.push_back(retained_problem_.data());
      }
      problem_.SetProblemPartition(rings, ids);
    }
  }

  Problem &GetProblem() { return problem_; }
  const StateBatch *Points() const { return point_batch_.get(); }
  VectorStateBatch<3> *PointBatch() { return point_batch_.get(); }
  /** The free tangent columns (poses, points, retained points). */
  size_t NumColumns() const {
    size_t n = (pose_batch_->NumActiveStates() - pose_batch_->NumConstStates()) * 6 +
               (point_batch_->NumActiveStates() - point_batch_->NumConstStates()) * 3;
    return retained_batch_ ? n + retained_batch_->NumActiveStates() * 3 : n;
  }

 private:
  dvector<SE3Transform> poses_;
  dvector<Vector<3>> points_, retained_;
  dvector<int> pose_const_, point_const_;
  dvector<int> pose_problem_, point_problem_, retained_problem_;
  std::unique_ptr<SE3StateBatch> pose_batch_;
  std::unique_ptr<VectorStateBatch<3>> point_batch_, retained_batch_;
  dvector<Vector<2>> observations_[3];
  dvector<SE3Transform> extrinsics_[3];
  std::unique_ptr<ReprojectionFactorBatch> reprojection_[3];
  CauchyLossFunctionBatch cauchy_{1.f, 0.01f};
  dvector<SE3Transform> deltas_;
  std::unique_ptr<SE3BetweenFactorBatch> between_;
  Problem problem_;
};

std::vector<float> ToHost(const dvector<float> &v) {
  std::vector<float> out(v.size());
  v.CopyToHost(out.data(), out.size());
  return out;
}

/** ‖a - b‖ / ‖b‖. */
double RelativeError(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-300));
}

/** One structure setup and system build of the minimizer for `type`. */
struct BuiltSystem {
  BuiltSystem(cudaStream_t stream, Problem &problem, LinearSystemType type, ColumnScaling scaling) {
    options.linear_system = type;
    // The exact reference solve for Assembled (SchurComplement has its own solver).
    options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    options.column_scaling = scaling;
    system.linear_system = internal::CreateLinearSystem(options);
    internal::SetUpStructure(stream, problem, options, state_ops, system);
    state.Recreate(stream, problem);
    internal::BuildSystem(stream, problem, state, options, system, scratch);
    system.step.resize(system.rhs.size());
  }

  MinimizerOptions options;
  StateBatchOps state_ops;
  internal::MinimizerSystem system;
  internal::MinimizerScratch scratch;
  MinimizerState state;
};

/** A bundle variant, by name. */
struct Variant {
  const char *name;
  BundleOptions options;
};

const Variant kVariants[] = {
    {"Plain", {}},
    {"TwoSubproblems", {2, 0, false}},
    {"RetainedPoints", {1, 30, false}},
};

class SchurSystemTest : public ::testing::TestWithParam<std::tuple<Variant, ColumnScaling>> {};

/**
 * Same rhs, Hessian diagonal and damped (and masked) step as the assembled
 * system, and the same step curvature.
 */
TEST_P(SchurSystemTest, MatchesAssembledSystem) {
  const auto [variant, scaling] = GetParam();
  SmallBundle bundle(variant.options);
  Problem &problem = bundle.GetProblem();
  ASSERT_TRUE(problem.CheckConsistency());
  problem.SetEliminatedStateBatches({bundle.Points()});
  CudaStream cuda_stream;
  cudaStream_t stream = cuda_stream.GetStream();
  BuiltSystem assembled(stream, problem, LinearSystemType::Assembled, scaling);
  BuiltSystem schur(stream, problem, LinearSystemType::SchurComplement, scaling);
  const size_t n = assembled.system.rhs.size();
  ASSERT_EQ(n, schur.system.rhs.size());
  ASSERT_EQ(n, bundle.NumColumns());
  EXPECT_LT(RelativeError(ToHost(schur.system.rhs), ToHost(assembled.system.rhs)), 1e-5);
  dvector<float> diagonal_a, diagonal_s;
  assembled.system.linear_system->ExtractScaledHessianDiagonal(stream, diagonal_a);
  schur.system.linear_system->ExtractScaledHessianDiagonal(stream, diagonal_s);
  const std::vector<float> diagonal = ToHost(diagonal_a);
  EXPECT_LT(RelativeError(ToHost(diagonal_s), diagonal), 1e-5);

  // LM-like damping, and a mask holding a few pose and point coordinates.
  std::vector<float> damping(n), mask(n, 1.f);
  for (size_t i = 0; i < n; ++i) damping[i] = 1e-3f * diagonal[i] + 1e-6f;
  for (size_t i : {size_t(3), size_t(20), size_t(66 + 4), size_t(66 + 100)}) mask[i] = 0.f;
  dvector<float> d_damping(damping), d_mask(mask);
  for (bool masked : {false, true}) {
    std::vector<std::vector<float>> steps;
    for (BuiltSystem *built : {&assembled, &schur}) {
      LinearSystem &system = *built->system.linear_system;
      system.SetDamping(&d_damping);
      system.SetMask(masked ? &d_mask : nullptr);
      dvector<float> rhs(n);
      std::vector<float> host_rhs = ToHost(built->system.rhs);
      if (masked) {
        for (size_t i = 0; i < n; ++i) host_rhs[i] *= mask[i];
      }
      rhs.CopyFromHost(host_rhs.data(), n);
      dvector<float> step(n);
      ASSERT_TRUE(system.PrepareSolver(stream, problem, rhs, step));
      ASSERT_TRUE(system.Solve(stream, rhs, step));
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));
      steps.push_back(ToHost(step));
    }
    EXPECT_LT(RelativeError(steps[1], steps[0]), 2e-3) << "masked " << masked;
    if (masked) {
      for (size_t i = 0; i < n; ++i) {
        if (mask[i] == 0.f) EXPECT_EQ(steps[1][i], 0.f) << "held column " << i;
      }
    }
  }

  // δᵀHδ per subproblem.
  ProblemPartition partition;
  MinimizerState state;
  state.Recreate(stream, problem);
  partition.Build(stream, problem, state, n);
  std::vector<float> host_step(n);
  std::mt19937 rng(3);
  std::normal_distribution<float> g;
  for (float &v : host_step) v = g(rng);
  const size_t np = problem.NumProblems();
  dvector<float> step(host_step), curvature(2 * np);
  assembled.system.linear_system->StepCurvature(stream, step, partition, curvature.data());
  schur.system.linear_system->StepCurvature(stream, step, partition, curvature.data() + np);
  const std::vector<float> c = ToHost(curvature);
  for (size_t p = 0; p < np; ++p) EXPECT_NEAR(c[np + p], c[p], 1e-4f * c[p]) << "subproblem " << p;
}

INSTANTIATE_TEST_SUITE_P(Variants, SchurSystemTest,
                         ::testing::Combine(::testing::ValuesIn(kVariants),
                                            ::testing::Values(ColumnScaling::None,
                                                              ColumnScaling::HessianDiagonal)),
                         [](const ::testing::TestParamInfo<SchurSystemTest::ParamType> &info) {
                           return std::string(std::get<0>(info.param).name) +
                                  (std::get<1>(info.param) == ColumnScaling::None ? "Unscaled"
                                                                                  : "Scaled");
                         });

class SchurSystemMinimizeTest : public ::testing::TestWithParam<Variant> {};

/** The same minimization as the assembled system. */
TEST_P(SchurSystemMinimizeTest, ConvergesLikeAssembled) {
  std::vector<float> final_costs;
  for (LinearSystemType type : {LinearSystemType::Assembled, LinearSystemType::SchurComplement}) {
    SmallBundle bundle(GetParam().options);
    Problem &problem = bundle.GetProblem();
    problem.SetEliminatedStateBatches({bundle.Points()});
    LevenbergMarquardtMinimizerOptions options;
    options.base_options.max_num_iterations = 30;
    options.base_options.linear_system = type;
    options.base_options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
    options.base_options.column_scaling = ColumnScaling::HessianDiagonal;
    LevenbergMarquardtMinimizer minimizer(options);
    CudaStream stream;
    const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    EXPECT_LT(summary.final_cost, 1e-2f * summary.initial_cost);
    final_costs.push_back(summary.final_cost);
  }
  EXPECT_NEAR(final_costs[1], final_costs[0], 1e-3f * final_costs[0]);
}

INSTANTIATE_TEST_SUITE_P(Variants, SchurSystemMinimizeTest, ::testing::ValuesIn(kVariants),
                         [](const ::testing::TestParamInfo<Variant> &info) {
                           return std::string(info.param.name);
                         });

/**
 * Runs `minimizer` on a bundle with a singular landmark. Returns whether all
 * points stay finite; `moved` receives how far the singular landmark moved.
 */
bool MinimizeSingular(Minimizer &minimizer, MinimizerSummary *summary, float *moved) {
  BundleOptions options;
  options.singular_landmark = true;
  SmallBundle bundle(options);
  Problem &problem = bundle.GetProblem();
  problem.SetEliminatedStateBatches({bundle.Points()});
  const VectorStateBatch<3> *points = bundle.PointBatch();
  std::vector<float> before(points->NumActiveStates() * 3), after(before.size());
  THROW_ON_CUDA_ERROR(cudaMemcpy(before.data(), points->StateDevicePtr(0),
                                 before.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CudaStream stream;
  *summary = minimizer.Minimize(stream.GetStream(), problem);
  THROW_ON_CUDA_ERROR(cudaMemcpy(after.data(), points->StateDevicePtr(0),
                                 after.size() * sizeof(float), cudaMemcpyDeviceToHost));
  *moved = 0.f;
  for (size_t i = after.size() - 3; i < after.size(); ++i) {
    *moved = std::max(*moved, std::fabs(after[i] - before[i]));
  }
  return std::all_of(after.begin(), after.end(), [](float v) { return std::isfinite(v); });
}

MinimizerOptions SingularOptions(LinearSystemType type, bool safety_checks) {
  MinimizerOptions options;
  options.max_num_iterations = 10;
  options.linear_system = type;
  options.sparse_linear_solver_type = SparseLinearSolverType::DenseCholesky;
  options.disable_safety_checks = !safety_checks;
  return options;
}

/**
 * A landmark whose Hessian block is singular (an exactly zero Jacobian
 * column, which no damping of the form λ diag(H) fixes) makes the full
 * system singular: with safety checks Gauss-Newton throws; Levenberg-Marquardt
 * rejects the failed step instead of throwing (with or without checks), and
 * the states stay finite.
 */
TEST(SchurSystemFailureTest, AssembledFailedSolveIsRejected) {
  MinimizerSummary summary;
  float moved = 0.f;
  {
    GaussNewtonMinimizer gn(SingularOptions(LinearSystemType::Assembled, true));
    EXPECT_THROW(MinimizeSingular(gn, &summary, &moved), std::runtime_error);
  }
  for (bool checks : {true, false}) {
    LevenbergMarquardtMinimizerOptions options;
    options.base_options = SingularOptions(LinearSystemType::Assembled, checks);
    LevenbergMarquardtMinimizer lm(options);
    bool finite = false;
    ASSERT_NO_THROW(finite = MinimizeSingular(lm, &summary, &moved)) << "checks " << checks;
    EXPECT_TRUE(finite) << "checks " << checks;
    EXPECT_TRUE(std::isfinite(summary.final_cost));
    EXPECT_LE(summary.final_cost, summary.initial_cost);
  }
}

/**
 * The Schur complement factors a singular landmark block with a small
 * diagonal shift (as Ceres uses a pseudo-inverse): the unobservable landmark
 * stays where it is, and both methods converge on the rest of the problem.
 */
TEST(SchurSystemFailureTest, SchurRegularizesSingularLandmarks) {
  for (bool checks : {true, false}) {
    MinimizerSummary summary;
    float moved = 1.f;
    GaussNewtonMinimizer gn(SingularOptions(LinearSystemType::SchurComplement, checks));
    ASSERT_NO_THROW(EXPECT_TRUE(MinimizeSingular(gn, &summary, &moved))) << "checks " << checks;
    EXPECT_LT(summary.final_cost, 1e-2f * summary.initial_cost);
    EXPECT_LT(moved, 1e-3f);
    LevenbergMarquardtMinimizerOptions options;
    options.base_options = SingularOptions(LinearSystemType::SchurComplement, checks);
    LevenbergMarquardtMinimizer lm(options);
    ASSERT_NO_THROW(EXPECT_TRUE(MinimizeSingular(lm, &summary, &moved))) << "checks " << checks;
    EXPECT_LT(summary.final_cost, 1e-2f * summary.initial_cost);
    EXPECT_LT(moved, 1e-3f);
  }
}

/** Setup errors name what to fix. */
TEST(SchurSystemSetupTest, RejectsInvalidPartitions) {
  CudaStream stream;
  LevenbergMarquardtMinimizerOptions options;
  options.base_options.linear_system = LinearSystemType::SchurComplement;
  {
    SmallBundle bundle;  // no eliminated batches declared
    LevenbergMarquardtMinimizer minimizer(options);
    EXPECT_THROW(minimizer.Minimize(stream.GetStream(), bundle.GetProblem()),
                 std::invalid_argument);
  }
  {
    // A factor between two points: they cannot both be eliminated.
    SmallBundle bundle;
    Problem &problem = bundle.GetProblem();
    dvector<Vector<3>> delta(std::vector<Vector<3>>{{0.f, 0.f, 0.f}});
    VectorBetweenFactorBatch<3> between(delta.data(), 1);
    between.SetNumActiveFactors(1);
    VectorStateBatch<3> *points = bundle.PointBatch();
    problem.AddFactorBatch(&between, {points->StateDevicePtr(1), points->StateDevicePtr(2)});
    problem.SetEliminatedStateBatches({bundle.Points()});
    LevenbergMarquardtMinimizer minimizer(options);
    EXPECT_THROW(minimizer.Minimize(stream.GetStream(), problem), std::invalid_argument);
  }
}

/** The Schur complement has its own reduced solver: sparse_linear_solver_type is not read. */
TEST(SchurSystemSetupTest, IgnoresSparseLinearSolverType) {
  MinimizerOptions options;
  options.linear_system = LinearSystemType::SchurComplement;
  options.sparse_linear_solver_type = SparseLinearSolverType::BlockTridiagonal;
  EXPECT_NO_THROW(internal::CreateLinearSystem(options));
}

}  // namespace
}  // namespace cunls
