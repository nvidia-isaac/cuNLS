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

// Articulations: ArticulationTopology validation, ArticulationStateBatch
// (forward kinematics against a float64 reference, Plus, bounds, refresh),
// the kinematic factors (Jacobians against central differences of Plus,
// evaluation items), and solves: bounded IK, a four-bar loop closure, a mimic
// joint.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/kinematics/link_target_factor_batch.h"
#include "cunls/factor/kinematics/loop_closure_factor_batch.h"
#include "cunls/factor/kinematics/mimic_joint_factor_batch.h"
#include "cunls/factor/kinematics/posture_factor_batch.h"
#include "cunls/minimizer/augmented_lagrangian_minimizer.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/state/articulation_state_batch.h"
#include "evaluate_items_check.h"

namespace cunls {
namespace {

using JT = ArticulationJointType;
constexpr float kInf = std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------

/** Appends a joint (frames as position + quaternion xyzw) to `d`. */
void AddJoint(ArticulationDescription &d, JT type, int parent, std::array<float, 7> X_p,
              std::array<float, 7> X_c, std::vector<std::array<float, 3>> axes,
              std::vector<std::array<float, 2>> limits = {}, int lin = 0, int ang = 0) {
  const int j = static_cast<int>(d.joint_type.size());
  if (d.joint_q_start.empty()) d.joint_q_start = d.joint_qd_start = {0};
  d.joint_type.push_back(static_cast<int>(type));
  d.joint_parent.push_back(parent);
  d.joint_child.push_back(j);
  d.joint_X_p.insert(d.joint_X_p.end(), X_p.begin(), X_p.end());
  d.joint_X_c.insert(d.joint_X_c.end(), X_c.begin(), X_c.end());
  d.joint_dof_dim.push_back(lin);
  d.joint_dof_dim.push_back(ang);
  int coords = 0, dofs = 0;
  switch (type) {
    case JT::kPrismatic:
    case JT::kRevolute:
      coords = dofs = 1;
      break;
    case JT::kBall:
      coords = 4, dofs = 3;
      break;
    case JT::kFree:
      coords = 7, dofs = 6;
      break;
    case JT::kD6:
      coords = dofs = lin + ang;
      break;
    default:
      break;
  }
  d.joint_q_start.push_back(d.joint_q_start.back() + coords);
  d.joint_qd_start.push_back(d.joint_qd_start.back() + dofs);
  for (int k = 0; k < dofs; ++k) {
    const auto a = k < static_cast<int>(axes.size()) ? axes[k] : std::array<float, 3>{0, 0, 0};
    d.joint_axis.insert(d.joint_axis.end(), a.begin(), a.end());
    const auto l =
        k < static_cast<int>(limits.size()) ? limits[k] : std::array<float, 2>{-kInf, kInf};
    d.limit_lower.push_back(l[0]);
    d.limit_upper.push_back(l[1]);
  }
}

std::array<float, 7> Pose(float x, float y, float z, std::array<float, 4> q = {0, 0, 0, 1}) {
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {x, y, z, q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

/** A tree with every joint type: FREE root, BALL, REVOLUTE, PRISMATIC, FIXED, two D6, a branch. */
ArticulationDescription MixedTree() {
  ArticulationDescription d;
  AddJoint(d, JT::kFree, -1, Pose(0.1f, -0.2f, 0.3f, {0.1f, 0.2f, 0.0f, 1.f}),
           Pose(0.f, 0.f, 0.05f), {});
  AddJoint(d, JT::kBall, 0, Pose(0.f, 0.f, 0.4f, {0.f, 0.3f, 0.f, 1.f}), Pose(0.f, 0.02f, 0.f), {});
  AddJoint(d, JT::kRevolute, 1, Pose(0.3f, 0.f, 0.f), Pose(0.f, 0.f, -0.1f, {0.2f, 0, 0, 1}),
           {{0.f, 0.6f, 0.8f}}, {{-1.5f, 1.5f}});
  AddJoint(d, JT::kPrismatic, 2, Pose(0.f, 0.25f, 0.f, {0, 0, 0.4f, 1}), Pose(0.f, 0.f, 0.f),
           {{1.f, 0.f, 0.f}}, {{-0.2f, 0.4f}});
  AddJoint(d, JT::kFixed, 3, Pose(0.1f, 0.1f, 0.1f, {0.3f, 0.1f, 0.f, 1.f}), Pose(0, 0, 0), {});
  AddJoint(d, JT::kD6, 4, Pose(0.f, 0.f, 0.2f), Pose(0.05f, 0.f, 0.f),
           {{0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}, {}, 1, 2);
  AddJoint(d, JT::kD6, 5, Pose(0.2f, 0.f, 0.f, {0, 0.2f, 0.1f, 1}), Pose(0.f, 0.f, 0.f),
           {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}, {}, 0, 3);
  AddJoint(d, JT::kRevolute, 1, Pose(-0.3f, 0.f, 0.1f), Pose(0.f, 0.f, 0.f), {{0.f, 0.f, 1.f}});
  return d;
}

/** A 7-DOF arm: revolute joints alternating about z and y, 0.3 m links, limits ±(2.4, 1.6). */
ArticulationDescription Arm() {
  ArticulationDescription d;
  for (int j = 0; j < 7; ++j) {
    const bool z = j % 2 == 0;
    AddJoint(d, JT::kRevolute, j - 1, Pose(0.f, 0.f, j == 0 ? 0.1f : 0.3f), Pose(0, 0, 0),
             {{0.f, z ? 0.f : 1.f, z ? 1.f : 0.f}}, {{z ? -2.4f : -1.6f, z ? 2.4f : 1.6f}});
  }
  return d;
}

// ---------------------------------------------------------------------------
// float64 reference forward kinematics
// ---------------------------------------------------------------------------

using M4 = std::array<double, 16>;

M4 Identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

M4 Mul(const M4 &a, const M4 &b) {
  M4 c{};
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k) {
      double s = 0;
      for (int i = 0; i < 4; ++i) s += a[4 * r + i] * b[4 * i + k];
      c[4 * r + k] = s;
    }
  return c;
}

M4 FromPose(const float *p) {  // (p, quat xyzw)
  double x = p[3], y = p[4], z = p[5], w = p[6];
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n, y /= n, z /= n, w /= n;
  return {1 - 2 * (y * y + z * z),
          2 * (x * y - z * w),
          2 * (x * z + y * w),
          p[0],
          2 * (x * y + z * w),
          1 - 2 * (x * x + z * z),
          2 * (y * z - x * w),
          p[1],
          2 * (x * z - y * w),
          2 * (y * z + x * w),
          1 - 2 * (x * x + y * y),
          p[2],
          0,
          0,
          0,
          1};
}

M4 Inverse(const M4 &T) {
  M4 r = Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r[4 * i + j] = T[4 * j + i];
    r[4 * i + 3] = -(T[i] * T[3] + T[4 + i] * T[7] + T[8 + i] * T[11]);
  }
  return r;
}

M4 AxisAngle(const float *a, double t) {
  const double c = std::cos(t), s = std::sin(t), k = 1 - c, x = a[0], y = a[1], z = a[2];
  return {c + k * x * x,
          k * x * y - s * z,
          k * x * z + s * y,
          0,
          k * x * y + s * z,
          c + k * y * y,
          k * y * z - s * x,
          0,
          k * x * z - s * y,
          k * y * z + s * x,
          c + k * z * z,
          0,
          0,
          0,
          0,
          1};
}

M4 Rotate(const M4 &R, const float *v, double *out) {
  for (int i = 0; i < 3; ++i) out[i] = R[4 * i] * v[0] + R[4 * i + 1] * v[1] + R[4 * i + 2] * v[2];
  return R;
}

/** World link poses (one 4x4 per link) of joint_q `q` by the documented convention. */
std::vector<M4> ReferenceFK(const ArticulationDescription &d, const float *q) {
  const int J = static_cast<int>(d.joint_type.size());
  std::vector<M4> poses(J);
  std::vector<bool> done(J, false);
  const M4 root = FromPose(d.root_pose.data());
  for (int pass = 0; pass < J; ++pass) {
    for (int j = 0; j < J; ++j) {
      const int c = d.joint_child[j], p = d.joint_parent[j];
      if (done[c] || (p >= 0 && !done[p])) continue;
      const M4 A = Mul(p < 0 ? root : poses[p], FromPose(d.joint_X_p.data() + 7 * j));
      const float *qj = q + d.joint_q_start[j];
      const float *ax = d.joint_axis.data() + 3 * d.joint_qd_start[j];
      M4 X = Identity();
      switch (static_cast<JT>(d.joint_type[j])) {
        case JT::kPrismatic:
          for (int i = 0; i < 3; ++i) X[4 * i + 3] = ax[i] * qj[0];
          break;
        case JT::kRevolute:
          X = AxisAngle(ax, qj[0]);
          break;
        case JT::kBall: {
          const float pq[7] = {0, 0, 0, qj[0], qj[1], qj[2], qj[3]};
          X = FromPose(pq);
          break;
        }
        case JT::kFree:
          X = FromPose(qj);
          break;
        case JT::kD6: {
          const int lin = d.joint_dof_dim[2 * j], ang = d.joint_dof_dim[2 * j + 1];
          M4 R = Identity();
          for (int k = 0; k < ang; ++k) {
            double a[3];
            Rotate(R, ax + 3 * (lin + k), a);
            const float af[3] = {float(a[0]), float(a[1]), float(a[2])};
            R = Mul(AxisAngle(af, qj[lin + k]), R);
          }
          X = R;
          for (int k = 0; k < lin; ++k)
            for (int i = 0; i < 3; ++i) X[4 * i + 3] += ax[3 * k + i] * qj[k];
          break;
        }
        default:
          break;
      }
      poses[c] = Mul(Mul(A, X), Inverse(FromPose(d.joint_X_c.data() + 7 * j)));
      done[c] = true;
    }
  }
  return poses;
}

/** Random joint_q (normalized quaternions, scalars within their limits or ±1). */
/** The larger of `worst` and `err`; infinity for a non-finite err (std::max would drop a NaN). */
double Worst(double worst, double err) {
  return std::isfinite(err) ? std::max(worst, err) : std::numeric_limits<double>::infinity();
}

std::vector<float> RandomQ(const ArticulationDescription &d, std::mt19937 &rng, float scale = 1.f) {
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::vector<float> q(d.joint_q_start.back());
  for (size_t j = 0; j < d.joint_type.size(); ++j) {
    const int q0 = d.joint_q_start[j], d0 = d.joint_qd_start[j];
    auto quat = [&](float *o) {
      float n = 0;
      for (int i = 0; i < 3; ++i) {
        o[i] = scale * 0.5f * u(rng);
        n += o[i] * o[i];
      }
      o[3] = 1.f;
      n = std::sqrt(n + 1.f);
      for (int i = 0; i < 4; ++i) o[i] /= n;
    };
    switch (static_cast<JT>(d.joint_type[j])) {
      case JT::kBall:
        quat(q.data() + q0);
        break;
      case JT::kFree:
        for (int i = 0; i < 3; ++i) q[q0 + i] = scale * u(rng);
        quat(q.data() + q0 + 3);
        break;
      default:
        for (int i = q0; i < d.joint_q_start[j + 1]; ++i) {
          const int k = d0 + (i - q0);
          const float lo = std::isfinite(d.limit_lower[k]) ? d.limit_lower[k] : -1.f;
          const float hi = std::isfinite(d.limit_upper[k]) ? d.limit_upper[k] : 1.f;
          q[i] = lo + (hi - lo) * 0.5f * (1.f + scale * 0.9f * u(rng));
        }
        break;
    }
  }
  return q;
}

/** Pose of link `link` (position + quaternion xyzw) from a float64 reference pose. */
std::array<float, 7> PoseOf(const M4 &T) {
  const double tr = T[0] + T[5] + T[10];
  double x, y, z, w;
  if (tr > 0) {
    const double s = std::sqrt(tr + 1.0) * 2;
    w = 0.25 * s, x = (T[9] - T[6]) / s, y = (T[2] - T[8]) / s, z = (T[4] - T[1]) / s;
  } else if (T[0] > T[5] && T[0] > T[10]) {
    const double s = std::sqrt(1.0 + T[0] - T[5] - T[10]) * 2;
    w = (T[9] - T[6]) / s, x = 0.25 * s, y = (T[1] + T[4]) / s, z = (T[2] + T[8]) / s;
  } else if (T[5] > T[10]) {
    const double s = std::sqrt(1.0 + T[5] - T[0] - T[10]) * 2;
    w = (T[2] - T[8]) / s, x = (T[1] + T[4]) / s, y = 0.25 * s, z = (T[6] + T[9]) / s;
  } else {
    const double s = std::sqrt(1.0 + T[10] - T[0] - T[5]) * 2;
    w = (T[4] - T[1]) / s, x = (T[2] + T[8]) / s, y = (T[6] + T[9]) / s, z = 0.25 * s;
  }
  return {float(T[3]), float(T[7]), float(T[11]), float(x), float(y), float(z), float(w)};
}

template <typename T>
std::vector<T> Host(const dvector<T> &d) {
  std::vector<T> h(d.size());
  if (!h.empty()) d.CopyToHost(h.data(), h.size());
  return h;
}

std::vector<float> HostState(const ArticulationStateBatch &s, size_t n) {
  std::vector<float> h(n * s.AmbientSize());
  THROW_ON_CUDA_ERROR(
      cudaMemcpy(h.data(), s.StateDevicePtr(0), h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

TEST(ArticulationTopology, SizesAndLimits) {
  const ArticulationDescription d = MixedTree();
  ArticulationTopology topo(d);
  EXPECT_EQ(topo.JointCount(), 8);
  EXPECT_EQ(topo.CoordCount(), 7 + 4 + 1 + 1 + 0 + 3 + 3 + 1);
  EXPECT_EQ(topo.DofCount(), 6 + 3 + 1 + 1 + 0 + 3 + 3 + 1);
  EXPECT_EQ(topo.AmbientSize(), topo.CoordCount() + 12 * 8 + 6 * topo.DofCount());
  EXPECT_TRUE(topo.HasLimits());
  EXPECT_TRUE(ArticulationTopology(Arm()).HasLimits());
}

void ExpectInvalid(const ArticulationDescription &d, const std::string &needle) {
  try {
    ArticulationTopology topo(d);
    FAIL() << "expected std::invalid_argument containing '" << needle << "'";
  } catch (const std::invalid_argument &e) {
    const std::string what = e.what();
    EXPECT_NE(what.find(needle), std::string::npos) << what;
    EXPECT_NE(what.find("Fix:"), std::string::npos) << what;
  }
}

TEST(ArticulationTopology, RejectsInvalidDescriptions) {
  {
    auto d = Arm();
    d.joint_child[3] = 2;  // link 2 gets two tree joints
    ExpectInvalid(d, "child of two tree joints");
  }
  {
    auto d = Arm();
    d.joint_parent[0] = 6;  // cycle: no root
    ExpectInvalid(d, "do not form a tree");
  }
  {
    auto d = Arm();
    d.joint_type[2] = 5;  // DISTANCE
    ExpectInvalid(d, "unsupported type");
  }
  {
    auto d = Arm();
    d.joint_axis[3] = 2.f;  // non-unit axis
    ExpectInvalid(d, "non-unit axis");
  }
  {
    auto d = MixedTree();
    d.limit_lower[0] = -1.f;  // FREE DOF with a finite limit
    ExpectInvalid(d, "finite limit");
  }
  {
    auto d = Arm();
    d.joint_q_start[3] += 1;
    ExpectInvalid(d, "offsets do not match");
  }
}

// ---------------------------------------------------------------------------
// State: forward kinematics, Plus, bounds
// ---------------------------------------------------------------------------

TEST(ArticulationStateBatch, ForwardKinematicsMatchesReference) {
  const ArticulationDescription d = MixedTree();
  ArticulationTopology topo(d);
  const int n = 200, C = topo.CoordCount();
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  std::mt19937 rng(3);
  std::vector<float> q(size_t(n) * C);
  for (int s = 0; s < n; ++s) {
    const auto qs = RandomQ(d, rng);
    std::copy(qs.begin(), qs.end(), q.begin() + size_t(s) * C);
  }
  dvector<float> d_q(q);
  CudaStream stream;
  states.SetJointQ(d_q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto h = HostState(states, n);
  double worst = 0;
  for (int s = 0; s < n; ++s) {
    const auto ref = ReferenceFK(d, q.data() + size_t(s) * C);
    for (int l = 0; l < topo.LinkCount(); ++l) {
      const float *P = h.data() + size_t(s) * topo.AmbientSize() + topo.LinkPoseOffset(l);
      for (int i = 0; i < 12; ++i) worst = Worst(worst, std::fabs(P[i] - ref[l][i]));
    }
  }
  EXPECT_LT(worst, 2e-5);
}

// Instances inside a larger (scene-wide) array: joint_q read and written in place with a stride,
// link transforms (position + quaternion) written with a stride; the gaps stay untouched.
TEST(ArticulationStateBatch, StridedJointQAndLinkTransforms) {
  const ArticulationDescription d = MixedTree();
  ArticulationTopology topo(d);
  const int n = 50, C = topo.CoordCount(), L = topo.LinkCount();
  const int q_stride = C + 5, x_stride = 7 * L + 3;
  constexpr float kSentinel = -123.f;
  std::mt19937 rng(11);
  std::vector<float> scene(size_t(n) * q_stride, kSentinel);
  for (int s = 0; s < n; ++s) {
    const auto q = RandomQ(d, rng);
    std::copy(q.begin(), q.end(), scene.begin() + size_t(s) * q_stride);
  }
  dvector<float> d_scene(scene), d_back(std::vector<float>(scene.size(), kSentinel));
  dvector<float> d_x(std::vector<float>(size_t(n) * x_stride, kSentinel));
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  CudaStream stream;
  states.SetJointQ(d_scene.data(), n, stream.GetStream(), q_stride);
  states.GetJointQ(d_back.data(), n, stream.GetStream(), q_stride);
  states.GetLinkTransforms(d_x.data(), n, stream.GetStream(), x_stride);
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  EXPECT_EQ(Host(d_back), scene);  // same values, same untouched gaps

  const auto x = Host(d_x);
  double worst = 0;
  for (int s = 0; s < n; ++s) {
    const auto ref = ReferenceFK(d, scene.data() + size_t(s) * q_stride);
    for (int l = 0; l < L; ++l) {
      const float *X = x.data() + size_t(s) * x_stride + 7 * l;
      const auto r = PoseOf(ref[l]);
      const float sign = X[3] * r[3] + X[4] * r[4] + X[5] * r[5] + X[6] * r[6] < 0 ? -1.f : 1.f;
      for (int i = 0; i < 7; ++i) {
        worst = Worst(worst, double(std::fabs((i < 3 ? 1.f : sign) * X[i] - r[i])));
      }
    }
    for (int i = 7 * L; i < x_stride; ++i) EXPECT_EQ(x[size_t(s) * x_stride + i], kSentinel);
  }
  EXPECT_LT(worst, 5e-5);

  EXPECT_THROW(states.SetJointQ(d_scene.data(), n, stream.GetStream(), C - 1),
               std::invalid_argument);
  EXPECT_THROW(states.GetLinkTransforms(d_x.data(), n + 1, stream.GetStream()),
               std::invalid_argument);
}

TEST(ArticulationStateBatch, ZeroStepKeepsStateAndBoundsClampMovedComponents) {
  const ArticulationDescription d = Arm();
  ArticulationTopology topo(d);
  const int n = 4, T = topo.DofCount(), A = topo.AmbientSize();
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  ASSERT_NE(states.LowerBounds(), nullptr);
  CudaStream stream;
  dvector<float> out(size_t(n) * A);
  std::vector<float> delta(size_t(n) * T, 0.f);
  delta[0] = 10.f;  // pushes DOF 0 of state 0 beyond +2.4
  delta[T + 1] = -10.f;
  dvector<float> d_delta(delta);
  states.Plus(states.StateDevicePtr(0), d_delta.data(), out.data(), stream.GetStream());
  const auto h = Host(out);
  EXPECT_FLOAT_EQ(h[0], 2.4f);
  EXPECT_FLOAT_EQ(h[A + 1], -1.6f);
  EXPECT_FLOAT_EQ(h[2], 0.f);
  states.PlusIgnoringBounds(states.StateDevicePtr(0), d_delta.data(), out.data(),
                            stream.GetStream());
  EXPECT_FLOAT_EQ(Host(out)[0], 10.f);
}

// ---------------------------------------------------------------------------
// Factors: Jacobians against central differences of Plus
// ---------------------------------------------------------------------------

/**
 * Evaluates `factor` (one factor per instance t of `states`, state t) and
 * compares its Jacobian with central differences of PlusIgnoringBounds.
 */
void CheckJacobians(const FactorBatch &factor, ArticulationStateBatch &states, int n,
                    const char *label, double tolerance = 3e-3) {
  CudaStream stream;
  const int m = static_cast<int>(factor.ResidualsSize());
  const int T = static_cast<int>(states.TangentSize()), A = static_cast<int>(states.AmbientSize());
  auto pointers = [&](float *base) {
    std::vector<float *> p(n);
    for (int t = 0; t < n; ++t) p[t] = base + size_t(t) * A;
    return dvector<float *>(p);
  };
  dvector<float *> ptrs = pointers(states.StateDevicePtr(0));
  dvector<float> res(size_t(n) * m), jac(size_t(n) * m * T);
  ASSERT_TRUE(factor.Evaluate(res.data(), jac.data(), ptrs.data(), stream.GetStream()));
  const auto J = Host(jac);
  dvector<float> plus(size_t(n) * A), minus(size_t(n) * A), rp(size_t(n) * m), rm(size_t(n) * m);
  dvector<float *> pp = pointers(plus.data()), pm = pointers(minus.data());
  const float h = 1e-3f;
  double worst = 0;
  for (int k = 0; k < T; ++k) {
    std::vector<float> delta(size_t(n) * T, 0.f);
    for (int t = 0; t < n; ++t) delta[size_t(t) * T + k] = h;
    dvector<float> dp(delta);
    for (float &x : delta) x = -x;
    dvector<float> dm(delta);
    states.PlusIgnoringBounds(states.StateDevicePtr(0), dp.data(), plus.data(), stream.GetStream());
    states.PlusIgnoringBounds(states.StateDevicePtr(0), dm.data(), minus.data(),
                              stream.GetStream());
    factor.Evaluate(rp.data(), nullptr, pp.data(), stream.GetStream());
    factor.Evaluate(rm.data(), nullptr, pm.data(), stream.GetStream());
    const auto a = Host(rp), b = Host(rm);
    for (int t = 0; t < n; ++t) {
      for (int r = 0; r < m; ++r) {
        const double numeric = (a[size_t(t) * m + r] - b[size_t(t) * m + r]) / (2.0 * h);
        const double analytic = J[(size_t(t) * m + r) * T + k];
        const double err = std::fabs(numeric - analytic) / (1.0 + std::fabs(analytic));
        worst = Worst(worst, err);
        if (!(err <= tolerance)) {  // NaN fails too
          ADD_FAILURE() << label << ": item " << t << " row " << r << " column " << k
                        << " analytic " << analytic << " numeric " << numeric;
          return;
        }
      }
    }
  }
  ::testing::Test::RecordProperty(std::string(label) + "_worst", std::to_string(worst));
}

struct MixedFixture {
  ArticulationDescription d = MixedTree();
  ArticulationTopology topo{d};
  int n = 64;
  ArticulationStateBatch states{topo, static_cast<size_t>(n), false};
  std::mt19937 rng{11};

  MixedFixture() {
    states.SetNumActiveStates(n);
    std::vector<float> q;
    for (int s = 0; s < n; ++s) {
      const auto qs = RandomQ(d, rng);
      q.insert(q.end(), qs.begin(), qs.end());
    }
    dvector<float> d_q(q);
    CudaStream stream;
    states.SetJointQ(d_q.data(), n, stream.GetStream());
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  }

  std::vector<float> Uniform(size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(count);
    for (float &x : v) x = u(rng);
    return v;
  }

  std::vector<float> Quats(size_t count, size_t stride, size_t at) {
    std::vector<float> v = Uniform(count * stride, -1.f, 1.f);
    for (size_t i = 0; i < count; ++i) {
      float *q = v.data() + i * stride + at;
      const float nrm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      for (int k = 0; k < 4; ++k) q[k] /= nrm;
    }
    return v;
  }

  /** World pose of `link` of instance `s` from the state's cache. */
  M4 LinkPose(int s, int link) {
    std::vector<float> h(12);
    THROW_ON_CUDA_ERROR(cudaMemcpy(h.data(), states.StateDevicePtr(s) + topo.LinkPoseOffset(link),
                                   12 * sizeof(float), cudaMemcpyDeviceToHost));
    M4 T = Identity();
    for (int i = 0; i < 12; ++i) T[i] = h[i];
    return T;
  }

  /** A transform with a random translation (±`dist`) and a rotation of at most ~0.6 rad. */
  M4 SmallTransform(float dist) {
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    const float p[7] = {dist * u(rng),
                        dist * u(rng),
                        dist * u(rng),
                        0.3f * u(rng),
                        0.3f * u(rng),
                        0.3f * u(rng),
                        1.f};
    return FromPose(p);
  }

  std::vector<int> Links() {
    std::vector<int> l(n);
    for (int t = 0; t < n; ++t) l[t] = t % topo.LinkCount();
    return l;
  }
};

TEST(KinematicFactors, LinkTargetJacobians) {
  MixedFixture f;
  const std::vector<int> link_ids = f.Links();
  dvector<int> links(link_ids);
  dvector<float> points(f.Uniform(3 * f.n, -0.2f, 0.2f)), pos_targets(f.Uniform(3 * f.n, -1, 1));
  dvector<float> weights(f.Uniform(6 * f.n, 0.5f, 2.f));
  LinkPositionFactorBatch position(f.topo, links.data(), points.data(), pos_targets.data(),
                                   weights.data(), f.n);
  position.SetNumActiveFactors(f.n);
  CheckJacobians(position, f.states, f.n, "position");

  // Offsets and targets near the current link frames: rotation residuals well below π.
  std::vector<float> offsets, rot_targets, frames, targets;
  for (int t = 0; t < f.n; ++t) {
    const M4 off = f.SmallTransform(0.2f);
    const M4 frame = Mul(f.LinkPose(t, link_ids[t]), off);
    const auto po = PoseOf(off), pt = PoseOf(Mul(frame, f.SmallTransform(0.3f)));
    offsets.insert(offsets.end(), po.begin() + 3, po.end());
    rot_targets.insert(rot_targets.end(), pt.begin() + 3, pt.end());
    frames.insert(frames.end(), po.begin(), po.end());
    targets.insert(targets.end(), pt.begin(), pt.end());
  }
  dvector<float> d_offsets(offsets), d_rot_targets(rot_targets);
  LinkOrientationFactorBatch orientation(f.topo, links.data(), d_offsets.data(),
                                         d_rot_targets.data(), nullptr, f.n);
  orientation.SetNumActiveFactors(f.n);
  CheckJacobians(orientation, f.states, f.n, "orientation");

  dvector<float> d_frames(frames), d_targets(targets);
  LinkPoseFactorBatch pose(f.topo, links.data(), d_frames.data(), d_targets.data(), weights.data(),
                           f.n);
  pose.SetNumActiveFactors(f.n);
  CheckJacobians(pose, f.states, f.n, "pose");
  // Each copy reads the instances in another order, so an item that looked its state up by
  // factor id instead of by item index would read the wrong instance.
  evaluate_items_test::CheckEvaluateItems(pose, 2, [&](int k) {
    std::vector<float *> p(f.n);
    for (int t = 0; t < f.n; ++t) p[t] = f.states.StateDevicePtr((t + 7 * k) % f.n);
    return p;
  });
}

TEST(KinematicFactors, PostureJacobians) {
  MixedFixture f;
  std::vector<float> targets;
  for (int t = 0; t < f.n; ++t) {
    const auto q = RandomQ(f.d, f.rng);
    targets.insert(targets.end(), q.begin(), q.end());
  }
  dvector<float> d_targets(targets), weights(f.Uniform(size_t(f.n) * f.topo.DofCount(), 0.5f, 2));
  PostureFactorBatch posture(f.topo, d_targets.data(), weights.data(), f.n);
  posture.SetNumActiveFactors(f.n);
  CheckJacobians(posture, f.states, f.n, "posture");
}

TEST(KinematicFactors, LoopClosureJacobians) {
  MixedFixture f;
  std::vector<int> parents(f.n), children(f.n);
  std::vector<float> frames;
  for (int t = 0; t < f.n; ++t) {
    parents[t] = (t % 3 == 0) ? -1 : (t % f.topo.LinkCount());
    children[t] = (t * 5 + 3) % f.topo.LinkCount();
    // X_c chosen so that F_C's rotation is near F_P's (rotation rows away from π).
    const M4 Xp = f.SmallTransform(0.2f);
    const M4 FP = Mul(parents[t] >= 0 ? f.LinkPose(t, parents[t]) : Identity(), Xp);
    const M4 FC = Mul(FP, f.SmallTransform(0.3f));
    const M4 Xc = Mul(Inverse(f.LinkPose(t, children[t])), FC);
    const auto pp = PoseOf(Xp), pc = PoseOf(Xc);
    frames.insert(frames.end(), pp.begin(), pp.end());
    frames.insert(frames.end(), pc.begin(), pc.end());
  }
  dvector<int> d_parents(parents), d_children(children);
  dvector<float> d_frames(frames), axes(f.Uniform(3 * f.n, -1.f, 1.f));
  for (JT type : {JT::kBall, JT::kRevolute, JT::kPrismatic, JT::kFixed}) {
    LoopClosureFactorBatch loop(f.topo, type, d_parents.data(), d_children.data(), d_frames.data(),
                                axes.data(), f.n);
    loop.SetNumActiveFactors(f.n);
    CheckJacobians(loop, f.states, f.n,
                   ("loop closure type " + std::to_string(static_cast<int>(type))).c_str());
  }
}

TEST(KinematicFactors, MimicJacobian) {
  MixedFixture f;
  // Scalar DOFs of MixedTree: 9 (revolute), 10 (prismatic), 11..13 (D6), 14..16 (D6), 17.
  std::vector<int> mimic(f.n), ref(f.n);
  for (int t = 0; t < f.n; ++t) {
    mimic[t] = 9 + t % 9;
    ref[t] = 9 + (t + 4) % 9;
  }
  dvector<int> d_mimic(mimic), d_ref(ref);
  dvector<float> coeffs(f.Uniform(2 * f.n, -2.f, 2.f));
  MimicJointFactorBatch factor(f.topo, d_mimic.data(), d_ref.data(), coeffs.data(), f.n);
  factor.SetNumActiveFactors(f.n);
  CheckJacobians(factor, f.states, f.n, "mimic");
}

// ---------------------------------------------------------------------------
// Solves
// ---------------------------------------------------------------------------

TEST(ArticulationSolve, BoundedIKReachesTargetsWithinLimits) {
  const ArticulationDescription d = Arm();
  ArticulationTopology topo(d);
  const int n = 256, C = topo.CoordCount(), T = topo.DofCount(), ee = 6;
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  std::mt19937 rng(5);
  std::vector<float> targets, q0;
  for (int s = 0; s < n; ++s) {
    const auto q = RandomQ(d, rng);  // reachable: FK of an in-limit configuration
    const auto p = PoseOf(ReferenceFK(d, q.data())[ee]);
    targets.insert(targets.end(), p.begin(), p.end());
    // Initial guess: the true configuration perturbed by ±0.4 rad (local convergence;
    // global search is multi-start's job), possibly outside the limits.
    std::uniform_real_distribution<float> u(-0.4f, 0.4f);
    for (int k = 0; k < C; ++k) q0.push_back(q[k] + u(rng));
  }
  // One unreachable-without-violating target: its configuration exceeds DOF 1's limit.
  std::vector<float> outside(C, 0.3f);
  outside[1] = 2.4f;  // limit 1.6
  const auto po = PoseOf(ReferenceFK(d, outside.data())[ee]);
  std::copy(po.begin(), po.end(), targets.begin());
  std::copy(outside.begin(), outside.end(), q0.begin());  // starts beyond the limit
  dvector<float> d_targets(targets), d_q0(q0);
  dvector<int> links(std::vector<int>(n, ee));
  CudaStream stream;
  states.SetJointQ(d_q0.data(), n, stream.GetStream());
  LinkPoseFactorBatch pose(topo, links.data(), nullptr, d_targets.data(), nullptr, n);
  pose.SetNumActiveFactors(n);
  std::vector<float> nominal(size_t(n) * C, 0.f);
  dvector<float> d_nominal(nominal), weights(std::vector<float>(size_t(n) * T, 1e-3f));
  PostureFactorBatch posture(topo, d_nominal.data(), weights.data(), n);
  posture.SetNumActiveFactors(n);

  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs(n);
  for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
  problem.AddFactorBatch(&pose, ptrs);
  problem.AddFactorBatch(&posture, ptrs);
  std::vector<int> ids(n);
  for (int s = 0; s < n; ++s) ids[s] = s;
  dvector<int> d_ids(ids);
  problem.SetProblemPartition(n, {d_ids.data()});

  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options.max_num_iterations = 100;
  LevenbergMarquardtMinimizer inner(lm);
  AugmentedLagrangianMinimizer solver(inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_NE(summary.status, AugmentedLagrangianMinimizerStatus::kInnerSolveFailed);

  dvector<float> q(size_t(n) * C);
  states.GetJointQ(q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hq = Host(q);
  int reached = 0;
  for (int s = 0; s < n; ++s) {
    for (int k = 0; k < T; ++k) {
      EXPECT_GE(hq[size_t(s) * C + k], topo.LowerLimits()[k] - 1e-6f) << "state " << s;
      EXPECT_LE(hq[size_t(s) * C + k], topo.UpperLimits()[k] + 1e-6f) << "state " << s;
    }
    const auto fk = ReferenceFK(d, hq.data() + size_t(s) * C)[ee];
    const float *tp = targets.data() + 7 * s;
    const double err = std::sqrt(std::pow(fk[3] - tp[0], 2) + std::pow(fk[7] - tp[1], 2) +
                                 std::pow(fk[11] - tp[2], 2));
    if (err < 1e-3) ++reached;
  }
  EXPECT_GE(reached, n * 97 / 100) << reached << " of " << n;
  EXPECT_FLOAT_EQ(hq[1], 1.6f);  // the unreachable target ends at the limit
}

// A DOF held by a zero-width box (lower == upper == its start value) stays there while the others
// move within their limits to reach the target.
TEST(ArticulationSolve, ZeroWidthBoundsFreezeADof) {
  const ArticulationDescription d = Arm();
  ArticulationTopology topo(d);
  const int n = 64, C = topo.CoordCount(), T = topo.DofCount(), ee = 6, frozen = 2;
  std::mt19937 rng(9);
  std::vector<float> targets, q0, lower, upper;
  for (int s = 0; s < n; ++s) {
    const auto q = RandomQ(d, rng);
    const auto p = PoseOf(ReferenceFK(d, q.data())[ee]);
    targets.insert(targets.end(), p.begin(), p.end());
    std::uniform_real_distribution<float> u(-0.3f, 0.3f);
    for (int k = 0; k < C; ++k) {
      q0.push_back(k == frozen
                       ? q[k]
                       : std::clamp(q[k] + u(rng), topo.LowerLimits()[k], topo.UpperLimits()[k]));
    }
    for (int k = 0; k < T; ++k) {
      lower.push_back(k == frozen ? q[k] : topo.LowerLimits()[k]);
      upper.push_back(k == frozen ? q[k] : topo.UpperLimits()[k]);
    }
  }
  dvector<float> d_targets(targets), d_q0(q0), d_lower(lower), d_upper(upper);
  dvector<int> links(std::vector<int>(n, ee));
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  states.SetBounds(d_lower.data(), d_upper.data());
  CudaStream stream;
  states.SetJointQ(d_q0.data(), n, stream.GetStream());
  LinkPoseFactorBatch pose(topo, links.data(), nullptr, d_targets.data(), nullptr, n);
  pose.SetNumActiveFactors(n);
  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs(n);
  for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
  problem.AddFactorBatch(&pose, ptrs);
  std::vector<int> ids(n);
  for (int s = 0; s < n; ++s) ids[s] = s;
  dvector<int> d_ids(ids);
  problem.SetProblemPartition(n, {d_ids.data()});
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options.max_num_iterations = 100;
  LevenbergMarquardtMinimizer inner(lm);
  AugmentedLagrangianMinimizer solver(inner);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_NE(summary.status, AugmentedLagrangianMinimizerStatus::kInnerSolveFailed);

  dvector<float> q(size_t(n) * C);
  states.GetJointQ(q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hq = Host(q);
  int reached = 0;
  for (int s = 0; s < n; ++s) {
    EXPECT_EQ(hq[size_t(s) * C + frozen], q0[size_t(s) * C + frozen]) << "state " << s;
    for (int k = 0; k < T; ++k) {
      EXPECT_GE(hq[size_t(s) * C + k], lower[size_t(s) * T + k] - 1e-6f) << "state " << s;
      EXPECT_LE(hq[size_t(s) * C + k], upper[size_t(s) * T + k] + 1e-6f) << "state " << s;
    }
    const auto fk = ReferenceFK(d, hq.data() + size_t(s) * C)[ee];
    const float *tp = targets.data() + 7 * s;
    const double err = std::sqrt(std::pow(fk[3] - tp[0], 2) + std::pow(fk[7] - tp[1], 2) +
                                 std::pow(fk[11] - tp[2], 2));
    if (err < 1e-3) ++reached;
  }
  EXPECT_GE(reached, n * 9 / 10) << reached << " of " << n;
}

TEST(ArticulationSolve, FourBarLoopClosureAssembles) {
  // Ground (link 0, fixed root), crank (1) at the origin, coupler (2) at the
  // crank tip, rocker (3) at (G, 0, 0); the coupler tip and the rocker tip are
  // joined by a loop-closing REVOLUTE joint about z.
  const float G = 1.0f, L1 = 0.4f, L2 = 1.0f, L3 = 0.8f;
  ArticulationDescription d;
  AddJoint(d, JT::kFixed, -1, Pose(0, 0, 0), Pose(0, 0, 0), {});
  AddJoint(d, JT::kRevolute, 0, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(d, JT::kRevolute, 1, Pose(L1, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(d, JT::kRevolute, 0, Pose(G, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  ArticulationTopology topo(d);
  const int n = 32, C = topo.CoordCount();
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  std::vector<float> q0, nominal;
  for (int s = 0; s < n; ++s) {
    const float crank = 0.3f + 0.08f * s;
    q0.insert(q0.end(), {crank, 0.f, 1.2f});  // coupler and rocker angles: rough guess
    nominal.insert(nominal.end(), {crank, 0.f, 1.2f});
  }
  dvector<float> d_q0(q0), d_nominal(nominal);
  std::vector<float> w;
  for (int s = 0; s < n; ++s) w.insert(w.end(), {100.f, 1e-2f, 1e-2f});
  dvector<float> d_w(w);
  CudaStream stream;
  states.SetJointQ(d_q0.data(), n, stream.GetStream());
  PostureFactorBatch posture(topo, d_nominal.data(), d_w.data(), n);
  posture.SetNumActiveFactors(n);
  dvector<int> parents(std::vector<int>(n, 2)), children(std::vector<int>(n, 3));
  std::vector<float> frames;
  for (int s = 0; s < n; ++s)
    frames.insert(frames.end(), {L2, 0, 0, 0, 0, 0, 1, L3, 0, 0, 0, 0, 0, 1});
  dvector<float> d_frames(frames);
  std::vector<float> axes;
  for (int s = 0; s < n; ++s) axes.insert(axes.end(), {0.f, 0.f, 1.f});
  dvector<float> d_axes(axes);
  LoopClosureFactorBatch loop(topo, JT::kRevolute, parents.data(), children.data(), d_frames.data(),
                              d_axes.data(), n);
  loop.SetNumActiveFactors(n);
  ConstraintFactorBatch closed(&loop, ConstraintKind::kEquality);

  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs(n);
  for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
  problem.AddFactorBatch(&posture, ptrs);
  problem.AddFactorBatch(&closed, ptrs);
  std::vector<int> ids(n);
  for (int s = 0; s < n; ++s) ids[s] = s;
  dvector<int> d_ids(ids);
  problem.SetProblemPartition(n, {d_ids.data()});
  LevenbergMarquardtMinimizerOptions lm;
  lm.base_options.max_num_iterations = 100;
  LevenbergMarquardtMinimizer inner(lm);
  AugmentedLagrangianMinimizerOptions al;
  al.constraint_tolerance = 1e-5f;
  AugmentedLagrangianMinimizer solver(inner, al);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged) << summary.message;
  EXPECT_LT(summary.max_violation, 1e-5f);

  dvector<float> q(size_t(n) * C);
  states.GetJointQ(q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hq = Host(q);
  for (int s = 0; s < n; ++s) {
    const auto poses = ReferenceFK(d, hq.data() + size_t(s) * C);
    // Coupler tip (link 2 at (L2, 0, 0)) and rocker tip (link 3 at (L3, 0, 0)) coincide.
    const double cx = poses[2][0] * L2 + poses[2][3], cy = poses[2][4] * L2 + poses[2][7];
    const double rx = poses[3][0] * L3 + poses[3][3], ry = poses[3][4] * L3 + poses[3][7];
    EXPECT_LT(std::hypot(cx - rx, cy - ry), 1e-4) << "state " << s;
    EXPECT_NEAR(hq[size_t(s) * C], 0.3f + 0.08f * s, 1e-3f) << "crank moved, state " << s;
  }
}

TEST(ArticulationSolve, MimicJointHolds) {
  ArticulationDescription d;
  AddJoint(d, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(d, JT::kRevolute, 0, Pose(0.2f, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  ArticulationTopology topo(d);
  const int n = 8;
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  dvector<float> nominal(std::vector<float>(2 * n, 0.3f));
  PostureFactorBatch posture(topo, nominal.data(), nullptr, n);
  posture.SetNumActiveFactors(n);
  dvector<int> m(std::vector<int>(n, 1)), r(std::vector<int>(n, 0));
  std::vector<float> coeffs;
  for (int s = 0; s < n; ++s) coeffs.insert(coeffs.end(), {0.1f, 2.f});
  dvector<float> d_coeffs(coeffs);
  MimicJointFactorBatch mimic(topo, m.data(), r.data(), d_coeffs.data(), n);
  mimic.SetNumActiveFactors(n);
  ConstraintFactorBatch coupled(&mimic, ConstraintKind::kEquality);
  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs(n);
  for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
  problem.AddFactorBatch(&posture, ptrs);
  problem.AddFactorBatch(&coupled, ptrs);
  LevenbergMarquardtMinimizer inner;
  AugmentedLagrangianMinimizer solver(inner);
  CudaStream stream;
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  EXPECT_EQ(summary.status, AugmentedLagrangianMinimizerStatus::kConverged) << summary.message;
  dvector<float> q(2 * n);
  states.GetJointQ(q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto hq = Host(q);
  for (int s = 0; s < n; ++s) {
    EXPECT_NEAR(hq[2 * s + 1], 0.1f + 2.f * hq[2 * s], 2e-4f);
    // Least squares on the constraint line: q0 = (0.3 + 2 (0.3 - 0.1)) / 5 = 0.14.
    EXPECT_NEAR(hq[2 * s], 0.14f, 2e-3f);
  }
}

}  // namespace
}  // namespace cunls

namespace cunls {
namespace {

/** Problem counts of DISABLED_IkThroughput: IK_BENCH_N (one value) or 512, 4096, 32768. */
std::vector<int> IkBenchSizes() {
  if (const char *n = std::getenv("IK_BENCH_N")) return {std::atoi(n)};
  return {512, 4096, 32768};
}

/**
 * Throughput of batched bounded IK on the 7-DOF arm (the size of Newton's FastIKSolve FR3
 * benchmark): N problems, one subproblem each, a LinkPose target, a fixed budget of 16 LM
 * iterations, plain LM: no line search and no bound refinement (real time), as Newton's captured
 * IKSolver.step(iterations=16); IK_BENCH_LS / IK_BENCH_REFINE override them.
 * Prints the median over 10 solves; success = position error < 5 mm and limits respected.
 */
TEST(ArticulationBenchmark, DISABLED_IkThroughput) {
  const ArticulationDescription d = Arm();
  ArticulationTopology topo(d);
  const int C = topo.CoordCount(), T = topo.DofCount(), ee = 6;
  std::vector<SparseLinearSolverType> solvers = {SparseLinearSolverType::BlockTridiagonal,
                                                 SparseLinearSolverType::BlockSparseCholesky};
  if (std::getenv("IK_BENCH_TRIDIAGONAL_ONLY")) solvers.pop_back();
  for (SparseLinearSolverType solver_type : solvers) {
    for (int n : IkBenchSizes()) {
      ArticulationStateBatch states(topo, n);
      states.SetNumActiveStates(n);
      std::mt19937 rng(9);
      std::vector<float> targets, q0;
      for (int s = 0; s < n; ++s) {
        const auto q = RandomQ(d, rng);
        const auto p = PoseOf(ReferenceFK(d, q.data())[ee]);
        targets.insert(targets.end(), p.begin(), p.end());
        std::uniform_real_distribution<float> u(-0.4f, 0.4f);
        for (int k = 0; k < C; ++k) q0.push_back(q[k] + u(rng));
      }
      dvector<float> d_targets(targets), d_q0(q0);
      dvector<int> links(std::vector<int>(n, ee));
      LinkPoseFactorBatch pose(topo, links.data(), nullptr, d_targets.data(), nullptr, n);
      pose.SetNumActiveFactors(n);
      Problem problem;
      problem.AddStateBatch(&states);
      std::vector<float *> ptrs(n);
      for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
      problem.AddFactorBatch(&pose, ptrs);
      std::vector<int> ids(n);
      for (int s = 0; s < n; ++s) ids[s] = s;
      dvector<int> d_ids(ids), stages(std::vector<int>(n, 0));
      problem.SetProblemPartition(n, {d_ids.data()});
      if (solver_type == SparseLinearSolverType::BlockTridiagonal) {
        problem.SetStateStages({stages.data()});
      }
      LevenbergMarquardtMinimizerOptions lm;
      lm.base_options.max_num_iterations = 16;
      lm.base_options.sparse_linear_solver_type = solver_type;
      LevenbergMarquardtMinimizer inner(lm);
      AugmentedLagrangianMinimizerOptions al;
      al.real_time = true;
      al.max_outer_iterations = 1;
      al.inner_iterations = 16;
      al.final_inner_iterations = 16;
      al.inner_line_search_steps =
          std::getenv("IK_BENCH_LS") ? std::atoi(std::getenv("IK_BENCH_LS")) : 0;
      al.max_bound_refinements =
          std::getenv("IK_BENCH_REFINE") ? std::atoi(std::getenv("IK_BENCH_REFINE")) : 0;
      al.reuse_structure = true;
      AugmentedLagrangianMinimizer solver(inner, al);
      CudaStream stream;
      std::vector<double> ms;
      for (int rep = 0; rep < 11; ++rep) {
        states.SetJointQ(d_q0.data(), n, stream.GetStream());
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        const auto start = std::chrono::steady_clock::now();
        solver.Minimize(stream.GetStream(), problem);
        THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
        if (rep > 0) {
          ms.push_back(
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                  .count());
        }
      }
      std::sort(ms.begin(), ms.end());
      dvector<float> q(size_t(n) * C);
      states.GetJointQ(q.data(), n, stream.GetStream());
      THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
      const auto hq = Host(q);
      int ok = 0, violations = 0;
      for (int s = 0; s < n; ++s) {
        bool inside = true;
        for (int k = 0; k < T; ++k) {
          const float v = hq[size_t(s) * C + k];
          inside =
              inside && v >= topo.LowerLimits()[k] - 1e-6f && v <= topo.UpperLimits()[k] + 1e-6f;
        }
        violations += inside ? 0 : 1;
        const auto fk = ReferenceFK(d, hq.data() + size_t(s) * C)[ee];
        const float *tp = targets.data() + 7 * s;
        const double err = std::sqrt(std::pow(fk[3] - tp[0], 2) + std::pow(fk[7] - tp[1], 2) +
                                     std::pow(fk[11] - tp[2], 2));
        ok += inside && err < 5e-3 ? 1 : 0;  // a solution outside the limits is no success
      }
      printf("BENCH ik_arm7_%-6d %-20s median_ms=%8.3f success=%5.1f%% outside_limits=%d\n", n,
             solver_type == SparseLinearSolverType::BlockTridiagonal ? "BlockTridiagonal"
                                                                     : "BlockSparseCholesky",
             ms[ms.size() / 2], 100.0 * ok / n, violations);
    }
  }
}

// A factor built for one topology must not read the state of another with the same DOF count:
// the cache layout (ambient size) differs, so the kernels would read the wrong floats.
TEST(ArticulationSolve, ProblemRejectsStateOfAnotherLayout) {
  ArticulationDescription one, two;
  AddJoint(one, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(two, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(two, JT::kFixed, 0, Pose(0.3f, 0, 0), Pose(0, 0, 0), {});
  ArticulationTopology topo_one(one), topo_two(two);
  ASSERT_EQ(topo_one.DofCount(), topo_two.DofCount());
  ASSERT_NE(topo_one.AmbientSize(), topo_two.AmbientSize());
  ArticulationStateBatch states(topo_two, 1);
  states.SetNumActiveStates(1);
  dvector<float> targets(std::vector<float>{0.2f});
  CudaStream stream;
  for (const ArticulationTopology *topo : {&topo_one, &topo_two}) {
    PostureFactorBatch posture(*topo, targets.data(), nullptr, 1);
    posture.SetNumActiveFactors(1);
    ConstraintFactorBatch constrained(&posture, ConstraintKind::kEquality);
    for (FactorBatch *factor :
         {static_cast<FactorBatch *>(&posture), static_cast<FactorBatch *>(&constrained)}) {
      Problem problem;
      problem.AddStateBatch(&states);
      problem.AddFactorBatch(factor, std::vector<float *>{states.StateDevicePtr(0)});
      EXPECT_EQ(problem.Validate(stream.GetStream()), topo == &topo_two);
    }
  }
}

// A REVOLUTE loop closure must not be satisfied by axes that point in opposite directions.
TEST(KinematicFactors, RevoluteLoopRejectsOppositeAxes) {
  ArticulationDescription d;  // link 0 turns about x; the loop joint (about z) ties it to the world
  AddJoint(d, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{1, 0, 0}});
  ArticulationTopology topo(d);
  ArticulationStateBatch states(topo, 2);
  states.SetNumActiveStates(2);
  dvector<float> q(std::vector<float>{0.f, 3.14159265f});  // aligned, then flipped by π about x
  CudaStream stream;
  states.SetJointQ(q.data(), 2, stream.GetStream());
  dvector<int> parents(std::vector<int>{-1}), children(std::vector<int>{0});
  dvector<float> frames(std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});
  dvector<float> axes(std::vector<float>{0, 0, 1});
  LoopClosureFactorBatch loop(topo, JT::kRevolute, parents.data(), children.data(), frames.data(),
                              axes.data(), 1);
  loop.SetNumActiveFactors(1);
  dvector<float *> ptrs(std::vector<float *>{states.StateDevicePtr(0), states.StateDevicePtr(1)});
  const int m = static_cast<int>(loop.ResidualsSize());
  dvector<float> r(2 * m);
  ASSERT_TRUE(loop.Evaluate(r.data(), nullptr, ptrs.data(), stream.GetStream(), nullptr, 2));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto h = Host(r);
  double aligned = 0, flipped = 0;
  for (int i = 0; i < m; ++i) {
    aligned += h[i] * h[i];
    flipped += h[m + i] * h[m + i];
  }
  EXPECT_LT(std::sqrt(aligned), 1e-6);
  EXPECT_NEAR(std::sqrt(flipped), 2.0, 1e-5);  // |a_C - a_P| = |-z - z|
}

// A topology whose staged arrays and one instance exceed 48 KB of shared memory takes the
// unstaged forward-kinematics path (one revolute joint, then 299 fixed links).
TEST(ArticulationStateBatch, ForwardKinematicsBeyondSharedMemory) {
  ArticulationDescription d;
  AddJoint(d, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  for (int j = 1; j < 300; ++j)
    AddJoint(d, JT::kFixed, j - 1, Pose(0.01f, 0, 0), Pose(0, 0, 0), {});
  ArticulationTopology topo(d);
  const int n = 3, C = topo.CoordCount();
  ArticulationStateBatch states(topo, n);
  states.SetNumActiveStates(n);
  const std::vector<float> q = {0.f, 0.7f, -2.f};
  dvector<float> d_q(q);
  CudaStream stream;
  states.SetJointQ(d_q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto h = HostState(states, n);
  double worst = 0;
  for (int s = 0; s < n; ++s) {
    const auto ref = ReferenceFK(d, q.data() + size_t(s) * C);
    for (int l = 0; l < topo.LinkCount(); ++l) {
      const float *P = h.data() + size_t(s) * topo.AmbientSize() + topo.LinkPoseOffset(l);
      for (int i = 0; i < 12; ++i) worst = Worst(worst, std::fabs(P[i] - ref[l][i]));
    }
  }
  EXPECT_LT(worst, 2e-5);
}

}  // namespace
}  // namespace cunls
