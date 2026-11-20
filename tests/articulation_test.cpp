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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/between/se3_between_factor_batch.h"
#include "cunls/factor/between/so3_between_factor_batch.h"
#include "cunls/factor/between/vector_between_factor_batch.h"
#include "cunls/factor/constraint_factor_batch.h"
#include "cunls/factor/halfspace_factor_batch.h"
#include "cunls/factor/kinematics/link_frame_factor_batch.h"
#include "cunls/factor/kinematics/link_target_factor_batch.h"
#include "cunls/factor/kinematics/mimic_joint_factor_batch.h"
#include "cunls/factor/kinematics/posture_factor_batch.h"
#include "cunls/factor/prior/prior_vector_factor_batch.h"
#include "cunls/factor/prior/se3_prior_factor_batch.h"
#include "cunls/factor/prior/so3_prior_factor_batch.h"
#include "cunls/factor/se3_joint_axis_factor_batch.h"
#include "cunls/linear_solver/block_tridiagonal_solver.h"
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

/** A 7-DOF arm (or `joints`): revolute joints alternating about z and y, 0.3 m links, limits
 * ±(2.4, 1.6). */
ArticulationDescription Arm(int joints = 7) {
  ArticulationDescription d;
  for (int j = 0; j < joints; ++j) {
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
/** Row-major 4x4 floats of `T`, appended to `out`. */
void AppendMatrix(const M4 &T, std::vector<float> &out) {
  for (double v : T) out.push_back(static_cast<float>(v));
}

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
  const auto back = Host(d_back);  // same values (quaternions renormalized), untouched gaps
  ASSERT_EQ(back.size(), scene.size());
  for (size_t i = 0; i < scene.size(); ++i) {
    if (scene[i] == kSentinel) {
      EXPECT_EQ(back[i], kSentinel) << i;
    } else {
      EXPECT_NEAR(back[i], scene[i], 1e-6) << i;
    }
  }

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
 * Evaluates `factor` (one factor per instance t of `states`, state t, then
 * extra[t] when `extra` is given: a second state of `extra_cols` tangent
 * columns, held fixed) and compares its DOF columns with central differences
 * of PlusIgnoringBounds.
 */
void CheckJacobians(const FactorBatch &factor, ArticulationStateBatch &states, int n,
                    const char *label, double tolerance = 3e-3,
                    const std::vector<float *> &extra = {}, int extra_cols = 0) {
  CudaStream stream;
  const int m = static_cast<int>(factor.ResidualsSize());
  const int T = static_cast<int>(states.TangentSize()), A = static_cast<int>(states.AmbientSize());
  const int Jc = T + extra_cols;
  auto pointers = [&](float *base) {
    std::vector<float *> p;
    for (int t = 0; t < n; ++t) {
      p.push_back(base + size_t(t) * A);
      if (!extra.empty()) p.push_back(extra[t]);
    }
    return dvector<float *>(p);
  };
  dvector<float *> ptrs = pointers(states.StateDevicePtr(0));
  dvector<float> res(size_t(n) * m), jac(size_t(n) * m * Jc);
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
        const double analytic = J[(size_t(t) * m + r) * Jc + k];
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

  AugmentedLagrangianMinimizerOptions options;
  options.base_options.max_num_iterations = 100;
  AugmentedLagrangianMinimizer solver(options);
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
  AugmentedLagrangianMinimizerOptions options;
  options.base_options.max_num_iterations = 100;
  AugmentedLagrangianMinimizer solver(options);
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
  AugmentedLagrangianMinimizer solver;
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

/** Joints of the arm of DISABLED_IkThroughput: IK_BENCH_JOINTS or 7. */
int IkBenchJoints() {
  const char *joints = std::getenv("IK_BENCH_JOINTS");
  return joints ? std::atoi(joints) : 7;
}

/** Problem counts of DISABLED_IkThroughput: IK_BENCH_N (one value) or 512, 4096, 32768. */
std::vector<int> IkBenchSizes() {
  if (const char *n = std::getenv("IK_BENCH_N")) return {std::atoi(n)};
  return {512, 4096, 32768};
}

/**
 * Throughput of batched bounded IK on the 7-DOF arm (IK_BENCH_JOINTS: another joint count): N
 * problems, one subproblem each, a LinkPose
 * target, a fixed budget of 16 LM iterations, plain LM: no line search and no bound refinement
 * (real time); IK_BENCH_LS / IK_BENCH_REFINE override them. Objectives (IK_BENCH_OBJECTIVES, a
 * comma list, default all): "fused" LinkPoseFactorBatch; "views" a Position-view vector prior plus
 * an Orientation-view SO(3) prior through LinkFrameFactorBatch (the same decoupled objective);
 * "pose_view" an SE(3) prior on a Pose view (coupled SE(3) log residual).
 * Prints the median over 10 solves; success = position error < 5 mm and limits respected.
 */
TEST(ArticulationBenchmark, DISABLED_IkThroughput) {
  const ArticulationDescription d = Arm(IkBenchJoints());
  ArticulationTopology topo(d);
  const int C = topo.CoordCount(), T = topo.DofCount(), ee = T - 1;
  std::vector<SparseLinearSolverType> solvers = {SparseLinearSolverType::BlockTridiagonal,
                                                 SparseLinearSolverType::BlockSparseCholesky};
  if (std::getenv("IK_BENCH_TRIDIAGONAL_ONLY")) solvers.pop_back();
  if (T > BlockTridiagonalSolver::kMaxStageSize) solvers.erase(solvers.begin());
  if (solvers.empty()) {
    GTEST_SKIP() << "IK_BENCH_TRIDIAGONAL_ONLY with " << T
                 << " DOF (BlockTridiagonal takes at most " << BlockTridiagonalSolver::kMaxStageSize
                 << "): no solver to measure";
  }
  std::vector<std::string> objectives = {"fused", "views", "pose_view"};
  if (const char *o = std::getenv("IK_BENCH_OBJECTIVES")) {
    objectives.clear();
    std::string list(o);
    for (size_t at = 0; at <= list.size();) {
      const size_t comma = std::min(list.find(',', at), list.size());
      if (comma > at) objectives.push_back(list.substr(at, comma - at));
      at = comma + 1;
    }
  }
  for (SparseLinearSolverType solver_type : solvers) {
    for (const std::string &objective : objectives) {
      for (int n : IkBenchSizes()) {
        ArticulationStateBatch states(topo, n);
        states.SetNumActiveStates(n);
        std::mt19937 rng(9);
        std::vector<float> targets, q0, positions, rotations, matrices;
        for (int s = 0; s < n; ++s) {
          const auto q = RandomQ(d, rng);
          const M4 X = ReferenceFK(d, q.data())[ee];
          const auto p = PoseOf(X);
          targets.insert(targets.end(), p.begin(), p.end());
          positions.insert(positions.end(), {float(X[3]), float(X[7]), float(X[11])});
          for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) rotations.push_back(float(X[4 * r + c]));
          AppendMatrix(X, matrices);
          std::uniform_real_distribution<float> u(-0.4f, 0.4f);
          for (int k = 0; k < C; ++k) q0.push_back(q[k] + u(rng));
        }
        dvector<float> d_targets(targets), d_q0(q0), d_positions(positions), d_rotations(rotations),
            d_matrices(matrices);
        dvector<int> links(std::vector<int>(n, ee));
        LinkPoseFactorBatch pose(topo, links.data(), nullptr, d_targets.data(), nullptr, n);
        PriorVectorFactorBatch<3> position_prior(
            reinterpret_cast<const Vector<3> *>(d_positions.data()), n);
        SO3PriorFactorBatch rotation_prior(
            reinterpret_cast<const SO3Rotation *>(d_rotations.data()), n);
        SE3PriorFactorBatch pose_prior(reinterpret_cast<const SE3Transform *>(d_matrices.data()),
                                       n);
        LinkFrameFactorBatch position_view(topo, &position_prior, {{0, FrameView::kPosition}},
                                           links.data());
        LinkFrameFactorBatch orientation_view(topo, &rotation_prior, {{0, FrameView::kOrientation}},
                                              links.data());
        LinkFrameFactorBatch pose_view(topo, &pose_prior, {{0, FrameView::kPose}}, links.data());
        Problem problem;
        problem.AddStateBatch(&states);
        std::vector<float *> ptrs(n);
        for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
        std::vector<FactorBatch *> factors;
        if (objective == "fused") factors = {&pose};
        if (objective == "views") factors = {&position_view, &orientation_view};
        if (objective == "pose_view") factors = {&pose_view};
        ASSERT_FALSE(factors.empty()) << "unknown objective " << objective;
        for (FactorBatch *factor : factors) {
          factor->SetNumActiveFactors(n);
          problem.AddFactorBatch(factor, ptrs);
        }
        std::vector<int> ids(n);
        for (int s = 0; s < n; ++s) ids[s] = s;
        dvector<int> d_ids(ids), stages(std::vector<int>(n, 0));
        problem.SetProblemPartition(n, {d_ids.data()});
        if (solver_type == SparseLinearSolverType::BlockTridiagonal) {
          problem.SetStateStages({stages.data()});
        }
        AugmentedLagrangianMinimizerOptions al;
        al.base_options.sparse_linear_solver_type = solver_type;
        al.base_options.real_time = true;
        al.base_options.reuse_structure = true;
        al.base_options.max_num_iterations = 16;
        al.base_options.max_line_search_steps =
            std::getenv("IK_BENCH_LS") ? std::atoi(std::getenv("IK_BENCH_LS")) : 0;
        al.max_outer_iterations = 1;
        al.max_bound_refinements =
            std::getenv("IK_BENCH_REFINE") ? std::atoi(std::getenv("IK_BENCH_REFINE")) : 0;
        AugmentedLagrangianMinimizer solver(al);
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
        printf(
            "BENCH ik_arm%d_%-6d %-20s %-10s median_ms=%8.3f success=%5.1f%% outside_limits=%d\n",
            T, n,
            solver_type == SparseLinearSolverType::BlockTridiagonal ? "BlockTridiagonal"
                                                                    : "BlockSparseCholesky",
            objective.c_str(), ms[ms.size() / 2], 100.0 * ok / n, violations);
      }
    }
  }
}

// A factor built for one topology must not read the state of another, even one of the same DOF
// count or the same sizes: the cache layout differs, so the kernels would read the wrong floats.
TEST(ArticulationSolve, ProblemRejectsStateOfAnotherTopology) {
  ArticulationDescription one, two;
  AddJoint(one, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(two, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(two, JT::kFixed, 0, Pose(0.3f, 0, 0), Pose(0, 0, 0), {});
  // topo_twin: another robot of exactly the same sizes (e.g. the mirrored arm of a bimanual
  // robot); sizes alone cannot tell it apart, the topology object does.
  ArticulationTopology topo_one(one), topo_two(two), topo_twin(two);
  ASSERT_EQ(topo_one.DofCount(), topo_two.DofCount());
  ASSERT_NE(topo_one.AmbientSize(), topo_two.AmbientSize());
  ASSERT_EQ(topo_twin.AmbientSize(), topo_two.AmbientSize());
  ArticulationStateBatch states(topo_two, 1);
  states.SetNumActiveStates(1);
  dvector<float> targets(std::vector<float>{0.2f});
  CudaStream stream;
  for (const ArticulationTopology *topo : {&topo_one, &topo_two, &topo_twin}) {
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

// Written BALL / FREE quaternions are stored normalized and the cache is rigid; a zero quaternion
// (an uninitialized root) becomes the identity and stays finite through a step.
TEST(ArticulationStateBatch, NormalizesWrittenQuaternions) {
  const ArticulationDescription d = MixedTree();  // coords: FREE 0..6 (quat 3..6), BALL 7..10
  ArticulationTopology topo(d);
  const int n = 2, C = topo.CoordCount(), T = topo.DofCount();
  std::mt19937 rng(5);
  std::vector<float> unit(size_t(n) * C);
  for (int s = 0; s < n; ++s) {
    const auto qs = RandomQ(d, rng);
    std::copy(qs.begin(), qs.end(), unit.begin() + size_t(s) * C);
  }
  std::vector<float> q = unit;
  for (int i = 3; i < 7; ++i) q[i] *= 2.5f;    // instance 0: drifted FREE quaternion
  for (int i = 7; i < 11; ++i) q[i] *= 0.3f;   // and BALL quaternion
  for (int i = 3; i < 7; ++i) q[C + i] = 0.f;  // instance 1: zero FREE quaternion
  dvector<float> d_q(q), d_back(q.size());
  ArticulationStateBatch states(topo, n, /*apply_joint_limits=*/false);
  states.SetNumActiveStates(n);
  CudaStream stream;
  states.SetJointQ(d_q.data(), n, stream.GetStream());
  states.GetJointQ(d_back.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto back = Host(d_back);
  for (int i = 0; i < C; ++i) EXPECT_NEAR(back[i], unit[i], 1e-6) << "instance 0, coord " << i;
  const std::vector<float> identity = {0.f, 0.f, 0.f, 1.f};
  for (int i = 0; i < 4; ++i) EXPECT_EQ(back[C + 3 + i], identity[i]);

  const auto h = HostState(states, 1);
  const auto ref = ReferenceFK(d, unit.data());
  double worst = 0;
  for (int l = 0; l < topo.LinkCount(); ++l) {
    const float *P = h.data() + topo.LinkPoseOffset(l);
    for (int i = 0; i < 12; ++i) worst = Worst(worst, std::fabs(P[i] - ref[l][i]));
  }
  EXPECT_LT(worst, 2e-5);

  dvector<float> delta(std::vector<float>(size_t(n) * T, 0.1f)),
      out(size_t(n) * topo.AmbientSize());
  states.Plus(states.StateDevicePtr(0), delta.data(), out.data(), stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  for (float v : Host(out)) ASSERT_TRUE(std::isfinite(v));
}

// Link and DOF indices outside the topology give NaN residuals and zero Jacobians instead of
// reading past an instance's cache; valid factors of the same batch are unaffected.
TEST(KinematicFactors, OutOfRangeIndicesGiveNaN) {
  const ArticulationDescription d = MixedTree();
  ArticulationTopology topo(d);
  const int L = topo.LinkCount(), T = topo.DofCount(), k = 3;
  ArticulationStateBatch states(topo, 1);
  states.SetNumActiveStates(1);
  CudaStream stream;
  std::vector<float *> ptr_host(k, states.StateDevicePtr(0));
  dvector<float *> ptrs(ptr_host);

  auto check = [&](const FactorBatch &factor, const std::vector<bool> &valid, const char *what) {
    const int m = static_cast<int>(factor.ResidualsSize());
    dvector<float> r(std::vector<float>(size_t(k) * m, 7.f));
    dvector<float> J(std::vector<float>(size_t(k) * m * T, 7.f));
    ASSERT_TRUE(factor.Evaluate(r.data(), J.data(), ptrs.data(), stream.GetStream()));
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const auto hr = Host(r), hJ = Host(J);
    for (int t = 0; t < k; ++t) {
      for (int i = 0; i < m; ++i) {
        EXPECT_EQ(std::isfinite(hr[size_t(t) * m + i]), bool(valid[t])) << what << " factor " << t;
      }
      for (int i = 0; i < m * T; ++i) {
        const float v = hJ[size_t(t) * m * T + i];
        if (valid[t]) {
          EXPECT_TRUE(std::isfinite(v)) << what;
        } else {
          EXPECT_EQ(v, 0.f) << what << " factor " << t;
        }
      }
    }
  };

  dvector<int> links(std::vector<int>{-1, L, L - 1});
  std::vector<float> target_poses;
  for (int t = 0; t < k; ++t) target_poses.insert(target_poses.end(), {0, 0, 0, 0, 0, 0, 1});
  dvector<float> targets(target_poses);
  LinkPoseFactorBatch pose(topo, links.data(), nullptr, targets.data(), nullptr, k);
  pose.SetNumActiveFactors(k);
  check(pose, {false, false, true}, "pose");

  // A link frame with topology links -2 (below the world, -1) and L; the third is valid.
  dvector<int> frame_links(std::vector<int>{-2, L, L - 1});
  std::vector<float> zero_targets(3 * k, 0.f);
  dvector<float> d_zero_targets(zero_targets);
  PriorVectorFactorBatch<3> point_prior(reinterpret_cast<const Vector<3> *>(d_zero_targets.data()),
                                        k);
  LinkFrameFactorBatch frame(topo, &point_prior, {{0, FrameView::kPosition}}, frame_links.data());
  frame.SetNumActiveFactors(k);
  check(frame, {false, false, true}, "link frame");

  // Two position slots, one valid and one not: the whole row block is zero, not only the
  // invalid slot's contribution (VectorBetween writes [I, -I] for the two slots).
  dvector<int> pair_links(std::vector<int>{L - 1, -2, L - 1, L, 0, L - 1});
  dvector<float> zero_deltas(std::vector<float>(3 * k, 0.f));
  VectorBetweenFactorBatch<3> between(reinterpret_cast<const Vector<3> *>(zero_deltas.data()), k);
  LinkFrameFactorBatch pair(topo, &between, {{0, FrameView::kPosition}, {1, FrameView::kPosition}},
                            pair_links.data());
  pair.SetNumActiveFactors(k);
  check(pair, {false, false, true}, "link frame pair");

  // DOF T: out of range; DOF 6: a BALL DOF (no scalar coordinate); 9 -> 10: revolute, prismatic.
  dvector<int> mimic(std::vector<int>{T, 6, 9}), ref(std::vector<int>{9, 9, 10});
  dvector<float> coeffs(std::vector<float>{0.f, 1.f, 0.f, 1.f, 0.f, 1.f});
  MimicJointFactorBatch mimic_factor(topo, mimic.data(), ref.data(), coeffs.data(), k);
  mimic_factor.SetNumActiveFactors(k);
  check(mimic_factor, {false, false, true}, "mimic");
}

// ---------------------------------------------------------------------------
// LinkFrameFactorBatch: SE(3) factors on link frames
// ---------------------------------------------------------------------------

// An SE(3) prior on a frame of a link: the residuals equal the prior evaluated on that frame's
// pose, the DOF columns match central differences, and items follow the item contract.
TEST(LinkFrameFactor, PriorOnLinkFrames) {
  MixedFixture f;
  const std::vector<int> link_ids = f.Links();
  std::vector<float> frames, targets, poses;
  for (int t = 0; t < f.n; ++t) {
    const M4 off = f.SmallTransform(0.2f);
    const M4 F = Mul(f.LinkPose(t, link_ids[t]), off);
    const auto po = PoseOf(off);
    frames.insert(frames.end(), po.begin(), po.end());
    AppendMatrix(Mul(F, f.SmallTransform(0.3f)), targets);
    AppendMatrix(F, poses);
  }
  dvector<int> links(link_ids);
  dvector<float> d_frames(frames), d_targets(targets), d_poses(poses);
  SE3PriorFactorBatch prior(reinterpret_cast<const SE3Transform *>(d_targets.data()), f.n);
  LinkFrameFactorBatch frame(f.topo, &prior, {{0, FrameView::kPose}}, links.data(),
                             d_frames.data());
  frame.SetNumActiveFactors(f.n);
  EXPECT_EQ(prior.NumActiveFactors(), size_t(f.n));
  EXPECT_EQ(frame.ResidualsSize(), 6u);
  EXPECT_EQ(frame.StateSizes(), std::vector<size_t>{size_t(f.topo.DofCount())});

  CudaStream stream;
  std::vector<float *> art(f.n), direct(f.n);
  for (int t = 0; t < f.n; ++t) {
    art[t] = f.states.StateDevicePtr(t);
    direct[t] = d_poses.data() + 16 * size_t(t);
  }
  dvector<float *> d_art(art), d_direct(direct);
  dvector<float> r_frame(6 * f.n), r_direct(6 * f.n);
  ASSERT_TRUE(frame.Evaluate(r_frame.data(), nullptr, d_art.data(), stream.GetStream()));
  ASSERT_TRUE(prior.Evaluate(r_direct.data(), nullptr, d_direct.data(), stream.GetStream()));
  const auto a = Host(r_frame), b = Host(r_direct);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 2e-5) << i;

  CheckJacobians(frame, f.states, f.n, "frame prior");
  evaluate_items_test::CheckEvaluateItems(frame, 2, [&](int k) {
    std::vector<float *> p(f.n);
    for (int t = 0; t < f.n; ++t) p[t] = f.states.StateDevicePtr((t + 5 * k) % f.n);
    return p;
  });
}

// Two link slots of one factor (the relative pose of two links of an instance): DOFs that move
// both links contribute through both slots.
TEST(LinkFrameFactor, RelativePoseOfTwoLinks) {
  MixedFixture f;
  const int L = f.topo.LinkCount();
  std::vector<int> link_ids;
  std::vector<float> deltas;
  for (int t = 0; t < f.n; ++t) {
    const int a = t % L, b = (t * 3 + 2) % L;
    link_ids.push_back(a);
    link_ids.push_back(b);
    // Delta near the current relative pose: rotation residuals well below π.
    const M4 rel = Mul(Inverse(f.LinkPose(t, b)), f.LinkPose(t, a));
    AppendMatrix(Mul(f.SmallTransform(0.1f), rel), deltas);
  }
  dvector<int> links(link_ids);
  dvector<float> d_deltas(deltas);
  SE3BetweenFactorBatch between(reinterpret_cast<const SE3Transform *>(d_deltas.data()), f.n);
  LinkFrameFactorBatch frame(f.topo, &between, {{0, FrameView::kPose}, {1, FrameView::kPose}},
                             links.data());
  frame.SetNumActiveFactors(f.n);
  EXPECT_EQ(frame.StateSizes(), std::vector<size_t>{size_t(f.topo.DofCount())});
  CheckJacobians(frame, f.states, f.n, "two links");
}

// A link frame related to a free SE(3) state (an object pose): the object's columns and the
// residuals are the inner factor's, the DOF columns match central differences.
TEST(LinkFrameFactor, LinkToObjectPose) {
  MixedFixture f;
  const int T = f.topo.DofCount();
  const std::vector<int> link_ids = f.Links();
  std::vector<float> objects, deltas, poses;
  for (int t = 0; t < f.n; ++t) {
    const M4 X = f.LinkPose(t, link_ids[t]);
    const M4 object = Mul(X, f.SmallTransform(0.5f));
    AppendMatrix(object, objects);
    AppendMatrix(Mul(f.SmallTransform(0.1f), Mul(Inverse(object), X)), deltas);
    AppendMatrix(X, poses);
  }
  dvector<int> links(link_ids);
  dvector<float> d_objects(objects), d_deltas(deltas), d_poses(poses);
  SE3BetweenFactorBatch between(reinterpret_cast<const SE3Transform *>(d_deltas.data()), f.n);
  LinkFrameFactorBatch frame(f.topo, &between, {{0, FrameView::kPose}}, links.data());
  frame.SetNumActiveFactors(f.n);
  EXPECT_EQ(frame.StateSizes(), (std::vector<size_t>{size_t(T), 6}));

  std::vector<float *> object_ptrs(f.n), outer, direct;
  for (int t = 0; t < f.n; ++t) {
    object_ptrs[t] = d_objects.data() + 16 * size_t(t);
    outer.push_back(f.states.StateDevicePtr(t));
    outer.push_back(object_ptrs[t]);
    direct.push_back(d_poses.data() + 16 * size_t(t));
    direct.push_back(object_ptrs[t]);
  }
  CheckJacobians(frame, f.states, f.n, "link to object", 3e-3, object_ptrs, 6);

  CudaStream stream;
  dvector<float *> d_outer(outer), d_direct(direct);
  dvector<float> r1(6 * f.n), r2(6 * f.n), j1(size_t(f.n) * 6 * (T + 6)), j2(size_t(f.n) * 36 * 2);
  ASSERT_TRUE(frame.Evaluate(r1.data(), j1.data(), d_outer.data(), stream.GetStream()));
  ASSERT_TRUE(between.Evaluate(r2.data(), j2.data(), d_direct.data(), stream.GetStream()));
  const auto hr1 = Host(r1), hr2 = Host(r2), hj1 = Host(j1), hj2 = Host(j2);
  for (size_t i = 0; i < hr1.size(); ++i) EXPECT_NEAR(hr1[i], hr2[i], 2e-5) << i;
  for (int t = 0; t < f.n; ++t) {
    for (int r = 0; r < 6; ++r) {
      for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(hj1[(size_t(t) * 6 + r) * (T + 6) + T + k],
                    hj2[(size_t(t) * 6 + r) * 12 + 6 + k], 2e-5)
            << "item " << t << " row " << r << " object column " << k;
      }
    }
  }
}

// A zero-DOF topology (one FIXED joint) whose inner slots are all frame slots has no Jacobian
// columns: evaluating with a Jacobian buffer must succeed (no zero-size launch).
TEST(LinkFrameFactor, ZeroDofTopologyEvaluates) {
  ArticulationDescription d;
  AddJoint(d, JT::kFixed, -1, Pose(0.1f, 0.2f, 0.3f), Pose(0, 0, 0), {});
  ArticulationTopology topo(d);
  ASSERT_EQ(topo.DofCount(), 0);
  ArticulationStateBatch states(topo, 1);
  states.SetNumActiveStates(1);
  std::vector<float> target;
  AppendMatrix(Identity(), target);
  dvector<float> d_target(target);
  dvector<int> links(std::vector<int>{0});
  SE3PriorFactorBatch prior(reinterpret_cast<const SE3Transform *>(d_target.data()), 1);
  LinkFrameFactorBatch frame(topo, &prior, {{0, FrameView::kPose}}, links.data());
  frame.SetNumActiveFactors(1);
  CudaStream stream;
  dvector<float *> ptrs(std::vector<float *>{states.StateDevicePtr(0)});
  dvector<float> r(6), J(1);
  ASSERT_TRUE(frame.Evaluate(r.data(), J.data(), ptrs.data(), stream.GetStream()));
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  const auto h = Host(r);
  EXPECT_NEAR(h[3], 0.1f, 1e-6);  // translation part of Log(I^-1 X): X is a pure translation
}

TEST(LinkFrameFactor, RejectsInvalidSlots) {
  MixedFixture f;
  dvector<int> links(std::vector<int>(2 * f.n, 0));
  dvector<float> deltas(std::vector<float>(16 * f.n, 0.f));
  SE3BetweenFactorBatch between(reinterpret_cast<const SE3Transform *>(deltas.data()), f.n);
  auto make = [&](std::vector<int> slots, FactorBatch *inner, const int *l,
                  FrameView view = FrameView::kPose) {
    std::vector<LinkFrameSlot> frame_slots;
    for (int s : slots) frame_slots.push_back({s, view});
    LinkFrameFactorBatch frame(f.topo, inner, frame_slots, l);
  };
  EXPECT_THROW(make({}, &between, links.data()), std::invalid_argument);
  EXPECT_THROW(make({2}, &between, links.data()), std::invalid_argument);
  EXPECT_THROW(make({0, 0}, &between, links.data()), std::invalid_argument);
  EXPECT_THROW(make({0}, nullptr, links.data()), std::invalid_argument);
  EXPECT_THROW(make({0}, &between, nullptr), std::invalid_argument);
  MimicJointFactorBatch mimic(f.topo, links.data(), links.data(), deltas.data(), f.n);
  EXPECT_THROW(make({0}, &mimic, links.data()), std::invalid_argument);  // tangent T, not 6
  ConstraintFactorBatch constraint(&between, ConstraintKind::kEquality);
  EXPECT_THROW(make({0}, &constraint, links.data()), std::invalid_argument);  // hidden from AL
  // A view must match its slot's tangent: SE(3) slots (6) cannot take kPosition / kOrientation.
  EXPECT_THROW(make({0}, &between, links.data(), FrameView::kPosition), std::invalid_argument);
  EXPECT_THROW(make({0}, &between, links.data(), FrameView::kOrientation), std::invalid_argument);
  EXPECT_THROW(make({0}, &between, links.data(), static_cast<FrameView>(7)), std::invalid_argument);
  try {
    make({1, 3}, &between, links.data());
    FAIL() << "expected a throw";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("Fix:"), std::string::npos) << e.what();
  }
}

// The Position view with a vector prior is LinkPositionFactorBatch: same residuals and the same
// Jacobian (both are W (X_L c - p*) in the world frame).
TEST(LinkFrameFactor, PositionViewMatchesLinkPosition) {
  MixedFixture f;
  const std::vector<int> link_ids = f.Links();
  const std::vector<float> points = f.Uniform(3 * f.n, -0.2f, 0.2f);
  const std::vector<float> targets = f.Uniform(3 * f.n, -1.f, 1.f);
  std::vector<float> offsets;
  for (int t = 0; t < f.n; ++t) {
    offsets.insert(offsets.end(), {points[3 * t], points[3 * t + 1], points[3 * t + 2]});
    offsets.insert(offsets.end(), {0.f, 0.f, 0.f, 1.f});
  }
  dvector<int> links(link_ids);
  dvector<float> d_points(points), d_targets(targets), d_offsets(offsets);
  PriorVectorFactorBatch<3> prior(reinterpret_cast<const Vector<3> *>(d_targets.data()), f.n);
  LinkFrameFactorBatch frame(f.topo, &prior, {{0, FrameView::kPosition}}, links.data(),
                             d_offsets.data());
  frame.SetNumActiveFactors(f.n);
  LinkPositionFactorBatch position(f.topo, links.data(), d_points.data(), d_targets.data(), nullptr,
                                   f.n);
  position.SetNumActiveFactors(f.n);
  EXPECT_EQ(frame.StateSizes(), std::vector<size_t>{size_t(f.topo.DofCount())});

  CudaStream stream;
  const int T = f.topo.DofCount();
  std::vector<float *> art(f.n);
  for (int t = 0; t < f.n; ++t) art[t] = f.states.StateDevicePtr(t);
  dvector<float *> ptrs(art);
  dvector<float> r1(3 * f.n), r2(3 * f.n), j1(size_t(f.n) * 3 * T), j2(size_t(f.n) * 3 * T);
  ASSERT_TRUE(frame.Evaluate(r1.data(), j1.data(), ptrs.data(), stream.GetStream()));
  ASSERT_TRUE(position.Evaluate(r2.data(), j2.data(), ptrs.data(), stream.GetStream()));
  const auto hr1 = Host(r1), hr2 = Host(r2), hj1 = Host(j1), hj2 = Host(j2);
  for (size_t i = 0; i < hr1.size(); ++i) EXPECT_NEAR(hr1[i], hr2[i], 1e-5) << "residual " << i;
  for (size_t i = 0; i < hj1.size(); ++i) EXPECT_NEAR(hj1[i], hj2[i], 1e-5) << "jacobian " << i;
  CheckJacobians(frame, f.states, f.n, "position view");
  evaluate_items_test::CheckEvaluateItems(frame, 2, [&](int k) {
    std::vector<float *> p(f.n);
    for (int t = 0; t < f.n; ++t) p[t] = f.states.StateDevicePtr((t + 3 * k) % f.n);
    return p;
  });
}

// The Orientation view: an SO(3) prior on a frame's rotation (body-frame residual), and an
// SO(3) between of two links' orientations; DOF columns against central differences.
TEST(LinkFrameFactor, OrientationView) {
  MixedFixture f;
  const int L = f.topo.LinkCount();
  const std::vector<int> link_ids = f.Links();
  std::vector<float> offsets, targets, rotations;
  for (int t = 0; t < f.n; ++t) {
    const M4 off = f.SmallTransform(0.2f);
    const M4 frame = Mul(f.LinkPose(t, link_ids[t]), off);
    const auto po = PoseOf(off);
    offsets.insert(offsets.end(), po.begin(), po.end());
    const M4 target = Mul(frame, f.SmallTransform(0.3f));
    const M4 *both[2] = {&target, &frame};
    for (int k = 0; k < 2; ++k) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          (k == 0 ? targets : rotations).push_back(static_cast<float>((*both[k])[4 * a + b]));
        }
      }
    }
  }
  dvector<int> links(link_ids);
  dvector<float> d_offsets(offsets), d_targets(targets), d_rotations(rotations);
  SO3PriorFactorBatch prior(reinterpret_cast<const SO3Rotation *>(d_targets.data()), f.n);
  LinkFrameFactorBatch frame(f.topo, &prior, {{0, FrameView::kOrientation}}, links.data(),
                             d_offsets.data());
  frame.SetNumActiveFactors(f.n);

  // Residuals equal the prior evaluated on the frames' rotations.
  CudaStream stream;
  std::vector<float *> art(f.n), direct(f.n);
  for (int t = 0; t < f.n; ++t) {
    art[t] = f.states.StateDevicePtr(t);
    direct[t] = d_rotations.data() + 9 * size_t(t);
  }
  dvector<float *> d_art(art), d_direct(direct);
  dvector<float> r1(3 * f.n), r2(3 * f.n);
  ASSERT_TRUE(frame.Evaluate(r1.data(), nullptr, d_art.data(), stream.GetStream()));
  ASSERT_TRUE(prior.Evaluate(r2.data(), nullptr, d_direct.data(), stream.GetStream()));
  const auto a = Host(r1), b = Host(r2);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 2e-5) << i;
  CheckJacobians(frame, f.states, f.n, "orientation view");

  // Relative orientation of two links of one instance.
  std::vector<int> pair_ids;
  std::vector<float> deltas;
  for (int t = 0; t < f.n; ++t) {
    const int l0 = t % L, l1 = (t * 3 + 2) % L;
    pair_ids.insert(pair_ids.end(), {l0, l1});
    const M4 rel = Mul(Inverse(f.LinkPose(t, l1)), f.LinkPose(t, l0));  // Delta = R_1^T R_0
    const M4 delta = Mul(f.SmallTransform(0.f), rel);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) deltas.push_back(static_cast<float>(delta[4 * r + c]));
  }
  dvector<int> pairs(pair_ids);
  dvector<float> d_deltas(deltas);
  SO3BetweenFactorBatch between(reinterpret_cast<const SO3Rotation *>(d_deltas.data()), f.n);
  LinkFrameFactorBatch relative(
      f.topo, &between, {{0, FrameView::kOrientation}, {1, FrameView::kOrientation}}, pairs.data());
  relative.SetNumActiveFactors(f.n);
  CheckJacobians(relative, f.states, f.n, "relative orientation");
}

// Position views with existing vector factors: a halfspace on a frame's origin, and a ball loop
// closure as a vector between of two frames' origins (against the host frames).
TEST(LinkFrameFactor, PositionViewHalfspaceAndBallLoop) {
  MixedFixture f;
  const int L = f.topo.LinkCount();
  const std::vector<int> link_ids = f.Links();
  std::vector<float> normals = f.Uniform(3 * f.n, -1.f, 1.f), offsets_b = f.Uniform(f.n, -1, 1);
  dvector<int> links(link_ids);
  dvector<float> d_normals(normals), d_b(offsets_b);
  HalfspaceFactorBatch<3> halfspace(d_normals.data(), d_b.data(), f.n);
  LinkFrameFactorBatch above(f.topo, &halfspace, {{0, FrameView::kPosition}}, links.data());
  above.SetNumActiveFactors(f.n);
  CheckJacobians(above, f.states, f.n, "halfspace");

  // Ball loop: the origins of child frame X_C X_c and parent frame X_P X_p coincide.
  std::vector<int> pc;
  std::vector<float> frame_offsets, expected;
  for (int t = 0; t < f.n; ++t) {
    const int parent = t % L, child = (t * 5 + 3) % L;
    pc.insert(pc.end(), {child, parent});  // left = child, right = parent
    const auto xp = f.Uniform(3, -0.2f, 0.2f), xc = f.Uniform(3, -0.2f, 0.2f);
    frame_offsets.insert(frame_offsets.end(), {xc[0], xc[1], xc[2], 0, 0, 0, 1});
    frame_offsets.insert(frame_offsets.end(), {xp[0], xp[1], xp[2], 0, 0, 0, 1});
    const M4 XC = f.LinkPose(t, child), XP = f.LinkPose(t, parent);
    for (int i = 0; i < 3; ++i) {  // VectorBetween: left - right - delta = p(F_C) - p(F_P)
      const double pc_i =
          XC[4 * i] * xc[0] + XC[4 * i + 1] * xc[1] + XC[4 * i + 2] * xc[2] + XC[4 * i + 3];
      const double pp_i =
          XP[4 * i] * xp[0] + XP[4 * i + 1] * xp[1] + XP[4 * i + 2] * xp[2] + XP[4 * i + 3];
      expected.push_back(static_cast<float>(pc_i - pp_i));
    }
  }
  dvector<int> d_pc(pc);
  dvector<float> d_frame_offsets(frame_offsets);
  dvector<float> zeros(std::vector<float>(3 * f.n, 0.f));
  VectorBetweenFactorBatch<3> between(reinterpret_cast<const Vector<3> *>(zeros.data()), f.n);
  LinkFrameFactorBatch ball(f.topo, &between,
                            {{0, FrameView::kPosition}, {1, FrameView::kPosition}}, d_pc.data(),
                            d_frame_offsets.data());
  ball.SetNumActiveFactors(f.n);
  CheckJacobians(ball, f.states, f.n, "ball loop");

  CudaStream stream;
  std::vector<float *> art(f.n);
  for (int t = 0; t < f.n; ++t) art[t] = f.states.StateDevicePtr(t);
  dvector<float *> ptrs(art);
  dvector<float> r(3 * f.n);
  ASSERT_TRUE(ball.Evaluate(r.data(), nullptr, ptrs.data(), stream.GetStream()));
  const auto a = Host(r);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], expected[i], 2e-5) << i;
}

// ---------------------------------------------------------------------------
// Loop closures as compositions of existing factors on link-frame views
// ---------------------------------------------------------------------------

/**
 * A loop-closing joint of type `type` as equality constraints on frame views (what
 * pycunls.articulation.loop_closure builds): slot 0 the parent frame X_P X_p, slot 1 the child
 * frame X_C X_c. `links` holds (parent, child) per factor (parent -1: the world), `frames`
 * (X_p, X_c) per factor (14 floats: exactly frame_offsets), `axes` 3 floats per factor.
 */
struct LoopComposition {
  dvector<float> zeros, identity3, identity4;
  std::vector<std::unique_ptr<FactorBatch>> inner, views;
  std::vector<std::unique_ptr<ConstraintFactorBatch>> constraints;

  LoopComposition(const ArticulationTopology &topo, JT type, const int *links, const float *frames,
                  const float *axes, int n) {
    std::vector<float> z(3 * n, 0.f), i3, i4;
    for (int t = 0; t < n; ++t) {
      i3.insert(i3.end(), {1, 0, 0, 0, 1, 0, 0, 0, 1});
      i4.insert(i4.end(), {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    }
    zeros = dvector<float>(z);
    identity3 = dvector<float>(i3);
    identity4 = dvector<float>(i4);
    auto add = [&](std::unique_ptr<FactorBatch> factor, FrameView view) {
      auto frame = std::make_unique<LinkFrameFactorBatch>(
          topo, factor.get(), std::vector<LinkFrameSlot>{{0, view}, {1, view}}, links, frames);
      frame->SetNumActiveFactors(n);
      constraints.push_back(
          std::make_unique<ConstraintFactorBatch>(frame.get(), ConstraintKind::kEquality));
      inner.push_back(std::move(factor));
      views.push_back(std::move(frame));
    };
    const auto *z3 = reinterpret_cast<const Vector<3> *>(zeros.data());
    if (type == JT::kBall || type == JT::kRevolute) {
      add(std::make_unique<VectorBetweenFactorBatch<3>>(z3, n), FrameView::kPosition);
    }
    if (type == JT::kRevolute) {
      add(std::make_unique<SE3AxisAlignmentFactorBatch>(axes, n), FrameView::kPose);
    }
    if (type == JT::kPrismatic) {
      add(std::make_unique<SO3BetweenFactorBatch>(
              reinterpret_cast<const SO3Rotation *>(identity3.data()), n),
          FrameView::kOrientation);
      add(std::make_unique<SE3AxisOffsetFactorBatch>(axes, n), FrameView::kPose);
    }
    if (type == JT::kFixed) {
      add(std::make_unique<SE3BetweenFactorBatch>(
              reinterpret_cast<const SE3Transform *>(identity4.data()), n),
          FrameView::kPose);
    }
    for (auto &factor : inner) factor->SetNumActiveFactors(n);
  }
};

/**
 * The four-bar (ground, crank at the origin, coupler at the crank tip, rocker at (1, 0, 0)), crank
 * angles held by a heavy posture weight, closed by the constraints `add_loop` returns; returns
 * the summary and the solved joint_q.
 */
template <typename AddLoop>
std::pair<AugmentedLagrangianMinimizerSummary, std::vector<float>> SolveFourBar(AddLoop add_loop) {
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
  std::vector<float> q0, w;
  for (int s = 0; s < n; ++s) {
    q0.insert(q0.end(), {0.3f + 0.08f * s, 0.f, 1.2f});
    w.insert(w.end(), {100.f, 1e-2f, 1e-2f});
  }
  dvector<float> d_q0(q0), d_w(w);
  CudaStream stream;
  states.SetJointQ(d_q0.data(), n, stream.GetStream());
  PostureFactorBatch posture(topo, d_q0.data(), d_w.data(), n);
  posture.SetNumActiveFactors(n);
  std::vector<float> frames, axes;
  std::vector<int> pc;
  for (int s = 0; s < n; ++s) {
    frames.insert(frames.end(), {L2, 0, 0, 0, 0, 0, 1, L3, 0, 0, 0, 0, 0, 1});
    axes.insert(axes.end(), {0.f, 0.f, 1.f});
    pc.insert(pc.end(), {2, 3});
  }
  dvector<float> d_frames(frames), d_axes(axes);
  dvector<int> d_pc(pc);

  Problem problem;
  problem.AddStateBatch(&states);
  std::vector<float *> ptrs(n);
  for (int s = 0; s < n; ++s) ptrs[s] = states.StateDevicePtr(s);
  problem.AddFactorBatch(&posture, ptrs);
  auto keep = add_loop(topo, d_pc.data(), d_frames.data(), d_axes.data(), n);
  for (FactorBatch *c : keep.second) problem.AddFactorBatch(c, ptrs);
  std::vector<int> ids(n);
  for (int s = 0; s < n; ++s) ids[s] = s;
  dvector<int> d_ids(ids);
  problem.SetProblemPartition(n, {d_ids.data()});
  AugmentedLagrangianMinimizerOptions al;
  al.base_options.max_num_iterations = 100;
  al.constraint_tolerance = 1e-5f;
  AugmentedLagrangianMinimizer solver(al);
  const auto summary = solver.Minimize(stream.GetStream(), problem);
  dvector<float> q(size_t(n) * C);
  states.GetJointQ(q.data(), n, stream.GetStream());
  THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
  return {summary, Host(q)};
}

// A four-bar linkage (crank, coupler, rocker) closed by a revolute loop composition (position
// equality of the two joint frames + SE3AxisAlignmentFactorBatch) assembles at the requested
// crank angles. (Before LoopClosureFactorBatch was removed this matched it exactly: 5 outer / 20
// inner iterations each, joint angles within 1e-4.)
TEST(LoopComposition, FourBarAssembles) {
  const float L2 = 1.0f, L3 = 0.8f;
  std::unique_ptr<LoopComposition> composition;
  const auto result = SolveFourBar([&](const ArticulationTopology &topo, const int *pc,
                                       const float *frames, const float *axes, int n) {
    composition = std::make_unique<LoopComposition>(topo, JT::kRevolute, pc, frames, axes, n);
    std::vector<FactorBatch *> out;
    for (auto &c : composition->constraints) out.push_back(c.get());
    return std::make_pair(0, out);
  });
  EXPECT_EQ(result.first.status, AugmentedLagrangianMinimizerStatus::kConverged)
      << result.first.message;
  EXPECT_LT(result.first.max_violation, 1e-5f);
  ArticulationDescription d;
  AddJoint(d, JT::kFixed, -1, Pose(0, 0, 0), Pose(0, 0, 0), {});
  AddJoint(d, JT::kRevolute, 0, Pose(0, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(d, JT::kRevolute, 1, Pose(0.4f, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  AddJoint(d, JT::kRevolute, 0, Pose(1.0f, 0, 0), Pose(0, 0, 0), {{0, 0, 1}});
  const int C = 3, n = static_cast<int>(result.second.size()) / C;
  for (int s = 0; s < n; ++s) {
    const auto poses = ReferenceFK(d, result.second.data() + size_t(s) * C);
    // Coupler tip (link 2 at (L2, 0, 0)) and rocker tip (link 3 at (L3, 0, 0)) coincide.
    const double cx = poses[2][0] * L2 + poses[2][3], cy = poses[2][4] * L2 + poses[2][7];
    const double rx = poses[3][0] * L3 + poses[3][3], ry = poses[3][4] * L3 + poses[3][7];
    EXPECT_LT(std::hypot(cx - rx, cy - ry), 1e-4) << "state " << s;
    EXPECT_NEAR(result.second[size_t(s) * C], 0.3f + 0.08f * s, 1e-3f)
        << "crank moved, state " << s;
  }
}

// Every loop type's composition (parents including the world, -1) against central differences,
// and every row vanishing at a closed configuration.
TEST(LoopComposition, JacobiansAndZeroSets) {
  MixedFixture f;
  std::vector<int> parents(f.n), children(f.n), pc;
  std::vector<float> frames, closed_frames;
  for (int t = 0; t < f.n; ++t) {
    parents[t] = (t % 3 == 0) ? -1 : (t % f.topo.LinkCount());
    children[t] = (t * 5 + 3) % f.topo.LinkCount();
    pc.insert(pc.end(), {parents[t], children[t]});
    const M4 Xp = f.SmallTransform(0.2f);
    const M4 FP = Mul(parents[t] >= 0 ? f.LinkPose(t, parents[t]) : Identity(), Xp);
    const M4 FC = Mul(FP, f.SmallTransform(0.3f));  // near, not on, the closed configuration
    const M4 Xc = Mul(Inverse(f.LinkPose(t, children[t])), FC);
    const M4 Xc_closed = Mul(Inverse(f.LinkPose(t, children[t])), FP);  // F_C == F_P: closed
    for (const M4 *X : {&Xp, &Xc}) {
      const auto pose = PoseOf(*X);
      frames.insert(frames.end(), pose.begin(), pose.end());
    }
    for (const M4 *X : {&Xp, &Xc_closed}) {
      const auto pose = PoseOf(*X);
      closed_frames.insert(closed_frames.end(), pose.begin(), pose.end());
    }
  }
  dvector<int> d_pc(pc);
  dvector<float> d_frames(frames), d_closed(closed_frames), axes(f.Uniform(3 * f.n, -1.f, 1.f));
  CudaStream stream;
  std::vector<float *> art(f.n);
  for (int t = 0; t < f.n; ++t) art[t] = f.states.StateDevicePtr(t);
  dvector<float *> ptrs(art);
  for (JT type : {JT::kBall, JT::kRevolute, JT::kPrismatic, JT::kFixed}) {
    const std::string label = "loop composition type " + std::to_string(static_cast<int>(type));
    LoopComposition loop(f.topo, type, d_pc.data(), d_frames.data(), axes.data(), f.n);
    for (auto &view : loop.views) CheckJacobians(*view, f.states, f.n, label.c_str());
    // At a closed configuration (F_C == F_P) every row is zero.
    LoopComposition closed(f.topo, type, d_pc.data(), d_closed.data(), axes.data(), f.n);
    for (auto &view : closed.views) {
      const int m = static_cast<int>(view->ResidualsSize());
      dvector<float> r(size_t(f.n) * m);
      ASSERT_TRUE(view->Evaluate(r.data(), nullptr, ptrs.data(), stream.GetStream()));
      for (float v : Host(r)) EXPECT_NEAR(v, 0.f, 2e-5f) << label;
    }
  }
}

// The revolute composition does not accept a flipped axis (rotation by pi about a perpendicular):
// SE3AxisAlignmentFactorBatch compares the axes whole.
TEST(LoopComposition, RevoluteRejectsOppositeAxes) {
  ArticulationDescription d;  // link 0 turns about x; the loop joint (about z) ties it to the world
  AddJoint(d, JT::kRevolute, -1, Pose(0, 0, 0), Pose(0, 0, 0), {{1, 0, 0}});
  ArticulationTopology topo(d);
  ArticulationStateBatch states(topo, 2);
  states.SetNumActiveStates(2);
  dvector<float> q(std::vector<float>{0.f, 3.14159265f});  // aligned, then flipped by π about x
  CudaStream stream;
  states.SetJointQ(q.data(), 2, stream.GetStream());
  dvector<int> pc(std::vector<int>{-1, 0});
  dvector<float> frames(std::vector<float>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});
  dvector<float> axes(std::vector<float>{0, 0, 1});
  LoopComposition loop(topo, JT::kRevolute, pc.data(), frames.data(), axes.data(), 1);
  dvector<float *> ptrs(std::vector<float *>{states.StateDevicePtr(0), states.StateDevicePtr(1)});
  LinkFrameFactorBatch &alignment = *static_cast<LinkFrameFactorBatch *>(loop.views[1].get());
  dvector<float> r(6);
  ASSERT_TRUE(alignment.Evaluate(r.data(), nullptr, ptrs.data(), stream.GetStream(), nullptr, 2));
  const auto h = Host(r);
  EXPECT_LT(std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]), 1e-6);
  EXPECT_NEAR(std::sqrt(h[3] * h[3] + h[4] * h[4] + h[5] * h[5]), 2.0, 1e-5);  // |-z - z|
}

}  // namespace
}  // namespace cunls
