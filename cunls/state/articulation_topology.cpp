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

#include "cunls/state/articulation_topology.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cunls {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
/** Limits at or beyond this magnitude mean "unbounded" (a common simulator sentinel). */
constexpr float kUnboundedLimit = 1e5f;

[[noreturn]] void Fail(const std::string &what, const std::string &got, const std::string &expected,
                       const std::string &fix) {
  throw std::invalid_argument("ArticulationTopology: " + what + " (got " + got + ", expected " +
                              expected + "). Fix: " + fix + ".");
}

/** Coordinates and DOFs of joint `j` of the description. */
void JointSizes(const ArticulationDescription &d, int j, int *coords, int *dofs) {
  switch (static_cast<ArticulationJointType>(d.joint_type[j])) {
    case ArticulationJointType::kPrismatic:
    case ArticulationJointType::kRevolute:
      *coords = *dofs = 1;
      return;
    case ArticulationJointType::kBall:
      *coords = 4;
      *dofs = 3;
      return;
    case ArticulationJointType::kFixed:
      *coords = *dofs = 0;
      return;
    case ArticulationJointType::kFree:
      *coords = 7;
      *dofs = 6;
      return;
    case ArticulationJointType::kD6: {
      if (d.joint_dof_dim.size() != 2 * d.joint_type.size()) {
        Fail("joint_dof_dim is required with D6 joints",
             std::to_string(d.joint_dof_dim.size()) + " entries",
             std::to_string(2 * d.joint_type.size()) + " (linear, angular) per joint",
             "pass joint_dof_dim with a (linear, angular) axis count per joint");
      }
      const int lin = d.joint_dof_dim[2 * j], ang = d.joint_dof_dim[2 * j + 1];
      if (lin < 0 || lin > 3 || ang < 0 || ang > 3) {
        Fail("D6 joint " + std::to_string(j) + " has invalid axis counts",
             "(" + std::to_string(lin) + ", " + std::to_string(ang) + ")",
             "0..3 linear and 0..3 angular axes", "fix joint_dof_dim");
      }
      *coords = *dofs = lin + ang;
      return;
    }
  }
  Fail("joint " + std::to_string(j) + " has an unsupported type", std::to_string(d.joint_type[j]),
       "PRISMATIC=0, REVOLUTE=1, BALL=2, FIXED=3, FREE=4 or D6=6",
       "model DISTANCE/ROD joints differently, or loop-closing joints with "
       "LoopClosureFactorBatch");
}

/** Unit quaternion (x, y, z, w) from `q`, or an error naming `what`. */
void NormalizedQuat(const float *q, double *out, const std::string &what) {
  const double n = std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] + double(q[2]) * q[2] +
                             double(q[3]) * q[3]);
  if (!(std::isfinite(n) && n > 1e-6)) {
    Fail(what + " has a zero or non-finite quaternion", "norm " + std::to_string(n),
         "a unit quaternion (x, y, z, w)", "pass a valid rotation");
  }
  for (int i = 0; i < 4; ++i) out[i] = q[i] / n;
}

/** Row-major 3x4 [R | p] of the transform (p, q xyzw); inverse when `invert`. */
void TransformMatrix(const float *pq, bool invert, const std::string &what, float *M) {
  double q[4];
  NormalizedQuat(pq + 3, q, what);
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                       2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
  const double p[3] = {pq[0], pq[1], pq[2]};
  for (int r = 0; r < 3; ++r) {
    double t = 0.0;
    for (int c = 0; c < 3; ++c) {
      const double v = invert ? R[3 * c + r] : R[3 * r + c];
      M[4 * r + c] = static_cast<float>(v);
      t += v * p[c];
    }
    M[4 * r + 3] = static_cast<float>(invert ? -t : p[r]);
  }
}

}  // namespace

ArticulationTopology::ArticulationTopology(const ArticulationDescription &d) {
  const int J = static_cast<int>(d.joint_type.size());
  if (J == 0) Fail("the articulation has no joints", "0 joints", "at least 1", "add the joints");
  auto check_size = [&](const char *name, size_t size, size_t expected, const char *unit) {
    if (size != expected) {
      Fail(std::string(name) + " has the wrong size", std::to_string(size),
           std::to_string(expected) + " (" + unit + ")",
           std::string("pass ") + unit + " for " + name);
    }
  };
  check_size("joint_parent", d.joint_parent.size(), J, "one per joint");
  check_size("joint_child", d.joint_child.size(), J, "one per joint");
  check_size("joint_X_p", d.joint_X_p.size(), 7 * size_t(J), "7 floats per joint");
  check_size("joint_X_c", d.joint_X_c.size(), 7 * size_t(J), "7 floats per joint");
  check_size("joint_q_start", d.joint_q_start.size(), J + 1, "one per joint plus the total");
  check_size("joint_qd_start", d.joint_qd_start.size(), J + 1, "one per joint plus the total");

  // Coordinate and DOF layout: consecutive per joint, in the description's order.
  if (d.joint_q_start[0] != 0 || d.joint_qd_start[0] != 0) {
    Fail(
        "joint_q_start / joint_qd_start must start at 0",
        "(" + std::to_string(d.joint_q_start[0]) + ", " + std::to_string(d.joint_qd_start[0]) + ")",
        "(0, 0)", "pass offsets relative to this articulation's first joint");
  }
  for (int j = 0; j < J; ++j) {
    int coords = 0, dofs = 0;
    JointSizes(d, j, &coords, &dofs);
    if (d.joint_q_start[j + 1] - d.joint_q_start[j] != coords ||
        d.joint_qd_start[j + 1] - d.joint_qd_start[j] != dofs) {
      Fail("joint " + std::to_string(j) + "'s coordinate/DOF offsets do not match its type",
           std::to_string(d.joint_q_start[j + 1] - d.joint_q_start[j]) + " coordinates, " +
               std::to_string(d.joint_qd_start[j + 1] - d.joint_qd_start[j]) + " DOFs",
           std::to_string(coords) + " coordinates, " + std::to_string(dofs) + " DOFs",
           "list each joint's coordinates consecutively, in joint order");
    }
  }
  const int C = d.joint_q_start[J], T = d.joint_qd_start[J];
  check_size("joint_axis", d.joint_axis.size(), 3 * size_t(T), "3 floats per DOF");

  // Links: every link the child of exactly one joint; parents valid.
  std::vector<int> link_joint(J, -1);
  for (int j = 0; j < J; ++j) {
    const int c = d.joint_child[j], p = d.joint_parent[j];
    if (c < 0 || c >= J) {
      Fail("joint " + std::to_string(j) + " has an invalid child link", std::to_string(c),
           "a link index in [0, " + std::to_string(J) + ")",
           "number the links 0 .. J-1 (link i is the child of one joint)");
    }
    if (link_joint[c] != -1) {
      Fail("link " + std::to_string(c) + " is the child of two tree joints",
           "joints " + std::to_string(link_joint[c]) + " and " + std::to_string(j),
           "exactly one tree joint per link",
           "leave the loop-closing joint out of the tree and add it as a "
           "LoopClosureFactorBatch");
    }
    if (p < -1 || p >= J || p == c) {
      Fail("joint " + std::to_string(j) + " has an invalid parent link", std::to_string(p),
           "-1 (root frame) or another link in [0, " + std::to_string(J) + ")", "fix joint_parent");
    }
    link_joint[c] = j;
  }

  // Parents-first order (breadth first from the root joints).
  std::vector<int> order;
  order.reserve(J);
  for (int j = 0; j < J; ++j) {
    if (d.joint_parent[j] == -1) order.push_back(j);
  }
  for (size_t k = 0; k < order.size(); ++k) {
    const int link = d.joint_child[order[k]];
    for (int j = 0; j < J; ++j) {
      if (d.joint_parent[j] == link) order.push_back(j);
    }
  }
  if (static_cast<int>(order.size()) != J) {
    Fail("the joints do not form a tree rooted at the root frame",
         std::to_string(order.size()) + " of " + std::to_string(J) + " joints reachable",
         "every joint reachable from a joint with parent -1",
         "remove the cycle (close loops with LoopClosureFactorBatch)");
  }

  // Axes: unit length; D6 angular axes orthogonal (they are used as given, not orthonormalized).
  std::vector<float> axis(3 * size_t(T), 0.f);
  std::vector<int> scalar_coord(T, -1), dof_joint(T, -1);
  for (int j = 0; j < J; ++j) {
    const auto type = static_cast<ArticulationJointType>(d.joint_type[j]);
    const int q0 = d.joint_q_start[j], d0 = d.joint_qd_start[j];
    const int n = d.joint_qd_start[j + 1] - d0;
    const bool scalar = type == ArticulationJointType::kPrismatic ||
                        type == ArticulationJointType::kRevolute ||
                        type == ArticulationJointType::kD6;
    for (int k = 0; k < n; ++k) {
      dof_joint[d0 + k] = j;
      if (!scalar) continue;
      scalar_coord[d0 + k] = q0 + k;
      const float *a = d.joint_axis.data() + 3 * (d0 + k);
      const double len = std::sqrt(double(a[0]) * a[0] + double(a[1]) * a[1] + double(a[2]) * a[2]);
      if (!(std::fabs(len - 1.0) < 1e-3)) {
        Fail("DOF " + std::to_string(d0 + k) + " (joint " + std::to_string(j) +
                 ") has a non-unit axis",
             "length " + std::to_string(len), "1", "normalize joint_axis");
      }
      for (int i = 0; i < 3; ++i) axis[3 * (d0 + k) + i] = static_cast<float>(a[i] / len);
    }
    if (type == ArticulationJointType::kD6) {
      const int lin = d.joint_dof_dim[2 * j], ang = d.joint_dof_dim[2 * j + 1];
      for (int u = 0; u < ang; ++u) {
        for (int v = u + 1; v < ang; ++v) {
          const float *a = axis.data() + 3 * (d0 + lin + u);
          const float *b = axis.data() + 3 * (d0 + lin + v);
          const double dot = double(a[0]) * b[0] + double(a[1]) * b[1] + double(a[2]) * b[2];
          if (std::fabs(dot) > 1e-3) {
            Fail("D6 joint " + std::to_string(j) + " has non-orthogonal angular axes",
                 "dot " + std::to_string(dot), "orthogonal axes", "use orthogonal angular axes");
          }
        }
      }
    }
  }

  // Limits.
  lower_.assign(T, -kInf);
  upper_.assign(T, kInf);
  if (!d.limit_lower.empty() || !d.limit_upper.empty()) {
    check_size("limit_lower", d.limit_lower.size(), T, "one per DOF");
    check_size("limit_upper", d.limit_upper.size(), T, "one per DOF");
    for (int k = 0; k < T; ++k) {
      const float lo = d.limit_lower[k], hi = d.limit_upper[k];
      if (std::isnan(lo) || std::isnan(hi)) {
        Fail("DOF " + std::to_string(k) + " has a NaN limit",
             "[" + std::to_string(lo) + ", " + std::to_string(hi) + "]",
             "numbers (±inf or a magnitude of 1e5 or more: unbounded)", "fix the limits");
      }
      lower_[k] = (std::isfinite(lo) && std::fabs(lo) < kUnboundedLimit) ? lo : -kInf;
      upper_[k] = (std::isfinite(hi) && std::fabs(hi) < kUnboundedLimit) ? hi : kInf;
      if (lower_[k] > upper_[k]) {
        Fail("DOF " + std::to_string(k) + " has lower > upper limit",
             std::to_string(lower_[k]) + " > " + std::to_string(upper_[k]), "lower <= upper",
             "fix the limits");
      }
      const bool bounded = std::isfinite(lower_[k]) || std::isfinite(upper_[k]);
      if (bounded && scalar_coord[k] < 0) {
        Fail("DOF " + std::to_string(k) + " of BALL/FREE joint " + std::to_string(dof_joint[k]) +
                 " has a finite limit",
             "[" + std::to_string(lower_[k]) + ", " + std::to_string(upper_[k]) + "]",
             "unbounded (box limits on rotation-vector DOFs have no meaning)",
             "pass ±inf (or ±1e5 or beyond) for BALL and FREE DOFs");
      }
      has_limits_ = has_limits_ || bounded;
    }
  }

  // Ancestor DOFs of every link.
  const int W = (T + 31) / 32;
  std::vector<unsigned> ancestors(size_t(J) * (W > 0 ? W : 1), 0u);
  for (int link = 0; link < J; ++link) {
    for (int l = link; l != -1;) {
      const int j = link_joint[l];
      for (int k = d.joint_qd_start[j]; k < d.joint_qd_start[j + 1]; ++k) {
        ancestors[size_t(link) * W + k / 32] |= 1u << (k % 32);
      }
      l = d.joint_parent[j];
    }
  }

  // Transforms as 3x4 matrices.
  std::vector<float> X_p(12 * size_t(J)), X_c_inv(12 * size_t(J)), root(12);
  for (int j = 0; j < J; ++j) {
    TransformMatrix(d.joint_X_p.data() + 7 * j, false, "joint_X_p of joint " + std::to_string(j),
                    X_p.data() + 12 * j);
    TransformMatrix(d.joint_X_c.data() + 7 * j, true, "joint_X_c of joint " + std::to_string(j),
                    X_c_inv.data() + 12 * j);
  }
  TransformMatrix(d.root_pose.data(), false, "root_pose", root.data());

  // Host copies.
  joint_type_ = d.joint_type;
  q_start_ = d.joint_q_start;
  qd_start_ = d.joint_qd_start;
  scalar_coord_ = scalar_coord;
  link_joint_ = link_joint;

  // Upload: integer arrays and float arrays back to back.
  std::vector<int> dof_dim(2 * size_t(J), 0);
  for (int j = 0; j < J; ++j) {
    if (static_cast<ArticulationJointType>(d.joint_type[j]) == ArticulationJointType::kD6) {
      dof_dim[2 * j] = d.joint_dof_dim[2 * j];
      dof_dim[2 * j + 1] = d.joint_dof_dim[2 * j + 1];
    }
  }
  std::vector<int> ints;
  auto append_ints = [&](const std::vector<int> &v) {
    const size_t at = ints.size();
    ints.insert(ints.end(), v.begin(), v.end());
    return at;
  };
  const size_t o_order = append_ints(order), o_type = append_ints(d.joint_type),
               o_parent = append_ints(d.joint_parent), o_child = append_ints(d.joint_child),
               o_q = append_ints(d.joint_q_start), o_qd = append_ints(d.joint_qd_start),
               o_dim = append_ints(dof_dim), o_scalar = append_ints(scalar_coord),
               o_dof_joint = append_ints(dof_joint);
  std::vector<float> floats;
  auto append_floats = [&](const std::vector<float> &v) {
    const size_t at = floats.size();
    floats.insert(floats.end(), v.begin(), v.end());
    return at;
  };
  const size_t o_xp = append_floats(X_p), o_xc = append_floats(X_c_inv),
               o_axis = append_floats(axis), o_root = append_floats(root);
  d_ints_ = dvector<int>(ints);
  d_floats_ = dvector<float>(floats);
  d_ancestors_ = dvector<unsigned>(ancestors);

  view_.num_joints = J;
  view_.num_coords = C;
  view_.num_dofs = T;
  view_.ambient = C + 12 * J + 6 * T;
  view_.ancestor_words = W;
  const int *ip = d_ints_.data();
  const float *fp = d_floats_.data();
  view_.order = ip + o_order;
  view_.type = ip + o_type;
  view_.parent = ip + o_parent;
  view_.child = ip + o_child;
  view_.q_start = ip + o_q;
  view_.qd_start = ip + o_qd;
  view_.dof_dim = ip + o_dim;
  view_.scalar_coord = ip + o_scalar;
  view_.dof_joint = ip + o_dof_joint;
  view_.X_p = fp + o_xp;
  view_.X_c_inv = fp + o_xc;
  view_.axis = T > 0 ? fp + o_axis : nullptr;
  view_.root = fp + o_root;
  view_.ancestors = d_ancestors_.data();
}

}  // namespace cunls
