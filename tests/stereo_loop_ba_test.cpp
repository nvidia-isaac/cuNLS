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
 * @file stereo_loop_ba_test.cpp
 * @brief Synthetic replica of a cuSFM Mapper bundle adjustment on a closed
 *        stereo loop.
 *
 * Mirrors the first Mapper bundle adjustment of a cuSFM stereo dataset
 * (Galileo "figure8", October 2026; the cuSFM adapter's factor graph):
 * 724 vehicle rig poses on two laps of a figure-eight in a cluttered room, a
 * forward stereo pair (0.15 m baseline) with fixed camera_from_rig transforms,
 * about 90k landmarks mostly 0.5 to 3 m away, 1.2M reprojection factors in
 * normalized coordinates weighted by focal / 4 px with a Cauchy loss, a triple
 * smooth factor on every three consecutive poses, and the first pose held
 * constant.
 *
 * The initial state is what the front end hands the Mapper: poses integrated
 * from smoothly drifting visual odometry (about 0.35 m mean, 0.65 m max error
 * after alignment, as on the cuSFM data), every landmark triangulated in those
 * poses from the run of frames that created it, and observations from matched
 * revisits kept only within cuSFM's 25 px reprojection gate. The map is then
 * locally consistent, and the few loop observations that survive the gate are
 * almost switched off by the Cauchy loss. Closing the loop is a smooth
 * deformation of the whole trajectory along a nearly flat valley of the cost:
 * the slowest mode of the normal equations, which an inexact inner linear
 * solve resolves only a little per Levenberg-Marquardt iteration.
 *
 * On the cuSFM data the default BlockSparsePCG (200 iterations, relative
 * tolerance 1e-3) hits the 200-iteration cap with about 0.6 m of the
 * correction left, while an exact linear solve closes the loop within one
 * minimization. Here: cuDSS ends near 6 mm, BlockSparsePCG at 150 to 175 mm
 * (max position error against the ground truth after a rigid alignment; no
 * scale, the stereo baseline fixes it).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "cunls/common/cuda_stream.h"
#include "cunls/common/device_vector.h"
#include "cunls/common/helper.h"
#include "cunls/common/types.h"
#include "cunls/factor/information/information_factor_batch.h"
#include "cunls/factor/reprojection_factor_batch.h"
#include "cunls/minimizer/levenberg_marquardt_minimizer.h"
#include "cunls/minimizer/problem.h"
#include "cunls/robustifier/cauchy_loss_function_batch.h"
#include "cunls/state/se3_state_batch.h"
#include "cunls/state/vector_state_batch.h"
#include "tests/triple_smooth_factor_batch.h"

namespace cunls {
namespace {

// -----------------------------------------------------------------------------
// Host rigid-body math in double precision.
// -----------------------------------------------------------------------------

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;  // row-major

/** p -> rotation * p + translation. */
struct Rigid {
  Mat3 rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  Vec3 translation{0, 0, 0};
};

Vec3 Apply(const Mat3 &m, const Vec3 &v) {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

Mat3 Multiply(const Mat3 &a, const Mat3 &b) {
  Mat3 out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
    }
  }
  return out;
}

Mat3 Transpose(const Mat3 &m) { return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]}; }

Vec3 Apply(const Rigid &t, const Vec3 &p) {
  Vec3 out = Apply(t.rotation, p);
  for (int i = 0; i < 3; ++i) out[i] += t.translation[i];
  return out;
}

Rigid Compose(const Rigid &a, const Rigid &b) {
  return {Multiply(a.rotation, b.rotation), Apply(a, b.translation)};
}

Rigid Inverse(const Rigid &t) {
  Rigid out;
  out.rotation = Transpose(t.rotation);
  const Vec3 rt = Apply(out.rotation, t.translation);
  out.translation = {-rt[0], -rt[1], -rt[2]};
  return out;
}

/** Rotation matrix of a rotation vector (Rodrigues). */
Mat3 ExpSO3(const Vec3 &w) {
  const double theta = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  const double a = theta > 1e-9 ? std::sin(theta) / theta : 1.0;
  const double b = theta > 1e-9 ? (1.0 - std::cos(theta)) / (theta * theta) : 0.5;
  const Mat3 k{0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
  const Mat3 k2 = Multiply(k, k);
  Mat3 out{};
  for (int i = 0; i < 9; ++i) out[i] = (i % 4 == 0 ? 1.0 : 0.0) + a * k[i] + b * k2[i];
  return out;
}

/** Rotation from yaw (about z), then pitch (about y), then roll (about x). */
Mat3 YawPitchRoll(double yaw, double pitch, double roll) {
  return Multiply(ExpSO3({0, 0, yaw}), Multiply(ExpSO3({0, pitch, 0}), ExpSO3({roll, 0, 0})));
}

SE3Transform ToSE3(const Rigid &t) {
  SE3Transform out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[r * 4 + c] = static_cast<float>(t.rotation[r * 3 + c]);
    out[r * 4 + 3] = static_cast<float>(t.translation[r]);
  }
  out[15] = 1.f;
  return out;
}

Vec3 Position(const SE3Transform &t) { return {t[3], t[7], t[11]}; }

/**
 * Translation errors [m] of `estimate` against `truth` after the rigid
 * alignment (Kabsch, no scale) that minimizes their squared sum.
 */
std::vector<double> AlignedPositionErrors(const std::vector<Vec3> &estimate,
                                          const std::vector<Vec3> &truth) {
  const size_t n = truth.size();
  Vec3 ce{0, 0, 0}, ct{0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      ce[k] += estimate[i][k] / n;
      ct[k] += truth[i][k] / n;
    }
  }
  // Horn's method: the rotation is the unit quaternion of the largest
  // eigenvalue of a symmetric 4x4 matrix built from the cross covariance
  // C = sum (e - ce)(t - ct)^T; cyclic Jacobi rotations diagonalize it.
  Mat3 c{};
  for (size_t i = 0; i < n; ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int s = 0; s < 3; ++s) {
        c[r * 3 + s] += (estimate[i][r] - ce[r]) * (truth[i][s] - ct[s]);
      }
    }
  }
  const double sxx = c[0], sxy = c[1], sxz = c[2], syx = c[3], syy = c[4], syz = c[5], szx = c[6],
               szy = c[7], szz = c[8];
  double a[4][4] = {{sxx + syy + szz, syz - szy, szx - sxz, sxy - syx},
                    {syz - szy, sxx - syy - szz, sxy + syx, szx + sxz},
                    {szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy},
                    {sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz}};
  double v[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    for (int p = 0; p < 3; ++p) {
      for (int q = p + 1; q < 4; ++q) {
        if (std::fabs(a[p][q]) < 1e-300) continue;
        const double theta = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
        const double cs = std::cos(theta), sn = std::sin(theta);
        for (int k = 0; k < 4; ++k) {  // A <- A J
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = cs * akp - sn * akq;
          a[k][q] = sn * akp + cs * akq;
        }
        for (int k = 0; k < 4; ++k) {  // A <- J^T A
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = cs * apk - sn * aqk;
          a[q][k] = sn * apk + cs * aqk;
        }
        for (int k = 0; k < 4; ++k) {  // V <- V J
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = cs * vkp - sn * vkq;
          v[k][q] = sn * vkp + cs * vkq;
        }
      }
    }
  }
  int best = 0;
  for (int k = 1; k < 4; ++k) {
    if (a[k][k] > a[best][best]) best = k;
  }
  const double q[4] = {v[0][best], v[1][best], v[2][best], v[3][best]};
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const Mat3 rot{1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                 2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  std::vector<double> errors(n);
  for (size_t i = 0; i < n; ++i) {
    const Vec3 d{estimate[i][0] - ce[0], estimate[i][1] - ce[1], estimate[i][2] - ce[2]};
    const Vec3 aligned = Apply(rot, d);
    double sq = 0.0;
    for (int k = 0; k < 3; ++k) sq += std::pow(aligned[k] + ct[k] - truth[i][k], 2);
    errors[i] = std::sqrt(sq);
  }
  return errors;
}

// -----------------------------------------------------------------------------
// Scene.
// -----------------------------------------------------------------------------

constexpr int kNumPoses = 724;
constexpr double kLaps = 2.0;
constexpr double kHalfWidth = 2.75;  // x extent of the figure eight [m]
constexpr double kLength = 4.3;      // y extent [m]
constexpr double kRigHeight = 1.0;
constexpr double kBaseline = 0.15;  // stereo baseline [m]
// Camera model: pixels per normalized unit, image half extents in normalized
// coordinates (1280 x 800 at f = 830).
constexpr double kFocal = 830.0;
constexpr double kHalfFovX = 640.0 / kFocal;
constexpr double kHalfFovY = 400.0 / kFocal;
constexpr double kMinDepth = 0.3, kMaxDepth = 10.0;
constexpr double kPixelNoise = 0.6;      // observation noise [px]
constexpr double kSigmaPixels = 4.0;     // reprojection_error_stddev_pixels of cuSFM
constexpr double kOutlierRatio = 0.005;  // observations moved by up to 30 px
// Room around the trajectory and boxes standing in it: landmarks on their
// surfaces, mostly 0.5 to 3 m from the cameras.
constexpr double kRoomX = 4.0, kRoomY = 3.4, kRoomZ = 2.6;
constexpr int kNumBoxes = 60;
constexpr double kBoxClearance = 0.35;  // from the trajectory [m]
constexpr int kNumLandmarks = 45000;
// Feature tracks: observed for a geometric number of consecutive frames
// within each visibility run (one in five features tracks longer); a revisit
// is matched to the landmark with this probability, otherwise the front end
// creates a new landmark for it.
constexpr double kMeanTrackLength = 3.0;
constexpr double kLongTrackProbability = 0.2;  // tracks of a well-textured feature
constexpr double kMeanLongTrackLength = 25.0;
constexpr double kRevisitMatchProbability = 0.5;
constexpr int kMinObservedFrames = 2;
// cuSFM's map initialization: observations further than this from the
// triangulated point are dropped, and a point needs this many observations.
constexpr double kInitialMaxReprojectionErrorPixels = 25.0;
constexpr size_t kMinObservations = 3;
// A landmark observed more than this many frames apart links a loop.
constexpr int kLoopFrameSpan = 100;
// Visual odometry drifts smoothly: its rotation bias wanders (random walk),
// its scale is slightly off, and the white noise per frame is small, so the
// initial map is locally consistent (about 1 px) while the loop is not.
constexpr double kDriftYawBias = 1.1e-3;         // initial bias [rad / frame]
constexpr double kDriftBiasWalk = 3e-5;          // bias random walk [rad / frame]
constexpr double kDriftScale = 4e-3;             // translation scale error
constexpr double kDriftRotationNoise = 3e-5;     // [rad]
constexpr double kDriftTranslationNoise = 2e-4;  // [m]
// cuSFM relative_pose_*_stddev for the triple smooth factors.
constexpr double kSmoothRotationSigma = 1.0 * M_PI / 180.0;
constexpr double kSmoothTranslationSigma = 0.03;
constexpr float kZThreshold = 1e-3f;

/** Ground truth, initial values and measurements of the synthetic problem. */
struct StereoLoopScene {
  std::vector<Rigid> truth_poses;    ///< world_from_rig.
  std::vector<Rigid> initial_poses;  ///< Drifted world_from_rig.
  std::vector<Vector<3>> initial_points;
  std::vector<Vector<2>> observations;        ///< Normalized coordinates.
  std::vector<SE3Transform> camera_from_rig;  ///< Per observation.
  std::vector<int> pose_ids, point_ids;       ///< Per observation.
  std::vector<int> camera_ids;                ///< Per observation: 0 left, 1 right.
  size_t num_loop_points = 0;  ///< Landmarks observed over more than kLoopFrameSpan frames.
};

/** Camera i (0 left, 1 right) of the forward stereo pair, camera_from_rig. */
Rigid CameraFromRig(int camera) {
  // Camera z forward (rig x), x right (rig -y), y down (rig -z).
  Rigid rig_from_camera;
  rig_from_camera.rotation = {0, 0, 1, -1, 0, 0, 0, -1, 0};
  rig_from_camera.translation = {0.1, camera == 0 ? 0.5 * kBaseline : -0.5 * kBaseline, 0.336};
  return Inverse(rig_from_camera);
}

/**
 * Least-squares intersection of the observation rays (midpoint method):
 * minimizes the sum of squared distances of `point` to the rays through the
 * camera centers. `track` entries need frame, camera, u and v.
 */
template <class Track>
bool Triangulate(const Track &track, const std::vector<Rigid> (&world_from_camera)[2],
                 Vec3 *point) {
  double a[9] = {}, b[3] = {};
  for (const auto &o : track) {
    const Rigid &t = world_from_camera[o.camera][o.frame];
    Vec3 d = Apply(t.rotation, {o.u, o.v, 1.0});
    const double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (double &x : d) x /= norm;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        const double m = (r == c ? 1.0 : 0.0) - d[r] * d[c];
        a[r * 3 + c] += m;
        b[r] += m * t.translation[c];
      }
    }
  }
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                     a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!(std::fabs(det) > 1e-12)) return false;
  const double inv[9] = {
      a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
      a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
      a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3]};
  for (int r = 0; r < 3; ++r) {
    (*point)[r] = (inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]) / det;
  }
  return true;
}

StereoLoopScene MakeScene() {
  std::mt19937 rng(20261015);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  StereoLoopScene scene;

  // Two laps of a Gerono lemniscate, heading along the path, gentle pitch and
  // roll from the floor.
  for (int k = 0; k < kNumPoses; ++k) {
    const double s = 2.0 * M_PI * kLaps * k / kNumPoses;
    const double x = kHalfWidth * std::sin(s);
    const double y = kLength * std::sin(s) * std::cos(s);
    const double dx = kHalfWidth * std::cos(s), dy = kLength * std::cos(2.0 * s);
    Rigid pose;
    pose.rotation =
        YawPitchRoll(std::atan2(dy, dx), 0.02 * std::sin(3.0 * s), 0.015 * std::cos(5.0 * s));
    pose.translation = {x, y, kRigHeight + 0.03 * std::sin(2.0 * s)};
    scene.truth_poses.push_back(pose);
  }

  // Odometry: relative motions with a wandering rotation bias, a scale error
  // and small white noise, integrated from the (exact) first pose.
  scene.initial_poses.push_back(scene.truth_poses[0]);
  Vec3 bias{0.2 * kDriftYawBias, -0.1 * kDriftYawBias, kDriftYawBias};
  for (int k = 1; k < kNumPoses; ++k) {
    Rigid motion = Compose(Inverse(scene.truth_poses[k - 1]), scene.truth_poses[k]);
    for (double &b : bias) b += kDriftBiasWalk * gauss(rng);
    Vec3 w;
    for (int i = 0; i < 3; ++i) w[i] = bias[i] + kDriftRotationNoise * gauss(rng);
    motion.rotation = Multiply(motion.rotation, ExpSO3(w));
    for (double &t : motion.translation) {
      t = (1.0 + kDriftScale) * t + kDriftTranslationNoise * gauss(rng);
    }
    scene.initial_poses.push_back(Compose(scene.initial_poses.back(), motion));
  }

  const Rigid camera_from_rig[2] = {CameraFromRig(0), CameraFromRig(1)};
  const SE3Transform camera_from_rig_se3[2] = {ToSE3(camera_from_rig[0]),
                                               ToSE3(camera_from_rig[1])};
  std::vector<Rigid> camera_from_world[2];
  for (int c = 0; c < 2; ++c) {
    for (const Rigid &pose : scene.truth_poses) {
      camera_from_world[c].push_back(Compose(camera_from_rig[c], Inverse(pose)));
    }
  }
  std::vector<Rigid> drifted_camera_from_world[2], drifted_world_from_camera[2];
  for (int c = 0; c < 2; ++c) {
    for (const Rigid &pose : scene.initial_poses) {
      drifted_camera_from_world[c].push_back(Compose(camera_from_rig[c], Inverse(pose)));
      drifted_world_from_camera[c].push_back(Inverse(drifted_camera_from_world[c].back()));
    }
  }
  auto project = [&](int camera, int frame, const Vec3 &p, Vec3 &out) {
    const Vec3 pc = Apply(camera_from_world[camera][frame], p);
    if (!(pc[2] > kMinDepth && pc[2] < kMaxDepth)) return false;
    const double u = pc[0] / pc[2], v = pc[1] / pc[2];
    if (std::fabs(u) > kHalfFovX || std::fabs(v) > kHalfFovY) return false;
    out = {u, v, pc[2]};
    return true;
  };

  // Landmark surfaces: rectangles origin + a u + b v, a, b in [0, 1].
  struct Surface {
    Vec3 origin, u, v;
  };
  std::vector<Surface> surfaces = {
      {{-kRoomX, -kRoomY, 0}, {2 * kRoomX, 0, 0}, {0, 2 * kRoomY, 0}},       // floor
      {{-kRoomX, -kRoomY, kRoomZ}, {2 * kRoomX, 0, 0}, {0, 2 * kRoomY, 0}},  // ceiling
      {{-kRoomX, -kRoomY, 0}, {2 * kRoomX, 0, 0}, {0, 0, kRoomZ}},
      {{-kRoomX, kRoomY, 0}, {2 * kRoomX, 0, 0}, {0, 0, kRoomZ}},
      {{-kRoomX, -kRoomY, 0}, {0, 2 * kRoomY, 0}, {0, 0, kRoomZ}},
      {{kRoomX, -kRoomY, 0}, {0, 2 * kRoomY, 0}, {0, 0, kRoomZ}}};
  // Boxes on the floor, clear of the trajectory: their top and four sides.
  for (int placed = 0; placed < kNumBoxes;) {
    const double sx = 0.3 + 0.7 * uniform(rng), sy = 0.3 + 0.7 * uniform(rng),
                 sz = 0.3 + 1.3 * uniform(rng);
    const double cx = (2.0 * uniform(rng) - 1.0) * (kRoomX - sx),
                 cy = (2.0 * uniform(rng) - 1.0) * (kRoomY - sy);
    double clearance = 1e9;
    for (const Rigid &pose : scene.truth_poses) {
      clearance =
          std::min(clearance, std::hypot(pose.translation[0] - cx, pose.translation[1] - cy));
    }
    if (clearance < kBoxClearance + 0.5 * std::hypot(sx, sy)) continue;
    const Vec3 o{cx - 0.5 * sx, cy - 0.5 * sy, 0.0};
    surfaces.push_back({{o[0], o[1], sz}, {sx, 0, 0}, {0, sy, 0}});
    surfaces.push_back({o, {sx, 0, 0}, {0, 0, sz}});
    surfaces.push_back({{o[0], o[1] + sy, 0}, {sx, 0, 0}, {0, 0, sz}});
    surfaces.push_back({o, {0, sy, 0}, {0, 0, sz}});
    surfaces.push_back({{o[0] + sx, o[1], 0}, {0, sy, 0}, {0, 0, sz}});
    ++placed;
  }
  std::vector<double> area_cdf;
  for (const Surface &f : surfaces) {
    const Vec3 &u = f.u, &v = f.v;
    const Vec3 n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    area_cdf.push_back((area_cdf.empty() ? 0.0 : area_cdf.back()) +
                       std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]));
  }
  std::geometric_distribution<int> track_extra(1.0 / kMeanTrackLength);
  std::geometric_distribution<int> long_track_extra(1.0 / kMeanLongTrackLength);
  std::vector<int> visible;
  for (int i = 0; i < kNumLandmarks; ++i) {
    const double pick = uniform(rng) * area_cdf.back();
    const Surface &f =
        surfaces[std::lower_bound(area_cdf.begin(), area_cdf.end(), pick) - area_cdf.begin()];
    const double a = uniform(rng), b = uniform(rng);
    const Vec3 p{f.origin[0] + a * f.u[0] + b * f.v[0], f.origin[1] + a * f.u[1] + b * f.v[1],
                 f.origin[2] + a * f.u[2] + b * f.v[2]};

    // Frames whose left camera sees it, split into runs of consecutive frames.
    visible.clear();
    Vec3 uvz;
    for (int k = 0; k < kNumPoses; ++k) {
      if (project(0, k, p, uvz)) visible.push_back(k);
    }
    std::vector<std::pair<int, int>> runs;  // [first, last]
    for (size_t j = 0; j < visible.size(); ++j) {
      if (j == 0 || visible[j] != visible[j - 1] + 1) runs.push_back({visible[j], visible[j]});
      runs.back().second = visible[j];
    }
    // One track per run: a random stretch of geometric length. Later runs
    // (revisits) are matched to the landmark with kRevisitMatchProbability;
    // the front end makes an unmatched one a new landmark of its own.
    // Per landmark: its frames, the first `local` of them from its first run.
    struct Landmark {
      std::vector<int> frames;
      size_t local = 0;
    };
    std::vector<Landmark> landmarks(1);
    const bool long_track = uniform(rng) < kLongTrackProbability;
    for (size_t r = 0; r < runs.size(); ++r) {
      Landmark &landmark = r == 0 || uniform(rng) < kRevisitMatchProbability
                               ? landmarks.front()
                               : landmarks.emplace_back();
      const int run_length = runs[r].second - runs[r].first + 1;
      const int length =
          std::min(run_length, 2 + (long_track ? long_track_extra(rng) : track_extra(rng)));
      const int start = runs[r].first + static_cast<int>(uniform(rng) * (run_length - length + 1));
      for (int k = start; k < start + length; ++k) landmark.frames.push_back(k);
      if (landmark.local == 0) landmark.local = landmark.frames.size();
    }
    for (const Landmark &landmark : landmarks) {
      if (static_cast<int>(landmark.frames.size()) < kMinObservedFrames) continue;
      struct Observation {
        int frame, camera;
        double u, v;
      };
      std::vector<Observation> track, local;
      for (size_t j = 0; j < landmark.frames.size(); ++j) {
        const int k = landmark.frames[j];
        for (int c = 0; c < 2; ++c) {
          if (!project(c, k, p, uvz)) continue;
          double u = uvz[0] + kPixelNoise / kFocal * gauss(rng);
          double v = uvz[1] + kPixelNoise / kFocal * gauss(rng);
          if (uniform(rng) < kOutlierRatio) {
            u += 30.0 / kFocal * (2.0 * uniform(rng) - 1.0);
            v += 30.0 / kFocal * (2.0 * uniform(rng) - 1.0);
          }
          track.push_back({k, c, u, v});
          if (j < landmark.local) local.push_back(track.back());
        }
      }
      // The front end's map: the point triangulated in the drifted poses from
      // the run that created it (locally consistent), the observations of
      // matched revisits added when within the reprojection gate.
      Vec3 init{0, 0, 0};
      if (local.size() < 2 || !Triangulate(local, drifted_world_from_camera, &init)) continue;
      std::vector<Observation> kept;
      for (const Observation &o : track) {
        const Vec3 pc = Apply(drifted_camera_from_world[o.camera][o.frame], init);
        if (pc[2] > kMinDepth && kFocal * std::hypot(pc[0] / pc[2] - o.u, pc[1] / pc[2] - o.v) <=
                                     kInitialMaxReprojectionErrorPixels) {
          kept.push_back(o);
        }
      }
      track.swap(kept);
      if (track.size() < kMinObservations) continue;

      const int point_id = static_cast<int>(scene.initial_points.size());
      int first = kNumPoses, last = -1;
      for (const Observation &o : track) {
        scene.observations.push_back({static_cast<float>(o.u), static_cast<float>(o.v)});
        scene.camera_from_rig.push_back(camera_from_rig_se3[o.camera]);
        scene.pose_ids.push_back(o.frame);
        scene.camera_ids.push_back(o.camera);
        scene.point_ids.push_back(point_id);
        first = std::min(first, o.frame);
        last = std::max(last, o.frame);
      }
      scene.initial_points.push_back(
          {static_cast<float>(init[0]), static_cast<float>(init[1]), static_cast<float>(init[2])});
      scene.num_loop_points += last - first > kLoopFrameSpan ? 1 : 0;
    }
  }
  return scene;
}

/** Device data, batches and problem of one StereoLoopScene. */
class StereoLoopProblem {
 public:
  explicit StereoLoopProblem(const StereoLoopScene &scene) {
    const size_t num_obs = scene.observations.size();
    std::vector<SE3Transform> poses;
    for (const Rigid &pose : scene.initial_poses) poses.push_back(ToSE3(pose));
    poses_ = dvector<SE3Transform>(poses);
    points_ = dvector<Vector<3>>(scene.initial_points);
    const_pose_ids_ = dvector<int>(std::vector<int>{0});
    pose_batch_ = std::make_unique<SE3StateBatch>(reinterpret_cast<const float *>(poses_.data()),
                                                  poses.size(), const_pose_ids_.data(), 1);
    pose_batch_->SetNumActiveStates(pose_batch_->Capacity(), pose_batch_->ConstCapacity());
    point_batch_ = std::make_unique<VectorStateBatch<3>>(
        reinterpret_cast<const float *>(points_.data()), scene.initial_points.size());
    point_batch_->SetNumActiveStates(point_batch_->Capacity());
    problem_.AddStateBatch(pose_batch_.get());
    problem_.AddStateBatch(point_batch_.get());

    // Reprojection: residuals in normalized coordinates scaled to pixels /
    // sigma, Cauchy loss of scale 1 (cuSFM's mapper defaults).
    const float info = static_cast<float>(kFocal / kSigmaPixels);
    observations_ = dvector<Vector<2>>(scene.observations);
    camera_from_rig_ = dvector<SE3Transform>(scene.camera_from_rig);
    reprojection_info_ = dvector<Matrix<2>>(std::vector<Matrix<2>>(num_obs, {info, 0, 0, info}));
    reprojection_ = std::make_unique<InformationFactorBatch<ReprojectionFactorBatch>>(
        reprojection_info_.data(), num_obs, observations_.data(), camera_from_rig_.data(), num_obs,
        kZThreshold);
    reprojection_->SetNumActiveFactors(num_obs);
    std::vector<float *> reprojection_states;
    reprojection_states.reserve(2 * num_obs);
    for (size_t i = 0; i < num_obs; ++i) {
      reprojection_states.push_back(pose_batch_->StateDevicePtr(scene.pose_ids[i]));
      reprojection_states.push_back(point_batch_->StateDevicePtr(scene.point_ids[i]));
    }
    cauchy_ = std::make_unique<CauchyLossFunctionBatch>(1.f, 1.f);
    problem_.AddFactorBatch(reprojection_.get(), cauchy_.get(), reprojection_states);

    // Triple smooth factors over consecutive poses (uniform timestamps:
    // ratio 1/2), information from cuSFM's relative pose sigmas.
    const size_t num_smooth = kNumPoses - 2;
    Matrix<6> smooth_info{};
    for (int i = 0; i < 3; ++i) {
      smooth_info[i * 7] = static_cast<float>(1.0 / kSmoothRotationSigma);
      smooth_info[(i + 3) * 7] = static_cast<float>(1.0 / kSmoothTranslationSigma);
    }
    smooth_ratios_ = dvector<float>(std::vector<float>(num_smooth, 0.5f));
    smooth_info_ = dvector<Matrix<6>>(std::vector<Matrix<6>>(num_smooth, smooth_info));
    smooth_ = std::make_unique<InformationFactorBatch<test_utils::TripleSmoothFactorBatch>>(
        smooth_info_.data(), num_smooth, smooth_ratios_.data(), num_smooth);
    smooth_->SetNumActiveFactors(num_smooth);
    std::vector<float *> smooth_states;
    for (size_t k = 0; k < num_smooth; ++k) {
      for (size_t j = 0; j < 3; ++j) smooth_states.push_back(pose_batch_->StateDevicePtr(k + j));
    }
    problem_.AddFactorBatch(smooth_.get(), smooth_states);
  }

  Problem &GetProblem() { return problem_; }

  /** The landmark state batch (eliminated by LinearSystemType::SchurComplement). */
  const StateBatch *Points() const { return point_batch_.get(); }

  /** Current pose positions (synchronous copy). */
  std::vector<Vec3> Positions() const {
    std::vector<SE3Transform> poses(poses_.size());
    poses_.CopyToHost(poses.data(), poses.size());
    std::vector<Vec3> out;
    for (const SE3Transform &pose : poses) out.push_back(Position(pose));
    return out;
  }

 private:
  dvector<SE3Transform> poses_;
  dvector<Vector<3>> points_;
  dvector<int> const_pose_ids_;
  dvector<Vector<2>> observations_;
  dvector<SE3Transform> camera_from_rig_;
  dvector<Matrix<2>> reprojection_info_;
  dvector<float> smooth_ratios_;
  dvector<Matrix<6>> smooth_info_;
  std::unique_ptr<SE3StateBatch> pose_batch_;
  std::unique_ptr<VectorStateBatch<3>> point_batch_;
  std::unique_ptr<InformationFactorBatch<ReprojectionFactorBatch>> reprojection_;
  std::unique_ptr<CauchyLossFunctionBatch> cauchy_;
  std::unique_ptr<InformationFactorBatch<test_utils::TripleSmoothFactorBatch>> smooth_;
  Problem problem_;
};

/** Mean and max of a vector. */
std::pair<double, double> MeanMax(const std::vector<double> &values) {
  double sum = 0.0, max = 0.0;
  for (double v : values) {
    sum += v;
    max = std::max(max, v);
  }
  return {sum / values.size(), max};
}

class StereoLoopBaTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { scene_ = std::make_unique<StereoLoopScene>(MakeScene()); }
  static void TearDownTestSuite() { scene_.reset(); }

  static std::vector<Vec3> Positions(const std::vector<Rigid> &poses) {
    std::vector<Vec3> out;
    for (const Rigid &pose : poses) out.push_back(pose.translation);
    return out;
  }
  std::vector<Vec3> TruthPositions() const { return Positions(scene_->truth_poses); }
  std::vector<Vec3> InitialPositions() const { return Positions(scene_->initial_poses); }

  /**
   * One cuSFM-like Levenberg-Marquardt solve (the adapter's options) with the
   * given linear solver, from the drifted initial state. Returns the summary;
   * `before` and `after` receive the aligned position errors [m].
   */
  MinimizerSummary Solve(SparseLinearSolverType solver, std::vector<double> *before,
                         std::vector<double> *after,
                         LinearSystemType linear_system = LinearSystemType::Assembled) {
    StereoLoopProblem setup(*scene_);
    Problem &problem = setup.GetProblem();
    EXPECT_TRUE(problem.CheckConsistency());
    problem.SetEliminatedStateBatches({setup.Points()});
    *before = AlignedPositionErrors(setup.Positions(), TruthPositions());

    MinimizerOptions options;
    options.max_num_iterations = 200;
    options.cost_tolerance = 1e-9f;
    options.state_tolerance = 1e-9f;
    options.column_scaling = ColumnScaling::HessianDiagonal;
    options.sparse_linear_solver_type = solver;
    options.linear_system = linear_system;
    LevenbergMarquardtMinimizerOptions lm_options;
    lm_options.base_options = options;
    lm_options.relative_reduction_tolerance = 1e-9f;
    LevenbergMarquardtMinimizer minimizer(lm_options);

    CudaStream stream;
    const auto start = std::chrono::steady_clock::now();
    const MinimizerSummary summary = minimizer.Minimize(stream.GetStream(), problem);
    THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream.GetStream()));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    *after = AlignedPositionErrors(setup.Positions(), TruthPositions());
    const auto [mean0, max0] = MeanMax(*before);
    const auto [mean1, max1] = MeanMax(*after);
    std::printf(
        "[stereo loop BA] %zu poses, %zu points (%zu spanning > 100 frames), %zu observations\n"
        "  iterations %zu, cost %.6g -> %.6g, %.2f s\n"
        "  position error vs truth [mm]: mean %.1f max %.1f -> mean %.1f max %.1f\n",
        scene_->truth_poses.size(), scene_->initial_points.size(), scene_->num_loop_points,
        scene_->observations.size(), summary.num_iterations, summary.initial_cost,
        summary.final_cost, seconds, 1e3 * mean0, 1e3 * max0, 1e3 * mean1, 1e3 * max1);
    RecordProperty("iterations", static_cast<int>(summary.num_iterations));
    RecordProperty("max_error_mm", static_cast<int>(std::lround(1e3 * max1)));
    return summary;
  }

  static std::unique_ptr<StereoLoopScene> scene_;
};

std::unique_ptr<StereoLoopScene> StereoLoopBaTest::scene_;

/** The scene has the scale and structure of the cuSFM problem. */
TEST_F(StereoLoopBaTest, SceneMatchesCuSfmScale) {
  EXPECT_EQ(scene_->truth_poses.size(), 724u);
  EXPECT_GT(scene_->initial_points.size(), 50000u);
  EXPECT_GT(scene_->observations.size(), 1000000u);
  EXPECT_LT(scene_->observations.size(), 1500000u);
  EXPECT_GT(scene_->num_loop_points, 100u) << "a few observations link the loop";
  std::vector<int> per_frame(kNumPoses, 0);
  for (int k : scene_->pose_ids) ++per_frame[k];
  EXPECT_GE(*std::min_element(per_frame.begin(), per_frame.end()), 50)
      << "every frame needs a usable number of observations (cuSFM's fewest: 54)";
  // The front end's map is locally consistent: small initial reprojection
  // errors (cuSFM's: median 1.05 px, p90 4.5 px).
  std::vector<double> errors;
  for (size_t i = 0; i < scene_->observations.size(); ++i) {
    const Vector<3> &x = scene_->initial_points[scene_->point_ids[i]];
    const Rigid camera_from_world = Compose(CameraFromRig(scene_->camera_ids[i]),
                                            Inverse(scene_->initial_poses[scene_->pose_ids[i]]));
    const Vec3 pc = Apply(camera_from_world, {x[0], x[1], x[2]});
    errors.push_back(kFocal * std::hypot(pc[0] / pc[2] - scene_->observations[i][0],
                                         pc[1] / pc[2] - scene_->observations[i][1]));
  }
  std::sort(errors.begin(), errors.end());
  const double p50 = errors[errors.size() / 2], p90 = errors[errors.size() * 9 / 10];
  std::printf("[stereo loop BA] initial reprojection error [px]: p50 %.2f p90 %.2f\n", p50, p90);
  EXPECT_LT(p50, 2.5);
  EXPECT_LT(p90, 10.0);
  const auto [mean, max] = MeanMax(AlignedPositionErrors(InitialPositions(), TruthPositions()));
  EXPECT_GT(mean, 0.25) << "the initial drift must be loop-scale";
  EXPECT_GT(max, 0.5);
  EXPECT_LT(max, 1.0);
}

#ifdef CUNLS_ENABLE_CUDSS
/** An exact linear solve closes the loop within one solve. */
TEST_F(StereoLoopBaTest, CuDssClosesTheLoop) {
  std::vector<double> before, after;
  const MinimizerSummary summary = Solve(SparseLinearSolverType::cuDSS, &before, &after);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_LT(summary.num_iterations, 200u) << "LM should converge before the iteration cap";
  EXPECT_LT(MeanMax(after).second, 0.02) << "max position error [m]";
}
#endif

/**
 * The landmarks eliminated, the reduced camera system (4338 columns) solved
 * by the supernodal Cholesky: exact like cuDSS on the full system.
 */
TEST_F(StereoLoopBaTest, SchurClosesTheLoop) {
  std::vector<double> before, after;
  const MinimizerSummary summary = Solve(SparseLinearSolverType::BlockSparsePCG, &before, &after,
                                         LinearSystemType::SchurComplement);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_LT(summary.num_iterations, 200u) << "LM should converge before the iteration cap";
  EXPECT_LT(MeanMax(after).second, 0.02) << "max position error [m]";
}

/**
 * The default inner solver (block-Jacobi PCG on the full system, 200
 * iterations, relative tolerance 1e-3) under-solves the loop closure mode and
 * creeps: it reaches the iteration cap with most of the drift left (use the
 * Schur complement or cuDSS for such problems). Disabled; run it with
 * --gtest_also_run_disabled_tests to see the gap.
 */
TEST_F(StereoLoopBaTest, DISABLED_BlockSparsePcgClosesTheLoop) {
  std::vector<double> before, after;
  const MinimizerSummary summary = Solve(SparseLinearSolverType::BlockSparsePCG, &before, &after);
  EXPECT_LT(summary.final_cost, summary.initial_cost);
  EXPECT_LT(MeanMax(after).second, 0.02) << "max position error [m]";
}

}  // namespace
}  // namespace cunls
