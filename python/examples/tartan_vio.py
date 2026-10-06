# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A small RGB-D inertial odometry on TartanGround, with injected outliers.

The legged robot of the TartanGround ``OldTownFall/Data_anymal/P2000``
sequence walks 82 m in 129 s. Its front camera (640 x 640, 10 Hz, with
depth) and IMU (100 Hz) drive a minimal odometry:

* front end (OpenCV): KLT feature tracks; a track's landmark is created from
  the depth image at its first frame, through the pose estimated for that
  frame, and stays fixed (a frame-to-map odometry, it drifts slowly);
* back end (cuNLS): every frame, an inertial RANSAC PnP on a two-frame
  window: the pose, velocity and IMU bias of the previous and the current
  frame (30 free tangent dimensions); ``ImuFactorBatch`` between them and
  priors on the previous frame from the last solve, always on; one
  ``PnPFactorBatch`` factor per tracked landmark, sampled and classified by
  ``RansacLevenbergMarquardtMinimizer``. Because the IMU predicts the motion,
  two matches per hypothesis determine the pose, and a group of mismatches
  that agrees with a motion the IMU rules out is rejected.

Outliers are injected into the 2D matches the back end sees: swapped matches
(a track reports another track's feature), a coherent shift of a third of
the tracks (repetitive texture: they agree with a pose rotated by 5 deg), a
camera blackout, and 75% swaps. White noise and biases are added to the
dataset's ideal IMU. Three baselines run on the same data, each with its own
map: visual-only RANSAC PnP (constant-velocity prediction), the same inertial
window solved by Levenberg-Marquardt with a Huber loss instead of RANSAC, and
IMU dead reckoning.

Download (needs ``pip install tartanair``; about 1.1 GB)::

    import tartanair as ta
    ta.init("dataset/tartan_ground")
    ta.download_ground(env=["OldTownFall"], version=["anymal"], traj=["P2000"],
                       modality=["image", "depth", "imu"], camera_name=["lcam_front"],
                       unzip=True)

Usage::

    python tartan_vio.py --data dataset/tartan_ground/OldTownFall/Data_anymal/P2000 \\
        [--rrd tartan_vio.rrd | --spawn]

``--rrd`` / ``--spawn`` log the run to Rerun (``pip install rerun-sdk``);
open a recording with ``rerun tartan_vio.rrd``. Needs ``opencv-python``.
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import cupy as cp
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import pycunls
from example_utils import report

FOCAL, CENTER, IMAGE_SIZE = 320.0, 320.0, 640  # lcam_front pinhole
GRAVITY = np.array([0.0, 0.0, 9.8])  # TartanAir world frame is NED: z down
SAMPLES_PER_FRAME = 10  # IMU 100 Hz, camera 10 Hz
# The rig (and the IMU) is the lcam_front body frame: x forward, y right, z down.
# OpenCV camera axes: x right = +y, y down = +z, z forward = +x.
CAMERA_FROM_RIG = np.eye(4)
CAMERA_FROM_RIG[:3, :3] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

# Noise and biases added to the dataset's ideal IMU (a consumer-grade MEMS IMU).
GYRO_NOISE_DENSITY = 2e-3  # rad/s/sqrt(Hz)
ACCEL_NOISE_DENSITY = 2e-2  # m/s^2/sqrt(Hz)
GYRO_BIAS = np.array([0.010, -0.008, 0.012])  # rad/s
ACCEL_BIAS = np.array([0.08, -0.06, 0.10])  # m/s^2

MIN_TRACKS, MAX_TRACKS = 180, 300
MATCH_SIGMA_PX = 1.0
MAX_DEPTH = 40.0  # m; farther pixels (sky) do not make landmarks
GENUINE, SWAPPED, SHIFTED = 0, 1, 2  # match kinds
COHERENT_SHIFT_PX = np.array([28.0, 0.0])  # ~5 deg of yaw


@dataclass
class Segment:
    start: float
    end: float
    name: str
    swapped: float = 0.0  # fraction of matches swapped with another track's feature
    shifted: float = 0.0  # fraction shifted by COHERENT_SHIFT_PX
    blackout: bool = False  # the camera delivers black frames


SEGMENTS = [
    Segment(0, 12, "nominal", 0.10),
    Segment(12, 27, "repetitive texture", 0.15, 0.35),
    Segment(27, 40, "nominal", 0.10),
    Segment(40, 43, "camera blackout", blackout=True),
    Segment(43, 55, "nominal", 0.10),
    Segment(55, 70, "heavy clutter", 0.75),
    Segment(70, 1e9, "nominal", 0.10),
]


def segment_at(t):
    return next(s for s in SEGMENTS if s.start <= t < s.end)


# --------------------------------------------------------------------------- SE(3) and IMU helpers
def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w):
    """Rodrigues' formula."""
    theta = np.linalg.norm(w)
    K = skew(w)
    if theta < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * K @ K


def pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def integrate_imu(R, v, p, samples, bias, gravity=GRAVITY):
    """Euler integration of ``ImuFactorBatch``: the state after the samples
    (rows [gyro, accel, dt]) from (R, v, p) with the bias [b_g, b_a] removed."""
    for s in samples.astype(np.float64):
        w, a, dt = s[:3] - bias[:3], s[3:6] - bias[3:], s[6]
        acc = R @ a + gravity
        p = p + v * dt + 0.5 * acc * dt * dt
        v = v + acc * dt
        R = R @ so3_exp(w * dt)
    return R, v, p


def project(world_from_rig, camera_from_rig, points):
    """Points in camera coordinates and their normalized projections."""
    camera_from_world = camera_from_rig @ np.linalg.inv(world_from_rig)
    pc = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pc[:, :2] / pc[:, 2:3]
    return pc, uv


# --------------------------------------------------------------------------- back end
INLIER_THRESHOLD = 3.5  # on the whitened residual |r| / sigma (chi2_2: 99.8% of inliers)
MAX_MATCHES = 2048  # capacity of the match buffers, allocated once
MAX_IMU_SAMPLES = 64  # capacity of the IMU sample buffer (samples between two frames)


@dataclass
class Sensors:
    """What the estimators know about the rig. Matches passed to ``track`` are
    any object with ``observations`` (n, 2, normalized image coordinates),
    ``points`` (n, 3, world) and ``camera`` (n, index into ``cameras``)."""
    cameras: np.ndarray  # (num_cameras, 4, 4) camera_from_rig
    sigma: float  # match noise in normalized image units
    imu: pycunls.ImuParameters  # sensor model of the IMU factor (gravity included)

    @property
    def gravity(self):
        return np.asarray(self.imu.gravity, dtype=np.float64)


@dataclass
class FrameEstimate:
    """One method's output for one camera frame."""
    pose: np.ndarray  # world_from_rig
    velocity: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    bias: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    inlier_mask: np.ndarray = None  # one byte per match
    solve_ms: float = 0.0
    num_hypotheses: int = 0


def pnp_information(world_from_rig, matches, mask, sensors):
    """Gauss-Newton information J^T J of the inlier matches on the rig pose
    (tangent [phi; rho], perturbed on the right, as SE3StateBatch)."""
    keep = mask.astype(bool)
    pts, cam = matches.points[keep], matches.camera[keep]
    R, t = world_from_rig[:3, :3], world_from_rig[:3, 3]
    q = (pts - t) @ R  # in the rig frame
    E = sensors.cameras[cam]  # camera_from_rig per match
    pc = np.einsum("nij,nj->ni", E[:, :3, :3], q) + E[:, :3, 3]
    z = pc[:, 2]
    dproj = np.zeros((len(q), 2, 3))
    dproj[:, 0, 0] = dproj[:, 1, 1] = 1 / z
    dproj[:, :, 2] = -pc[:, :2] / z[:, None] ** 2
    dq = np.zeros((len(q), 3, 6))  # d q / d [phi; rho] = [[q]x, -I]
    dq[:, :, 3:] = -np.eye(3)
    dq[:, 0, 1], dq[:, 0, 2] = -q[:, 2], q[:, 1]
    dq[:, 1, 0], dq[:, 1, 2] = q[:, 2], -q[:, 0]
    dq[:, 2, 0], dq[:, 2, 1] = -q[:, 1], q[:, 0]
    J = dproj @ E[:, :3, :3] @ dq / sensors.sigma
    return np.einsum("nri,nrj->ij", J, J)


class MatchBuffers:
    """Device buffers of the matches of one frame, sized once for MAX_MATCHES:
    observations, matched world points and the camera_from_rig of each match."""

    def __init__(self, sensors):
        self.sensors = sensors
        self.observations = cp.zeros((MAX_MATCHES, 2), dtype=cp.float32)
        self.points = cp.zeros((MAX_MATCHES, 3), dtype=cp.float32)
        self.camera_from_rig = cp.zeros((MAX_MATCHES, 4, 4), dtype=cp.float32)

    def upload(self, matches):
        n = min(len(matches.camera), MAX_MATCHES)
        self.observations[:n] = cp.asarray(matches.observations[:n], dtype=cp.float32)
        self.points[:n] = cp.asarray(matches.points[:n], dtype=cp.float32)
        self.camera_from_rig[:n] = cp.asarray(self.sensors.cameras[matches.camera[:n]],
                                              dtype=cp.float32)
        return n

    def pnp_batch(self):
        """PnP factors whitened by the match noise: |r| is in units of sigma."""
        pnp = pycunls.PnPFactorBatch(self.observations, self.camera_from_rig, self.points,
                                     MAX_MATCHES)
        return pycunls.WeightedFactorBatch(pnp, 1.0 / self.sensors.sigma)


def classify(world_from_rig, matches, sensors):
    """Matches within INLIER_THRESHOLD of their projection (in front of the camera)."""
    mask = np.zeros(len(matches.camera), np.uint8)
    for c, camera_from_rig in enumerate(sensors.cameras):
        sel = matches.camera == c
        pc, uv = project(world_from_rig, camera_from_rig, matches.points[sel])
        r = np.linalg.norm(uv - matches.observations[sel], axis=1) / sensors.sigma
        mask[sel] = (pc[:, 2] > 0) & (r <= INLIER_THRESHOLD)
    return mask


def ransac_options(sample_size):
    options = pycunls.RansacLevenbergMarquardtMinimizerOptions()
    ransac = options.base_options  # a reference: edits change `options`
    ransac.hypotheses_per_round = 256
    ransac.max_rounds = 8
    ransac.sample_size = sample_size  # 0 = ceil(D / 2)
    ransac.hypothesis_iterations = 5
    ransac.final_iterations = 10
    ransac.seed = 1
    return options


class InertialPnP:
    """Two-frame inertial PnP window: slot 0 is the previous frame (with priors
    from the previous solve), slot 1 the current frame. Built once; every
    frame rewrites the buffers and the active counts."""

    # Priors on the previous frame (an approximate marginalization): its pose
    # carries the covariance of the previous solve; velocity and bias use fixed
    # widths, so the bias estimate is a running average of what the frames observe.
    VELOCITY_SIGMA = 0.05  # m/s
    BIAS_SIGMA = np.array([5e-3] * 3 + [1e-1] * 3)  # rad/s, m/s^2
    POSE_PROCESS_SIGMA = np.array([1e-3] * 3 + [5e-3] * 3)  # IMU prediction per frame: rad, m

    def __init__(self, sensors, robust, sample_size=2):
        self.sensors = sensors
        self.stream = pycunls.CudaStream()

        # --- States: two slots each, all free ---
        self.pose_buf = cp.zeros((2, 4, 4), dtype=cp.float32)
        self.vel_buf = cp.zeros((2, 3), dtype=cp.float32)
        self.bias_buf = cp.zeros((2, 6), dtype=cp.float32)
        self.poses = pycunls.SE3StateBatch(self.pose_buf, 2)
        self.vels = pycunls.VectorStateBatch3(self.vel_buf, 2)
        self.biases = pycunls.VectorStateBatch6(self.bias_buf, 2)
        for s in (self.poses, self.vels, self.biases):
            s.set_num_active_states(2)

        # --- Matches: every factor reads the current pose ---
        self.matches = MatchBuffers(sensors)
        self.pnp = self.matches.pnp_batch()

        # --- IMU factor between the two slots; offsets [0, n] rewritten per frame ---
        self.imu_samples = cp.zeros((MAX_IMU_SAMPLES, 7), dtype=cp.float32)
        self.imu_offsets = cp.zeros(2, dtype=cp.int32)
        self.imu = pycunls.ImuFactorBatch(self.imu_samples, self.imu_offsets, MAX_IMU_SAMPLES,
                                          sensors.imu, 1)
        self.imu.set_num_active_factors(1)

        # --- Priors on slot 0, whitened by their square-root information ---
        self.pose_prior_obs = cp.zeros((1, 4, 4), dtype=cp.float32)
        self.pose_prior_sqrt_info = cp.zeros((1, 6, 6), dtype=cp.float32)
        self.pose_prior = pycunls.InformationFactorBatch(
            pycunls.SE3PriorFactorBatch(self.pose_prior_obs, 1), self.pose_prior_sqrt_info)
        self.vel_prior_obs = cp.zeros((1, 3), dtype=cp.float32)
        self.vel_prior = pycunls.WeightedFactorBatch(
            pycunls.PriorVectorFactorBatch3(self.vel_prior_obs, 1), 1.0 / self.VELOCITY_SIGMA)
        self.bias_prior_obs = cp.zeros((1, 6), dtype=cp.float32)
        self.bias_prior_sqrt_info = cp.asarray(np.diag(1.0 / self.BIAS_SIGMA)[None],
                                               dtype=cp.float32)
        self.bias_prior = pycunls.InformationFactorBatch(
            pycunls.PriorVectorFactorBatch6(self.bias_prior_obs, 1), self.bias_prior_sqrt_info)
        for prior in (self.pose_prior, self.vel_prior, self.bias_prior):
            prior.set_num_active_factors(1)

        # --- Problem: residual batch 0 is the matches, the rest is always on ---
        self.problem = pycunls.Problem()
        for s in (self.poses, self.vels, self.biases):
            self.problem.add_state_batch(s)
        if robust == "huber":
            self.problem.add_factor_batch(self.pnp, pycunls.HuberLossFunctionBatch(2.0), [])
        else:
            self.problem.add_factor_batch(self.pnp, [])
        P, V, B = self.poses.state_device_ptr, self.vels.state_device_ptr, \
            self.biases.state_device_ptr
        self.problem.add_factor_batch(self.imu, [P(0), V(0), B(0), P(1), V(1), B(1)])
        self.problem.add_factor_batch(self.pose_prior, [P(0)])
        self.problem.add_factor_batch(self.vel_prior, [V(0)])
        self.problem.add_factor_batch(self.bias_prior, [B(0)])

        # --- Minimizers, constructed once and reused every frame ---
        self.sample_size = sample_size
        self.ransac = None
        if robust == "ransac":
            options = ransac_options(sample_size)
            options.base_options.factor_batches = [  # one entry per residual batch, in order
                pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, INLIER_THRESHOLD)
            ] + [pycunls.RansacFactorBatchOptions(pycunls.RansacRole.always_on)] * 4
            self.ransac = pycunls.RansacLevenbergMarquardtMinimizer(options)
        lm_options = pycunls.LevenbergMarquardtMinimizerOptions()
        lm_options.base_options.max_num_iterations = 15
        self.lm = pycunls.LevenbergMarquardtMinimizer(lm_options)

    def start(self, world_from_rig, velocity, pose_sigma):
        """Initial state of the track (from an initializer; bias unknown, zero)."""
        self.state = (world_from_rig, velocity, np.zeros(6))
        self.pose_cov = np.diag(np.asarray(pose_sigma) ** 2)

    def predict(self, imu_samples):
        """The current frame's (world_from_rig, velocity) integrated from the last estimate."""
        T0, v0, b0 = self.state
        R1, v1, p1 = integrate_imu(T0[:3, :3], v0, T0[:3, 3], imu_samples, b0,
                                   self.sensors.gravity)
        return pose(R1, p1), v1

    def track(self, matches, imu_samples):
        T0, v0, b0 = self.state
        # Prediction for the current frame: integrate the IMU from the previous one.
        T1, v1 = self.predict(imu_samples)
        self.pose_buf[...] = cp.asarray(np.stack([T0, T1]), dtype=cp.float32)
        self.vel_buf[...] = cp.asarray(np.stack([v0, v1]), dtype=cp.float32)
        self.bias_buf[...] = cp.asarray(np.stack([b0, b0]), dtype=cp.float32)
        num_samples = len(imu_samples)
        assert 0 < num_samples <= MAX_IMU_SAMPLES
        self.imu_samples[:num_samples] = cp.asarray(imu_samples, dtype=cp.float32)
        self.imu_offsets[1] = num_samples

        # Priors: the previous estimate and its pose covariance.
        self.pose_prior_obs[0] = cp.asarray(T0, dtype=cp.float32)
        self.pose_prior_sqrt_info[0] = cp.asarray(
            np.linalg.cholesky(np.linalg.inv(self.pose_cov)).T, dtype=cp.float32)
        self.vel_prior_obs[0] = cp.asarray(v0, dtype=cp.float32)
        self.bias_prior_obs[0] = cp.asarray(b0, dtype=cp.float32)

        # Matches: rewrite the buffers, the active count and the connectivity.
        n = self.matches.upload(matches)
        self.pnp.set_num_active_factors(n)
        self.problem.set_state_pointers(0, [self.poses.state_device_ptr(1)] * n)

        start = time.perf_counter()
        mask, hypotheses = None, 0
        if self.ransac is not None and n > self.sample_size:
            summary = self.ransac.minimize(self.stream, self.problem)  # writes the states back
            mask = self.ransac.inlier_mask(0)[:n].copy()  # numpy uint8, 1 = inlier
            hypotheses = summary.num_hypotheses
        else:  # Huber, or too few matches to sample from: LM on everything
            self.lm.minimize(self.stream, self.problem)
        solve_ms = 1e3 * (time.perf_counter() - start)

        T1 = cp.asnumpy(self.pose_buf[1]).astype(np.float64)
        v1 = cp.asnumpy(self.vel_buf[1]).astype(np.float64)
        b1 = cp.asnumpy(self.bias_buf[1]).astype(np.float64)
        if mask is None:  # classify for the report: the Huber "inliers" are within the threshold
            mask = classify(T1, matches, self.sensors)
        info = pnp_information(T1, matches, mask, self.sensors) if n else np.zeros((6, 6))
        # Covariance of the new pose: the IMU prediction's, updated with the inliers.
        predicted = self.pose_cov + np.diag(self.POSE_PROCESS_SIGMA ** 2)
        self.pose_cov = np.linalg.inv(np.linalg.inv(predicted) + info)
        self.state = (T1, v1, b1)
        return FrameEstimate(T1, v1, b1, mask, solve_ms, hypotheses)


class VisualPnP:
    """Visual-only RANSAC PnP: one free pose, the matches, no IMU. The initial
    guess is the constant-velocity extrapolation of the last two estimates."""

    MIN_INLIERS = 8  # fewer: the frame failed, keep the prediction

    def __init__(self, sensors):
        self.stream = pycunls.CudaStream()
        self.pose_buf = cp.zeros((1, 4, 4), dtype=cp.float32)
        self.pose = pycunls.SE3StateBatch(self.pose_buf, 1)
        self.pose.set_num_active_states(1)
        self.matches = MatchBuffers(sensors)
        self.pnp = self.matches.pnp_batch()
        self.problem = pycunls.Problem()
        self.problem.add_state_batch(self.pose)
        self.problem.add_factor_batch(self.pnp, [])
        options = ransac_options(0)  # 0: ceil(6 / 2) = 3 matches per sample
        options.base_options.default_inlier_threshold = INLIER_THRESHOLD
        self.ransac = pycunls.RansacLevenbergMarquardtMinimizer(options)

    def start(self, world_from_rig):
        self.history = [world_from_rig, world_from_rig]

    def predict(self):
        prev2, prev = self.history[-2:]
        return prev @ np.linalg.inv(prev2) @ prev  # constant velocity

    def track(self, matches):
        guess = self.predict()
        n = self.matches.upload(matches)
        estimate = FrameEstimate(guess, inlier_mask=np.zeros(n, np.uint8))
        if n > 3:
            self.pose_buf[0] = cp.asarray(guess, dtype=cp.float32)
            self.pnp.set_num_active_factors(n)
            self.problem.set_state_pointers(0, [self.pose.state_device_ptr(0)] * n)
            start = time.perf_counter()
            summary = self.ransac.minimize(self.stream, self.problem)
            estimate.solve_ms = 1e3 * (time.perf_counter() - start)
            estimate.num_hypotheses = summary.num_hypotheses
            if summary.num_inliers >= self.MIN_INLIERS:
                estimate.pose = cp.asnumpy(self.pose_buf[0]).astype(np.float64)
                estimate.inlier_mask = self.ransac.inlier_mask(0)[:n].copy()
        # Too few matches or inliers: the frame is a failure, coast on the prediction.
        self.history.append(estimate.pose)
        return estimate


# --------------------------------------------------------------------------- dataset
class TartanSequence:
    """Ground truth, IMU samples (with added noise and bias) and images."""

    def __init__(self, path, rng):
        self.path = path
        gt = np.loadtxt(os.path.join(path, "pose_lcam_front.txt"))  # x y z qx qy qz qw
        imu = os.path.join(path, "imu")
        gyro, acc = np.load(os.path.join(imu, "gyro.npy")), np.load(os.path.join(imu, "acc.npy"))
        vel = np.load(os.path.join(imu, "vel_global.npy"))
        n = min(len(gt), len(gyro) // SAMPLES_PER_FRAME + 1)
        self.num_frames = n
        self.true_poses = np.stack([pose(Rotation.from_quat(q).as_matrix(), t)
                                    for t, q in zip(gt[:n, :3], gt[:n, 3:])])
        self.true_velocities = vel[::SAMPLES_PER_FRAME][:n]
        dt = 0.01
        m = (n - 1) * SAMPLES_PER_FRAME
        gyro = gyro[:m] + GYRO_BIAS + rng.normal(0, GYRO_NOISE_DENSITY / np.sqrt(dt), (m, 3))
        acc = acc[:m] + ACCEL_BIAS + rng.normal(0, ACCEL_NOISE_DENSITY / np.sqrt(dt), (m, 3))
        self.imu = np.concatenate([gyro, acc, np.full((m, 1), dt)], 1).astype(np.float32)
        self.true_bias = np.concatenate([GYRO_BIAS, ACCEL_BIAS])

    def imu_between(self, k):
        return self.imu[(k - 1) * SAMPLES_PER_FRAME:k * SAMPLES_PER_FRAME]

    def image(self, k):
        return cv2.imread(os.path.join(self.path, "image_lcam_front", f"{k:06d}_lcam_front.png"))

    def depth(self, k):
        """Planar depth in meters: a float32 per pixel, its bytes stored in the four
        channels of a PNG in B, G, R, A order (the official TartanAir reader takes
        them from cv2.imread, which returns BGRA). PIL returns RGBA: swap R and B back
        before reinterpreting the bytes, or every value is garbled."""
        rgba = np.asarray(Image.open(os.path.join(self.path, "depth_lcam_front",
                                                  f"{k:06d}_lcam_front_depth.png")))
        return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]]).view("<f4")[..., 0]


def tartan_sensors():
    params = pycunls.ImuParameters()
    params.gravity = GRAVITY.tolist()
    params.gyro_noise_density = GYRO_NOISE_DENSITY
    params.accel_noise_density = ACCEL_NOISE_DENSITY
    params.integration_noise_density = 1e-3
    params.gyro_bias_random_walk = 1e-3
    params.accel_bias_random_walk = 1e-2
    return Sensors(CAMERA_FROM_RIG[None], MATCH_SIGMA_PX / FOCAL, params)


# --------------------------------------------------------------------------- front end
class KltTracker:
    """KLT tracks with a forward-backward check, replenished with corners."""

    def __init__(self):
        self.prev = None
        self.points = np.zeros((0, 2), np.float32)
        self.ids = np.zeros(0, np.int64)
        self.next_id = 0

    def step(self, gray):
        if self.prev is not None and len(self.points):
            p0 = self.points.reshape(-1, 1, 2)
            p1, ok, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, winSize=(21, 21),
                                                 maxLevel=3)
            back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None,
                                                        winSize=(21, 21), maxLevel=3)
            p1, back = p1.reshape(-1, 2), back.reshape(-1, 2)
            good = (ok[:, 0] == 1) & (ok_back[:, 0] == 1) & \
                (np.linalg.norm(back - self.points, axis=1) < 0.5) & \
                np.all((p1 >= 0) & (p1 < IMAGE_SIZE - 1), axis=1)
            self.points, self.ids = p1[good], self.ids[good]
        if len(self.points) < MIN_TRACKS:
            mask = np.full(gray.shape, 255, np.uint8)
            for x, y in self.points:
                cv2.circle(mask, (int(x), int(y)), 12, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, MAX_TRACKS - len(self.points), 0.01, 12,
                                          mask=mask, blockSize=7)
            if new is not None:
                new = new.reshape(-1, 2)
                self.points = np.concatenate([self.points, new]).astype(np.float32)
                self.ids = np.concatenate([self.ids, self.next_id + np.arange(len(new))])
                self.next_id += len(new)
        self.prev = gray
        return self.points.copy(), self.ids.copy()


def inject_outliers(pixels, seg, rng):
    """The 2D matches the back end sees, and their kind (GENUINE / SWAPPED / SHIFTED)."""
    obs, kind = pixels.copy(), np.full(len(pixels), GENUINE)
    order = rng.permutation(len(pixels))
    n_shift = int(round(seg.shifted * len(pixels)))
    n_swap = int(round(seg.swapped * len(pixels)))
    shifted, swapped = order[:n_shift], order[n_shift:n_shift + n_swap]
    obs[shifted] += COHERENT_SHIFT_PX
    kind[shifted] = SHIFTED
    if len(swapped) > 1:  # each reports the feature of the next one: a wrong association
        obs[swapped] = pixels[np.roll(swapped, 1)]
        kind[swapped] = SWAPPED
    return obs, kind


@dataclass
class Matches:
    """Matches of one frame, in the form the estimators take."""
    observations: np.ndarray  # (n, 2) normalized image coordinates
    points: np.ndarray  # (n, 3) world points
    camera: np.ndarray  # (n,) camera index (one camera here)
    kind: np.ndarray  # (n,) GENUINE / SWAPPED / SHIFTED
    ids: np.ndarray  # (n,) track ids


class LandmarkMap:
    """One method's map: a world point per track id, from depth at the track's first frame."""

    def __init__(self):
        self.points = {}

    def matches(self, obs_px, ids, kind):
        has = np.array([i in self.points for i in ids], bool)
        pts = np.array([self.points[i] for i in ids[has]]).reshape(-1, 3)
        return Matches((obs_px[has] - CENTER) / FOCAL, pts, np.zeros(int(has.sum()), int),
                       kind[has], ids[has])

    def add(self, pixels, ids, depth, world_from_rig):
        world_from_camera = world_from_rig @ np.linalg.inv(CAMERA_FROM_RIG)
        for (u, v), i in zip(pixels, ids):
            if i in self.points:
                continue
            z = depth[int(round(v)), int(round(u))]
            if 0.3 < z < MAX_DEPTH:
                pc = np.array([(u - CENTER) / FOCAL * z, (v - CENTER) / FOCAL * z, z])
                self.points[i] = world_from_camera[:3, :3] @ pc + world_from_camera[:3, 3]


# --------------------------------------------------------------------------- odometry
METHODS = ("inertial RANSAC", "visual RANSAC", "inertial LM + Huber", "IMU dead reckoning")


@dataclass
class FrameLog:
    """What the report and the visualization need about one frame."""
    t: float
    segment: str
    pixels: np.ndarray  # tracked feature positions
    obs_px: np.ndarray  # what the back end saw (after injection)
    kind: np.ndarray
    ids: np.ndarray
    estimates: dict  # method -> FrameEstimate
    matched: dict  # method -> Matches


def run(seq, num_frames, on_frame=None):
    """Runs every method over the sequence; ``on_frame(k, image, FrameLog)`` is
    called after each frame (for the visualization)."""
    rng = np.random.default_rng(5)
    sensors = tartan_sensors()
    T0, v0 = seq.true_poses[0], seq.true_velocities[0]
    inertial, huber = InertialPnP(sensors, "ransac"), InertialPnP(sensors, "huber")
    visual = VisualPnP(sensors)
    for m in (inertial, huber):
        m.start(T0, v0, [1e-3] * 3 + [1e-3] * 3)
    visual.start(T0)
    dead = (T0[:3, :3], v0, T0[:3, 3])
    maps = {m: LandmarkMap() for m in METHODS[:3]}
    tracker = KltTracker()
    logs = []
    for k in range(num_frames):
        t = 0.1 * k
        seg = segment_at(t)
        image = seq.image(k)
        if seg.blackout:
            image = np.zeros_like(image)
        pixels, ids = tracker.step(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        obs_px, kind = inject_outliers(pixels, seg, rng)
        estimates, matched = {}, {}
        for m in METHODS[:3]:
            matched[m] = maps[m].matches(obs_px, ids, kind)
        if k == 0:
            for m in METHODS:
                estimates[m] = FrameEstimate(T0, v0, np.zeros(6),
                                             classify(T0, matched["inertial RANSAC"], sensors))
        else:
            samples = seq.imu_between(k)
            estimates["inertial RANSAC"] = inertial.track(matched["inertial RANSAC"], samples)
            estimates["inertial LM + Huber"] = huber.track(matched["inertial LM + Huber"], samples)
            estimates["visual RANSAC"] = visual.track(matched["visual RANSAC"])
            dead = integrate_imu(*dead, samples, np.zeros(6), GRAVITY)  # bias unknown
            estimates["IMU dead reckoning"] = FrameEstimate(pose(dead[0], dead[2]), dead[1])
        if not seg.blackout:
            depth = seq.depth(k)
            for m in METHODS[:3]:
                maps[m].add(pixels, ids, depth, estimates[m].pose)
        log = FrameLog(t, seg.name, pixels, obs_px, kind, ids, estimates, matched)
        logs.append(log)
        if on_frame:
            on_frame(k, image, log)
        if k % 100 == 0:
            e = {m: np.linalg.norm(estimates[m].pose[:3, 3] - seq.true_poses[k][:3, 3])
                 for m in METHODS}
            print(f"  frame {k:4d}  t={t:5.1f}s  {seg.name:<18}" +
                  "  ".join(f"{m.split()[0][:8]} {v:6.2f} m" for m, v in e.items()))
    return logs


def position_errors(seq, logs, method):
    est = np.stack([log.estimates[method].pose[:3, 3] for log in logs])
    return np.linalg.norm(est - seq.true_poses[:len(logs), :3, 3], axis=1)


def print_report(seq, logs):
    t = np.array([log.t for log in logs])
    errs = {m: position_errors(seq, logs, m) for m in METHODS}
    path = np.sum(np.linalg.norm(np.diff(seq.true_poses[:len(logs), :3, 3], axis=0), axis=1))
    print(f"\nTartanGround VIO: {len(logs)} frames, {t[-1]:.0f} s, {path:.1f} m walked")
    print(f"{'segment':<20}" + "".join(f"{m:>22}" for m in METHODS[:3]))
    print(f"{'':<20}" + f"{'position RMSE [m]':>22}" * 3)
    for seg in SEGMENTS:
        sel = (t >= seg.start) & (t < seg.end)
        if sel.any():
            print(f"{seg.name:<20}" + "".join(
                f"{np.sqrt(np.mean(errs[m][sel] ** 2)):>22.3f}" for m in METHODS[:3]))
    print()
    for m in METHODS:
        print(f"{m:<20}: final position error {errs[m][-1]:7.3f} m "
              f"({100 * errs[m][-1] / path:.2f}% of the path), max {errs[m].max():7.3f} m")
    kept = injected = injected_kept = genuine = 0
    for log in logs[1:]:
        mask = log.estimates["inertial RANSAC"].inlier_mask.astype(bool)
        bad = log.matched["inertial RANSAC"].kind != GENUINE
        injected += bad.sum()
        injected_kept += (mask & bad).sum()
        genuine += (~bad).sum()
        kept += (mask & ~bad).sum()
    print(f"\ninertial RANSAC: kept {kept} of {genuine} genuine matches, accepted "
          f"{injected_kept} of {injected} injected outliers")
    b = logs[-1].estimates["inertial RANSAC"].bias
    print(f"bias at the end: gyro {np.round(b[:3], 4)} (true {GYRO_BIAS})")
    print(f"                 accel {np.round(b[3:], 3)} (true {ACCEL_BIAS})")
    report.check(errs["inertial RANSAC"][-1] < 0.02 * path, "inertial RANSAC drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="dataset/tartan_ground/OldTownFall/Data_anymal/P2000")
    parser.add_argument("--frames", type=int, default=0, help="limit the frames (0: all)")
    parser.add_argument("--rrd", help="save a Rerun recording")
    parser.add_argument("--spawn", action="store_true", help="stream to a Rerun viewer")
    args = parser.parse_args()

    seq = TartanSequence(args.data, np.random.default_rng(11))
    n = min(args.frames or seq.num_frames, seq.num_frames)
    on_frame = None
    if args.rrd or args.spawn:
        from example_utils import tartan_vio_rerun
        on_frame = tartan_vio_rerun.Logger(seq, METHODS, SEGMENTS, args.rrd, args.spawn,
                                           CAMERA_FROM_RIG, FOCAL, CENTER, IMAGE_SIZE,
                                           seq.true_bias).log
    logs = run(seq, n, on_frame)
    print_report(seq, logs)


if __name__ == "__main__":
    main()
