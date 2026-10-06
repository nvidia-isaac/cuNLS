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

"""Rerun logging of ``tartan_vio.py``.

Layout: the 3D scene (ground truth, every method's trajectory, the rig with
its camera, the matched landmarks and their rays) on the left; on the right
the camera image with the matches classified by the inertial RANSAC, the
position error of every method, the injected outlier ratio against the
fraction RANSAC rejected, the IMU bias estimates and a status panel.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

GREEN = [118, 185, 0]
RED = [255, 79, 94]
AMBER = [255, 191, 60]
MAGENTA = [255, 61, 245]
SLATE = [120, 140, 160]
WHITE = [235, 240, 245]
COLORS = {
    "inertial RANSAC": GREEN,
    "visual RANSAC": [255, 143, 46],
    "inertial LM + Huber": [199, 125, 255],
    "IMU dead reckoning": [140, 153, 166],
}
AXIS = ([255, 107, 107], [77, 213, 153], [77, 163, 255])
GENUINE, SWAPPED, SHIFTED = 0, 1, 2  # as in tartan_vio.py
ERROR_CLIP_M = 5.0  # the error plot saturates here (dead reckoning leaves within seconds)
HERO = "inertial RANSAC"


DRAWN_PERCENT = 25  # share of the tracks drawn in the image and the 3D view


def _drawn(ids):
    """A fixed pseudo-random DRAWN_PERCENT of the track ids."""
    return (np.asarray(ids, np.uint64) * np.uint64(2654435761) % np.uint64(100)) < DRAWN_PERCENT


def _path(name):
    """Entity path segment of a display name: "inertial LM + Huber" -> "inertial_lm_huber"."""
    return "_".join(name.lower().replace("+", " ").replace("(", " ").replace(")", " ").split())


def blueprint(true_positions, image_size, play_state=None):
    """The layout; the 3D eye looks down on the whole ground-truth path (NED: up is -z),
    tilted slightly so the orbit controls keep a well-defined up direction."""
    lo, hi = true_positions.min(axis=0), true_positions.max(axis=0)
    target = (lo + hi) / 2
    height = 1.35 * np.max(hi[:2] - lo[:2])
    eye = target + np.array([-0.15 * height, 0.0, -height])
    return rrb.Blueprint(
        rrb.Horizontal(
            column_shares=[0.55, 0.45],
            contents=[
                rrb.Spatial3DView(
                    name="3D", origin="world",
                    background=rrb.Background(kind=rrb.BackgroundKind.SolidColor,
                                              color=[11, 14, 19]),
                    eye_controls=rrb.EyeControls3D(
                        kind=rrb.Eye3DKind.Orbital, position=eye, look_target=target,
                        eye_up=[0.0, 0.0, -1.0])),
                rrb.Vertical(
                    row_shares=[0.36, 0.2, 0.14, 0.14, 0.16],
                    contents=[
                        rrb.Spatial2DView(name="front camera: matches classified by inertial "
                                          "RANSAC", origin="world/rig/camera/image",
                                          # Fixed to the image: drawn matches cannot resize it.
                                          visual_bounds=rrb.VisualBounds2D(
                                              x_range=[0, image_size], y_range=[0, image_size])),
                        rrb.TimeSeriesView(name="position error [m]", origin="plots/error",
                                           axis_y=rrb.ScalarAxis(range=(0.0, 3.0))),
                        rrb.TimeSeriesView(name="matches: injected outliers vs rejected",
                                           origin="plots/matches",
                                           axis_y=rrb.ScalarAxis(range=(0.0, 1.0))),
                        rrb.Horizontal(contents=[
                            rrb.TimeSeriesView(name="gyro bias [rad/s]", origin="plots/gyro_bias"),
                            rrb.TimeSeriesView(name="accel bias [m/s²]",
                                               origin="plots/accel_bias"),
                        ]),
                        rrb.TextDocumentView(name="status", origin="status"),
                    ]),
            ]),
        rrb.TimePanel(state="collapsed", timeline="time", play_state=play_state),
        collapse_panels=True,
    )


class Logger:
    def __init__(self, seq, methods, segments, rrd_path=None, spawn=False, camera_from_rig=None,
                 focal=320.0, center=320.0, size=640, true_bias=None):
        self.seq, self.methods, self.segments = seq, methods, segments
        self.camera_from_rig = camera_from_rig
        self.focal, self.center, self.size = focal, center, size
        self.true_bias = true_bias
        rr.init("cunls_tartan_vio")
        sinks = ([rr.GrpcSink()] if spawn else []) + ([rr.FileSink(rrd_path)] if rrd_path else [])
        if spawn:
            rr.spawn(connect=False)
        rr.set_sinks(*sinks)
        rr.send_blueprint(blueprint(seq.true_poses[:, :3, 3], size))
        self.trails = {m: [] for m in methods}
        self.map_points = {}
        self._static()

    def _static(self):
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # NED
        gt = self.seq.true_poses[:, :3, 3]
        rr.log("world/ground_truth", rr.LineStrips3D([gt], colors=[WHITE], radii=0.05),
               static=True)
        for m, c in COLORS.items():
            rr.log(f"plots/error/{_path(m)}", rr.SeriesLines(colors=[c], names=[m], widths=[2]),
                   static=True)
        rr.log("plots/matches/injected", rr.SeriesLines(colors=[WHITE], names=["injected"]),
               static=True)
        rr.log("plots/matches/rejected", rr.SeriesLines(colors=[GREEN], names=["rejected"]),
               static=True)
        for name, sl in (("gyro_bias", slice(0, 3)), ("accel_bias", slice(3, 6))):
            for i, axis in enumerate("xyz"):
                rr.log(f"plots/{name}/{axis}", rr.SeriesLines(colors=[AXIS[i]], names=[axis]),
                       static=True)
                rr.log(f"plots/{name}/{axis}_true",
                       rr.SeriesLines(colors=[[c // 2 for c in AXIS[i]]], names=[f"{axis} true"]),
                       static=True)
        rig_from_camera = np.linalg.inv(self.camera_from_rig)
        rr.log("world/rig/camera", rr.Transform3D(translation=rig_from_camera[:3, 3],
                                                  mat3x3=rig_from_camera[:3, :3]), static=True)
        rr.log("world/rig/camera", rr.Pinhole(focal_length=self.focal,
                                              principal_point=[self.center, self.center],
                                              width=self.size, height=self.size,
                                              image_plane_distance=0.6), static=True)

    def log(self, k, image, frame):
        rr.set_time("frame", sequence=k)
        rr.set_time("time", duration=frame.t)
        true_pose = self.seq.true_poses[k]
        hero = frame.estimates[HERO]
        T = hero.pose

        # Trajectories (each only while within 10 m of the truth).
        errors = {}
        for m in self.methods:
            p = frame.estimates[m].pose[:3, 3]
            errors[m] = float(np.linalg.norm(p - true_pose[:3, 3]))
            if errors[m] < 10:
                self.trails[m].append(p)
            if self.trails[m]:
                rr.log(f"world/trajectory/{_path(m)}",
                       rr.LineStrips3D([np.array(self.trails[m])], colors=[COLORS[m]],
                                       radii=0.12 if m == HERO else 0.08))
            rr.log(f"plots/error/{_path(m)}", rr.Scalars(min(errors[m], ERROR_CLIP_M)))
        rr.log("world/rig", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("world/rig/camera/image", rr.Image(image[..., ::-1]).compress(jpeg_quality=80))

        # Matches of the inertial RANSAC: classified in the image, rays to their landmarks.
        # Only a fixed subset of the tracks is drawn (chosen by track id, so a drawn track
        # stays drawn), large enough to tell kept from rejected; accepted outliers always.
        matches = frame.matched[HERO]
        kept = hero.inlier_mask.astype(bool) if len(matches.kind) else np.zeros(0, bool)
        kind = matches.kind
        px = matches.observations * self.focal + self.center
        inside = np.all((px >= 0) & (px < self.size), axis=1)
        shown = inside & (_drawn(matches.ids) | (kept & (kind != GENUINE)))
        px = px[shown]
        kept, kind = kept[shown], kind[shown]
        classes = [
            ("kept", kept & (kind == GENUINE), GREEN, 10),
            ("rejected swapped", ~kept & (kind == SWAPPED), RED, 10),
            ("rejected shifted", ~kept & (kind == SHIFTED), AMBER, 10),
            ("rejected genuine (track or map error)", ~kept & (kind == GENUINE), SLATE, 7),
            ("ACCEPTED OUTLIER", kept & (kind != GENUINE), MAGENTA, 16),
        ]
        for name, sel, color, radius in classes:
            rr.log(f"world/rig/camera/image/{_path(name)}",
                   rr.Points2D(px[sel], colors=[color], radii=radius))
        # Injected coherent shifts: arrows from the tracked feature to what the back end saw.
        shifted = (frame.kind == SHIFTED) & _drawn(frame.ids) & \
            np.all((frame.obs_px >= 0) & (frame.obs_px < self.size), axis=1)
        rr.log("world/rig/camera/image/injected_shift",
               rr.Arrows2D(origins=frame.pixels[shifted],
                           vectors=frame.obs_px[shifted] - frame.pixels[shifted],
                           colors=[AMBER], radii=2.5))

        camera_center = (T @ np.linalg.inv(self.camera_from_rig))[:3, 3]
        points = matches.points[shown]
        colors = np.where(kept[:, None], GREEN, np.where((kind == GENUINE)[:, None], SLATE, RED))
        rr.log("world/rays", rr.LineStrips3D([np.stack([camera_center, p]) for p in points],
                                             colors=colors, radii=0.012))
        rr.log("world/landmarks/matched", rr.Points3D(points, colors=colors, radii=0.12))
        for i, p in zip(matches.ids, matches.points):
            self.map_points[i] = p
        if k % 10 == 0 and self.map_points:
            rr.log("world/landmarks/map",
                   rr.Points3D(np.array(list(self.map_points.values())), colors=[SLATE],
                               radii=0.02))

        # Plots.
        n = len(kind)
        if n:
            rr.log("plots/matches/injected", rr.Scalars(float(np.mean(kind != GENUINE))))
            rr.log("plots/matches/rejected", rr.Scalars(float(np.mean(~kept))))
        if np.all(np.isfinite(hero.bias)):
            for name, sl in (("gyro_bias", slice(0, 3)), ("accel_bias", slice(3, 6))):
                for i, axis in enumerate("xyz"):
                    rr.log(f"plots/{name}/{axis}", rr.Scalars(hero.bias[sl][i]))
                    rr.log(f"plots/{name}/{axis}_true", rr.Scalars(self.true_bias[sl][i]))

        # Status panel.
        rows = "\n".join(
            f"| {m} | {'**' if m == HERO else ''}"
            f"{'lost' if errors[m] > 50 else f'{errors[m]:.2f} m'}{'**' if m == HERO else ''} |"
            for m in self.methods)
        injected = int(np.sum(kind != GENUINE))
        accepted = int(np.sum(kept & (kind != GENUINE)))
        solve = (f"{hero.num_hypotheses} hypotheses (2-match samples), {hero.solve_ms:.1f} ms"
                 if hero.num_hypotheses else "IMU and priors only (no matches)")
        rr.log("status", rr.TextDocument(
            f"### t = {frame.t:5.1f} s · {frame.segment.upper()}\n\n"
            f"**{n}** matches to the map, **{injected}** injected outliers, "
            f"**{accepted}** of them accepted · {solve}\n\n"
            f"| method | position error |\n|---|---|\n{rows}\n",
            media_type=rr.MediaType.MARKDOWN))
