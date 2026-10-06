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

"""Rerun logging of ``supermarket_drones.py``.

Two panels side by side: the store from above (the fused map without its
ceiling, the shoppers with their personal space, every drone with its planned
second ahead, its route and its destination) and a third-person view: a
virtual camera behind drone 0 that turns with its heading. The closest
distances, the solve time, the deliveries and a status text are logged too
(``plots/*``, ``status``) for adding views in the viewer.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Saturated colors that stand out against the (dimmed) store; shoppers are all one color.
PALETTE = [[0, 255, 120], [0, 210, 255], [255, 150, 0], [255, 60, 230], [255, 245, 0],
           [150, 120, 255], [0, 255, 220], [255, 255, 255]]
SHOPPER = [255, 50, 60]
BODY = (([0, 0, 0.45], [0.2, 0.17, 0.45]), ([0, 0, 1.15], [0.24, 0.2, 0.32]),
        ([0, 0, 1.6], [0.12, 0.12, 0.13]))  # (center, half sizes) of legs, torso, head
MAP_BRIGHTNESS = 0.55  # the store's colors are scaled down so the overlays stand out
RED = [255, 79, 94]
WHITE = [235, 240, 245]
ARM = 0.17  # m, rotor distance from the center (QuadrotorParameters.arm_length)
FRAME, MOTOR, PROP = [48, 52, 60], [100, 106, 116], [225, 230, 235]  # drone part colors
PROP_SPIN = 25.0  # rad/s, drawn propeller rotation (slowed down so it reads at 40 frames/s)
MAP_CEILING = 3.0  # m: map points above are not drawn (the view looks down into the store)
TRAIL_S = 3.0  # s of flown path drawn behind each drone
FOLLOW = 0  # the drone the third-person camera follows
CAMERA_BACK, CAMERA_UP, CAMERA_AHEAD = 1.3, 0.55, 2.0  # m: behind, above, the point it looks at
CAMERA_SIZE, CAMERA_FOCAL = (1280, 720), 560.0  # pixels: a wide (~98 deg) virtual camera
CAMERA_TURN = 0.06  # per step: how fast the camera swings to the drone's heading


def blueprint():
    """Two panels side by side: the store from above and the third-person view."""
    overview = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, look_target=[0.0, -2.5, 0.0],
                                 position=[0.0, -22.0, 30.0], eye_up=[0.0, 0.0, 1.0])
    background = rrb.Background(kind=rrb.BackgroundKind.SolidColor, color=[11, 14, 19])
    # A virtual camera behind the drone, turning with it: the 3D scene projected into it.
    third_person = rrb.Spatial2DView(
        name=f"third person: drone {FOLLOW}", origin="world/third_person",
        contents=["/world/**"],
        visual_bounds=rrb.VisualBounds2D(x_range=[0, CAMERA_SIZE[0]], y_range=[0, CAMERA_SIZE[1]]))
    return rrb.Blueprint(
        rrb.Horizontal(contents=[
            rrb.Spatial3DView(name="supermarket", origin="world", background=background,
                              eye_controls=overview),
            third_person,
        ]),
        rrb.TimePanel(state="collapsed", timeline="time"),
        collapse_panels=True,
    )


class Logger:
    def __init__(self, store, num_drones, person_spheres, person_zone, shopper_clearance,
                 drone_separation, rrd_path=None, spawn=False):
        self.num_drones = num_drones
        self.person_height = max(z + r for z, r in person_spheres)
        self.person_zone = person_zone
        self.shopper_clearance = shopper_clearance  # drone center to a shopper's body sphere
        self.separation = drone_separation  # between drone centers
        rr.init("cunls_supermarket_drones")
        sinks = ([rr.GrpcSink()] if spawn else []) + ([rr.FileSink(rrd_path)] if rrd_path else [])
        if spawn:
            rr.spawn(connect=False)
        rr.set_sinks(*sinks)
        rr.send_blueprint(blueprint())
        self.routes = [None] * num_drones
        self.camera_yaw = None
        self.trails = [[] for _ in range(num_drones)]
        self._static(store)

    def _static(self, store):
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/third_person", rr.Pinhole(
            focal_length=CAMERA_FOCAL, width=CAMERA_SIZE[0], height=CAMERA_SIZE[1],
            image_plane_distance=0.2), static=True)
        below = store.points[:, 2] < MAP_CEILING
        colors = (store.colors[below] * MAP_BRIGHTNESS).astype(np.uint8)
        rr.log("world/store", rr.Points3D(store.points[below], colors=colors, radii=0.03),
               static=True)
        a = ARM / np.sqrt(2)
        self.motors = np.array([[a, -a, 0.0], [-a, a, 0.0], [a, a, 0.0], [-a, -a, 0.0]])
        for b in range(self.num_drones):
            self._drone_model(f"world/drones/{b}", PALETTE[b % len(PALETTE)])
            # Half the separation: two drones' bubbles touch at the closest allowed distance.
            rr.log(f"world/drones/{b}/bubble", rr.Ellipsoids3D(
                radii=[self.separation / 2], colors=[[*PALETTE[b % len(PALETTE)], 70]],
                fill_mode="majorwireframe", line_radii=0.005), static=True)
            rr.log(f"world/drones/{b}/label", rr.Points3D(
                [[0, 0, 0.35]], radii=0.001, labels=[f"drone {b}"],
                colors=[PALETTE[b % len(PALETTE)]]), static=True)
        rr.log("plots/drones/closest", rr.SeriesLines(colors=[PALETTE[1]], names=["closest"]),
               static=True)
        rr.log("plots/drones/kept", rr.SeriesLines(
            colors=[RED], names=[f"kept >= {self.separation:.2f}"]), static=True)
        rr.log("plots/shoppers/closest", rr.SeriesLines(colors=[PALETTE[2]], names=["closest"]),
               static=True)
        rr.log("plots/shoppers/kept", rr.SeriesLines(
            colors=[RED], names=[f"kept >= {self.shopper_clearance:.2f}"]), static=True)
        rr.log("plots/solve/ms", rr.SeriesLines(colors=[WHITE], names=["solve"]), static=True)
        rr.log("plots/solve/budget", rr.SeriesLines(colors=[RED], names=["25 ms period"]),
               static=True)
        rr.log("plots/deliveries/count", rr.SeriesLines(colors=[PALETTE[0]], names=["deliveries"]),
               static=True)

    def _drone_model(self, path, color):
        """A quadcopter in its body frame (x forward, z up), from boxes and ellipsoids
        (the primitives the third-person camera view draws): body with a colored cover,
        arms, motors, prop discs, landing skids, a front camera and navigation lights."""
        def yaw_quaternion(angle):
            return [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]

        m = self.motors
        rr.log(f"{path}/frame", rr.Boxes3D(
            centers=[[0, 0, 0]] + list(m / 2),
            half_sizes=[[0.08, 0.055, 0.03]] + [[ARM / 2, 0.012, 0.008]] * 4,
            quaternions=[[0, 0, 0, 1]] + [yaw_quaternion(np.arctan2(y, x)) for x, y, _ in m],
            colors=[FRAME], fill_mode="solid"), static=True)
        rr.log(f"{path}/cover", rr.Ellipsoids3D(
            centers=[[0.01, 0, 0.03]], half_sizes=[[0.075, 0.05, 0.025]], colors=[color],
            fill_mode="solid"), static=True)
        rr.log(f"{path}/motors", rr.Ellipsoids3D(
            centers=m + [0, 0, 0.015], half_sizes=[[0.022, 0.022, 0.025]] * 4,
            colors=[MOTOR], fill_mode="solid"), static=True)
        rr.log(f"{path}/prop_discs", rr.Ellipsoids3D(
            centers=m + [0, 0, 0.05], half_sizes=[[0.075, 0.075, 0.002]] * 4,
            colors=[[*PROP, 45]], fill_mode="solid"), static=True)
        rr.log(f"{path}/skids", rr.Boxes3D(
            centers=[[0, 0.07, -0.09], [0, -0.07, -0.09], [0.05, 0.065, -0.06],
                     [-0.05, 0.065, -0.06], [0.05, -0.065, -0.06], [-0.05, -0.065, -0.06]],
            half_sizes=[[0.11, 0.008, 0.008]] * 2 + [[0.006, 0.006, 0.03]] * 4,
            colors=[FRAME], fill_mode="solid"), static=True)
        rr.log(f"{path}/camera", rr.Ellipsoids3D(
            centers=[[0.085, 0, -0.035], [0.11, 0, -0.035]],
            half_sizes=[[0.025, 0.025, 0.025], [0.006, 0.014, 0.014]],
            colors=[[22, 22, 26], [90, 160, 255]], fill_mode="solid"), static=True)
        # Navigation lights: red on the left (port), green on the right, white at the back.
        rr.log(f"{path}/lights", rr.Ellipsoids3D(
            centers=m[[2, 0, 1, 3]] + [0, 0, -0.012], radii=[0.016] * 4,
            colors=[[255, 30, 30], [30, 255, 60], [255, 255, 255], [255, 255, 255]],
            fill_mode="solid"), static=True)

    def _third_person(self, pose, velocity):
        """Places the virtual camera behind and above the followed drone, facing its
        heading (the direction of flight when it moves), turning smoothly."""
        if np.linalg.norm(velocity[:2]) > 0.3:
            target = np.arctan2(velocity[1], velocity[0])
        else:
            target = np.arctan2(pose[1, 0], pose[0, 0])
        if self.camera_yaw is None:
            self.camera_yaw = target
        self.camera_yaw += CAMERA_TURN * np.angle(np.exp(1j * (target - self.camera_yaw)))
        forward_xy = np.array([np.cos(self.camera_yaw), np.sin(self.camera_yaw), 0.0])
        p = pose[:3, 3]
        eye = p - CAMERA_BACK * forward_xy + [0.0, 0.0, CAMERA_UP]
        forward = p + CAMERA_AHEAD * forward_xy - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        # Pinhole cameras look along +z with x right and y down (RDF).
        rr.log("world/third_person", rr.Transform3D(translation=eye,
                                                    mat3x3=np.stack([right, down, forward], 1)))

    def log(self, i, s):
        rr.set_time("step", sequence=i)
        rr.set_time("time", duration=s.t)
        self._third_person(s.poses[FOLLOW], s.velocities[FOLLOW])
        for b in range(self.num_drones):
            c = PALETTE[b % len(PALETTE)]
            T = s.poses[b]
            rr.log(f"world/drones/{b}", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            # Two-blade propellers, spinning in the rotor directions of the X layout.
            spin = PROP_SPIN * s.t * np.array([-1.0, -1.0, 1.0, 1.0]) + b
            rr.log(f"world/drones/{b}/props", rr.Boxes3D(
                centers=self.motors + [0, 0, 0.052], half_sizes=[[0.075, 0.009, 0.002]] * 4,
                quaternions=[[0, 0, np.sin(x / 2), np.cos(x / 2)] for x in spin],
                colors=[PROP], fill_mode="solid"))
            # Line widths in screen points: thin up close (third person), visible from above.
            rr.log(f"world/plans/{b}", rr.LineStrips3D([s.plans[b]], colors=[c],
                                                       radii=rr.Radius.ui_points(4.0)))
            self.trails[b].append(T[:3, 3])
            trail = np.array(self.trails[b][-int(TRAIL_S / 0.025):])
            rr.log(f"world/trails/{b}", rr.LineStrips3D([trail], colors=[c],
                                                        radii=rr.Radius.ui_points(2.5)))
            if self.routes[b] is not s.routes[b]:  # a new delivery: its route and destination
                self.routes[b] = s.routes[b]
                rr.log(f"world/routes/{b}", rr.LineStrips3D([s.routes[b]], colors=[[*c, 200]],
                                                            radii=rr.Radius.ui_points(2.0)))
                goal = s.goals[b]
                rr.log(f"world/goals/{b}", rr.Cylinders3D(
                    lengths=[0.02], radii=[0.5], centers=[[goal[0], goal[1], 0.02]],
                    colors=[c], fill_mode="solid"))
        # Shoppers: legs, torso and head as ellipsoids (capsules are not drawn in the
        # third-person camera view; ellipsoids are), and where they will be in 1 s.
        n = len(s.shoppers)
        xy = np.repeat(np.asarray(s.shoppers), len(BODY), axis=0)
        parts = np.tile(np.array([c for c, _ in BODY]), (len(s.shoppers), 1))
        rr.log("world/shoppers/bodies", rr.Ellipsoids3D(
            centers=np.column_stack([xy, parts[:, 2]]),
            half_sizes=np.tile(np.array([h for _, h in BODY]), (len(s.shoppers), 1)),
            colors=[SHOPPER], fill_mode="solid"))
        # Their personal space (the body spheres grown by the zone): no drone enters it.
        top = self.person_height + self.person_zone
        rr.log("world/shoppers/zones", rr.Ellipsoids3D(
            centers=[[x, y, top / 2] for x, y in s.shoppers],
            half_sizes=[[0.3 + self.person_zone, 0.3 + self.person_zone, top / 2]] * n,
            colors=[[*SHOPPER, 60]], fill_mode="solid"))
        rr.log("world/shoppers/heading", rr.Arrows3D(
            origins=[[x, y, 0.05] for x, y in s.shoppers],
            vectors=[[vx, vy, 0.0] for vx, vy in s.shopper_velocities],
            colors=[[255, 255, 255]], radii=0.05))

        rr.log("plots/drones/closest", rr.Scalars(min(s.min_drone_distance, 4.0)))
        rr.log("plots/drones/kept", rr.Scalars(self.separation))
        rr.log("plots/shoppers/closest", rr.Scalars(min(s.min_shopper_clearance, 4.0)))
        rr.log("plots/shoppers/kept", rr.Scalars(self.shopper_clearance))
        rr.log("plots/solve/ms", rr.Scalars(s.solve_ms))
        rr.log("plots/solve/budget", rr.Scalars(25.0))
        rr.log("plots/deliveries/count", rr.Scalars(s.deliveries))
        speeds = np.linalg.norm(s.velocities, axis=1)
        rr.log("status", rr.TextDocument(
            f"### t = {s.t:5.1f} s · {s.deliveries} deliveries\n\n"
            f"**{self.num_drones} drones** in one batched MPC problem: "
            f"**{s.solve_ms:.1f} ms** per step · speeds {np.round(speeds, 1)} m/s\n\n"
            f"closest drones **{s.min_drone_distance:.2f} m** · closest shopper "
            f"**{s.min_shopper_clearance:.2f} m** · closest map point "
            f"**{s.min_map_clearance:.2f} m**",
            media_type=rr.MediaType.MARKDOWN))
