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

"""A fleet of delivery quadrotors in a supermarket full of shoppers (pycunls.mpc).

The store is the TartanGround ``Supermarket`` environment, fused from the
depth images of its ten trajectories into a 3D map (``example_utils/
tartan_map.py``). Shoppers walk the paths the dataset's robot drove. A fleet
of quadrotors flies deliveries between stations in the store at shoulder
height, along routes from a grid planner (A* through the aisles).

All drones are one batched MPC problem (``mpc.Horizon(batch=B)``): one
trajectory of 40 steps of 25 ms per drone, solved together on the GPU every
control step, each drone its own subproblem. Each drone keeps clear of, on
every step of its horizon,

* the store: the occupied voxels of the map nearest to its previous plan at
  that step (shelves, pillars, the floor, the ceiling);
* the shoppers: three spheres each (legs, torso, head) grown by a personal
  space of PERSON_ZONE, predicted at constant velocity;
* the other drones: their previous plans, shifted one step (each drone
  solves against the others' latest intentions, as in distributed MPC),
  with a separation zone of DRONE_ZONE.

The obstacles are rewritten in ``Horizon.obstacles`` every step; the
problem's structure never changes.

Usage::

    python supermarket_drones.py --env dataset/tartan_ground/Supermarket \\
        [--drones 6] [--rrd supermarket_drones.rrd | --spawn]

The first run fuses the map (about a minute) and caches it next to the
environment (``--map`` to choose the file). Needs scipy; ``--rrd`` /
``--spawn`` need rerun-sdk.
"""

import argparse
import heapq
import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

import pycunls
from pycunls import mpc
from example_utils import tartan_map

STEPS, DT = 40, 0.025  # 1 s horizon at 40 Hz
F_MAX = 8.0  # N per rotor
DRONE_RADIUS = 0.25  # m
MARGIN = 0.15  # m
CRUISE = 1.5  # m/s
ALTITUDE = 1.5  # m above the floor: shoulder height, among the shoppers
VOXEL = 0.2  # m, occupancy grid
K_STATIC = 24  # map voxels per drone and step
PERSON_SPHERES = ((0.45, 0.28), (1.0, 0.3), (1.55, 0.2))  # (height, radius): legs, torso, head
# Keep-out zones on top of the body sizes and MARGIN: the drones' personal space around a
# shopper and around each other (the map keeps only DRONE_RADIUS + MARGIN).
PERSON_ZONE = 0.6  # m
DRONE_ZONE = 0.5  # m
SHOPPER_CLEARANCE = DRONE_RADIUS + MARGIN + PERSON_ZONE  # drone center to a body sphere
DRONE_SEPARATION = 2 * DRONE_RADIUS + MARGIN + DRONE_ZONE  # between drone centers
WALK_SPEED = 1.2  # m/s
DRAG = 0.15  # 1/s: aerodynamic drag of the simulated drones, unknown to the model


# --------------------------------------------------------------------------- the drones
def plant(params, T, v, w, f, dt, substeps=10):
    """A 'real' quadrotor: fine Euler steps of the rigid body (exact rotation steps),
    with the aerodynamic drag the MPC model does not know."""
    a = params.arm_length / np.sqrt(2)
    tau = np.array([a * (-f[0] + f[1] + f[2] - f[3]), a * (-f[0] + f[1] - f[2] + f[3]),
                    params.torque_coefficient * (-f[0] - f[1] + f[2] + f[3])])
    J = np.array(params.inertia, dtype=np.float64)
    h = dt / substeps
    for _ in range(substeps):
        R = T[:3, :3]
        acc = R[:, 2] * f.sum() / params.mass - DRAG * v - [0, 0, params.gravity]
        wdot = (tau - np.cross(w, J * w)) / J
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) * h
        th = np.linalg.norm(w) * h
        dR = np.eye(3) + (np.sin(th) / th if th > 1e-12 else 1.0) * K + \
            ((1 - np.cos(th)) / th**2 if th > 1e-12 else 0.5) * K @ K
        T = T.copy()
        T[:3, 3] += h * v
        T[:3, :3] = R @ dR
        v = v + h * acc
        w = w + h * wdot
    return T, v, w


# --------------------------------------------------------------------------- the store
class Store:
    """The fused map: points for display, occupied voxels for the constraints, a 2D
    grid for the route planner."""

    def __init__(self, points, colors, trajectories):
        low = points[points[:, 2] < np.median(points[:, 2]), 2]
        h, edges = np.histogram(low, bins=200)
        self.floor = edges[np.argmax(h)]  # the densest level of the lower half
        shift = np.array([0, 0, self.floor], np.float32)
        self.points, self.colors = points - shift, colors
        self.trajectories = [t - shift for t in trajectories]
        lo, hi = np.array([-20.0, -19.0, -0.1]), np.array([20.0, 14.0, 4.5])
        inside = np.all((self.points > lo) & (self.points < hi), axis=1)
        self.points, self.colors = self.points[inside], self.colors[inside]
        # Occupied voxels: at least two map points (single points are noise).
        keys, counts = np.unique(np.floor((self.points - lo) / VOXEL).astype(np.int32), axis=0,
                                 return_counts=True)
        occupied = keys[counts >= 2]
        self.voxels = (occupied + 0.5) * VOXEL + lo
        self.voxel_radius = VOXEL * np.sqrt(3) / 2  # a sphere covering the voxel
        self.tree = cKDTree(self.voxels)
        # Planner grid: a cell is blocked if anything occupies it near the flight altitude,
        # inflated by the drone's clearance.
        self.lo, shape = lo[:2], np.ceil((hi[:2] - lo[:2]) / VOXEL).astype(int)
        band = np.abs(self.voxels[:, 2] - ALTITUDE) < 0.6
        blocked = np.zeros(shape, bool)
        blocked[tuple(occupied[band, :2].T)] = True
        r = int(np.ceil((DRONE_RADIUS + MARGIN + 0.3) / VOXEL))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        self.blocked = binary_dilation(blocked, disk)
        free = np.argwhere(~self.blocked)
        self.free_tree, self.free_cells = cKDTree(self.center(free)), free

    def snap(self, xy):
        """The nearest free planner cell's center."""
        return self.center(self.free_cells[self.free_tree.query(xy)[1]])

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy) - self.lo) / VOXEL).astype(int))

    def center(self, cell):
        return (np.asarray(cell) + 0.5) * VOXEL + self.lo

    def route(self, start, goal):
        """A* on the planner grid (8-connected), shortcut by line of sight; xy waypoints."""
        s, g = self.cell(start), self.cell(goal)
        moves = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if dx or dy]
        came, cost, frontier = {s: None}, {s: 0.0}, [(0.0, s)]
        while frontier:
            _, c = heapq.heappop(frontier)
            if c == g:
                break
            for dx, dy, w in moves:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < self.blocked.shape[0] and 0 <= n[1] < self.blocked.shape[1]):
                    continue
                if self.blocked[n] and n != g:
                    continue
                new = cost[c] + w
                if new < cost.get(n, np.inf):
                    cost[n], came[n] = new, c
                    heapq.heappush(frontier, (new + np.hypot(n[0] - g[0], n[1] - g[1]), n))
        if g not in came:
            return None
        path, c = [], g
        while c is not None:
            path.append(c)
            c = came[c]
        path = path[::-1]
        keep, i = [path[0]], 0  # shortcut: jump to the farthest cell still in sight
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._visible(path[i], path[j]):
                j -= 1
            keep.append(path[j])
            i = j
        return np.array([self.center(c) for c in keep])

    def _visible(self, a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        xs = np.round(np.linspace(a[0], b[0], n)).astype(int)
        ys = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return not self.blocked[xs, ys].any()

    def free(self, xy):
        c = self.cell(xy)
        return 0 <= c[0] < self.blocked.shape[0] and 0 <= c[1] < self.blocked.shape[1] and \
            not self.blocked[c]


# --------------------------------------------------------------------------- shoppers
class Shopper:
    """Walks one of the dataset's robot paths at walking speed, back and forth."""

    def __init__(self, path, speed, offset):
        seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        self.path = path[:, :2]
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.speed, self.offset = speed, offset

    def position(self, t):
        length = self.s[-1]
        u = (self.offset + self.speed * t) % (2 * length)
        u = u if u < length else 2 * length - u  # back and forth
        return np.array([np.interp(u, self.s, self.path[:, 0]),
                         np.interp(u, self.s, self.path[:, 1])])

    def velocity(self, t):
        return (self.position(t + 0.05) - self.position(t - 0.05)) / 0.1


def make_shoppers(store, count, rng):
    paths = [t for t in store.trajectories if len(t) > 200]
    return [Shopper(paths[i % len(paths)], WALK_SPEED * rng.uniform(0.8, 1.15),
                    rng.uniform(0, 60)) for i in range(count)]


def person_spheres(xy, zone=0.0):
    """[3, 4] spheres (x, y, z, r) of a shopper standing at xy, radii grown by zone."""
    return np.array([[xy[0], xy[1], z, r + zone] for z, r in PERSON_SPHERES])


# --------------------------------------------------------------------------- deliveries
STATIONS = np.array([  # xy: aisle ends, the bakery, freezers, checkout
    [-16.0, 9.0], [-16.0, -2.0], [-16.0, -14.0], [-8.0, 2.5], [0.0, 2.5], [8.0, 2.5],
    [15.0, 9.0], [15.0, -2.0], [12.0, -16.0], [-7.0, -9.0], [3.0, -8.5], [-6.0, 11.5],
    [6.0, 11.5],
])


class Courier:
    """One drone's deliveries: a route to the next station, followed by a carrot that
    moves along it at cruise speed (and waits when the drone falls behind)."""

    def __init__(self, store, stations, start, rng):
        self.store, self.stations, self.rng = store, stations, rng
        self.station = int(np.argmin(np.linalg.norm(stations - start[:2], axis=1)))
        self.deliveries = 0
        self.plan_route(start)

    def plan_route(self, position):
        while True:
            nxt = int(self.rng.integers(len(self.stations)))
            if nxt == self.station:
                continue
            route = self.store.route(position[:2], self.stations[nxt])
            if route is not None:
                break
        self.station = nxt
        self.route = np.column_stack([route, np.full(len(route), ALTITUDE)])
        self.route[0] = position
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        self.s = 0.0

    def point(self, s):
        s = np.clip(s, 0, self.cumulative[-1])
        i = min(np.searchsorted(self.cumulative, s, side="right") - 1, len(self.route) - 2)
        u = (s - self.cumulative[i]) / max(self.cumulative[i + 1] - self.cumulative[i], 1e-9)
        d = self.route[i + 1] - self.route[i]
        return self.route[i] + u * d, np.arctan2(d[1], d[0])

    def update(self, position):
        """Advance the carrot; at the station, count the delivery and plan the next one."""
        if self.s >= self.cumulative[-1] and np.linalg.norm(position - self.route[-1]) < 0.3:
            self.deliveries += 1
            self.plan_route(position)
        carrot, _ = self.point(self.s)
        if np.linalg.norm(carrot - position) < 0.8 * CRUISE * STEPS * DT:
            self.s = min(self.s + CRUISE * DT, self.cumulative[-1])

    def reference(self):
        """Poses [N, 4, 4] and velocities [N, 3] of steps 1..N."""
        poses, vels = np.tile(np.eye(4), (STEPS, 1, 1)), np.zeros((STEPS, 3))
        yaw0 = self.point(self.s)[1]
        for k in range(STEPS):
            s = self.s + CRUISE * DT * (k + 1)
            p, yaw = self.point(s)
            yaw = yaw0 + np.angle(np.exp(1j * (yaw - yaw0)))  # no 2 pi jumps
            poses[k, :3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]]
            poses[k, :3, 3] = p
            if s < self.cumulative[-1]:
                vels[k] = CRUISE * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return poses, vels


# --------------------------------------------------------------------------- the fleet
def build_controller(num_drones, num_shoppers, params):
    """One horizon for the whole fleet; obstacles per drone and step:
    K_STATIC voxels, 3 spheres per shopper, one sphere per other drone."""
    B = num_drones
    h = mpc.Horizon(mpc.Quadrotor(parameters=params), steps=STEPS, dt=DT, batch=B)
    hover = params.mass * params.gravity / 4
    h.track_pose(weight=[0.3, 0.3, 0.4, 1.5, 1.5, 2.5])
    h.track_vector("velocity", weight=0.4)
    h.track_vector("rates", weight=0.05)
    h.controls[...] = hover
    h.control_effort(weight=0.05, nominal=np.full((B, STEPS, 4), hover))
    h.control_rate(weight=0.1)
    h.control_bounds(0.0, F_MAX)
    K = K_STATIC + 3 * num_shoppers + (B - 1)
    far = np.tile([1e3, 1e3, 1e3, 0.1], (B, K, 1)).astype(np.float32)  # placeholders
    h.disk_obstacles(far, margin=DRONE_RADIUS + MARGIN)
    return h, h.build(), K


@dataclass
class FleetStep:
    """The fleet at one control step (for the report and the visualization)."""
    t: float
    poses: np.ndarray  # [B, 4, 4]
    velocities: np.ndarray  # [B, 3]
    thrusts: np.ndarray  # [B, 4]
    plans: np.ndarray  # [B, N + 1, 3]
    routes: list  # per drone [n, 3]
    goals: np.ndarray  # [B, 3]
    shoppers: np.ndarray  # [P, 2]
    shopper_velocities: np.ndarray  # [P, 2]
    min_drone_distance: float  # between drone centers
    min_shopper_clearance: float  # drone center to a shopper sphere surface
    min_map_clearance: float  # drone center to the nearest map point
    solve_ms: float
    deliveries: int


def fly(store, num_drones=6, num_shoppers=10, duration=45.0, seed=3, on_step=None):
    rng = np.random.default_rng(seed)
    params = pycunls.QuadrotorParameters()
    params.linear_drag = 0.0
    h, ctrl, K = build_controller(num_drones, num_shoppers, params)
    shoppers = make_shoppers(store, num_shoppers, rng)
    points_tree = cKDTree(store.points[store.points[:, 2] > 0.05])  # the map, without the floor
    # Drones take off from free spots near the stations.
    B = num_drones
    stations = np.array([store.snap(s) for s in STATIONS])
    starts = [np.array([*stations[(3 * b) % len(stations)], ALTITUDE]) for b in range(B)]
    couriers = [Courier(store, stations, starts[b], rng) for b in range(B)]
    T = np.tile(np.eye(4), (B, 1, 1))
    T[:, :3, 3] = starts
    v, w = np.zeros((B, 3)), np.zeros((B, 3))
    stream = pycunls.CudaStream()
    history = []
    for i in range(int(duration / DT)):
        t = i * DT
        refs = [c.reference() for c in couriers]
        for b, c in enumerate(couriers):
            c.update(T[b, :3, 3])
        h.pose_reference[...] = cp.asarray(np.stack([r[0] for r in refs]), dtype=cp.float32)
        h.vector_reference["velocity"][...] = cp.asarray(np.stack([r[1] for r in refs]),
                                                         dtype=cp.float32)

        # Obstacles of steps 1..N, from the previous plans (shifted one step).
        plans = cp.asnumpy(h.poses[:, :, :3, 3]).astype(np.float64)
        if i == 0:
            plans[:] = T[:, None, :3, 3]
        ahead = np.concatenate([plans[:, 2:], plans[:, -1:]], axis=1)  # [B, N, 3]
        obstacles = np.zeros((B, STEPS, K, 4))
        _, nearest = store.tree.query(ahead.reshape(-1, 3), k=K_STATIC)
        obstacles[:, :, :K_STATIC, :3] = store.voxels[nearest].reshape(B, STEPS, K_STATIC, 3)
        obstacles[:, :, :K_STATIC, 3] = store.voxel_radius
        times = t + DT * np.arange(1, STEPS + 1)
        for j, s in enumerate(shoppers):  # constant-velocity prediction
            xy = s.position(t)[None] + s.velocity(t)[None] * (times - t)[:, None]
            spheres = np.stack([person_spheres(p, PERSON_ZONE) for p in xy])  # [N, 3, 4]
            obstacles[:, :, K_STATIC + 3 * j:K_STATIC + 3 * j + 3] = spheres[None]
        for b in range(B):
            others = [o for o in range(B) if o != b]
            obstacles[b, :, K_STATIC + 3 * len(shoppers):, :3] = ahead[others].transpose(1, 0, 2)
            obstacles[b, :, K_STATIC + 3 * len(shoppers):, 3] = DRONE_RADIUS + DRONE_ZONE
        h.obstacles[...] = cp.asarray(obstacles, dtype=cp.float32)

        start = time.perf_counter()
        u = cp.asnumpy(ctrl.step(stream, pose=T.astype(np.float32),
                                 velocity=v.astype(np.float32),
                                 rates=w.astype(np.float32))).astype(np.float64)
        solve_ms = 1e3 * (time.perf_counter() - start)

        p = T[:, :3, 3]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(B) * 1e9
        people = np.concatenate([person_spheres(s.position(t)) for s in shoppers])
        shopper_clear = np.min(np.linalg.norm(p[:, None] - people[None, :, :3], axis=-1) -
                               people[None, :, 3])
        step = FleetStep(t, T.copy(), v.copy(), u, cp.asnumpy(h.poses[:, :, :3, 3]),
                         [c.route for c in couriers], np.stack([c.route[-1] for c in couriers]),
                         np.stack([s.position(t) for s in shoppers]),
                         np.stack([s.velocity(t) for s in shoppers]), float(d.min()),
                         float(shopper_clear), float(points_tree.query(p)[0].min()), solve_ms,
                         sum(c.deliveries for c in couriers))
        history.append((step.min_drone_distance, step.min_shopper_clearance,
                        step.min_map_clearance, solve_ms))
        if on_step:
            on_step(i, step)
        if i % 200 == 0:
            print(f"  t={t:5.1f}s  solve {solve_ms:5.1f} ms  drones >= "
                  f"{step.min_drone_distance:.2f} m apart, shoppers >= "
                  f"{step.min_shopper_clearance:.2f} m, map >= "
                  f"{step.min_map_clearance:.2f} m, {step.deliveries} deliveries")
        for b in range(B):
            T[b], v[b], w[b] = plant(params, T[b], v[b], w[b], u[b], DT)
    return step, np.array(history)


def print_report(step, history, num_drones, num_shoppers):
    drone, shopper, store, ms = history.T
    print(f"Supermarket fleet: {num_drones} drones, {num_shoppers} shoppers, "
          f"{len(history) * DT:.0f} s, {step.deliveries} deliveries")
    print(f"  closest drones      {drone.min():.2f} m between centers "
          f"(kept >= {DRONE_SEPARATION:.2f})")
    print(f"  closest shopper     {shopper.min():.2f} m from a body sphere "
          f"(kept >= {SHOPPER_CLEARANCE:.2f})")
    print(f"  closest map point   {store.min():.2f} m (drone radius {DRONE_RADIUS})")
    print(f"  fleet solve time    median {np.median(ms[1:]):.1f} ms, "
          f"p95 {np.percentile(ms[1:], 95):.1f} ms (all drones, one batched problem)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="dataset/tartan_ground/Supermarket")
    parser.add_argument("--map", help="fused map cache (default: next to the environment)")
    parser.add_argument("--drones", type=int, default=6)
    parser.add_argument("--shoppers", type=int, default=10)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--rrd", help="save a Rerun recording")
    parser.add_argument("--spawn", action="store_true", help="stream to a Rerun viewer")
    args = parser.parse_args()
    cache = args.map or os.path.join(args.env, "fused_map.npz")
    store = Store(*tartan_map.build_map(args.env, cache))
    on_step = None
    if args.rrd or args.spawn:
        from example_utils import supermarket_drones_rerun
        on_step = supermarket_drones_rerun.Logger(store, args.drones, PERSON_SPHERES,
                                                  PERSON_ZONE, SHOPPER_CLEARANCE,
                                                  DRONE_SEPARATION, args.rrd,
                                                  args.spawn).log
    step, history = fly(store, args.drones, args.shoppers, args.duration, on_step=on_step)
    print_report(step, history, args.drones, args.shoppers)


if __name__ == "__main__":
    main()
