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

"""Model predictive control on top of pycunls.

A thin builder over the general API: :class:`Horizon` creates the state and
control batches of ``batch`` independent trajectories of ``steps`` steps, the
dynamics factors of a :class:`Model`, cost factors (existing priors,
betweens, weights), constraint wrappers and the subproblem partition (one
subproblem per trajectory). Everything it builds is an ordinary
:class:`pycunls.Problem`; :meth:`Horizon.build` returns a :class:`Controller`
that solves it with :class:`pycunls.AugmentedLagrangianMinimizer` and runs the
receding horizon (:meth:`Controller.step`: shift, set the measured state,
solve, return the first control).

Layout (B = batch, N = steps): poses ``[B, N + 1, 3, 3]`` (SE(2)) or
``[B, N + 1, 4, 4]`` (SE(3)), each vector state ``[B, N + 1, d]``, controls
``[B, N, nu]`` (``wp.array`` buffers owned by the horizon; ``assign()`` to them to
set initial guesses). The first state of every trajectory is constant (the
measured state).

Example::

    from pycunls import mpc

    h = mpc.Horizon(mpc.Carter(wheel_radius=0.125, track_width=0.5), steps=30, dt=0.1, batch=1)
    h.track_pose(reference, weight=1.0)           # reference [1, 31, 3, 3]
    h.control_effort(weight=0.05)
    h.control_bounds(-20.0, 20.0)
    h.disk_obstacles(disks, margin=0.3)           # disks [1, K, 3]: x, y, radius
    ctrl = h.build()
    for t in range(T):
        u0 = ctrl.step(stream, pose=measured_pose)  # [1, 2]

The solver is local: it finds a solution near the initial guess (the
previous plan, after :meth:`Controller.step`'s shift). An obstacle almost
centered on the reference path is close to a saddle (going left or right is
equally good); start such problems from a guess that already passes on one
side, or from the zero-motion default rather than a path through the obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import warp as wp

import pycunls
from pycunls._arrays import to_numpy, warp_stream

__all__ = [
    "Model",
    "Carter",
    "Car",
    "Quadrotor",
    "Quadruped",
    "Horizon",
    "Controller",
]

Weight = Union[float, Sequence[float]]

#: Largest augmented-Lagrangian penalty of Horizon.build's default options. In
#: float32, larger penalties (relative to cost weights of order 1) make exact
#: Gauss-Newton steps unreliable; the multipliers enforce the constraints.
MAX_PENALTY = 1e4

#: Largest augmented-Lagrangian penalty in real-time mode (Horizon.build(real_time=...)).
REAL_TIME_MAX_PENALTY = 1e3

_VECTOR_STATE = {
    1: pycunls.VectorStateBatch1,
    2: pycunls.VectorStateBatch2,
    3: pycunls.VectorStateBatch3,
    4: pycunls.VectorStateBatch4,
    6: pycunls.VectorStateBatch6,
    12: pycunls.VectorStateBatch12,
}
_VECTOR_PRIOR = {
    1: pycunls.PriorVectorFactorBatch1,
    2: pycunls.PriorVectorFactorBatch2,
    3: pycunls.PriorVectorFactorBatch3,
    4: pycunls.PriorVectorFactorBatch4,
    6: pycunls.PriorVectorFactorBatch6,
    12: pycunls.PriorVectorFactorBatch12,
}
_VECTOR_BETWEEN = {
    1: pycunls.VectorBetweenFactorBatch1,
    2: pycunls.VectorBetweenFactorBatch2,
    3: pycunls.VectorBetweenFactorBatch3,
    4: pycunls.VectorBetweenFactorBatch4,
    6: pycunls.VectorBetweenFactorBatch6,
    12: pycunls.VectorBetweenFactorBatch12,
}
_POSE = {
    "SE2": (pycunls.SE2StateBatch, pycunls.SE2PriorFactorBatch, (3, 3), 3),
    "SE3": (pycunls.SE3StateBatch, pycunls.SE3PriorFactorBatch, (4, 4), 6),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Model:
    """State layout of a model and the factory of its dynamics factor batch.

    ``pose`` is ``"SE2"`` or ``"SE3"``; ``vectors`` lists the vector states
    after the pose (name, dimension), in the slot order of the dynamics
    factor; ``control`` is the control dimension.
    """

    pose: str = "SE2"
    vectors: tuple = ()
    control: int = 0

    def dynamics(self, horizon: "Horizon") -> pycunls.FactorBatch:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Carter(Model):
    """Differential drive (NVIDIA Carter): controls are the wheel speeds [rad/s].

    ``terrain=True`` uses an SE(3) pose (slopes, ramps: roll and pitch come
    from other factors).
    """

    wheel_radius: float = 0.125
    track_width: float = 0.5
    terrain: bool = False

    def __post_init__(self):
        self.pose = "SE3" if self.terrain else "SE2"
        self.vectors = ()
        self.control = 2

    def dynamics(self, h):
        cls = (pycunls.SE3DifferentialDriveFactorBatch if self.terrain
               else pycunls.SE2DifferentialDriveFactorBatch)
        return cls(h.time_steps, self.wheel_radius, self.track_width, h.batch * h.steps)


@dataclass
class Car(Model):
    """Kinematic bicycle: vector state ``speed_steer`` = (v, δ), controls (a, δ̇)."""

    wheelbase: float = 2.75
    terrain: bool = False

    def __post_init__(self):
        self.pose = "SE3" if self.terrain else "SE2"
        self.vectors = (("speed_steer", 2),)
        self.control = 2

    def dynamics(self, h):
        cls = (pycunls.SE3KinematicBicycleFactorBatch if self.terrain
               else pycunls.SE2KinematicBicycleFactorBatch)
        return cls(h.time_steps, self.wheelbase, h.batch * h.steps)


@dataclass
class Quadrotor(Model):
    """Quadrotor (X layout): vector states ``velocity`` (world), ``rates`` (body);
    controls the four rotor thrusts [N]."""

    parameters: pycunls.QuadrotorParameters = field(default_factory=pycunls.QuadrotorParameters)

    def __post_init__(self):
        self.pose = "SE3"
        self.vectors = (("velocity", 3), ("rates", 3))
        self.control = 4

    def dynamics(self, h):
        return pycunls.QuadrotorFactorBatch(h.time_steps, self.parameters, h.batch * h.steps)


@dataclass
class Quadruped(Model):
    """Single-rigid-body quadruped: vector states ``velocity``, ``rates``;
    controls the four world-frame foot forces (12). The gait inputs are
    :attr:`Horizon.contacts` ``[B, N, 4]`` and :attr:`Horizon.foot_positions`
    ``[B, N, 4, 3]``."""

    parameters: pycunls.QuadrupedParameters = field(default_factory=pycunls.QuadrupedParameters)

    def __post_init__(self):
        self.pose = "SE3"
        self.vectors = (("velocity", 3), ("rates", 3))
        self.control = 12

    def dynamics(self, h):
        h.contacts = _device(np.ones((h.batch, h.steps, 4), np.float32))
        h.foot_positions = _device(np.zeros((h.batch, h.steps, 4, 3), np.float32))
        return pycunls.QuadrupedFactorBatch(h.time_steps, h.contacts, h.foot_positions,
                                            self.parameters, h.batch * h.steps)


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------


def _identity(shape):
    return np.eye(shape[0], dtype=np.float32)


def _device(a) -> wp.array:
    """Device copy of a host array (float32 or int32 kept)."""
    a = np.ascontiguousarray(a)
    return wp.array(a, dtype=wp.dtype_from_numpy(a.dtype), device=wp.get_cuda_device())


def _float32(value, name: str) -> np.ndarray:
    return to_numpy(value, np.float32, name, "mpc.Horizon")


@wp.kernel
def _shift_steps(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    # dst[:, k] = src[:, k + 1]; the last step repeats.
    b, k, f = wp.tid()
    dst[b, k, f] = src[b, wp.min(k + 1, src.shape[1] - 1), f]


@wp.kernel
def _write_steps(value: wp.array2d(dtype=float), all_steps: int, dst: wp.array3d(dtype=float)):
    # dst[:, 0] = value, or every step when all_steps.
    b, k, f = wp.tid()
    if all_steps or k == 0:
        dst[b, k, f] = value[b, f]


def _shift(a: wp.array) -> None:
    """Shifts a [B, T, ...] buffer one step toward the start, in place (last step repeats)."""
    view = a.reshape((a.shape[0], a.shape[1], -1))
    wp.launch(_shift_steps, dim=view.shape, inputs=[wp.clone(view), view], device=a.device)


def _write(a: wp.array, value, all_steps: bool, name: str) -> None:
    """Writes ``value`` ([B, ...] one step) into step 0 (or every step) of a [B, T, ...] buffer."""
    if not isinstance(value, wp.array):
        value = _device(_float32(value, name))
    view = a.reshape((a.shape[0], a.shape[1], -1))
    value = value.reshape((a.shape[0], view.shape[2]))
    wp.launch(_write_steps, dim=view.shape, inputs=[value, int(all_steps), view],
              device=a.device)


def _weighted(inner, weight: Weight, residual_size: int, capacity: int, keep: list):
    """Scalar weight -> WeightedFactorBatch; per-component weights -> InformationFactorBatch."""
    if np.isscalar(weight):
        return pycunls.WeightedFactorBatch(inner, float(weight))
    w = np.asarray(weight, dtype=np.float32)
    if w.shape != (residual_size,):
        raise ValueError(f"weight needs {residual_size} components, got {w.shape}")
    sqrt_info = _device(np.tile(np.diag(w).reshape(-1), capacity))
    keep.append(sqrt_info)
    return pycunls.InformationFactorBatch(inner, sqrt_info)


class Horizon:
    """``batch`` trajectories of ``steps`` steps of a model (see the module docs).

    Parameters
    ----------
    model : Model
    steps : int
        Number of steps N (N + 1 states, N controls per trajectory).
    dt : float
        Step duration (:attr:`time_steps` holds one per step; ``assign()`` it for
        non-uniform steps).
    batch : int
        Number of independent trajectories B (one subproblem each).
    hard_dynamics : bool
        Dynamics as equality constraints (default) or as soft factors with
        ``dynamics_weight``.
    """

    def __init__(self, model: Model, steps: int, dt: float, batch: int = 1,
                 hard_dynamics: bool = True, dynamics_weight: float = 100.0):
        if steps < 1 or batch < 1:
            raise ValueError("steps and batch must be positive")
        self.model, self.steps, self.batch = model, steps, batch
        B, N = batch, steps
        state_cls, self._pose_prior_cls, self._pose_shape, self._pose_tangent = _POSE[model.pose]
        self.poses = _device(np.broadcast_to(_identity(self._pose_shape),
                                             (B, N + 1) + self._pose_shape))
        self.vectors = {name: _device(np.zeros((B, N + 1, d), np.float32))
                        for name, d in model.vectors}
        self.controls = _device(np.zeros((B, N, model.control), np.float32))
        self.time_steps = _device(np.full(B * N, dt, np.float32))
        self._keep = []  # buffers the bound factors read from

        # State batches: the first state of every trajectory is constant.
        self._const_ids = _device(np.arange(B, dtype=np.int32) * (N + 1))
        self.pose_states = state_cls(self.poses, B * (N + 1), self._const_ids, B)
        self.pose_states.set_num_active_states(B * (N + 1), B)
        self.vector_states = {}
        for name, d in model.vectors:
            sb = _VECTOR_STATE[d](self.vectors[name], B * (N + 1), self._const_ids, B)
            sb.set_num_active_states(B * (N + 1), B)
            self.vector_states[name] = sb
        self.control_states = _VECTOR_STATE[model.control](self.controls, B * N)
        self.control_states.set_num_active_states(B * N)

        self._factors = []      # (factor_batch, slot_batches, indices)
        self._constraints = []  # (constraint batch, factors per trajectory) for shifting
        self.contacts = None
        self.foot_positions = None
        self.control_lower = self.control_upper = None
        self.vector_lower, self.vector_upper = {}, {}

        # Dynamics: slots [pose_k, vectors_k..., u_k, pose_{k+1}, vectors_{k+1}...].
        dyn = model.dynamics(self)
        dyn.set_num_active_factors(B * N)
        k = np.arange(N, dtype=np.int32)[None, :]
        b = np.arange(B, dtype=np.int32)[:, None]
        state_k = (b * (N + 1) + k).reshape(-1)
        control_k = (b * N + k).reshape(-1)
        step_batches = [self.pose_states] + [self.vector_states[n] for n, _ in model.vectors]
        slots = (step_batches + [self.control_states] + step_batches)
        cols = ([state_k] * len(step_batches) + [control_k] + [state_k + 1] * len(step_batches))
        indices = np.stack(cols, axis=1).astype(np.int32).reshape(-1)
        self._keep.append(dyn)
        if hard_dynamics:
            factor = pycunls.ConstraintFactorBatch(dyn, pycunls.ConstraintKind.Equality)
            self._constraints.append((factor, N))
        else:
            factor = pycunls.WeightedFactorBatch(dyn, float(dynamics_weight))
        self._add(factor, slots, indices)
        self.dynamics_factor = dyn

    # --- helpers ---------------------------------------------------------------

    def _add(self, factor, slots, indices):
        indices = _device(np.asarray(indices, np.int32))
        self._keep.append(indices)
        self._factors.append((factor, slots, indices))

    def _step_states(self, include_first=False):
        """State indices of steps 1..N (or 0..N) of every trajectory."""
        B, N = self.batch, self.steps
        k = np.arange(0 if include_first else 1, N + 1, dtype=np.int32)[None, :]
        return (np.arange(B, dtype=np.int32)[:, None] * (N + 1) + k).reshape(-1)

    # --- costs -------------------------------------------------------------------

    def track_pose(self, reference=None, weight: Weight = 1.0):
        """Pose tracking on steps 1..N: prior factors toward :attr:`pose_reference`
        (``[B, N, *pose]``, initialized from ``reference[:, 1:]`` when given; ``assign()``
        to it between solves)."""
        B, N = self.batch, self.steps
        ref = np.broadcast_to(_identity(self._pose_shape), (B, N) + self._pose_shape)
        if reference is not None:
            ref = np.broadcast_to(_float32(reference, "reference")[:, 1:], ref.shape)
        self.pose_reference = _device(ref)
        prior = self._pose_prior_cls(self.pose_reference, B * N)
        prior.set_num_active_factors(B * N)
        self._keep.append(prior)
        factor = _weighted(prior, weight, self._pose_tangent, B * N, self._keep)
        self._add(factor, [self.pose_states], self._step_states())
        return self

    def track_vector(self, name: str, reference=None, weight: Weight = 1.0):
        """Tracking of vector state ``name`` on steps 1..N toward
        :attr:`vector_reference[name]` (``[B, N, d]``)."""
        B, N = self.batch, self.steps
        d = self.vectors[name].shape[-1]
        if not hasattr(self, "vector_reference"):
            self.vector_reference = {}
        ref = np.zeros((B, N, d), np.float32)
        if reference is not None:
            ref = np.broadcast_to(_float32(reference, "reference")[:, 1:], ref.shape)
        ref = self.vector_reference[name] = _device(ref)
        prior = _VECTOR_PRIOR[d](ref, B * N)
        prior.set_num_active_factors(B * N)
        self._keep.append(prior)
        factor = _weighted(prior, weight, d, B * N, self._keep)
        self._add(factor, [self.vector_states[name]], self._step_states())
        return self

    def control_effort(self, weight: Weight = 1.0, nominal=None):
        """Controls pulled toward ``nominal`` (default 0)."""
        B, N, nu = self.batch, self.steps, self.model.control
        values = np.zeros((B, N, nu), np.float32)
        if nominal is not None:
            values = np.broadcast_to(_float32(nominal, "nominal"), values.shape)
        self.control_nominal = _device(values)
        prior = _VECTOR_PRIOR[nu](self.control_nominal, B * N)
        prior.set_num_active_factors(B * N)
        self._keep.append(prior)
        factor = _weighted(prior, weight, nu, B * N, self._keep)
        self._add(factor, [self.control_states], np.arange(B * N, dtype=np.int32))
        return self

    def control_rate(self, weight: Weight = 1.0):
        """Smoothness: consecutive controls pulled together."""
        B, N, nu = self.batch, self.steps, self.model.control
        if N < 2:
            return self
        n = B * (N - 1)
        zero = _device(np.zeros((n, nu), np.float32))
        between = _VECTOR_BETWEEN[nu](zero, n)
        between.set_num_active_factors(n)
        self._keep += [zero, between]
        factor = _weighted(between, weight, nu, n, self._keep)
        k = np.arange(N - 1, dtype=np.int32)[None, :]
        first = (np.arange(B, dtype=np.int32)[:, None] * N + k).reshape(-1)
        self._add(factor, [self.control_states, self.control_states],
                  np.stack([first, first + 1], axis=1).reshape(-1))
        return self

    # --- constraints ---------------------------------------------------------------

    def control_bounds(self, lower, upper):
        """lower <= u <= upper (scalars or per-component sequences; ±inf: none).

        Enforced by projection (:meth:`pycunls.VectorStateBatch2.set_bounds`):
        the controls never leave the box, at any iteration. The bounds live in
        :attr:`control_lower` / :attr:`control_upper` (``[B, N, nu]``) and may
        be reassigned between solves.
        """
        B, N, nu = self.batch, self.steps, self.model.control
        self.control_lower = self._bound_buffer(lower, (B, N, nu))
        self.control_upper = self._bound_buffer(upper, (B, N, nu))
        self.control_states.set_bounds(self.control_lower, self.control_upper)
        return self

    def vector_bounds(self, name: str, lower, upper):
        """lower <= x <= upper on vector state ``name``, by projection (see
        :meth:`control_bounds`); the bounds live in ``vector_lower[name]`` /
        ``vector_upper[name]`` (``[B, N + 1, d]``). The measured first state is
        constant and never projected."""
        B, N = self.batch, self.steps
        d = self.vectors[name].shape[-1]
        self.vector_lower[name] = self._bound_buffer(lower, (B, N + 1, d))
        self.vector_upper[name] = self._bound_buffer(upper, (B, N + 1, d))
        self.vector_states[name].set_bounds(self.vector_lower[name], self.vector_upper[name])
        return self

    @staticmethod
    def _bound_buffer(value, shape):
        return _device(np.broadcast_to(_float32(value, "bounds"), shape))

    def goal_pose(self, goal):
        """The last pose of every trajectory equals ``goal`` (``[B, *pose]``) exactly."""
        B, N = self.batch, self.steps
        self.goal = _device(np.broadcast_to(_float32(goal, "goal"), (B,) + self._pose_shape))
        prior = self._pose_prior_cls(self.goal, B)
        prior.set_num_active_factors(B)
        self._keep.append(prior)
        factor = pycunls.ConstraintFactorBatch(prior, pycunls.ConstraintKind.Equality)
        self._constraints.append((factor, None))
        self._add(factor, [self.pose_states], np.arange(B, dtype=np.int32) * (N + 1) + N)
        return self

    def disk_obstacles(self, obstacles, margin: float = 0.0):
        """Keep the pose origin ``margin`` away from every obstacle on steps 1..N.

        ``obstacles``: ``[B, K, 3]`` (x, y, radius) for SE(2) models or
        ``[B, K, 4]`` (x, y, z, radius) for SE(3) models. :attr:`obstacles` holds
        them per step (``[B, N, K, c]``) and may be reassigned between solves.
        """
        B, N = self.batch, self.steps
        obs = _float32(obstacles, "obstacles")
        K, c = obs.shape[1], obs.shape[2]
        if (self.model.pose == "SE2") != (c == 3):
            raise ValueError("SE(2) models take [B, K, 3] disks, SE(3) models [B, K, 4] spheres")
        self.obstacles = _device(np.broadcast_to(obs[:, None], (B, N, K, c)))
        cls = (pycunls.SE2DiskClearanceFactorBatch if c == 3 else pycunls.SE3SphereClearanceFactorBatch)
        clearance = cls(self.obstacles, float(margin), B * N * K)
        clearance.set_num_active_factors(B * N * K)
        self._keep.append(clearance)
        factor = pycunls.ConstraintFactorBatch(clearance, pycunls.ConstraintKind.Inequality)
        self._constraints.append((factor, N))
        idx = np.repeat(self._step_states(), K)  # factor (b, k, j) -> pose of step k
        self._add(factor, [self.pose_states], idx)
        return self

    # --- build -------------------------------------------------------------------------

    def build(self, minimizer=None,
              options: Optional[pycunls.AugmentedLagrangianMinimizerOptions] = None,
              real_time: Optional[Tuple[int, int]] = None):
        """Assemble the problem and return a :class:`Controller`.

        ``minimizer`` defaults to Levenberg-Marquardt with the block-tridiagonal
        linear solver (the problem declares its stages) and a state tolerance of
        1e-5 (pass your own minimizer to change either).

        ``real_time=(outer, inner)``: after the first solve (which converges),
        every solve runs a fixed budget of ``outer`` augmented-Lagrangian
        iterations of ``inner`` minimizer iterations each (two line-search
        halvings per iteration), without host synchronization but one read-back
        (:attr:`pycunls.AugmentedLagrangianMinimizerOptions.real_time`), with
        the penalty capped at :data:`REAL_TIME_MAX_PENALTY`. The warm start
        carries the solution and the multipliers forward, so the iterations of
        consecutive steps add up, as in real-time iteration schemes.
        """
        problem = pycunls.Problem()
        problem.add_state_batch(self.pose_states)
        for name, _ in self.model.vectors:
            problem.add_state_batch(self.vector_states[name])
        problem.add_state_batch(self.control_states)
        for factor, slots, indices in self._factors:
            problem.add_factor_batch(factor, slots, indices)
        B, N = self.batch, self.steps
        ids = [_device(np.arange(B * (N + 1), dtype=np.int32) // (N + 1))]
        ids += [ids[0]] * len(self.model.vectors)
        ids.append(_device(np.arange(B * N, dtype=np.int32) // N))
        self._keep += ids
        problem.set_problem_partition(B, ids)
        # Stages for the block-tridiagonal solver: x_k and u_k in stage k.
        stages = [_device(np.tile(np.arange(N + 1, dtype=np.int32), B))] * (
            1 + len(self.model.vectors))
        stages.append(_device(np.tile(np.arange(N, dtype=np.int32), B)))
        self._keep += stages
        problem.set_state_stages(stages)
        default_minimizer = minimizer is None
        if minimizer is None:
            mo = pycunls.MinimizerOptions()
            mo.sparse_linear_solver_type = pycunls.SparseLinearSolverType.BlockTridiagonal
            mo.max_num_iterations = 50
            # The default 1e-6 on the squared step norm (summed over every
            # unknown) is beyond what float32 iterative solves reach on a
            # horizon of thousands of unknowns. 1e-5 keeps a warm-started
            # solve at a few iterations; 1e-4 let Levenberg-Marquardt stop
            # short (quadrotors drifting off their hover points).
            mo.state_tolerance = 1e-5
            lm = pycunls.LevenbergMarquardtMinimizerOptions()
            lm.base_options = mo
            minimizer = pycunls.LevenbergMarquardtMinimizer(lm)
        if options is None:
            options = pycunls.AugmentedLagrangianMinimizerOptions()
            options.warm_start = True
            options.reuse_structure = True  # the horizon's structure never changes
            options.max_penalty = MAX_PENALTY
            if real_time is not None and default_minimizer:
                # Levenberg-Marquardt's damping keeps the projected steps
                # descending without the extra refinement solves (each a full
                # linear solve per iteration in real time).
                options.max_bound_refinements = 0
        return Controller(self, problem, minimizer, options, real_time)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class Controller:
    """Solves a :class:`Horizon`'s problem and runs the receding horizon."""

    def __init__(self, horizon: Horizon, problem, minimizer, options, real_time=None):
        self.horizon, self.problem, self.minimizer = horizon, problem, minimizer
        self.solver = pycunls.AugmentedLagrangianMinimizer(minimizer, options)
        self.summary = None
        self._started = False
        self._real_time = real_time

    def solve(self, stream):
        """Solve the current problem (warm-started from the current buffers).

        With ``real_time`` (see :meth:`Horizon.build`), the first call converges
        and switches the solver to the fixed budget for the following calls.
        """
        # Buffer writes (assign, Horizon setup) run on Warp's stream: the solve waits for them.
        warp_stream(stream).wait_stream(wp.get_stream(self.horizon.controls.device))
        self.summary = self.solver.minimize(stream, self.problem)
        if self._real_time is not None and not self.solver.options.real_time:
            outer, inner = self._real_time
            o = self.solver.options
            o.real_time = True
            o.max_outer_iterations = int(outer)
            o.inner_iterations = int(inner)
            o.final_inner_iterations = 0
            o.inner_line_search_steps = 2
            # A few iterations per step cannot absorb large penalties: the
            # penalty term swamps the costs and the closed loop drifts
            # (quadrotors). The multipliers carry the constraints across steps.
            o.max_penalty = min(o.max_penalty, REAL_TIME_MAX_PENALTY)
            o.initial_penalty = min(o.initial_penalty, o.max_penalty)
            o.warm_start = True
            o.reuse_structure = True
            self.solver.options = o
        return self.summary

    def shift(self):
        """Receding-horizon warm start: states, controls and the constraint
        multipliers and penalties advance one step; the last step repeats."""
        h = self.horizon
        for buffer in [h.poses, h.controls, *h.vectors.values()]:
            _shift(buffer)
        for batch, per_trajectory in h._constraints:
            if per_trajectory is None or per_trajectory < 2:
                continue
            n = batch.capacity
            rows = batch.residuals_size
            for ptr, width in ((batch.multipliers_ptr, rows), (batch.penalties_ptr, 1)):
                _shift(wp.array(ptr=ptr, dtype=float, shape=(h.batch, per_trajectory, n *
                                width // (h.batch * per_trajectory)), device=h.poses.device,
                                copy=False))

    def set_initial_state(self, pose=None, **vectors):
        """Write the measured state into the constant first state of every trajectory.

        The first call (cold start) also fills the rest of the horizon with it:
        the initial guess is "stay where you are" rather than the default
        identity poses and zero vectors.
        """
        h = self.horizon
        if pose is not None:
            _write(h.poses, pose, not self._started, "pose")
        for name, value in vectors.items():
            _write(h.vectors[name], value, not self._started, name)
        self._started = True

    def step(self, stream, pose=None, **vectors) -> wp.array:
        """Shift, set the measured state, solve; returns the first controls ``[B, nu]``.

        The shift and the state writes run on Warp's current stream, the solve on
        ``stream``; stream waits order them (the solve after the writes, the
        returned copy after the solve), also when the two streams differ.
        """
        current = wp.get_stream(self.horizon.controls.device)
        solver = warp_stream(stream)
        if self._started:
            self.shift()
        self.set_initial_state(pose, **vectors)
        solver.wait_stream(current)
        self.solve(stream)
        current.wait_stream(solver)
        return wp.clone(self.horizon.controls[:, 0])
