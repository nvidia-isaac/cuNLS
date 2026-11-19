# Module: factor/dynamics

> Notes for agents and contributors working in this directory. Shared factor contracts (item
> contract of `Evaluate`, capacity vs. active count, pose convention): `cunls/factor/AGENTS.md`.

## Purpose

Vehicle and body models as multiple-shooting defects between consecutive trajectory steps.

## Files

- `se2_differential_drive_factor_batch` (Carter: SE2 pose, wheel speeds, next pose) and
  `se2_kinematic_bicycle_factor_batch` (car: pose, (v, steer), (a, steer rate), next pose,
  next (v, steer)).
- `se3_differential_drive_factor_batch` / `se3_kinematic_bicycle_factor_batch`: terrain
  versions on SE3 poses; residual rows yaw, forward, lateral, vertical (roll and pitch are left
  to terrain factors).
- `se2_kinematics_factor_batch`, `so3_kinematics_factor_batch`, `se3_kinematics_factor_batch`:
  `X_{k+1} = X_k Exp(dt xi_k)` with the twist `xi_k` a vector state.
- `quadrotor_factor_batch` (SE3 pose, velocity, rates, 4 rotor thrusts; `QuadrotorParameters`)
  and `quadruped_factor_batch` (single rigid body, 4 foot forces, per-step contacts and foot
  positions; `QuadrupedParameters`).

## Conventions

- Euler defects with per-factor time steps, analytic Jacobians; each kernel in its factor's own
  `.cu`, Lie operations from `cunls/math/lie_device.cuh`.
- Not to be confused with `cunls/factor/kinematics/` (articulated robots, joint coordinates).
- Guide: `docs/sphinx/dynamics.rst`; batched MPC on top: `python/pycunls/mpc.py`.
