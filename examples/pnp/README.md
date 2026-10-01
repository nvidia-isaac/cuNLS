# PnP Example

This example shows how to use the shipped `PnPFactorBatch`
(`cunls/factor/pnp_factor_batch.h`) to solve a Perspective-n-Point problem:
recover a single camera pose from known 3D world points and their noisy 2D
normalized observations, with the 3D structure held fixed (unlike
`sparse_bundle_adjustment`, which also optimizes the points).

Residual per correspondence (see `PnPFactorBatch`'s doc comment):

```
P_cam = T_cam_from_world * P_world[i]
r_i = [P_cam.x/P_cam.z - obs_i.x, P_cam.y/P_cam.z - obs_i.y]
```

Jacobian: `2x6`, pose tangent only (no point derivatives, since the points
are fixed).

## Jacobian modes

The example demonstrates **both** `JacobianMode::kAnalytic` (the factor's
own hand-derived 2x6 Jacobian) and `JacobianMode::kNumeric` (finite
differences on the SE(3) tangent space, via `NumericDiffJacobianBuilder`) on
the same synthetic problem, and prints the final cost / pose error for each
so they can be compared directly.

```bash
./pnp_example                                  # runs both modes, default 2000 points
./pnp_example --num-points 20000               # scale up the correspondence count
./pnp_example --jacobian-mode analytic         # only the analytic path
./pnp_example --jacobian-mode numeric          # only the numeric-diff path
```

## Files

- `main.cpp`: the cuNLS workflow; `SolvePnP` builds and solves the problem
  once per Jacobian mode.
- `../utils/`: header-only helpers kept out of `main` so it reads as the cuNLS
  workflow: `datasets.h` (synthetic scenes), `validation.h` (error metrics),
  `report.h` (printing and the quality verdict), `cli.h` (command-line flags),
  `se3_utils.h` / `camera_utils.h` (host SE(3) math and projection).
- Built by the shared `examples/CMakeLists.txt`.

## Walkthrough

1. `examples::MakePnPScene` generates a ground-truth camera pose
   (`T_cam_from_world`), `--num-points` random 3D world points visible from
   it, their noisy normalized 2D observations, and a perturbed initial pose.
2. Build a single `SE3StateBatch` (one pose) and a `PnPFactorBatch` with one
   factor per correspondence, all referencing the same pose state block.
3. Solve with `LevenbergMarquardtMinimizer`, once per requested Jacobian
   mode, each on a fresh device copy of the perturbed pose so the two runs
   are directly comparable.
4. Compare initial vs. final cost and pose MSE for each mode.

Each batch is constructed with its **capacity** (how many blocks /
correspondences its bound device buffers hold, fixed for the batch's
lifetime) and starts with 0 active; `SetNumStateBlocks` / `SetNumFactors`
set the **active count** the next solve uses (host-only: no allocation, no
device work; a solve without it throws). Size the capacity once for the
largest problem you expect; the active count may change between solves up to
it, so a real-time application allocates once and reuses the same buffers
every frame while the problem size changes. The example keeps the two in
separate variables (`*_capacity` vs. `num_*`); it solves every slot once, so
each active count equals its capacity.

## Build locally (all examples)

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

Output binary: `pnp_example`.
