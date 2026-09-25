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

- `main.cpp`: synthetic PnP dataset generation and the analytic-vs-numeric
  optimization pipeline.
- `../utils/`: shared host-side utilities (`camera_utils.h` for
  projection/depth, `se3_utils.h` for pose composition and random SE(3)
  sampling, `validation.h` for MSE metrics).
- Built by the shared `examples/CMakeLists.txt`.

## Walkthrough

1. Generate a ground-truth camera pose (`T_cam_from_world`) and `--num-points`
   random 3D world points visible from it.
2. Project each point to a normalized 2D observation and add pixel noise.
3. Perturb the pose to create a non-trivial initial estimate.
4. Build a single `SE3StateBatch` (one pose) and a `PnPFactorBatch` with one
   factor per correspondence, all referencing the same pose state block.
5. Solve with `LevenbergMarquardtMinimizer`, once per requested Jacobian
   mode, each on a fresh device copy of the perturbed pose so the two runs
   are directly comparable.
6. Compare initial vs. final cost and pose MSE for each mode.

## Build locally (all examples)

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

Output binary: `pnp_example`.
