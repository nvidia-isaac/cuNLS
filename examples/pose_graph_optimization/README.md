# Pose Graph Optimization Example

This example demonstrates SE(3) pose graph optimization with:
- `SE3BetweenFactorBatch` for between-pose constraints
- `SE3StateBatch` for pose variables
- `LevenbergMarquardtMinimizer` for solving the nonlinear least-squares system

It builds a synthetic **pose chain**:
- A single chain of poses `T_0, T_1, ..., T_{N-1}`
- Between-factor constraints connect consecutive poses
- The first pose `T_0` is held fixed as the gauge anchor
- All remaining poses are optimized

## Files

- `main.cpp`: complete synthetic PGO pipeline.
- `../utils/`: shared host-side utilities (SE(3) math, validation).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## How the factor is used

`SE3BetweenFactorBatch` computes residuals with the convention:

`r = Log(delta * T_i^{-1} * T_{i+1})`

For zero residual, consecutive poses should satisfy:

`T_{i+1} = T_i * delta^{-1}`

The example constructs the ground-truth chain exactly this way, then
applies random disturbance to create initial estimates for all poses
except the fixed anchor.

## Problem setup

1. Generate a random anchor pose `T_0` and random `delta` constraints.
2. Compute the ground-truth chain: `T_{i+1} = T_i * inverse(delta_i)`.
3. Disturb all poses except `T_0`.
4. Add a single `SE3StateBatch` with `T_0` marked as constant.
5. Build one between factor per consecutive pair `(T_i, T_{i+1})`.

## Validation

After optimization, the example prints:
- initial/final solver cost
- iteration count
- chain constraint MSE for `delta * T_i^{-1} * T_{i+1}`

The binary returns non-zero if convergence quality is poor.

## Build locally (all examples)

```bash
cmake -S examples -B build/examples/all \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUNLS_INSTALL_DIR=/path/to/cunls_install
cmake --build build/examples/all -j
```

## Build inside Docker and export binaries

```bash
./examples/build_in_docker.sh Release ./artifacts/examples
```

Output binary:
- `./artifacts/examples/pose_graph_optimization_example`
