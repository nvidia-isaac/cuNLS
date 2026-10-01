# Custom Factor Example

This example shows how to implement a user-defined factor for `cuNLS`, in
**two** ways, both solving the same 1D chain problem:

- **Part 1** (`ScalarDifferenceFactorBatch`): implements both the residual
  and its analytic Jacobian by hand.
- **Part 2** (`ScalarDifferenceResidualOnlyFactorBatch`): implements only
  the residual and relies on cuNLS's numeric (finite-difference) Jacobian
  support to differentiate it. See
  [Numeric (finite-difference) Jacobians](../../docs/sphinx/numeric_jacobians.rst)
  for how this works in general (manifold-aware, mixing modes within one
  `Problem`, accuracy/performance tradeoffs).

Residual model for each factor, shared by both parts:

`r_i = (x_{i+1} - x_i) - m_i`

Part 1's analytic Jacobians:
- `dr/dx_i = -1`
- `dr/dx_{i+1} = +1`

Part 2's `Evaluate()` only ever writes `residuals` -- there is no Jacobian
code path at all. Its factor group is registered with
`JacobianMode::kNumeric` via `Problem::AddFactorBatch`'s per-group override,
so the minimizer differentiates it via `StateBatch::Plus` perturbations
instead.

### When to use which

| | Analytic (Part 1) | Numeric (Part 2) |
|---|---|---|
| Effort to write | Derive + hand-code the Jacobian | Residual only |
| Jacobian evaluation cost | Fastest | Several times slower (roughly 3-12x per Jacobian evaluation in internal benchmarks, problem-dependent) |
| Best for | Factors that ship / run at scale | Prototyping, one-off factors, or residuals that are awkward to differentiate by hand |

Both parts also add an anchor prior (`PriorFactorBatch<manifold::Vector<1>>`,
the manifold-generic facade specialized to `R^1`) on the first state to
remove global shift ambiguity; the anchor prior is always evaluated
analytically (it's a shipped factor), which in Part 2 also demonstrates
mixing Jacobian modes within a single `Problem`.

## Files

- `main.cu`: both custom factor classes and their kernels (the lesson), and
  the cuNLS workflow (`RunChainExample`, called once per part).
- `../utils/`: header-only helpers kept out of `main` so it reads as the cuNLS
  workflow: `datasets.h` (synthetic scenes), `validation.h` (error metrics),
  `report.h` (printing and the quality verdict), `cli.h` (command-line flags),
  `se3_utils.h` / `camera_utils.h` (host SE(3) math and projection).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## Walkthrough

1. `examples::MakeScalarChainScene` generates a 1D ground-truth chain
   `x_0..x_n`, measurements `m_i = x_{i+1} - x_i`, and a disturbed initial
   estimate.
2. Build `VectorStateBatch<1>` for all states.
3. Add:
   - the difference factor batch for all edges (analytic in Part 1,
     numeric-diff in Part 2)
   - anchor prior factor for the first node (always analytic)
4. Solve with `LevenbergMarquardtMinimizer` (the minimizer allocates GPU
   workspace during initialization).
5. Compare initial vs final MSE to validate improvement.

Each batch is constructed with its **capacity** (how many states /
measurements its bound device buffers hold, fixed for the batch's lifetime)
and starts with 0 active; `SetNumActiveStates` / `SetNumActiveFactors` set the
**active count** the next solve uses (host-only: no allocation, no device
work; a solve without it throws). Size the capacity once for the largest
problem you expect; the active count may change between solves up to it, so
a real-time application allocates once and reuses the same buffers every
frame while the problem size changes. The example keeps the two in separate
variables (`*_capacity` vs. `num_*`); it solves every slot once, so each
active count equals its capacity. The custom factors pass their capacity to
`SizedFactorBatch(capacity)` and read the active count with `NumActiveFactors()`
in `Evaluate`.

## Notes on memory layout

For this custom factor:
- residual size = 1
- state sizes = [1, 1]
- jacobian per factor is therefore `1 x 2` and written as:
  `[dres_dleft, dres_dright]` (Part 1 only -- Part 2 never writes to the
  Jacobian buffer)

State pointer layout for factor `i`:
- `state_pointers[2*i]   -> x_i`
- `state_pointers[2*i+1] -> x_{i+1}`

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
- `./artifacts/examples/custom_factor_example`
