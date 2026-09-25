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

- `main.cu`: both custom factor classes, kernels, and the shared
  optimization pipeline (`RunChainExample`, called once per part).
- `../utils/`: shared host-side utilities (validation metrics).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## Walkthrough

1. Generate a 1D ground-truth chain `x_0..x_n`.
2. Create measurements `m_i = x_{i+1} - x_i`.
3. Disturb all states to create an initial estimate.
4. Build `VectorStateBatch<1>` for all states.
5. Add:
   - the difference factor batch for all edges (analytic in Part 1,
     numeric-diff in Part 2)
   - anchor prior factor for the first node (always analytic)
6. Solve with `LevenbergMarquardtMinimizer` (the minimizer allocates GPU
   workspace during initialization).
7. Compare initial vs final MSE to validate improvement.

## Notes on memory layout

For this custom factor:
- residual size = 1
- state block sizes = [1, 1]
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
