# Custom Factor Example

This example shows how to implement a user-defined factor for `cuNLS`.

It defines:
- `ScalarDifferenceFactorBatch : SizedFactorBatch<1, 1, 1>`
- CUDA kernel `ScalarDifferenceKernel`

Residual model for each factor:

`r_i = (x_{i+1} - x_i) - m_i`

with Jacobians:
- `dr/dx_i = -1`
- `dr/dx_{i+1} = +1`

The example also adds an anchor prior (`PriorVectorFactorBatch<1>`) on the
first state to remove global shift ambiguity.

## Files

- `main.cu`: custom factor class, kernel, and optimization pipeline.
- `../utils/`: shared host-side utilities (validation metrics).
- Built by the shared `examples/CMakeLists.txt`.
- Exported by the shared `examples/build_in_docker.sh`.

## Walkthrough

1. Generate a 1D ground-truth chain `x_0..x_n`.
2. Create measurements `m_i = x_{i+1} - x_i`.
3. Disturb all states to create an initial estimate.
4. Build `VectorStateBatch<1>` for all states.
5. Add:
   - custom difference factor batch for all edges
   - anchor prior factor for the first node
6. Solve with `LevenbergMarquardtMinimizer` (the minimizer allocates GPU
   workspace during initialization).
7. Compare initial vs final MSE to validate improvement.

## Notes on memory layout

For this custom factor:
- residual size = 1
- state block sizes = [1, 1]
- jacobian per factor is therefore `1 x 2` and written as:
  `[dres_dleft, dres_dright]`

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
