# Changelog

## 0.4.3

### Changes

- Python vector batches take any size at run time: `VectorStateBatch(data, capacity, dim=n)`,
  `PriorVectorFactorBatch`, `VectorBetweenFactorBatch`, `BoundFactorBatch`,
  `HalfspaceFactorBatch` (dim from an `(N, n)` array or the `dim` keyword); `fit()` takes any
  number of parameters. `VectorStateBatch3` and the other numbered names remain.
- `BlockSparseCholesky` (the default linear solver) is 0-26% faster on an RTX A6000 and 4-19% on
  RTX PRO Blackwell, and much faster on batches of small subproblems.
- Builds compile native code for every GPU generation the CUDA toolkit supports (SM 75-90, and 100
  and 120 from CUDA 12.8): Blackwell no longer compiles the kernels at load time.

### Breaking changes (C++)

Before 1.0 a release may change the C++ API and ABI; rebuild against each release's headers.
From 0.4.3 the shared library's SONAME carries the full version (`libcunls.so.0.4.3`), so the
loader rejects a program built against another release instead of running it with the wrong
headers. The Python wheels link cuNLS statically and are not affected.

Removed from the installed headers, with replacements:

- `cunls/math/so_se_lie_math.h`: the host wrappers `ComputeSkewSO3`, `ComputeExpSO2/SO3/SE2`,
  `ComputeJacobianLeftSO3`, `ComputeJacobianRightSO3/SE3`, `ComputeJacobianLeftInverseSO3/SE3`,
  `ComputeInverseSE2/SE3`, `ComputeAdjointSE3`, `ComputeInverseAdjointSE3`,
  `ComputeTransposeSO2/SO3`, `ComputeNegateMatrix`. The exponential of a batch of tangents is
  `Plus` of the state batch (`SO3StateBatch`, `SE3StateBatch`, ...) applied to identities.
- `cunls/math/sim_lie_math.h`: `ComputeExpSim2/Sim3`, `ComputeInverseSim2/Sim3`,
  `ComputeAdjointSim3` (exponential: `Plus` of `Similarity2StateBatch` / `Similarity3StateBatch`).
- `cunls/math/sl_lie_math.h`: `ComputeInverseSL4`, `ComputeAdjointSL4`,
  `ComputeNegateMatrix15x15`.
- `cunls/math/dense_matrix_ops.h` (`ComputeSqrtMatrix`, `ScatterToRightBlock`) and
  `cunls/common/utils.h`: deleted.
- `cunls/state/articulation_topology.h`: `JointTypes()`, `JointQStart()`, `JointQdStart()`,
  `LinkJoint()`; the same arrays are in the `ArticulationDescription` the topology was built
  from. The class layout changed.
- `cunls/common/cudss_dynamic.h`: `cuDSSIsAvailable()`; to probe for cuDSS, call `GetCudssApi()`
  and catch its exception. `cunls/common/cudss_helper.h`: the device memory-pool functions.
- `cunls/minimizer/problem.h`: `Problem::GetStatePointers()` and `CheckForValidInputs()` (use
  `CheckSizes()` / `CheckConsistency()` / `Validate()`). A factor batch must now read at least
  one state.
- `cunls/minimizer/bsr_matrix.h`, `sparse_matrix.h`, `normal_equations.h`,
  `factor/weighted_factor_batch.h`, `factor/information/information_factor_batch.h`,
  `robustifier/scaled_loss_function_batch.h`: internal matrix, weighting and scaling helpers
  (e.g. `AddScaledDiagonal`, `ScaleSymmetric`, the `Copy*` functions, the `Apply*Weight*`
  functions); `ChooseHessianBlockSize` lost its `max_block_size` argument (the cap is 16).
  `LossFunctionBatch::Evaluate` returns nothing.
- The prior and between factor batches (Lie groups and vectors) and the point-to-point and
  point-to-plane factor batches have no default constructors.

### Breaking changes (Python)

- A bool where a device array or pointer is expected raises `TypeError`: e.g.
  `ArticulationStateBatch(topology, capacity, False)`, whose third argument is now
  `const_state_ids`; pass `apply_joint_limits=False` by keyword.
- `const_state_ids=None` with a nonzero `const_capacity` raises `ValueError`.
