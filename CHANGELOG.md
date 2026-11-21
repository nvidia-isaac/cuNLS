# Changelog

## 0.4.3

### Changes

- Python wheels are CUDA 13 only: one package, `pycunls-cu13` (R580+ driver), depending on
  `warp-lang>=1.18` (the `warp-lang<1.18` cap is gone, so pycunls installs next to current Warp).
  There is no `pycunls-cu12` any more; the C++ library still supports CUDA 12.6+ and 13 (Jetson
  Orin and Thor). Building the bindings with CUDA 12 stops at configure, and importing pycunls
  on a pre-R580 driver raises `ImportError` with the fix.
- `MinimizerOptions.real_time` (C++ `MinimizerOptions::real_time`): Gauss-Newton and
  Levenberg-Marquardt run exactly `max_num_iterations` iterations without host synchronization,
  so a solve can be captured in a CUDA graph after its first call.
- Devices: a problem runs on the device of its states. `minimize()`, `evaluate_batch()`,
  `validate()` and the articulation and state helpers make that device current themselves
  (`Problem.device` tells which), so callers need no `wp.ScopedDevice` around them; C++
  `Minimize()` rejects a call from another device or with a stream of another device. The Warp
  batch classes default to `device=None` (Warp's current device) instead of `"cuda:0"`.
- `pycunls.warp.WarpFunctionFactorBatch`: a factor from a Warp residual function over any number
  of states (`wp.transform`, `wp.quat`, `float`, `wp.vecN`), with Jacobians from Warp's automatic
  differentiation (checked against central differences), central differences, or an analytic
  Warp function. `diagnose()` and `ProblemBuilder` report a state batch connected to a slot
  whose layout it does not have (e.g. an `SO3StateBatch` on a `wp.vec3` argument).
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

- `AugmentedLagrangianMinimizer(options)` owns its Levenberg-Marquardt inner solver (Gauss-Newton
  is no longer an inner minimizer), built from `AugmentedLagrangianMinimizerOptions`, which now
  extends `LevenbergMarquardtMinimizerOptions` (see the Python section).
- `Minimizer::Minimize`, `AugmentedLagrangianMinimizer::Minimize` and `RansacMinimizer::Minimize`
  throw `std::invalid_argument` when the current device or the stream's device is not the device
  of the problem's states (`Problem::Device`), or when a minimizer is reused on another device.

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

- `pycunls.warp.WarpPoseFactorBatch` is removed: `WarpFunctionFactorBatch` with a
  `(X: wp.transform, data)` residual does the same.
- `AugmentedLagrangianMinimizer(options)` takes no minimizer: it owns a private
  Levenberg-Marquardt inner solver (Gauss-Newton is no longer an inner minimizer), built from
  `AugmentedLagrangianMinimizerOptions`, which now extends `LevenbergMarquardtMinimizerOptions`
  (damping fields and `base_options`) with the outer-loop fields. The options set the budget of a
  call, with or without constraint batches, through `base_options`: `max_num_iterations` caps a
  subproblem's full inner solve (was `final_inner_iterations`; 50; in real time it is the last
  outer iteration's budget: real-time callers set it, it no longer falls back to
  `inner_iterations`), `max_line_search_steps` is the inner line search (was
  `inner_line_search_steps`; 10 by default here), and `real_time` / `reuse_structure` replace the
  fields of the same names. Assigning `solver.options` changes the outer loop and the budget; the
  inner solver keeps the damping, linear solver and tolerances it was built with. A bounds-only
  problem runs the general loop: one outer iteration when its full inner solve converges, and a
  real-time call reads nothing back. `Horizon.build` (MPC) takes `options` only (no
  `minimizer=`). `fit(..., bounds=..., method="gn")` and `ProblemBuilder.solve(method="gn")` with
  bounds or constraint factors raise `ValueError`.
- A bool where a device array or pointer is expected raises `TypeError`: e.g.
  `ArticulationStateBatch(topology, capacity, False)`, whose third argument is now
  `const_state_ids`; pass `apply_joint_limits=False` by keyword.
- `const_state_ids=None` with a nonzero `const_capacity` raises `ValueError`.
