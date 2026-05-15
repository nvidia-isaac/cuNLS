################################################################################
Minimizer API
################################################################################

The minimizer module provides iterative solvers for non-linear least squares.
It implements **Gauss-Newton** and **Levenberg-Marquardt** algorithms, which
repeatedly linearize the residuals, solve a linear least-squares system for the
step in tangent space, and apply the step via the manifold :math:`\oplus`
operation. See `Gauss–Newton algorithm
<https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_ and
`Levenberg–Marquardt algorithm
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ for
background.

**C++** — ``cunls/minimizer``
  |  **Python** — ``pycunls``

================================================================================
Theory — Gauss-Newton and Levenberg-Marquardt
================================================================================

**Objective**

We minimize a sum of squared (and optionally robustified) residuals:

.. math::
   S(x) = \frac{1}{2} \sum_i \left\| f_i(x) \right\|^2
        = \frac{1}{2} \left\| f(x) \right\|^2

where :math:`x` is the state (on manifolds), :math:`f_i` are residual blocks,
and :math:`f` denotes the stacked residual vector. At the current estimate
:math:`x_0`, we linearize:

.. math::
   f(x_0 + \Delta x) \approx f(x_0) + J \Delta x

with :math:`J` the Jacobian of :math:`f` with respect to the tangent update
:math:`\Delta x`. Substituting into :math:`S` gives a quadratic model in
:math:`\Delta x`; minimizing it yields the **Gauss-Newton** step.

**Gauss-Newton**

The linearized least-squares problem is:

.. math::
   \Delta x^* = \arg\min_{\Delta x} \left\| J \Delta x + f(x_0) \right\|^2

Setting the gradient to zero gives the **normal equations**:

.. math::
   J^T J \, \Delta x^* = -J^T f(x_0)

So at each iteration we form :math:`H = J^T J` and :math:`b = -J^T f`, solve
:math:`H \Delta x = b`, then update :math:`x_{\mathrm{new}} = x_0 \oplus \Delta x`.
The matrix :math:`J^T J` is the Gauss-Newton approximation to the Hessian of
:math:`S`; no second derivatives of :math:`f` are needed. Convergence can be
quadratic when the residuals are small and the model is a good approximation.

**Levenberg-Marquardt**

When the initial guess is poor or the problem is badly scaled, Gauss-Newton
may diverge. The **Levenberg-Marquardt** method dampens the step by solving:

.. math::
   \left( J^T J + \lambda \, D \right) \Delta x = -J^T f(x_0)

where :math:`\lambda \ge 0` is a damping parameter and :math:`D` is often the
diagonal of :math:`J^T J` (so the step is scale-invariant). For :math:`\lambda = 0`
this is Gauss-Newton; for large :math:`\lambda` the step shrinks toward the
gradient-descent direction :math:`-J^T f`. The implementation adjusts
:math:`\lambda` each iteration: increase it when a step is rejected (cost
rises), decrease it when a step is very successful, so the method interpolates
between gradient descent and Gauss-Newton and is more robust. See the
Wikipedia links above for convergence and damping strategies.

**In cuNLS**

- **GaussNewtonMinimizer** solves :math:`J^T J \Delta x = -J^T r` each iteration
  and updates state with :math:`x \oplus \Delta x` until convergence (step norm
  or cost change below tolerance).
- **LevenbergMarquardtMinimizer** solves
  :math:`(J^T J + \lambda \operatorname{diag}(J^T J)) \Delta x = -J^T r`,
  adapts :math:`\lambda` from step quality (actual vs. predicted cost reduction),
  and accepts/rejects steps accordingly.

**Column scaling (optional)**

:code:`MinimizerOptions::column_scaling` can re-scale the normal equations with a
diagonal :math:`S`: the linear solve uses :math:`S H S \, z = S b` with
:math:`H = J^T J` and :math:`b = -J^T r`, then applies the physical tangent step
:math:`\Delta x = S z`. Modes are: no scaling (default); :math:`S_{ii} = 1/\sqrt{H_{ii}}`
(with a small floor for stability); or :math:`S_{jj} = 1/\|J_{:,j}\|_2` from the
CSR Jacobian. For Levenberg-Marquardt, damping uses the diagonal of the **scaled**
Hessian: :math:`S H S + \lambda \operatorname{diag}(S H S)`.

================================================================================
Structures
================================================================================

.. _cunls-minimizer-summary-label:

--------------------------------------------------------------------------------
:code:`MinimizerSummary`
--------------------------------------------------------------------------------

Returned by :cpp:func:`GaussNewtonMinimizer::Minimize` and
:cpp:func:`LevenbergMarquardtMinimizer::Minimize`. Holds solve statistics for
inspecting iteration count and cost history.

- **num_iterations** [out]: Number of nonlinear iterations performed.
- **initial_cost** [out]: Objective value before the first iteration.
- **final_cost** [out]: Objective value at termination.
- **iteration_costs** [out]: Per-iteration cost history (for plotting or debugging).

--------------------------------------------------------------------------------
:code:`MinimizerOptions`
--------------------------------------------------------------------------------

Options for the **Gauss-Newton** minimizer: iteration limit, convergence
tolerances, consecutive rejected-step limit, and sparse linear solver choice.
Used when constructing a :code:`GaussNewtonMinimizer`.

- **max_num_iterations** [in]: Maximum number of iterations. Default: 50.
- **state_tolerance** [in]: Convergence threshold on squared step norm; optimizer
  terminates when the step norm falls below this. Default: 1e-6.
- **cost_tolerance** [in]: Convergence threshold on cost; optimizer terminates
  when the cost falls below this. Default: 1e-6.
- **max_consecutive_rejected_steps** [in]: Maximum number of consecutive
  rejected steps before declaring convergence. When every trial step is rejected
  (cost increases or step quality below acceptance threshold) this many times
  in a row, the minimizer treats the current solution as converged. Set to 0 to
  disable. Default: 5.
- **sparse_linear_solver_type** [in]: Linear backend; options are
  ``BlockSparsePCG`` (block-Jacobi preconditioned conjugate gradient;
  layout auto-derived from the problem's state batches), ``cuDSS`` (sparse
  direct solver via NVIDIA's cuDSS library), ``DenseLDLT`` (converts CSR
  to dense and solves with a custom CUDA pivoted LDLT factorization),
  ``DenseCholesky`` (converts CSR to dense and solves via cuSOLVER Cholesky;
  requires SPD matrix), and ``DenseQR`` (converts CSR to dense and solves
  via cuSOLVER QR factorization; works for any non-singular matrix).
  Default: ``BlockSparsePCG``.
- **sparse_linear_solver_config** [in]: Backend-specific options. For
  ``BlockSparsePCG`` contains :code:`block_sparse_pcg_options`
  (``block_size`` / ``block_layout``, ``max_iterations``,
  ``relative_tolerance``, ``absolute_tolerance``, ``pivot_floor``,
  ``check_period``).  For ``cuDSS`` contains :code:`cudss_solver_options`
  (mode, ``nthreads``, optional ``threading_lib_path`` for multi-threaded
  cuDSS).  Dense backends take no extra configuration.
- **sparse_square_multiplier_type** [in]: Strategy for computing the approximate
  Hessian :math:`J^T J`; options are ``cuSPARSE`` (cuSPARSE SpGEMM reuse API)
  and ``Fast`` (warp-efficient CUDA kernels with bitmap pattern discovery).
  Default: ``Fast``.
- **column_scaling** [in]: Diagonal scaling of the normal equations; see the
  column-scaling note above. Values: ``None``, ``HessianDiagonal``,
  ``JacobianColumnNorm``. Default: ``None``.
- **disable_safety_checks** [in]: When ``false``, the minimizer enables all
  optional runtime validation.  Currently this covers post-factorization
  checks in the linear solver: Cholesky checks cuSOLVER ``devInfo`` after
  ``potrf`` and ``potrs``; QR inspects the diagonal of ``R`` for rank
  deficiency; LDLT performs in-kernel pivot and diagonal checks.  Future
  minimizer versions may add additional checks (e.g. NaN/Inf detection,
  cost-increase guards).  Failures cause ``Solve()`` to return ``false``
  with a diagnostic via ``LogError()``; the minimizer then throws
  ``std::runtime_error``.  When ``true``, every check listed above is
  skipped (no device-to-host memcpy, no stream synchronization, no in-kernel
  validation), which can reduce per-iteration latency for small systems
  but may produce silently incorrect results for singular or ill-conditioned
  matrices. Default: ``true``.

--------------------------------------------------------------------------------
:code:`LevenbergMarquardtMinimizerOptions`
--------------------------------------------------------------------------------

Options for the **Levenberg-Marquardt** minimizer. Extends
:code:`MinimizerOptions` with damping and step-acceptance parameters. Used when
constructing a :code:`LevenbergMarquardtMinimizer`.

- **base_options** [in]: Base Gauss-Newton options (:code:`MinimizerOptions`).
- **initial_lambda** [in]: Initial damping coefficient. Default: 1e-3.
- **relative_reduction_tolerance** [in]: Convergence threshold on predicted
  relative cost reduction. Default: 1e-6.
- **lambda_upscale** [in]: Factor by which :math:`\lambda` is increased after a
  rejected step. Default: 2.0.
- **lambda_downscale** [in]: Factor by which :math:`\lambda` is decreased after a
  very successful step. Default: 0.5.
- **lambda_max** [in]: Upper bound for :math:`\lambda`. Default: 1e+6.
- **lambda_min** [in]: Lower bound for :math:`\lambda`. Default: 1e-6.
- **step_accept_threshold** [in]: Minimum step quality (rho, actual/predicted
  cost reduction) to accept a step. Default: 0.25.
- **lambda_downscale_threshold** [in]: Step quality above which :math:`\lambda`
  is decreased. Default: 0.75.

================================================================================
Class APIs
================================================================================

.. _gauss-newton-minimizer-ctor-label:

--------------------------------------------------------------------------------
:code:`GaussNewtonMinimizer::GaussNewtonMinimizer`
--------------------------------------------------------------------------------

**Purpose:** Constructs a Gauss-Newton minimizer that will solve
:math:`J^T J \Delta x = -J^T r` each iteration and update the problem state
until convergence.

.. cpp:function:: GaussNewtonMinimizer(const MinimizerOptions& options = MinimizerOptions())

  :param ``options``: [in] Solver options (max iterations, tolerances, linear solver); copied into the minimizer.
  :returns: [out] Constructor has no return value.

.. _gauss-newton-minimize-label:

--------------------------------------------------------------------------------
:code:`GaussNewtonMinimizer::Minimize`
--------------------------------------------------------------------------------

**Purpose:** Runs the Gauss-Newton iteration on the given problem: at each step
evaluates residuals and Jacobians, assembles and solves the normal equations,
applies the tangent step via the state batches’ Plus, and checks convergence.
Updates the problem’s state in-place when steps are accepted.

.. cpp:function:: MinimizerSummary Minimize(cudaStream_t stream, Problem& problem)

  :param ``stream``: [in] CUDA stream for evaluation, linear algebra, and state updates.
  :param ``problem``: [in,out] Problem (factor graph + state batches); state memory is updated in-place.
  :returns: [out] :cpp:class:`MinimizerSummary` with iteration count and cost statistics.

  **Note:** A minimizer instance retains working buffers (normal-equation matrix,
  RHS, and internal state snapshots) across calls to :cpp:func:`Minimize`; when
  the problem size is unchanged, device memory is reused instead of reallocating
  each time.

.. _lm-minimizer-ctor-label:

--------------------------------------------------------------------------------
:code:`LevenbergMarquardtMinimizer::LevenbergMarquardtMinimizer`
--------------------------------------------------------------------------------

**Purpose:** Constructs a Levenberg-Marquardt minimizer that solves the damped
system :math:`(J^T J + \lambda D) \Delta x = -J^T r` and adapts :math:`\lambda`
from step quality. More robust than Gauss-Newton when the initial guess is far
from the solution.

.. cpp:function:: LevenbergMarquardtMinimizer(const LevenbergMarquardtMinimizerOptions& options = LevenbergMarquardtMinimizerOptions())

  :param ``options``: [in] LM options (damping, accept/reject thresholds, etc.).
  :returns: [out] Constructor has no return value.

:cpp:func:`LevenbergMarquardtMinimizer` also provides :cpp:func:`Minimize` with
the same signature as :cpp:func:`GaussNewtonMinimizer::Minimize`; it runs the LM
iteration instead of pure Gauss-Newton.

.. _problem-add-factor-label:

--------------------------------------------------------------------------------
:code:`Problem::AddFactorBatch`
--------------------------------------------------------------------------------

**Purpose:** Registers a factor batch (and optionally a robust loss batch) with
the problem and binds its factor instances to state block pointers. The
ordering of :code:`state_pointers` must match the factor batch’s expected
state layout (see :doc:`factor`).

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, const std::vector<float*>& state_pointers)

  :param ``factor_batch``: [in] Factor batch pointer (non-owning).
  :param ``state_pointers``: [in] Flattened device pointers: one per (factor index, state block), mapping factors to state. The problem stores a **host** copy of this list (each entry is still a device ``float*``); no device allocation is used for the table itself.
  :returns: [out] No return value.

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function_batch, const std::vector<float*>& state_pointers)

  :param ``factor_batch``: [in] Factor batch pointer (non-owning).
  :param ``loss_function_batch``: [in] Robust loss batch pointer (non-owning).
  :param ``state_pointers``: [in] Flattened state pointer mapping for all factors in the batch (stored on the host as above).
  :returns: [out] No return value.

.. _problem-add-state-label:

--------------------------------------------------------------------------------
:code:`Problem::AddStateBatch`
--------------------------------------------------------------------------------

**Purpose:** Registers a state batch with the problem. State batches supply the
manifold Plus operation and state block pointers used when building
:code:`state_pointers` for :cpp:func:`Problem::AddFactorBatch` and when
applying steps during :cpp:func:`Minimize`.

.. cpp:function:: void AddStateBatch(StateBatch* state_batch)

  :param ``state_batch``: [in] State batch pointer (non-owning).
  :returns: [out] No return value.

.. _problem-check-consistency-label:

--------------------------------------------------------------------------------
:code:`Problem::CheckConsistency`
--------------------------------------------------------------------------------

**Purpose:** Verifies that the problem’s factor batches, state batches, and
state pointer mappings are consistent (e.g. correct dimensions and
connectivity). Call before :cpp:func:`Minimize` to catch configuration errors.

.. cpp:function:: bool CheckConsistency() const

  :returns: [out] ``true`` when graph inputs and connectivity are valid.

.. _residual-batch-evaluate-label:

--------------------------------------------------------------------------------
:code:`ResidualBatch::Evaluate`
--------------------------------------------------------------------------------

**Purpose:** Evaluates the residual batch: computes residuals (and optionally
Jacobians) from the factor batch and applies the optional loss function to
produce robustified residuals and scaled Jacobians used by the minimizer.
Callers supply device scratch for per-factor squared norms and :math:`\rho`
triplets; see ``ResidualBatchWorkspaceSizeBytes`` / ``ResidualBatchWorkspaceNumFloats``.

.. cpp:function:: size_t ResidualBatchWorkspaceSizeBytes(size_t num_residuals)

  :param ``num_residuals``: [in] Number of factors (``NumFactors()`` for the batch).
  :returns: [out] Minimum device scratch size in bytes for ``Evaluate``\ 's ``workspace`` pointer.

.. cpp:function:: size_t ResidualBatchWorkspaceNumFloats(size_t num_residuals)

  :param ``num_residuals``: [in] Number of factors (``NumFactors()`` for the batch).
  :returns: [out] Same scratch as ``ResidualBatchWorkspaceSizeBytes``, expressed in ``float`` elements (rounded up), for sub-allocating inside a ``float`` arena.

.. cpp:function:: bool Evaluate(cudaStream_t stream, float* workspace, float* residuals, float const* const* state_pointers, float* cost, float* jacobians) const

  :param ``stream``: [in] CUDA stream for factor, loss, and scaling kernels.
  :param ``workspace``: [in] Device scratch; size at least ``ResidualBatchWorkspaceSizeBytes(NumFactors())`` bytes (see layout in the header). Must not overlap ``residuals``, ``jacobians``, or ``cost``.
  :param ``residuals``: [out] Residual output buffer of length ``NumFactors() * ResidualsSize()`` (after loss scaling if present).
  :param ``state_pointers``: [in] Device pointer array mapping factor inputs to state blocks.
  :param ``cost``: [out] Optional per-residual cost (e.g. :math:`\frac{1}{2}\rho(\|r\|^2)`); can be ``nullptr``.
  :param ``jacobians``: [out] Optional Jacobian output (after loss scaling); can be ``nullptr``.
  :returns: [out] ``true`` on successful evaluation.

.. _minimizer-state-copy-label:

--------------------------------------------------------------------------------
:code:`MinimizerState::Copy`
--------------------------------------------------------------------------------

**Purpose:** Snapshot or restore state. Copy state from a set of device vectors
into the minimizer state, or from another :cpp:class:`MinimizerState` into a
problem’s state storage. Useful for rollback or warm starts.

.. cpp:function:: void Copy(cudaStream_t stream, const std::vector<dvector<float>>& other)

  :param ``stream``: [in] CUDA stream for copy operations.
  :param ``other``: [in] Source state vectors (one per state batch segment).
  :returns: [out] No return value.

.. cpp:function:: void Copy(cudaStream_t stream, const MinimizerState& state, Problem& problem)

  :param ``stream``: [in] CUDA stream for copy operations.
  :param ``state``: [in] Source minimizer state snapshot.
  :param ``problem``: [out] Problem whose state storage is overwritten with the copied values.
  :returns: [out] No return value.

.. _minimizer-state-build-triplet-label:

--------------------------------------------------------------------------------
:code:`MinimizerState::BuildTripletSparseStructure`
--------------------------------------------------------------------------------

**Purpose:** Fills the row and column index arrays of a :code:`TripletSparseStructure`
for the problem’s Jacobian (COO / triplet layout). Implementation lives in
``jacobian_ops.cu``; the minimizer uploads the problem’s host-held state-pointer
lists to device internally before building column indices.

.. cpp:function:: void BuildTripletSparseStructure(cudaStream_t stream, const Problem& problem, TripletSparseStructure& structure)

  :param ``stream``: [in] CUDA stream for GPU work.
  :param ``problem``: [in] Factor graph and state-pointer mappings.
  :param ``structure``: [out] Row and column index device buffers sized to the Jacobian nonzeros.
  :returns: [out] No return value.

.. _sparse-square-multiplier-label:

--------------------------------------------------------------------------------
:code:`SparseMatrixMultiplier` (base class)
--------------------------------------------------------------------------------

Abstract interface for computing :math:`A^T A` of a sparse CSR matrix.
Two concrete implementations are provided:

- **cuSPARSESparseMatrixMultiplier** — uses the cuSPARSE SpGEMM reuse API
  (transpose + multiply). Select with
  :code:`SparseMatrixMultiplierType::cuSPARSE`.
- **FastSparseMatrixMultiplier** — uses custom warp-efficient CUDA
  kernels with bitmap-based sparsity pattern discovery. Select with
  :code:`SparseMatrixMultiplierType::Fast`.

.. cpp:function:: void SparseMatrixMultiplier::Initialize(cudaStream_t stream, const Problem& problem, const CSRSparseMatrix& input, CSRSparseMatrix& output)

  Analyzes the sparsity pattern of :math:`A^T A` and allocates the output
  matrix. Must be called once whenever the sparsity pattern changes.

  :param ``stream``: [in] CUDA stream for GPU operations.
  :param ``problem``: [in] Optimization problem providing structural hints.
  :param ``input``: [in] Input sparse matrix :math:`A`.
  :param ``output``: [out] Output sparse matrix :math:`A^T A` (structure allocated).
  :returns: [out] No return value.

.. cpp:function:: void SparseMatrixMultiplier::ComputeSquaredMatrix(cudaStream_t stream, const Problem& problem, const CSRSparseMatrix& input, CSRSparseMatrix& output)

  Computes the numerical values of :math:`A^T A`. The output structure must
  already be set by a prior call to :cpp:func:`Initialize`.

  :param ``stream``: [in] CUDA stream for GPU operations.
  :param ``problem``: [in] Optimization problem (may be used for kernel tuning).
  :param ``input``: [in] Input sparse matrix :math:`A`.
  :param ``output``: [out] Output sparse matrix :math:`A^T A`.
  :returns: [out] No return value.

.. _sparse-square-multiplier-type-label:

--------------------------------------------------------------------------------
:code:`SparseMatrixMultiplierType`
--------------------------------------------------------------------------------

Enum in ``cunls/minimizer/sparse_matrix_multiplier.h``:

- ``cuSPARSE`` — cuSPARSE SpGEMM reuse API (transpose + multiply).
- ``Fast`` — fast warp-efficient CUDA kernels with bitmap pattern
  discovery.

.. _sparse-square-multiplier-factory-label:

--------------------------------------------------------------------------------
:code:`CreateSparseMatrixMultiplier`
--------------------------------------------------------------------------------

Factory function in ``cunls/minimizer/sparse_matrix_multiplier.h``:

.. cpp:function:: SparseMatrixMultiplierPtr CreateSparseMatrixMultiplier(SparseMatrixMultiplierType type)

  :param ``type``: [in] Strategy to use for computing :math:`A^T A`.
  :returns: [out] Heap-allocated multiplier instance.

================================================================================
Python API (``pycunls``)
================================================================================

The Python bindings expose the same minimizer, problem, options, and summary
types through the ``pycunls`` package.  All GPU memory is managed via CuPy
arrays; every constructor argument documented as ``DevicePointer`` accepts
either a ``cupy.ndarray`` (the device pointer is extracted automatically) or
a raw ``int`` device address.

The utility types :ref:`CudaStream <py-cuda-stream-label>` and
:ref:`CublasHandle <py-cublas-handle-label>` are documented in
:doc:`common`.

.. _py-minimizer-options-label:

--------------------------------------------------------------------------------
``pycunls.MinimizerOptions``
--------------------------------------------------------------------------------

Configuration for the base Gauss-Newton iteration loop.  Create with default
values and then override individual fields.

**Constructor**

.. code-block:: python

   opts = pycunls.MinimizerOptions()

**Writable attributes**

- **max_num_iterations** (``int``, default ``50``) — upper bound on the
  number of nonlinear iterations.  The minimizer stops early if a
  convergence criterion is met.
- **state_tolerance** (``float``, default ``1e-6``) — convergence threshold
  on the squared norm of the tangent-space step :math:`\|\Delta x\|^2`.
  When the step is smaller than this value the minimizer declares
  convergence.
- **cost_tolerance** (``float``, default ``1e-6``) — convergence threshold
  on the absolute cost value :math:`S(x)`.  When the cost drops below this
  value the minimizer stops.
- **max_consecutive_rejected_steps** (``int``, default ``5``) — how many
  consecutive rejected steps (cost increased or step quality below
  acceptance threshold) are allowed before the minimizer treats the current
  estimate as converged.  Set to ``0`` to disable this criterion.
- **sparse_linear_solver_type** (``SparseLinearSolverType``, default
  ``BlockSparsePCG``) — selects the linear-system backend.
  ``BlockSparsePCG`` runs block-Jacobi preconditioned conjugate gradient
  with the block layout derived automatically from the problem's state
  batches.  ``cuDSS`` uses NVIDIA's sparse direct solver; ``DenseLDLT``
  converts to dense and factorizes with a custom pivoted LDLT kernel;
  ``DenseCholesky`` converts to dense and uses cuSOLVER Cholesky
  (requires SPD); ``DenseQR`` converts to dense and uses cuSOLVER QR
  factorization (works for any non-singular matrix).
- **sparse_square_multiplier_type** (``SparseMatrixMultiplierType``, default
  ``Fast``) — strategy for computing the approximate Hessian
  :math:`J^T J`.  ``cuSPARSE`` uses the cuSPARSE SpGEMM reuse API;
  ``Fast`` uses warp-efficient CUDA kernels with bitmap pattern discovery.
- **column_scaling** (``ColumnScaling``, default ``ColumnScaling.none``) —
  optional diagonal scaling :math:`S` for the normal equations
  (:math:`S H S\, z = S b`, then :math:`\Delta x = S z`). See
  :ref:`py-column-scaling-label`.
- **disable_safety_checks** (``bool``, default ``True``) — when ``False``,
  the minimizer enables all optional runtime validation.  Currently this
  covers post-factorization checks in the linear solver: Cholesky checks
  cuSOLVER ``devInfo`` after ``potrf`` / ``potrs``; QR inspects the diagonal
  of ``R`` for rank deficiency; LDLT performs in-kernel pivot and diagonal
  checks.  Future versions may add further checks (e.g. NaN/Inf detection,
  cost-increase guards).  Failures cause ``Solve()`` to return ``False`` and
  the minimizer raises ``RuntimeError``.  When ``True``, every check listed
  above is skipped (no device-to-host memcpy, no stream synchronization, no
  in-kernel validation), which can reduce per-iteration latency but may
  produce silently incorrect results for singular or ill-conditioned
  matrices.  Only set to ``True`` for well-conditioned, pre-validated
  systems where the extra overhead is a measurable bottleneck.

**Example**

.. code-block:: python

   opts = pycunls.MinimizerOptions()
   opts.max_num_iterations = 100
   opts.state_tolerance = 1e-8
   opts.cost_tolerance  = 1e-8
   opts.column_scaling = pycunls.ColumnScaling.hessian_diagonal

.. _py-column-scaling-label:

--------------------------------------------------------------------------------
``pycunls.ColumnScaling``
--------------------------------------------------------------------------------

Enum used by ``MinimizerOptions.column_scaling`` (and ``LevenbergMarquardtMinimizerOptions.base_options.column_scaling``):

- **none** — identity scaling (standard :math:`H \Delta x = -J^T r`).
- **hessian_diagonal** — :math:`S_{ii} = 1 / \sqrt{H_{ii}}` with a numerical floor.
- **jacobian_column_norm** — :math:`S_{jj} = 1 / \|J_{:,j}\|_2` from the CSR Jacobian.

For LM, damping uses the diagonal of the **scaled** Hessian. See the C++
column-scaling theory note earlier in this page.

.. _py-minimizer-summary-label:

--------------------------------------------------------------------------------
``pycunls.MinimizerSummary``
--------------------------------------------------------------------------------

Returned by ``GaussNewtonMinimizer.minimize`` and
``LevenbergMarquardtMinimizer.minimize``.  All fields are read-only
properties.

**Properties**

- **num_iterations** (``int``) — total number of nonlinear iterations
  executed (including rejected steps in LM).
- **initial_cost** (``float``) — objective value :math:`S(x_0)` evaluated
  before the first iteration.
- **final_cost** (``float``) — objective value at termination.
- **iteration_costs** (``list[float]``) — per-iteration cost history.  The
  list has ``num_iterations + 1`` entries: element 0 is ``initial_cost`` and
  element *i* is the cost after iteration *i*.  Useful for convergence
  plotting or debugging stalled solves.

``MinimizerSummary`` also supports ``repr()`` for quick inspection in a REPL.

.. _py-lm-options-label:

--------------------------------------------------------------------------------
``pycunls.LevenbergMarquardtMinimizerOptions``
--------------------------------------------------------------------------------

Extends the base ``MinimizerOptions`` with damping and step-acceptance
parameters for Levenberg-Marquardt.

**Constructor**

.. code-block:: python

   lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()

**Writable attributes**

- **base_options** (``MinimizerOptions``) — the underlying Gauss-Newton
  options (iteration limit, tolerances, linear solver).  Assign a
  pre-configured ``MinimizerOptions`` instance here.
- **initial_lambda** (``float``, default ``1e-3``) — starting damping
  coefficient :math:`\lambda`.  Larger values make the first step more
  like gradient descent; smaller values start closer to Gauss-Newton.
- **lambda_upscale** (``float``, default ``2.0``) — factor by which
  :math:`\lambda` is *increased* after a rejected step (cost went up).
- **lambda_downscale** (``float``, default ``0.5``) — factor by which
  :math:`\lambda` is *decreased* after a very successful step (step quality
  above ``lambda_downscale_threshold``).
- **lambda_max** (``float``, default ``1e+6``) — upper clamp for
  :math:`\lambda`.  Prevents the damping from growing unboundedly.
- **lambda_min** (``float``, default ``1e-6``) — lower clamp for
  :math:`\lambda`.
- **step_accept_threshold** (``float``, default ``0.25``) — minimum step
  quality :math:`\rho = \text{actual reduction} / \text{predicted
  reduction}` required to accept a step.  Steps with
  :math:`\rho < \text{threshold}` are rejected, :math:`\lambda` is
  increased, and the state is rolled back.
- **lambda_downscale_threshold** (``float``, default ``0.75``) — step
  quality above which :math:`\lambda` is decreased.  Steps with
  :math:`\rho \ge \text{threshold}` are considered "very successful" and
  the solver becomes more Gauss-Newton-like.

**Example**

.. code-block:: python

   opts = pycunls.MinimizerOptions()
   opts.max_num_iterations = 80
   opts.state_tolerance = 1e-8

   lm_opts = pycunls.LevenbergMarquardtMinimizerOptions()
   lm_opts.base_options   = opts
   lm_opts.initial_lambda = 1e-3

.. _py-gauss-newton-label:

--------------------------------------------------------------------------------
``pycunls.GaussNewtonMinimizer``
--------------------------------------------------------------------------------

Iterative Gauss-Newton solver.  At each iteration it evaluates residuals and
Jacobians, assembles and solves the normal equations
:math:`J^T J \,\Delta x = -J^T r`, applies the tangent-space step via each
state batch's Plus operation, and checks convergence.

**Constructor**

.. code-block:: python

   minimizer = pycunls.GaussNewtonMinimizer(options=pycunls.MinimizerOptions())

- **options** (``MinimizerOptions``, optional) — solver configuration.  When
  omitted, default options are used.

**Methods**

- ``minimize(stream: CudaStream, problem: Problem) -> MinimizerSummary`` —
  runs the Gauss-Newton iteration on *problem*.  The state memory owned by
  the state batches inside *problem* is updated **in-place** on the GPU.
  All GPU work is issued on *stream*; call
  ``cp.cuda.runtime.streamSynchronize(stream.get_stream())`` afterwards to
  ensure results are visible on the host.  Returns a
  :ref:`MinimizerSummary <py-minimizer-summary-label>` with iteration count
  and cost statistics.

.. _py-lm-label:

--------------------------------------------------------------------------------
``pycunls.LevenbergMarquardtMinimizer``
--------------------------------------------------------------------------------

Levenberg-Marquardt solver (damped Gauss-Newton).  Solves
:math:`(J^T J + \lambda\,\mathrm{diag}(J^T J))\,\Delta x = -J^T r` and
adapts :math:`\lambda` based on step quality.  More robust than pure
Gauss-Newton when the initial guess is far from the solution.

**Constructor**

.. code-block:: python

   minimizer = pycunls.LevenbergMarquardtMinimizer(
       options=pycunls.LevenbergMarquardtMinimizerOptions())

- **options** (``LevenbergMarquardtMinimizerOptions``, optional) — LM
  configuration including damping schedule.  When omitted, default options
  are used.

**Methods**

- ``minimize(stream: CudaStream, problem: Problem) -> MinimizerSummary`` —
  same interface as ``GaussNewtonMinimizer.minimize``.  Runs the LM
  iteration instead of pure Gauss-Newton.  State is updated in-place;
  rejected steps are automatically rolled back.

.. _py-problem-label:

--------------------------------------------------------------------------------
``pycunls.Problem``
--------------------------------------------------------------------------------

Assembles a factor graph from state batches and factor batches.  The problem
object is passed to a minimizer's ``minimize`` method.

**Constructor**

.. code-block:: python

   problem = pycunls.Problem()

Creates an empty problem with no states or factors.

**Methods**

- ``add_state_batch(state_batch: StateBatch) -> None`` — registers a state
  batch with the problem.  Every state batch whose blocks are referenced by
  any factor batch must be added here **before** calling ``minimize``.  The
  problem does **not** take ownership; the caller must keep the state batch
  alive for the lifetime of the problem.

- ``add_factor_batch(factor_batch, state_pointers) -> None`` — registers a
  factor batch and binds it to state blocks.  ``state_pointers`` is a flat
  ``list[int]`` of device pointers obtained from
  ``state_batch.state_block_device_ptr(i)``.  For a factor with *K* state
  block inputs and *N* factors, the list must contain *N* × *K* pointers in
  row-major order: ``[factor_0_block_0, factor_0_block_1, ...,
  factor_{N-1}_block_{K-1}]``.

- ``add_factor_batch(factor_batch, loss_function, state_pointers) -> None``
  — same as above, but also attaches a ``LossFunctionBatch`` (see
  :doc:`robustifier`) to robustify the residuals of this factor batch.

- ``check_consistency() -> bool`` — validates that all registered state
  batches and factor batches have matching dimensions and that every
  state-pointer entry belongs to a registered state batch.  Returns
  ``True`` when the graph is valid.  Call this before ``minimize`` to catch
  configuration errors early.

.. _py-enums-label:

--------------------------------------------------------------------------------
``pycunls.SparseLinearSolverType``
--------------------------------------------------------------------------------

Integer enum selecting the linear-system backend.

- ``SparseLinearSolverType.BlockSparsePCG`` (default) — block-Jacobi
  preconditioned conjugate gradient solver.  The block layout is derived
  automatically from the problem's state batches at minimizer-initialize
  time.
- ``SparseLinearSolverType.cuDSS`` — sparse direct solver via NVIDIA cuDSS.
- ``SparseLinearSolverType.DenseLDLT`` — converts CSR to dense and solves
  with a custom CUDA pivoted LDLT kernel.
- ``SparseLinearSolverType.DenseCholesky`` — converts CSR to dense and solves
  via cuSOLVER Cholesky factorization (requires SPD matrix).
- ``SparseLinearSolverType.DenseQR`` — converts CSR to dense and solves via
  cuSOLVER QR factorization (works for any non-singular matrix).

--------------------------------------------------------------------------------
``pycunls.SparseMatrixMultiplierType``
--------------------------------------------------------------------------------

Integer enum selecting the strategy for computing the approximate Hessian
:math:`J^T J`.

- ``SparseMatrixMultiplierType.cuSPARSE`` — cuSPARSE SpGEMM reuse API.
- ``SparseMatrixMultiplierType.Fast`` — warp-efficient CUDA kernels with
  bitmap pattern discovery.

.. _py-minimizer-example-label:

--------------------------------------------------------------------------------
Minimal Python example
--------------------------------------------------------------------------------

.. code-block:: python

   import cupy as cp
   import pycunls

   stream = pycunls.CudaStream()

   state_gpu = cp.array([0.0], dtype=cp.float32)
   obs_gpu   = cp.array([2.0], dtype=cp.float32)

   sb = pycunls.VectorStateBatch1(state_gpu, 1)
   fb = pycunls.PriorVectorFactorBatch1(obs_gpu, 1)

   problem = pycunls.Problem()
   problem.add_state_batch(sb)
   problem.add_factor_batch(fb, [sb.state_block_device_ptr(0)])

   minimizer = pycunls.LevenbergMarquardtMinimizer()
   summary   = minimizer.minimize(stream, problem)

   cp.cuda.runtime.streamSynchronize(stream.get_stream())
   print(summary.final_cost)  # ≈ 0.0
