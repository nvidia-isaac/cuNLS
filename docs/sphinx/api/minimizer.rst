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

For robust estimation with outliers (wrong matches, gross errors), the module
also provides the RANSAC minimizers :code:`RansacGaussNewtonMinimizer`
and :code:`RansacLevenbergMarquardtMinimizer`, which solve the same
:cpp:class:`Problem` while classifying every data factor as inlier or outlier.
See :doc:`../ransac` for the theory and a walkthrough; the reference is in
:ref:`py-ransac-label` (Python), :ref:`ransac-structures-label` and
:ref:`ransac-classes-label` (C++).

This page presents the theory shared by both APIs, then the Python API, then
the C++ API.

- **Python** — ``pycunls``
- **C++** — ``cunls/minimizer``

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

``MinimizerOptions.column_scaling`` (C++: :code:`MinimizerOptions::column_scaling`) can re-scale the normal equations with a
diagonal :math:`S`: the linear solve uses :math:`S H S \, z = S b` with
:math:`H = J^T J` and :math:`b = -J^T r`, then applies the physical tangent step
:math:`\Delta x = S z`. Modes are: no scaling (default); or
:math:`S_{ii} = 1/\sqrt{H_{ii}}` with a small floor for stability, which is
equivalently :math:`1/\|J_{:,j}\|_2` since :math:`H_{jj} = \|J_{:,j}\|_2^2`.
For Levenberg-Marquardt, damping uses the diagonal of the **scaled**
Hessian: :math:`S H S + \lambda \operatorname{diag}(S H S)`.

================================================================================
Python API (``pycunls``)
================================================================================

The Python bindings expose the same minimizer, problem, options, and summary
types as the C++ API (documented in :ref:`minimizer-cpp-api` below) through
the ``pycunls`` package.  All GPU memory is managed via CuPy
arrays; every constructor argument documented as ``DevicePointer`` accepts
either a ``cupy.ndarray`` (the device pointer is extracted automatically) or
a raw ``int`` device address.

The utility type :ref:`CudaStream <py-cuda-stream-label>` is documented in
:doc:`common`.

.. _py-minimizer-options-label:

--------------------------------------------------------------------------------
``pycunls.MinimizerOptions``
--------------------------------------------------------------------------------

Options common to ``GaussNewtonMinimizer`` and ``LevenbergMarquardtMinimizer``
(the latter takes them as ``LevenbergMarquardtMinimizerOptions.base_options``).
Every criterion is applied per subproblem (``Problem.set_problem_partition``); a
problem without a partition is one subproblem.  Create with default values and
then override individual fields.

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
  estimate as converged.  Levenberg-Marquardt adds the rejections its damping
  needs to escalate from ``lambda_min`` to ``lambda_max``, so the cap counts
  the rejections at full damping.  Set to ``0`` to disable this criterion.
- **sparse_linear_solver_type** (``SparseLinearSolverType``, default
  ``BlockSparsePCG``) — selects the linear-system backend.
  ``BlockSparsePCG`` runs block-Jacobi preconditioned conjugate gradient
  with the block layout derived automatically from the problem's state
  batches.  ``cuDSS`` uses NVIDIA's sparse direct solver; ``DenseLDLT``
  converts to dense and factorizes with a custom pivoted LDLT kernel;
  ``DenseCholesky`` converts to dense and uses cuSOLVER Cholesky
  (requires SPD); ``DenseQR`` converts to dense and uses cuSOLVER QR
  factorization (works for any non-singular matrix).
- **reuse_structure** (``bool``, default ``False``) — the problem's structure
  is unchanged since this minimizer's previous ``minimize`` on it (same
  batches, connectivity, active and constant counts, partition; only state
  values, factor data and bounds may differ).  Calls after the first then
  skip the structure setup (index expansion, Hessian pattern, symbolic
  analysis of the linear solver).  A size change falls back to the full
  setup; rewritten index tables at the same sizes must not be combined with
  this option.  Typical use: a real-time loop re-solving the same problem.
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
- **hessian_diagonal** — :math:`S_{ii} = 1 / \sqrt{H_{ii}}` with a numerical
  floor. Equivalently :math:`1 / \|J_{:,j}\|_2`, since
  :math:`H_{jj} = \|J_{:,j}\|_2^2`.

For LM, damping uses the diagonal of the **scaled** Hessian. See the
column-scaling note in the theory section earlier on this page.

.. _py-minimizer-summary-label:

--------------------------------------------------------------------------------
``pycunls.MinimizerSummary``
--------------------------------------------------------------------------------

Returned by ``Minimizer.minimize`` (``GaussNewtonMinimizer`` and
``LevenbergMarquardtMinimizer``).  All fields are read-only
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
  like gradient descent; smaller values start closer to Gauss-Newton.  Must
  not exceed ``lambda_max``.
- **lambda_upscale** (``float``, default ``2.0``) — factor by which
  :math:`\lambda` is *increased* after a rejected step; the :math:`k`-th
  consecutive rejection multiplies by ``lambda_upscale`` :math:`\cdot 2^{k-1}`
  (Nielsen's rule), so the damping escalates quickly when the model is poor.
  Must be greater than 1.
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

.. _py-minimizer-label:

--------------------------------------------------------------------------------
``pycunls.Minimizer``
--------------------------------------------------------------------------------

Common base of ``GaussNewtonMinimizer`` and ``LevenbergMarquardtMinimizer``.
Not constructible; use it to accept either minimizer (for example the inner
minimizer of ``AugmentedLagrangianMinimizer``).  Each iteration builds the
normal equations :math:`J^T J \,\Delta x = -J^T r` at the current states,
lets the subclass adjust them (Levenberg-Marquardt adds its damping), solves
for the step, evaluates the cost at the trial states (with optional line
search), lets the subclass classify the step, and takes or rejects it.  With a
problem partition every decision is taken per subproblem.

**Methods**

- ``minimize(stream: CudaStream, problem: Problem) -> MinimizerSummary`` —
  minimizes the cost starting from the problem's current states.  The state
  memory owned by the state batches inside *problem* is updated **in-place**
  on the GPU (also when the iteration limit is hit).  A problem with
  box-bounded states or constraint batches raises ``ValueError``: solve it
  with ``AugmentedLagrangianMinimizer``.  All GPU work is issued on *stream*, which is
  synchronized before the call returns.  Returns a
  :ref:`MinimizerSummary <py-minimizer-summary-label>` with iteration count
  and cost statistics.

**Properties**

- ``options`` (``MinimizerOptions``, read-only copy) — the options the
  minimizer runs with (for Levenberg-Marquardt the base options, with
  ``max_consecutive_rejected_steps`` widened by the damping's escalation
  room).

.. _py-gauss-newton-label:

--------------------------------------------------------------------------------
``pycunls.GaussNewtonMinimizer``
--------------------------------------------------------------------------------

Gauss-Newton (a :ref:`Minimizer <py-minimizer-label>`): solves the undamped
normal equations and takes every step that lowers the cost.  A step that does
not lower the cost is rejected and the subproblem stops (with line search it
is shortened first).  Converged when :math:`\|\Delta x\|^2` <
``state_tolerance``, the cost < ``cost_tolerance``, or the step does not lower
the cost.

**Constructor**

.. code-block:: python

   minimizer = pycunls.GaussNewtonMinimizer(options=pycunls.MinimizerOptions())

- **options** (``MinimizerOptions``, optional) — solver configuration.  When
  omitted, default options are used.

.. _py-lm-label:

--------------------------------------------------------------------------------
``pycunls.LevenbergMarquardtMinimizer``
--------------------------------------------------------------------------------

Levenberg-Marquardt (a :ref:`Minimizer <py-minimizer-label>`): Gauss-Newton
with adaptive damping, one :math:`\lambda` per subproblem.  Solves
:math:`(J^T J + \lambda\,\mathrm{diag}(J^T J))\,\Delta x = -J^T r` and
adapts :math:`\lambda` from the gain ratio :math:`\rho` (actual over
predicted cost reduction).  More robust than pure Gauss-Newton when the
initial guess is far from the solution.  Rejected steps leave the states
unchanged.

**Constructor**

.. code-block:: python

   minimizer = pycunls.LevenbergMarquardtMinimizer(
       options=pycunls.LevenbergMarquardtMinimizerOptions())

- **options** (``LevenbergMarquardtMinimizerOptions``, optional) — LM
  configuration including damping schedule.  When omitted, default options
  are used.
- **Raises** ``ValueError`` unless ``0 < lambda_min <= lambda_max``.

.. _py-ransac-label:

--------------------------------------------------------------------------------
RANSAC minimizers (``pycunls``)
--------------------------------------------------------------------------------

Python bindings of :ref:`ransac-structures-label` and
:ref:`ransac-classes-label`. Enum values are lowercase.

- ``pycunls.RansacRole`` — ``sampled`` / ``always_on``.
- ``pycunls.RansacScoring`` — ``msac`` / ``inlier_count``.
- ``pycunls.RansacLinearSolverType`` — ``cholesky`` / ``ldlt``.

**pycunls.RansacFactorBatchOptions(role=RansacRole.sampled,
inlier_threshold=1.0)** — attributes **role** (``RansacRole``) and
**inlier_threshold** (``float``, raw residual norm).

**pycunls.RansacMinimizerOptions()** — writable attributes with the C++ names
and defaults: **hypotheses_per_round** (``int``, 256), **max_rounds**
(``int``, 8), **sample_size** (``int``, 0 = automatic), **confidence**
(``float``, 0.999), **early_stop_inlier_ratio** (``float``, 1.0), **seed**
(``int``, 0), **factor_batches** (``list[RansacFactorBatchOptions]``, empty),
**default_inlier_threshold** (``float``, 1.0), **scoring**
(``RansacScoring``, ``msac``), **score_always_on** (``bool``, ``True``),
**require_informative_inliers** (``bool``, ``True``),
**scoring_memory_budget_bytes** (``int``, 64 MiB), **scoring_subset_size**
(``int``, 16384), **scoring_finalists** (``int``, 4),
**hypothesis_iterations** (``int``, 5), **final_iterations** (``int``, 20),
**state_tolerance** (``float``, 1e-10), **cost_tolerance** (``float``, 1e-7),
**linear_solver** (``RansacLinearSolverType``, ``ldlt``).

.. note::

   **factor_batches** is converted to and from a Python list, so assign a
   whole list (``opts.factor_batches = [...]``); appending to the list it
   returns (``opts.factor_batches.append(...)``) has no effect. Nested
   structs such as ``RansacLevenbergMarquardtMinimizerOptions.base_options``
   are returned by reference, so ``lm.base_options.max_rounds = 4`` does
   modify ``lm``.

**pycunls.RansacLevenbergMarquardtMinimizerOptions()** — **base_options**
(``RansacMinimizerOptions``), **initial_lambda** (1e-3), **lambda_upscale**
(2.0), **lambda_downscale** (0.5), **lambda_max** (1e6), **lambda_min**
(1e-6), **step_accept_threshold** (0.25), **lambda_downscale_threshold**
(0.75).

**pycunls.RansacSummary** — subclass of :ref:`MinimizerSummary
<py-minimizer-summary-label>` with read-only **num_rounds**,
**num_hypotheses**, **num_valid_hypotheses**, **num_inliers**,
**inlier_ratio**, **best_score**, **refinement_reverted**.

**pycunls.RansacMinimizer** — common base of the two RANSAC minimizers below;
not constructible, use it to accept either.

- ``minimize(stream: CudaStream, problem: Problem) -> RansacSummary`` — runs
  RANSAC; the estimate is written into the problem's state batches. Releases
  the GIL while running (custom Python factors re-acquire it). Invalid
  configurations raise ``ValueError``.
- ``inlier_mask(residual_batch_index: int) -> numpy.ndarray`` — host copy
  (``uint8``, 1 = inlier) of the mask of a ``sampled`` batch of the problem
  passed to the last ``minimize``, one entry per factor as that batch had in
  that run. Raises ``RuntimeError`` for an out-of-range index, an
  ``always_on`` batch, or before any run.
- ``options`` (``RansacMinimizerOptions``, read-only copy) — the options
  common to all RANSAC minimizers, as constructed.

**pycunls.RansacGaussNewtonMinimizer(options=RansacMinimizerOptions())** — a
``RansacMinimizer`` whose hypotheses and refinement take Gauss-Newton steps.

**pycunls.RansacLevenbergMarquardtMinimizer(options=RansacLevenbergMarquardtMinimizerOptions())**
— a ``RansacMinimizer`` whose hypotheses and refinement take
Levenberg-Marquardt steps, each hypothesis with its own damping.

**Example**

.. code-block:: python

   import pycunls

   opts = pycunls.RansacLevenbergMarquardtMinimizerOptions()
   opts.base_options.factor_batches = [
       pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, 0.01)
   ]
   opts.base_options.seed = 1

   ransac = pycunls.RansacLevenbergMarquardtMinimizer(opts)
   summary = ransac.minimize(stream, problem)   # problem built as usual
   mask = ransac.inlier_mask(0)        # numpy uint8, 1 = inlier
   print(summary.num_inliers, summary.inlier_ratio)

.. _py-problem-label:

--------------------------------------------------------------------------------
``pycunls.Problem``
--------------------------------------------------------------------------------

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``set_num_active_factors(n)`` /
   ``set_num_active_states(n)`` before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

Assembles a factor graph from state batches and factor batches.  The problem
object is passed to a minimizer's ``minimize`` method.

**Constructor**

.. code-block:: python

   problem = pycunls.Problem()

Creates an empty problem with no states or factors.

**Methods**

- ``add_state_batch(state_batch: StateBatch) -> None`` — registers a state
  batch with the problem.  Every state batch whose states are referenced by
  any factor batch must be added here **before** calling ``minimize``.  The
  problem does **not** take ownership; the caller must keep the state batch
  alive for the lifetime of the problem.

- ``add_factor_batch(factor_batch, state_pointers) -> None`` — registers a
  factor batch and binds it to states.  ``state_pointers`` is a flat
  ``list[int]`` of device pointers obtained from
  ``state_batch.state_device_ptr(i)``.  For a factor with *K* state
  inputs and *N* active factors, the list must contain *N* × *K* pointers in
  row-major order: ``[factor_0_state_0, factor_0_state_1, ...,
  factor_{N-1}_state_{K-1}]``.

- ``add_factor_batch(factor_batch, loss_function, state_pointers) -> None``
  — same as above, but also attaches a ``LossFunctionBatch`` (see
  :doc:`robustifier`) to robustify the residuals of this factor batch.

- ``add_factor_batch(factor_batch, *, state_pointer_table, loss_function=None,
  jacobian_mode_override=None) -> None`` — connectivity is a **device** table
  of state pointers (CuPy ``uint64`` array or int pointer) with
  ``capacity * K`` entries; bound once, rewritten in place between solves.
  Keyword-only, so a CuPy array is never mistaken for a host list.

- ``add_factor_batch(factor_batch, slot_state_batches, state_indices,
  loss_function=None, jacobian_mode_override=None) -> None`` — connectivity is
  a **device** table of state indices (CuPy ``int32`` array, ``capacity * K``
  entries): factor *f* reads state ``state_indices[f * K + k]`` of
  ``slot_state_batches[k]``.

- ``set_state_pointers(residual_batch_index, state_pointers) -> None`` —
  replaces the host-list connectivity of a residual batch
  (``num_active_factors * K`` pointers).

- ``validate(stream) -> bool`` — GPU check of every active connection (see
  ``Problem::Validate``); synchronizes the stream.

- ``check_consistency() -> bool`` — validates that all registered state
  batches and factor batches have matching dimensions and that every
  state-pointer entry belongs to a registered state batch.  Returns
  ``True`` when the graph is valid.  Call this before ``minimize`` to catch
  configuration errors early. Every ``minimize`` also runs a quick size check
  and raises ``ValueError`` when no factor batch has active factors
  (``set_num_active_factors`` never called) or a count exceeds its capacity.

- ``set_problem_partition(num_problems, state_problem_ids) -> None`` —
  declares the problem a batch of independent subproblems (e.g. one PnP per
  camera, one pose graph per training sample). ``state_problem_ids`` holds one
  device ``int32`` array per state batch (in the order the batches were added):
  state *s* of batch *b* belongs to subproblem ``state_problem_ids[b][s]``. Every
  factor must connect states of one subproblem (checked at ``minimize``;
  ``ValueError`` otherwise). Both minimizers then run their step control per
  subproblem: each accepts or rejects its own steps, keeps its own
  Levenberg-Marquardt damping and stops at its own convergence, exactly as if
  solved alone, while the linear system is still solved for all of them at
  once. Without a partition one shared decision covers the whole batch: a
  subproblem whose step increases its cost is carried along by the others (or
  holds them back). A subproblem that would stall alone stalls here too (raise
  ``max_consecutive_rejected_steps`` to give hard ones more attempts).
  ``num_problems <= 1`` clears the partition; ``num_problems`` reads it back.

  The linear system stays one solve, so choose the linear solver with the
  batch in mind: ``DenseCholesky`` runs without failure checks by default
  (``disable_safety_checks``), and a factorization that breaks down on one
  ill-conditioned subproblem's block corrupts the step of every subproblem;
  ``DenseLDLT`` / ``DenseQR`` (pivoted) and ``cuDSS`` keep the blocks
  independent. ``BlockSparsePCG`` stops on the residual of the whole batch, so
  a subproblem with a large residual sets the accuracy the others get.

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

   sb = pycunls.VectorStateBatch1(state_gpu, 1)       # capacity 1
   fb = pycunls.PriorVectorFactorBatch1(obs_gpu, 1)
   sb.set_num_active_states(1)                         # active sizes start at 0
   fb.set_num_active_factors(1)

   problem = pycunls.Problem()
   problem.add_state_batch(sb)
   problem.add_factor_batch(fb, [sb.state_device_ptr(0)])

   minimizer = pycunls.LevenbergMarquardtMinimizer()
   summary   = minimizer.minimize(stream, problem)

   cp.cuda.runtime.streamSynchronize(stream.get_stream())
   print(summary.final_cost)  # ≈ 0.0

.. _minimizer-cpp-api:

================================================================================
C++ API
================================================================================

--------------------------------------------------------------------------------
Structures
--------------------------------------------------------------------------------

.. _cunls-minimizer-summary-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`MinimizerSummary`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returned by :cpp:func:`Minimizer::Minimize`. Holds solve statistics for
inspecting iteration count and cost history. Costs are totals over all
subproblems.

- **num_iterations** [out]: Iterations performed (each builds and solves one
  linear system, including the last one that converged).
- **initial_cost** [out]: Cost of the states passed in.
- **final_cost** [out]: Cost of the states written back to the problem.
- **iteration_costs** [out]: Cost at the start of each iteration (for plotting
  or debugging).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`MinimizerOptions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Options common to :code:`GaussNewtonMinimizer` and
:code:`LevenbergMarquardtMinimizer` (header :code:`cunls/minimizer/minimizer.h`):
iteration limit, convergence tolerances, consecutive rejected-step limit,
sparse linear solver choice, bounds, line search and structure reuse. Every
criterion is applied per subproblem.

- **max_num_iterations** [in]: Maximum number of iterations. Default: 50.
- **state_tolerance** [in]: Convergence threshold on squared step norm; optimizer
  terminates when the step norm falls below this. Default: 1e-6.
- **cost_tolerance** [in]: Convergence threshold on cost; optimizer terminates
  when the cost falls below this. Default: 1e-6.
- **max_consecutive_rejected_steps** [in]: Maximum number of consecutive
  rejected steps before declaring convergence. When every trial step is rejected
  (cost increases or step quality below acceptance threshold) this many times
  in a row, the minimizer treats the current solution as converged
  (Levenberg-Marquardt: counted at full damping, after the rejections that
  escalate :math:`\lambda` to ``lambda_max``). Set to 0 to disable. Default: 5.
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
- **column_scaling** [in]: Diagonal scaling of the normal equations; see the
  column-scaling note above. Values: ``None``, ``HessianDiagonal``.
  Default: ``None``.
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
- **jacobian_mode** [in]: Global default :code:`JacobianMode` used to
  evaluate every factor batch's Jacobian, unless overridden per group via
  :cpp:func:`Problem::AddFactorBatch`. See
  :ref:`minimizer-jacobian-mode-label` below and :doc:`../numeric_jacobians`
  for the full picture. Default: ``kAnalytic``.
- **numeric_diff_options** [in]: Tuning knobs (finite-difference scheme, step
  size) used whenever a factor batch is evaluated with
  ``JacobianMode::kNumeric``; see :code:`NumericDiffOptions` below. Ignored
  for factor batches evaluated with ``kAnalytic``.

.. _minimizer-jacobian-mode-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`JacobianMode` / :code:`NumericDiffOptions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/minimizer/jacobian_mode.h`. Selects, per factor batch,
whether its Jacobian comes from the factor's own hand-derived
:cpp:func:`FactorBatch::Evaluate` output, or from cuNLS differentiating the
factor's residual numerically. See :doc:`../numeric_jacobians` for a full
walkthrough (manifold-aware perturbation, accuracy/performance tradeoffs,
worked example).

:code:`JacobianMode` (enum):

- **kAnalytic**: Use the factor batch's own Jacobian output. Default.
- **kNumeric**: Ignore any Jacobian the factor batch would compute; instead
  perturb each referenced state along its manifold tangent space (via
  :cpp:func:`StateBatch::Plus`) and finite-difference the residual. Requires
  only that the factor batch support residual-only evaluation
  (``jacobians == nullptr``), which every :cpp:class:`FactorBatch` must
  already do.

:code:`NumericDiffOptions` (struct, only consulted when a factor batch
resolves to ``kNumeric``):

- **method** [in]: ``kForward`` (one-sided, :math:`(f(x+\epsilon)-f(x))/\epsilon`,
  cheaper and less accurate) or ``kCentral`` (two-sided,
  :math:`(f(x+\epsilon)-f(x-\epsilon))/(2\epsilon)`). Default: ``kCentral``.
- **relative_step_size** [in]: Per-tangent-coordinate perturbation step
  :math:`\epsilon`. Default: 1e-4.

Where the mode for a given factor batch comes from:

.. cpp:function:: JacobianMode Problem::JacobianModeFor(size_t residual_batch_index, JacobianMode global_default) const

  :param ``residual_batch_index``: [in] Index into :cpp:func:`Problem::GetResidualBatches`.
  :param ``global_default``: [in] Typically :code:`MinimizerOptions::jacobian_mode`.
  :returns: [out] The per-group override passed to :cpp:func:`Problem::AddFactorBatch`, if one was given; otherwise ``global_default``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`LevenbergMarquardtMinimizerOptions`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Options for the **Levenberg-Marquardt** minimizer. Extends
:code:`MinimizerOptions` with damping and step-acceptance parameters. Used when
constructing a :code:`LevenbergMarquardtMinimizer`.

- **base_options** [in]: Base Gauss-Newton options (:code:`MinimizerOptions`).
- **initial_lambda** [in]: Initial damping coefficient; at most ``lambda_max``.
  Default: 1e-3.
- **relative_reduction_tolerance** [in]: Convergence threshold on predicted
  relative cost reduction. Default: 1e-6.
- **lambda_upscale** [in]: Factor by which :math:`\lambda` is increased after a
  rejected step, times :math:`2^{k-1}` at the :math:`k`-th consecutive rejection;
  greater than 1. Default: 2.0.
- **lambda_downscale** [in]: Factor by which :math:`\lambda` is decreased after a
  very successful step. Default: 0.5.
- **lambda_max** [in]: Upper bound for :math:`\lambda`. Default: 1e+6.
- **lambda_min** [in]: Lower bound for :math:`\lambda`. Default: 1e-6.
- **step_accept_threshold** [in]: Minimum step quality (rho, actual/predicted
  cost reduction) to accept a step. Default: 0.25.
- **lambda_downscale_threshold** [in]: Step quality above which :math:`\lambda`
  is decreased. Default: 0.75.

.. _ransac-structures-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RANSAC structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Header: :code:`cunls/minimizer/ransac_minimizer.h` (included by
:code:`cunls/cunls.h`). See :doc:`../ransac` for how each option is used.

**kMaxRansacTangentDim** — ``constexpr int kMaxRansacTangentDim = 64``. The
sum of ``TangentSize()`` over every **non-constant** state of the problem
(the free tangent dimension :math:`D`) must not exceed it. Constant states do
not count, whatever their number.

:code:`RansacRole` (enum) — role of a residual batch:

- ``kSampled`` (0): data factors that may be outliers. Minimal samples are
  drawn from them and each factor is classified as inlier / outlier.
- ``kAlwaysOn`` (1): trusted factors (priors, motion priors, extrinsic
  constraints). Included in every hypothesis solve and in the final
  refinement; never classified.

:code:`RansacScoring` (enum) — hypothesis scoring rule (lower is better):

- ``kMSAC`` (0): truncated quadratic
  :math:`\sum_i \min(\|r_i\|^2, \tau_i^2)` over the sampled factors.
- ``kInlierCount`` (1): number of inliers; ties broken by the lower MSAC score.

:code:`RansacLinearSolverType` (enum) — dense per-hypothesis solver:

- ``kCholesky`` (0): in-kernel Cholesky; a non-positive pivot marks the
  hypothesis invalid.
- ``kLDLT`` (1): in-kernel :math:`LDL^T` with symmetric diagonal pivoting;
  rank-deficient directions get a zero step instead of failing (suits the
  near-singular systems of minimal samples).

:code:`RansacFactorBatchOptions` — configuration of one residual batch:

- **role** [in]: :code:`RansacRole`. Default: ``kSampled``.
- **inlier_threshold** [in]: :math:`\tau` on the raw (loss-free) residual
  norm, in the batch's residual units; a factor is an inlier iff
  :math:`\|r\|^2 \le \tau^2`. Ignored for ``kAlwaysOn`` batches. Default: 1.0.

:code:`RansacMinimizerOptions` — options common to all RANSAC minimizers:

- **hypotheses_per_round** [in]: Hypotheses :math:`K` generated and scored
  together (in parallel) in one round. Default: 256.
- **max_rounds** [in]: Upper bound on the number of rounds. Default: 8.
- **sample_size** [in]: Sampled factors per minimal sample; ``0`` selects
  :math:`\lceil D / m_{\min} \rceil` (:math:`m_{\min}` = smallest residual
  dimension among sampled batches). Default: 0.
- **confidence** [in]: Target probability of drawing at least one all-inlier
  sample; drives adaptive stopping between rounds. Default: 0.999.
- **early_stop_inlier_ratio** [in]: Stop once the best inlier ratio reaches
  this value; 1 disables. Default: 1.0.
- **seed** [in]: Seed of the counter-based sampler; the same seed gives a
  bitwise identical result. Default: 0.
- **factor_batches** [in]: One :code:`RansacFactorBatchOptions` per residual
  batch, indexed like :cpp:func:`Problem::GetResidualBatches` (the order the
  batches were added). Empty means every batch is ``kSampled`` with
  **default_inlier_threshold**. Default: empty.
- **default_inlier_threshold** [in]: Threshold used when **factor_batches** is
  empty. Default: 1.0.
- **scoring** [in]: :code:`RansacScoring`. Default: ``kMSAC``.
- **score_always_on** [in]: Add 2 × (cost of the ``kAlwaysOn`` factors) to
  each hypothesis score. Default: true.
- **require_informative_inliers** [in]: Count a factor as an inlier only if its
  Jacobian has a non-zero entry on a free state. Guards against factors
  that report a zero residual for configurations they cannot evaluate (e.g.
  :code:`PnPFactorBatch` for points behind the camera). Costs one Jacobian
  evaluation per scored factor. Default: true.
- **scoring_memory_budget_bytes** [in]: Device memory budget for scoring
  buffers; bounds how many hypotheses are scored per chunk. Default: 64 MiB.
- **scoring_subset_size** [in]: Two-stage scoring for large problems: with
  more than 2 × this many sampled factors, every hypothesis is first scored on
  a random subset of this size (drawn anew each round), and only the best
  **scoring_finalists** are scored on all factors. 0 disables. Default: 16384.
- **scoring_finalists** [in]: Hypotheses scored on all factors in two-stage
  scoring (at most 64). Default: 4.
- **hypothesis_iterations** [in]: GN / LM iterations that turn a minimal
  sample into a hypothesis. Default: 5.
- **final_iterations** [in]: Iterations of the final refinement on the best
  inlier set. Default: 20.
- **state_tolerance** [in]: Per-hypothesis convergence on the squared step
  norm. Default: 1e-10.
- **cost_tolerance** [in]: Per-hypothesis convergence on the relative cost
  decrease. Default: 1e-7.
- **linear_solver** [in]: :code:`RansacLinearSolverType`. Default: ``kLDLT``.

:code:`RansacLevenbergMarquardtMinimizerOptions` — options of
:code:`RansacLevenbergMarquardtMinimizer`; every hypothesis carries its own
damping :math:`\lambda`:

- **base_options** [in]: :code:`RansacMinimizerOptions`.
- **initial_lambda** [in]: Damping each hypothesis starts from. Default: 1e-3.
- **lambda_upscale** [in]: Damping multiplier on a rejected step. Default: 2.0.
- **lambda_downscale** [in]: Damping multiplier on a very successful step.
  Default: 0.5.
- **lambda_max** [in]: Upper bound; a hypothesis that exceeds it stops.
  Default: 1e6.
- **lambda_min** [in]: Lower bound. Default: 1e-6.
- **step_accept_threshold** [in]: Accept a step if actual / predicted cost
  reduction is at least this. Default: 0.25.
- **lambda_downscale_threshold** [in]: Decrease damping if actual / predicted
  reduction exceeds this. Default: 0.75.

:code:`RansacSummary` — result of a RANSAC run; extends
:code:`MinimizerSummary` (whose **num_iterations** and **iteration_costs**
describe the final refinement, **initial_cost** is over all factors at the
initial guess and **final_cost** is the refined cost over the inliers and the
``kAlwaysOn`` factors):

- **num_rounds** [out]: Rounds executed.
- **num_hypotheses** [out]: Hypotheses generated across all rounds.
- **num_valid_hypotheses** [out]: Hypotheses whose first linear solve
  succeeded.
- **num_inliers** [out]: Inliers of the final estimate over all ``kSampled``
  factors.
- **inlier_ratio** [out]: **num_inliers** / number of ``kSampled`` factors.
- **best_score** [out]: Score (lower is better) of the final estimate.
- **refinement_reverted** [out]: True if the final refinement worsened the
  score and the best hypothesis (before refinement) was returned instead.

--------------------------------------------------------------------------------
Class APIs
--------------------------------------------------------------------------------

.. _minimizer-class-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Minimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Common base of :code:`GaussNewtonMinimizer` and
:code:`LevenbergMarquardtMinimizer` (header :code:`cunls/minimizer/minimizer.h`).
Holds everything the two share: the iteration, the linear system, line
search, structure reuse and the per-subproblem bookkeeping. Not
instantiable on its own; use a :code:`Minimizer&` to accept either. Each
iteration:

1. builds the normal equations :math:`H \Delta x = -g` at the current states
   (with column scaling),
2. lets the subclass update them (``UpdateSystem``: Levenberg-Marquardt adds
   :math:`\lambda_p \operatorname{diag}(H)` per subproblem :math:`p`),
3. solves for the step,
4. evaluates the cost at the trial states, shortening the step by line search
   if enabled,
5. lets the subclass classify each subproblem's step (``ClassifySteps``:
   reject, converged),
6. takes or rejects each step and stops each subproblem that converged or hit
   the rejection cap.

With a problem partition (:cpp:func:`Problem::SetProblemPartition`) the
linear system is solved for all subproblems together and every decision in
steps 4 to 6 is taken per subproblem; a problem without a partition is one
subproblem. Each iteration makes one small read-back to the host (total cost,
number of running subproblems), plus one per line-search step.

.. cpp:function:: MinimizerSummary Minimizer::Minimize(cudaStream_t stream, Problem& problem)

  Minimizes the problem's cost, starting from its current states.

  :param ``stream``: [in] CUDA stream for all device work; synchronized before
    the call returns.
  :param ``problem``: [in,out] Problem (factor graph + state batches); its
    states are updated in place, also when the iteration limit is hit.
  :returns: [out] :cpp:class:`MinimizerSummary` with iteration count and cost statistics.

  Throws ``std::invalid_argument`` if the problem has constraint factor batches
  or box-bounded states (solve those with :code:`AugmentedLagrangianMinimizer`)
  or invalid sizes,
  connectivity or partition; ``std::runtime_error`` if the linear solver fails.

  **Note:** A minimizer instance retains working buffers (normal-equation matrix,
  RHS, and internal state snapshots) across calls; when the problem size is
  unchanged, device memory is reused instead of reallocated. With
  ``MinimizerOptions::reuse_structure`` the structure setup is skipped too.

.. cpp:function:: const MinimizerOptions& Minimizer::Options() const

  :returns: [out] The options the minimizer runs with (for Levenberg-Marquardt
    the base options, with ``max_consecutive_rejected_steps`` widened by the
    damping's escalation room).

.. _gauss-newton-minimizer-ctor-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`GaussNewtonMinimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** A :code:`Minimizer` that solves the undamped normal equations
:math:`J^T J \Delta x = -J^T r` and takes every step that lowers the cost. Per
subproblem, a step is rejected if it does not lower the cost, and the
subproblem has converged if :math:`\|\Delta x\|^2 <` ``state_tolerance``,
the trial cost is below ``cost_tolerance``, or the step does not lower the
cost.

.. cpp:function:: explicit GaussNewtonMinimizer(const MinimizerOptions& options = MinimizerOptions())

  :param ``options``: [in] Solver options (max iterations, tolerances, linear solver); copied into the minimizer.

.. _lm-minimizer-ctor-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`LevenbergMarquardtMinimizer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** A :code:`Minimizer` that solves the damped system
:math:`(J^T J + \lambda \operatorname{diag}(J^T J)) \Delta x = -J^T r` with
one :math:`\lambda` per subproblem. With :math:`\rho` the actual over the
predicted cost reduction (:math:`\tfrac12 \Delta x^T H \Delta x + \lambda
\Delta x^T D \Delta x`): :math:`\rho \ge` ``step_accept_threshold`` takes the
step (and shrinks :math:`\lambda` by ``lambda_downscale`` when
:math:`\rho >` ``lambda_downscale_threshold``); otherwise the step is rejected
and the k-th consecutive rejection multiplies :math:`\lambda` by
``lambda_upscale`` :math:`\cdot 2^{k-1}`. Converged when
:math:`\|\Delta x\|^2 <` ``state_tolerance``, the predicted relative reduction
is below ``relative_reduction_tolerance``, or the trial cost is below
``cost_tolerance``. :math:`\lambda` stays in
[``lambda_min``, ``lambda_max``] and starts at ``initial_lambda`` on every call.
More robust than Gauss-Newton when the initial guess is far from the solution.

.. cpp:function:: explicit LevenbergMarquardtMinimizer(const LevenbergMarquardtMinimizerOptions& options = LevenbergMarquardtMinimizerOptions())

  :param ``options``: [in] LM options (damping, accept/reject thresholds, etc.).

  Throws ``std::invalid_argument`` unless ``0 < lambda_min <= lambda_max``.

.. _ransac-classes-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`RansacMinimizer` and its subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** RANSAC over an ordinary :cpp:class:`Problem`.
:code:`RansacMinimizer` is the common base (not instantiable; use a
:code:`RansacMinimizer&` to accept either); :code:`RansacGaussNewtonMinimizer`
and :code:`RansacLevenbergMarquardtMinimizer` decide how a hypothesis
iterates. Minimal samples of
``kSampled`` factors are turned into hypotheses by a few GN (or LM) iterations
from the current state values; every hypothesis is scored against all
``kSampled`` factors; the best one is refined on its inliers and written back
into the problem's state batches. :cpp:func:`RansacMinimizer::InlierMask` then exposes the
classification. See :doc:`../ransac`.

The problem is built exactly as for :code:`Minimizer`. The only
restriction is the free tangent dimension (``kMaxRansacTangentDim``); any
number of state batches of any supported types, and any number of factor
batches and factors, are allowed. Every factor and state batch must honor the
item parameters of :cpp:func:`FactorBatch::Evaluate` and the ``num_replicas``
parameter of :cpp:func:`StateBatch::Plus`; all built-in batches do, and
:doc:`../custom_factors_and_states` shows how to write custom ones.

.. cpp:function:: explicit RansacGaussNewtonMinimizer(const RansacMinimizerOptions& options = RansacMinimizerOptions())

  :param ``options``: [in] RANSAC options; copied into the minimizer.
  :returns: [out] Constructor has no return value.

.. cpp:function:: explicit RansacLevenbergMarquardtMinimizer(const RansacLevenbergMarquardtMinimizerOptions& options = RansacLevenbergMarquardtMinimizerOptions())

  :param ``options``: [in] Shared RANSAC options (``base_options``) plus the
    per-hypothesis damping policy. Hypotheses and refinement use LM; otherwise
    identical to :code:`RansacGaussNewtonMinimizer`.
  :returns: [out] Constructor has no return value.

.. cpp:function:: RansacSummary RansacMinimizer::Minimize(cudaStream_t stream, Problem& problem)

  Runs RANSAC and writes the refined estimate into the problem's state batches.

  :param ``stream``: [in] CUDA stream for all work. The call synchronizes it a
    few times (once per round and at the end) to read statistics.
  :param ``problem``: [in,out] The problem; its current state values are the
    initial guess for every hypothesis and receive the result.
  :returns: [out] :code:`RansacSummary`.

  Throws ``std::invalid_argument`` (with a message saying what to change) for
  an unsupported configuration: free tangent dimension :math:`D > 64` or
  :math:`D = 0`; no ``kSampled`` factors; fewer sampled factors than the
  sample size; a factor batch that requests numeric Jacobians; a
  **factor_batches** vector whose size does not match the problem's residual
  batches; invalid options (e.g. ``hypotheses_per_round == 0``,
  ``max_rounds == 0``, ``hypothesis_iterations == 0``, ``confidence``
  outside :math:`(0, 1)`).

  **Note:** The minimizer keeps its device buffers between calls and reuses
  them when the problem size is unchanged.

.. cpp:function:: const uint8_t* RansacMinimizer::InlierMask(size_t residual_batch_index) const

  :param ``residual_batch_index``: [in] Index into
    :cpp:func:`Problem::GetResidualBatches`.
  :returns: [out] Device pointer to one byte per factor of that batch
    (1 = inlier) for the estimate of the last :cpp:func:`Minimize`, valid until
    the next :cpp:func:`Minimize` or destruction; ``nullptr`` for ``kAlwaysOn``
    batches, an out-of-range index, or before any run.

.. cpp:function:: size_t RansacMinimizer::InlierMaskSize(size_t residual_batch_index) const

  :param ``residual_batch_index``: [in] Index into
    :cpp:func:`Problem::GetResidualBatches`.
  :returns: [out] Number of bytes of :cpp:func:`RansacMinimizer::InlierMask`:
    the factor count the batch had in the last :cpp:func:`Minimize`; ``0``
    whenever ``InlierMask`` returns ``nullptr``.

.. cpp:function:: const RansacMinimizerOptions& RansacMinimizer::Options() const

  :returns: [out] The options common to all RANSAC minimizers, as constructed.

**Example**

.. code-block:: cpp

   cunls::RansacLevenbergMarquardtMinimizerOptions options;
   // One entry per residual batch, in the order they were added.
   options.base_options.factor_batches = {{cunls::RansacRole::kSampled, 0.01f}};
   cunls::RansacLevenbergMarquardtMinimizer ransac(options);
   cunls::RansacSummary summary = ransac.Minimize(stream, problem);
   const uint8_t* inliers = ransac.InlierMask(0);  // device, one byte per factor

.. _problem-add-factor-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::AddFactorBatch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``set_num_active_factors(n)`` /
   ``set_num_active_states(n)`` (C++: ``SetNumActiveFactors`` /
   ``SetNumActiveStates``) before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

**Purpose:** Registers a factor batch (and optionally a robust loss batch) with
the problem and binds its factor instances to state pointers. The
ordering of :code:`state_pointers` must match the factor batch’s expected
state layout (see :doc:`factor`).

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, const std::vector<float*>& state_pointers, std::optional<JacobianMode> jacobian_mode_override = std::nullopt)

  :param ``factor_batch``: [in] Factor batch pointer (non-owning).
  :param ``state_pointers``: [in] Flattened device pointers: one per (factor index, state), mapping factors to state, at least ``NumActiveFactors() * B`` entries (B = ``StateSizes().size()``). The problem keeps the list and copies it once into a library-owned device table sized for the batch's capacity; replace it later with :cpp:func:`SetStatePointers`.
  :param ``jacobian_mode_override``: [in] When set, this factor batch always uses the given :code:`JacobianMode` regardless of the minimizer's :code:`MinimizerOptions::jacobian_mode` default; see :ref:`minimizer-jacobian-mode-label`. Default: ``std::nullopt`` (use the minimizer's global default).
  :returns: [out] No return value.

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function_batch, const std::vector<float*>& state_pointers, std::optional<JacobianMode> jacobian_mode_override = std::nullopt)

  :param ``factor_batch``: [in] Factor batch pointer (non-owning).
  :param ``loss_function_batch``: [in] Robust loss batch pointer (non-owning).
  :param ``state_pointers``: [in] Flattened state pointer mapping for all factors in the batch (stored as above).
  :param ``jacobian_mode_override``: [in] Same meaning as the other overload.
  :returns: [out] No return value.

**Device connectivity tables.** For problems rewritten every solve, the
connectivity can be a user-owned device table bound once and rewritten in place
(ordered before :cpp:func:`Minimize` on the GPU). Only the first
``NumActiveFactors() * B`` entries are read. See :ref:`capacity-and-active-count`.

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function_batch, float* const* device_state_pointers, std::optional<JacobianMode> jacobian_mode_override = std::nullopt)

  :param ``loss_function_batch``: [in] Robust loss batch, or ``nullptr``. An overload without this argument exists.
  :param ``device_state_pointers``: [in] Device array of ``Capacity() * B`` state pointers; entry ``f * B + b`` is the state factor ``f`` reads in slot ``b``. Not owned; must outlive the problem.
  :returns: [out] No return value.

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function_batch, const std::vector<StateBatch*>& slot_state_batches, const int* device_state_indices, std::optional<JacobianMode> jacobian_mode_override = std::nullopt)

  :param ``slot_state_batches``: [in] State batch of each state slot (B entries; registered with :cpp:func:`AddStateBatch`).
  :param ``device_state_indices``: [in] Device array of ``Capacity() * B`` ints: factor ``f`` reads state ``device_state_indices[f * B + b]`` of ``slot_state_batches[b]``. Indices must be below that batch's ``NumActiveStates()``. Not owned; must outlive the problem. An overload without the loss argument exists.
  :returns: [out] No return value.

**Example** (index table, rewritten every frame):

.. code-block:: cpp

   cunls::dvector<int> obs_indices(2 * max_observations);  // [pose, point] per factor
   problem.AddFactorBatch(&reprojection, {&pose_states, &point_states}, obs_indices.data());
   // every frame: write obs_indices[0 .. 2 * num_observations) on `stream`, then
   reprojection.SetNumActiveFactors(num_observations);
   minimizer.Minimize(stream, problem);

.. _problem-add-state-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::AddStateBatch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Registers a state batch with the problem. State batches supply the
manifold Plus operation and state pointers used when building
:code:`state_pointers` for :cpp:func:`Problem::AddFactorBatch` and when
applying steps during :cpp:func:`Minimize`.

.. cpp:function:: void AddStateBatch(StateBatch* state_batch)

  :param ``state_batch``: [in] State batch pointer (non-owning).
  :returns: [out] No return value.

.. _problem-check-consistency-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::CheckConsistency`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Verifies that the problem’s factor batches, state batches, and
state pointer mappings are consistent (e.g. correct dimensions and
connectivity). Call before :cpp:func:`Minimize` to catch configuration errors.

.. cpp:function:: bool CheckConsistency() const

  :returns: [out] ``true`` when graph inputs and connectivity are valid. Fails
    as well when no factor batch has active factors. Device tables are checked
    on the GPU (:cpp:func:`Validate`).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::SetProblemPartition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Declares the problem a batch of independent subproblems, so the
minimizers accept, damp and stop each one on its own (on the device; one
2-float read-back per iteration). See the Python
``set_problem_partition`` above for the semantics.

.. cpp:function:: void SetProblemPartition(size_t num_problems, const std::vector<const int *> &state_problem_ids)

  :param ``num_problems``: [in] Number of subproblems; 0 or 1 clears the partition.
  :param ``state_problem_ids``: [in] One device array per state batch (not owned,
    read at every solve): the subproblem of each state.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::SetStatePointers`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Replaces the host-list connectivity of a residual batch (copied
into its device table, synchronously), typically after its active count
changed.

.. cpp:function:: void SetStatePointers(size_t residual_batch_index, const std::vector<float*>& state_pointers)

  :param ``residual_batch_index``: [in] Index into :cpp:func:`GetResidualBatches`; the batch must have been registered with a host list (device tables are rewritten directly).
  :param ``state_pointers``: [in] ``NumActiveFactors() * B`` state pointers, at most ``Capacity() * B``.
  :returns: [out] No return value. Throws ``std::logic_error`` for device-table batches.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::CheckSizes`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Quick host-only check of the active sizes (loops over batches;
no device work). **Every minimizer calls it at the start of** :cpp:func:`Minimize`.

.. cpp:function:: void CheckSizes() const

  Throws ``std::invalid_argument`` with an actionable message when no factor
  batch has active factors (sizes never set), a factor or state count exceeds
  its capacity, the active constant ids exceed their capacity or the active
  states, or a connectivity table does not cover the active factors. Warns
  about residual batches with no active factors.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`Problem::Validate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Full GPU check of every active connection, for connectivity that
is rewritten on the device. Opt-in (one kernel per table and one readback).

.. cpp:function:: bool Validate(cudaStream_t stream) const

  Checks that every active pointer lies in an active state of a registered
  state batch with the slot's tangent size (index tables: every index is below
  the slot batch's ``NumActiveStates()``), that every active constant id is
  below ``NumActiveStates()``, that every active state is read by some
  factor, and that state batches do not overlap.

  :param ``stream``: [in] CUDA stream; synchronized before returning.
  :returns: [out] ``true`` when the problem is well formed; the first failure is logged.

**Advanced:** :cpp:func:`PrepareStatePointers` (expands index tables, called
by the minimizers at the start of every solve), :cpp:func:`DeviceStatePointers`
(device table of a residual batch), :cpp:func:`NumStatePointers`
(``NumActiveFactors() * B``) and :cpp:func:`HostStatePointers` (host view; downloads
device tables, which synchronizes).

.. _residual-batch-evaluate-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`ResidualBatch::Evaluate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Evaluates the residual batch: computes residuals (and optionally
Jacobians) from the factor batch and applies the optional loss function to
produce robustified residuals and scaled Jacobians used by the minimizer.
Callers supply device scratch for per-factor squared norms and :math:`\rho`
triplets; see ``ResidualBatchWorkspaceSizeBytes`` / ``ResidualBatchWorkspaceNumFloats``.

.. cpp:function:: size_t ResidualBatchWorkspaceSizeBytes(size_t num_residuals)

  :param ``num_residuals``: [in] Number of factors (``NumActiveFactors()`` for the batch).
  :returns: [out] Minimum device scratch size in bytes for ``Evaluate``\ 's ``workspace`` pointer.

.. cpp:function:: size_t ResidualBatchWorkspaceNumFloats(size_t num_residuals)

  :param ``num_residuals``: [in] Number of factors (``NumActiveFactors()`` for the batch).
  :returns: [out] Same scratch as ``ResidualBatchWorkspaceSizeBytes``, expressed in ``float`` elements (rounded up), for sub-allocating inside a ``float`` arena.

.. cpp:function:: bool Evaluate(cudaStream_t stream, float* workspace, float* residuals, float const* const* state_pointers, float* cost, float* jacobians) const

  :param ``stream``: [in] CUDA stream for factor, loss, and scaling kernels.
  :param ``workspace``: [in] Device scratch; size at least ``ResidualBatchWorkspaceSizeBytes(NumActiveFactors())`` bytes (see layout in the header). Must not overlap ``residuals``, ``jacobians``, or ``cost``.
  :param ``residuals``: [out] Residual output buffer of length ``NumActiveFactors() * ResidualsSize()`` (after loss scaling if present).
  :param ``state_pointers``: [in] Device pointer array mapping factor inputs to states.
  :param ``cost``: [out] Optional per-residual cost (e.g. :math:`\frac{1}{2}\rho(\|r\|^2)`); can be ``nullptr``.
  :param ``jacobians``: [out] Optional Jacobian output (after loss scaling); can be ``nullptr``.
  :returns: [out] ``true`` on successful evaluation.

.. _minimizer-state-copy-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:code:`MinimizerState::Copy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _hessian-assembly-label:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hessian assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normal equations are assembled directly from the per-factor Jacobian blocks
that each factor batch writes; no global sparse Jacobian is ever materialized.

:code:`HessianStructureBuilder` (``cunls/minimizer/hessian_structure.h``) derives
the sparsity pattern from factor-graph connectivity on the GPU: it resolves each
factor's state pointers to global columns, packs every candidate block pair into
a 64-bit key, then sorts and segments. It also returns, for each
``(factor, block_a, block_b)`` slot, the row-relative offset at which that tile
starts — the map the assembler scatters through.

:code:`BlockHessianAssembler` (``cunls/minimizer/block_hessian_assembler.h``)
runs one kernel per residual batch, one warp per factor. Each warp stages
:math:`J_f` and :math:`r_f` in shared memory, forms
:math:`H_f = J_f^T J_f` and :math:`b_f = -J_f^T r_f`, and scatter-adds both into
the global system.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hessian storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hessian of a factor graph is block structured, so it is stored as BSR — one
column index per dense tile instead of one per scalar entry. That is bandwidth
the iterative solver's SpMV no longer has to move.

Both the Hessian and the working left-hand side are owned by
:code:`NormalEquations` (``cunls/minimizer/normal_equations.h``), which is the
only place either layout is named; the minimizers work in terms of "the Hessian"
and "the left-hand side" and never branch on storage.

The layout is chosen automatically, with no user-facing switch. Block storage
requires a tile edge dividing every state's tangent dimension (the gcd of
the tangent sizes; see :code:`ChooseHessianBlockSize` in
``cunls/minimizer/bsr_matrix.h``) **and** a solver that reports
:code:`SparseLinearSolver::SupportsBlockStorage`. When either does not hold —
a gcd of one, or a backend such as cuDSS that needs CSR anyway — the minimizer
falls back to scalar CSR with no behavioural change. No conversion is ever
performed on the solver path; the fallback is assembled natively in CSR.

The dense factorizations (:code:`DenseLDLT`, :code:`DenseCholesky`,
:code:`DenseQR`) accept either layout. They scatter the coefficient matrix into
an ``n x n`` dense buffer and never consult the sparse form again, so the layout
is invisible to them past the first kernel. cuDSS is the only backend that
declines block storage.
