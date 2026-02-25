################################################################################
Minimizer API
################################################################################

The ``cunls/minimizer`` module provides iterative solvers for non-linear least
squares. It implements **Gauss-Newton** and **Levenberg-Marquardt** algorithms,
which repeatedly linearize the residuals, solve a linear least-squares system
for the step in tangent space, and apply the step via the manifold :math:`\oplus`
operation. See `Gauss–Newton algorithm
<https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_ and
`Levenberg–Marquardt algorithm
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ for
background.

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
tolerances, and sparse linear solver choice. Used when constructing a :code:`GaussNewtonMinimizer`.

- **max_num_iterations** [in]: Maximum number of iterations.
- **state_tolerance** [in]: Convergence threshold on step norm (squared).
- **cost_tolerance** [in]: Convergence threshold on cost change.
- **sparse_linear_solver_type** [in]: Sparse backend (e.g. Cholesky, iterative).
- **sparse_linear_solver_config** [in]: Backend-specific options.

--------------------------------------------------------------------------------
:code:`LevenbergMarquardtMinimizerOptions`
--------------------------------------------------------------------------------

Options for the **Levenberg-Marquardt** minimizer. Extends
:code:`MinimizerOptions` with damping and step-acceptance parameters. Used when
constructing a :code:`LevenbergMarquardtMinimizer`.

- **base_options** [in]: Base Gauss-Newton options.
- **initial_lambda** [in]: Initial damping coefficient.
- **relative_reduction_tolerance** [in]: Predicted reduction threshold.
- **lambda_upscale** [in]: Factor by which :math:`\lambda` is increased after a rejected step.
- **lambda_downscale** [in]: Factor by which :math:`\lambda` is decreased after a very successful step.
- **lambda_max** [in]: Upper bound for :math:`\lambda`.
- **lambda_min** [in]: Lower bound for :math:`\lambda`.
- **step_accept_threshold** [in]: Minimum step quality (e.g. rho) to accept a step.
- **lambda_downscale_threshold** [in]: Step quality above which :math:`\lambda` is decreased.

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
  :param ``state_pointers``: [in] Flattened device pointers: one per (factor index, state block), mapping factors to state.
  :returns: [out] No return value.

.. cpp:function:: void AddFactorBatch(FactorBatch* factor_batch, LossFunctionBatch* loss_function_batch, const std::vector<float*>& state_pointers)

  :param ``factor_batch``: [in] Factor batch pointer (non-owning).
  :param ``loss_function_batch``: [in] Robust loss batch pointer (non-owning).
  :param ``state_pointers``: [in] Flattened state pointer mapping for all factors in the batch.
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

.. cpp:function:: bool Evaluate(float* cost, float* residuals, float* jacobians, float const* const* state_pointers, cudaStream_t stream) const

  :param ``cost``: [out] Optional per-residual cost (e.g. :math:`\frac{1}{2}\rho(\|f\|^2)`); can be ``nullptr``.
  :param ``residuals``: [out] Residual output buffer (after loss scaling if present).
  :param ``jacobians``: [out] Optional Jacobian output (after loss scaling); can be ``nullptr``.
  :param ``state_pointers``: [in] Device pointer array mapping factor inputs to state blocks.
  :param ``stream``: [in] CUDA stream for factor/loss evaluation.
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

.. _sparse-matrix-multiplication-label:

--------------------------------------------------------------------------------
:code:`SparseMatrixMultiplication::ComputeSquaredMatrix`
--------------------------------------------------------------------------------

**Purpose:** Computes :math:`A^T A` for a sparse matrix :math:`A`. Used
internally to form the Gauss-Newton Hessian :math:`J^T J` from the sparse
Jacobian :math:`J`.

.. cpp:function:: void ComputeSquaredMatrix(cudaStream_t stream, const CSRSparseMatrix& input, CSRSparseMatrix& output)

  :param ``stream``: [in] CUDA stream for sparse transpose/multiplication.
  :param ``input``: [in] Input sparse matrix :math:`A`.
  :param ``output``: [out] Output sparse matrix storing :math:`A^T A`.
  :returns: [out] No return value.
