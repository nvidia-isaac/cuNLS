################################################################################
Diagnostics and Troubleshooting
################################################################################

cuNLS reports why a solve stopped, checks a problem before it is solved, and
explains argument mistakes where they are made. This page lists the tools and
the common failures with their fixes.

================================================================================
Why did the solve stop?
================================================================================

Every summary says why it stopped and, when it did not converge, what to try:

.. code-block:: python

   summary = minimizer.minimize(stream, problem)
   if not summary.converged():
       print(summary.message)
   summary.termination_reason          # pycunls.TerminationReason
   summary.to_dict()                   # JSON-ready
   print(summary.report())             # outcome first, then the numbers

``TerminationReason`` (C++: ``cunls::TerminationReason``), in increasing
severity:

.. list-table::
   :header-rows: 1

   * - Reason
     - Meaning
     - What to do
   * - ``NotDetermined``
     - real-time (fixed-iteration) solve: not read back
     - —
   * - ``NoFreeStates``
     - every state is constant
     - make fewer states constant
   * - ``CostTolerance``
     - converged: cost < ``cost_tolerance`` (absolute)
     - with exact data or tiny residual units, lower ``cost_tolerance``
   * - ``StateTolerance``
     - converged: squared step < ``state_tolerance``
     - —
   * - ``RelativeReductionTolerance``
     - converged (LM): predicted relative reduction < tolerance
     - —
   * - ``NoCostDecrease``
     - converged (GN): a full step no longer lowered the cost
     - if the cost is still high: check the Jacobian, use LM
   * - ``MaxIterations``
     - not converged: iteration cap reached
     - raise ``max_num_iterations``, improve the initial guess
   * - ``MaxRejectedSteps``
     - steps rejected in a row (LM: at full damping)
     - at a minimum this is float32 round-off; with a high cost check the
       Jacobian, the gauge, the initial guess
   * - ``NonFiniteStep``
     - the trial steps were not finite (failed linear solve or NaN residuals)
     - ``pycunls.diagnose``: gauge freedom, unconstrained states
   * - ``NonFiniteCost``
     - the initial cost is NaN or Inf
     - ``pycunls.diagnose`` names the batch and the first bad state or factor

With a problem partition (many independent subproblems), the summary reports the
most severe reason, ``num_problems_by_reason`` counts every reason, and
``Minimizer.subproblem_termination_reasons()`` (C++:
``Minimizer::SubproblemTerminationReasons()``) gives the reason of each
subproblem, e.g. ``np.flatnonzero(r >= int(TerminationReason.MaxIterations))``
lists those that did not converge. The reasons cost one small read-back at the
end of a solve; real-time solves skip it. ``RansacSummary`` and
``AugmentedLagrangianMinimizerSummary`` carry a ``message`` and
``converged()`` too; the latter reports ``status``
(``AugmentedLagrangianMinimizerStatus``), which is ``NotDetermined`` for a
real-time call without constraint batches (nothing is read back, so
``converged()`` is False).

================================================================================
Check a problem before solving: ``pycunls.diagnose``
================================================================================

.. code-block:: python

   for issue in pycunls.diagnose(problem):
       print(issue)        # [error] ... / [warning] ..., each with the fix

It checks, in order: the structure (``Problem.find_structure_error``, C++
``Problem::FindStructureError``: active counts, capacities, connectivity,
states no factor constrains), NaN or Inf state values, NaN or Inf residuals at
the current states (with the batch and the first bad factor), and gauge
freedoms (a group of free states that only between-type factors connect, with
no constant state or prior). ``ProblemBuilder.solve`` runs it before the first
solve.

================================================================================
Residuals: ``pycunls.evaluate``
================================================================================

.. code-block:: python

   print(pycunls.evaluate(problem))       # a table per factor batch
   ev = pycunls.evaluate(problem, keep_residuals=True)
   ev.batches[0].rms, ev.batches[0].worst_factor, ev.batches[0].factor_norms()

Before a solve it checks the data layout and the initial guess (a transposed
pose or a wrong convention shows up as a huge residual); after a solve the RMS
of each batch should be near its measurement noise, and ``worst_factor``
locates outliers. The costs are the plain squared costs, before any robust
loss.

================================================================================
Custom factors: ``pycunls.check_jacobians``
================================================================================

.. code-block:: python

   for check in pycunls.check_jacobians(problem):
       print(check)        # "#0 MyFactor: ok (relative error 2e-05, ...)"

It compares each factor batch's Jacobian with central differences along the
manifold update (``StateBatch.apply_plus``), i.e. exactly the derivative the
minimizers use. A wrong sign, a swapped column or the wrong state order gives a
relative error of order 1. Run it for every custom (Warp) factor.

================================================================================
Argument errors
================================================================================

Arrays are checked where they are passed, with the fix in the message: a NumPy
(host) array where GPU memory is needed, a wrong dtype, a strided view, an
array too small for the capacity (``data has 36 float32 values, but 48 are
needed (capacity 3 states x 16 float32 per state)``). Any object with
``__cuda_array_interface__`` (CuPy, PyTorch, Numba, Warp) is accepted; an
``int`` device pointer is passed through unchecked.

================================================================================
Logging
================================================================================

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.INFO)
   pycunls.log_to_python(verbosity=pycunls.Verbosity.Message)

``pycunls.set_log_verbosity(level)`` sets the level alone (console output).
``Message`` logs one line per iteration (cost, relative decrease, active
subproblems) and the final message; ``log_to_python`` sends the native log to
the Python logger ``"pycunls"``. Without code changes, set the environment
variable ``CUNLS_LOG_LEVEL`` (``silent``, ``error``, ``warning``, ``message``,
``debug``); C++ programs read it too, and ``cunls::SetLogCallback`` forwards
the log to any sink.

================================================================================
Environment: ``python -m pycunls doctor``
================================================================================

Checks Python, the GPU and driver, CuPy, the native module, a small solve and a
small fit, and the optional Warp and cuDSS, each with a fix when it fails
(``--json`` for a machine-readable report).
