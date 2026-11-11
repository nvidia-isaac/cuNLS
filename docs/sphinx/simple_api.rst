################################################################################
Quick Setup: fit, ProblemBuilder, solve
################################################################################

Three entry points build and solve a problem from plain arrays, without device
pointers, capacities or active counts. Everything they build is an ordinary
``pycunls.Problem``, so you can move to the full API at any point.

- ``pycunls.fit`` — fit the parameters of a model ``y ~ model(x, p)``: one
  curve, or thousands of independent fits in one GPU solve.
- ``pycunls.ProblemBuilder`` — states and built-in factors from arrays (pose
  graphs, bundle adjustment, registration), checked before solving.
- ``pycunls.solve`` — the builder in one call, for one state batch and one
  factor batch.

A runnable tour of ``fit`` is ``python/examples/model_fitting.py``.

================================================================================
Model fitting: ``pycunls.fit``
================================================================================

.. code-block:: python

   import numpy as np
   import warp as wp
   import pycunls

   @wp.func
   def decay(x: float, p: wp.vec3) -> float:   # y = a exp(-b x) + c
       return p[0] * wp.exp(-p[1] * x) + p[2]

   res = pycunls.fit(decay, x, y, p0=[1.0, 1.0, 0.0], sigma=0.02)
   print(res.report())
   res.params, res.param_errors, res.converged, res.reduced_chi2

**The model.** A **Warp function of one sample**, ``@wp.func def model(x, p)``,
with annotated types:

- ``x``: ``float``, or ``wp.vecN`` for ``N`` inputs per sample (``x`` of shape
  ``(n, N)``);
- ``p``: ``wp.vecP`` for ``P`` parameters (``wp.vec2``, ``wp.vec3``,
  ``wp.vec4``, ... or ``wp.types.vector(length=5, dtype=float)`` for sizes
  without a named type), or ``float`` for one parameter;
- the result: ``float``, or ``wp.vecM`` for ``M`` outputs per sample (``y`` of
  shape ``(n, M)``).

Write the formula with Warp's math (``wp.exp``, ``wp.sin``, ``wp.sqrt``, ...).
A loop with a constant trip count up to ``wp.config.max_unroll`` (16) is
unrolled and differentiated, e.g. a 16-step RK4 integration from 0 to ``x``
for a model given by an ODE; longer loops make ``jacobian="auto"`` fall back
to central differences. A Python or NumPy function raises ``TypeError``. The Jacobian comes
from Warp's automatic differentiation, checked against central differences at
``p0`` (``jacobian="auto"``: on a mismatch ``fit`` warns and uses central
differences); ``jacobian="autodiff"`` or ``"numeric"`` forces one.

.. code-block:: python

   P5 = wp.types.vector(length=5, dtype=float)

   @wp.func
   def peak(x: float, p: P5) -> float:     # Gaussian peak on a linear background
       d = (x - p[1]) / p[2]
       return p[0] * wp.exp(-0.5 * d * d) + p[3] + p[4] * x

   @wp.func
   def plane(x: wp.vec2, p: wp.vec3) -> float:   # two inputs per sample
       return p[0] * x[0] + p[1] * x[1] + p[2]

**Shapes.** With ``n`` samples, ``P`` parameters and ``B`` fits:

.. list-table::
   :header-rows: 1

   * - Argument
     - One fit
     - ``B`` independent fits
   * - ``y``
     - ``(n,)`` or ``(n, m)``
     - ``(B, n)`` or ``(B, n, m)``; NaN entries are skipped (ragged fits)
   * - ``x``
     - ``(n,)`` or ``(n, d)``
     - shared ``(n, ...)`` or per fit ``(B, n, ...)``
   * - ``p0``
     - ``(P,)``
     - ``(P,)`` for all, or ``(B, P)``
   * - results
     - ``params (P,)``, ``converged`` bool
     - ``params (B, P)``, ``converged (B,)``

The ``B`` fits are one problem with one subproblem per fit
(``Problem.set_problem_partition``): each fit takes its own steps and stops at
its own convergence, and ``res.reasons`` says why each one stopped. A fit that
has stopped costs no factorization work in later iterations with the default
``BlockSparseCholesky`` linear solver (other solvers factor the whole system
every iteration), so a batch where a few fits need many more iterations than
the rest costs little more than those few.

**Options.**

- ``sigma`` — measurement standard deviation (scalar or broadcastable to
  ``y``). Residuals become ``(model - y) / sigma``; then ``reduced_chi2`` is
  about 1 for a model that explains the data and the parameter errors are
  absolute. Without ``sigma`` the errors are scaled by the residual scatter, as
  in ``scipy.optimize.curve_fit``.
- ``loss`` / ``loss_scale`` — a robust loss (``"huber"``, ``"cauchy"``,
  ``"soft_l1"``, ``"tukey"``) against outliers; ``loss_scale`` is where it
  starts to down-weight, in units of the weighted residual.
- ``bounds=(lower, upper)`` — box bounds (scalars, ``(P,)`` or ``(B, P)``;
  ``±inf`` for none), solved by the augmented Lagrangian (projected) minimizer.
- ``starts`` — several initial guesses against local minima: an int ``K``
  draws ``K`` starts uniformly in finite ``bounds`` (the first is ``p0``), or a
  ``(K, P)`` array lists them. All starts are solved together; per fit the lowest
  objective (the robust one with ``loss``) wins (``res.start_index``), and starts
  where the model is not finite never win.
- ``method`` (``"lm"`` default, ``"gn"``), ``max_iterations``, ``jacobian``
  (``"auto"``, ``"autodiff"``, ``"numeric"``).

**Result** (``pycunls.FitResult``): ``params``, ``param_errors`` and
``covariance`` (from the Jacobian at the solution), ``converged`` and
``reasons`` per fit, ``cost``, ``reduced_chi2``, ``num_samples``,
``residuals`` (weighted, shaped like ``y``), ``message`` and the solver
``summary``; ``report()`` and ``to_dict()``.

**Is the model right?** A fit that converged is not necessarily a good model.
With the true ``sigma``, ``reduced_chi2`` well above 1 (say > 2) means the
model does not explain the data; look at ``res.residuals`` for structure.

================================================================================
Graph problems: ``pycunls.ProblemBuilder``
================================================================================

.. code-block:: python

   pb = pycunls.ProblemBuilder()
   poses = pb.add_states(pycunls.SE3StateBatch, initial=T0, constant=[0])  # (N, 4, 4)
   pb.add_factors(pycunls.SE3BetweenFactorBatch, deltas,                    # (M, 4, 4)
                  connect=(poses, edges))                                   # (M, 2)
   result = pb.solve()                     # Levenberg-Marquardt
   result.converged, result.message
   T = result[poses]                       # NumPy (N, 4, 4)
   print(result.evaluate())                # residuals per factor batch

- ``add_states(state_class, initial, constant=None, bounds=None)`` — the
  initial values set the number of states; ``constant`` lists states held
  fixed (fix the gauge of relative factors); ``bounds`` (vector states) solves
  with the augmented Lagrangian.
- ``add_factors(factor_class, *measurements, connect, loss=None)`` — the
  measurement arrays in constructor order (``pycunls.describe(factor_class)``
  shows them) and the connectivity: ``(handle, indices)`` with ``indices`` of
  shape ``(M, slots)`` when all slots are in one state batch, or a list of
  ``(handle, indices_m)`` pairs, one per slot, e.g.
  ``[(cameras, cam_ids), (points, point_ids)]`` for
  ``ReprojectionFactorBatch``. Shapes, slot types and index ranges are checked
  here, with the fix in the message.
- ``solve(method="lm", validate=True, **options)`` — runs ``pycunls.diagnose``
  before the first solve (errors raise ``ValueError``, warnings such as a gauge
  freedom warn), then solves. Options go by name (fields of
  ``MinimizerOptions``, ``LevenbergMarquardtMinimizerOptions`` and
  ``AugmentedLagrangianMinimizerOptions``); an unknown name raises with the
  valid ones.
- Repeated solves: ``handle.set(values)``, ``factor_handle.set_measurements(...)``,
  then ``solve()`` again. The builder keeps its minimizer while the method,
  the options and the structure stay the same, and then skips the structure
  setup (index expansion, symbolic analysis). ``add_states`` / ``add_factors``
  drop it (and ``diagnose`` runs again); ``solve(reuse_structure=False)``
  forces a fresh setup. Multipliers of constrained problems are not
  warm-started.
- The underlying objects: ``pb.problem``, ``handle.batch``, ``handle.data``
  (the ``wp.array`` the solve updates; ``handle.values`` is a NumPy copy).
- The result (``pycunls.SolveResult``): ``converged``, ``message``,
  ``summary``, ``result[handle]`` (NumPy states), ``result.states`` (with one
  state batch), ``evaluate()`` and ``report()``.

================================================================================
One call: ``pycunls.solve``
================================================================================

.. code-block:: python

   result = pycunls.solve(pycunls.SE3BetweenFactorBatch, deltas, initial=T0,
                          connect=edges, constant=[0])
   T = result.states

The state type comes from the factor (``state_class=`` overrides it).

================================================================================
Catalog
================================================================================

``pycunls.catalog(kind=None, query=None)`` lists the built-in state and factor
batches (states connected, residual, measurement layout, conventions);
``pycunls.describe(name)`` prints one, and ``help(pycunls.X)`` includes the same
text. ``python -m pycunls catalog between`` does the same from a shell.
