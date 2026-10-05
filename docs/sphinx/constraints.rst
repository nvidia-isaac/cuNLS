###############################################################################
Constraints
###############################################################################

cuNLS minimizes :math:`\tfrac12 \sum_i \|r_i(x)\|^2`. Many problems also need
constraints: control and joint limits, dynamics that must hold exactly,
obstacle clearance, goal conditions. cuNLS supports

- equality constraints :math:`c(x) = 0`,
- inequality constraints :math:`c(x) \le 0`,
- bounds :math:`l \le x \le u` on the components of vector states.

Bounds are a property of the state batch (``set_bounds``) and are enforced by
**projection** inside the Gauss-Newton and Levenberg-Marquardt minimizers: the
iterates never leave the box, and no multipliers or penalties are involved.
Equality and inequality constraints are solved by an **augmented Lagrangian**
(AL) outer loop around either minimizer. A constraint is a factor batch whose
"residual" is the constraint value, so every factor (built-in, custom, Warp,
numeric-diff) can be used as a constraint.

===============================================================================
Python example
===============================================================================

.. code-block:: python

   import cupy as cp
   import numpy as np
   import pycunls

   # Objective: pull 1000 3-D points towards their targets ...
   n = 1000
   x = cp.zeros(n * 3, dtype=cp.float32)
   targets = cp.asarray(np.random.randn(n * 3).astype(np.float32) * 3)
   states = pycunls.VectorStateBatch3(x, n)
   states.set_num_active_states(n)
   prior = pycunls.PriorVectorFactorBatch3(targets, n)
   prior.set_num_active_factors(n)

   # ... subject to -1 <= x <= 1 (per state and component; ±inf: unbounded) ...
   lo = cp.full(n * 3, -1.0, dtype=cp.float32)
   hi = cp.full(n * 3, 1.0, dtype=cp.float32)
   states.set_bounds(lo, hi)

   # ... and x0 + x1 + x2 = 0.5: any factor batch becomes constraint rows.
   normals = cp.ones(n * 3, dtype=cp.float32)
   offsets = cp.full(n, 0.5, dtype=cp.float32)
   plane = pycunls.HalfspaceFactorBatch3(normals, offsets, n)   # r = aᵀx - b
   plane.set_num_active_factors(n)
   on_plane = pycunls.ConstraintFactorBatch(plane, pycunls.ConstraintKind.Equality)

   problem = pycunls.Problem()
   problem.add_state_batch(states)
   ptrs = [states.state_device_ptr(i) for i in range(n)]
   for fb in (prior, on_plane):
       problem.add_factor_batch(fb, ptrs)

   solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer())
   summary = solver.minimize(pycunls.CudaStream(), problem)
   print(summary.status, summary.max_violation, summary.outer_iterations)

``AugmentedLagrangianMinimizer`` works on any problem: without constraint batches it
is the wrapped minimizer.

===============================================================================
Bounds on vector states
===============================================================================

``VectorStateBatchN.set_bounds(lower, upper)`` (C++ ``VectorStateBatch<N>::SetBounds``)
   Per-state, per-component bounds (device float32 arrays of
   ``capacity * N``; :math:`\pm\infty` leaves a side unbounded; ``None,
   None`` removes them). The arrays are read at every solve and may be
   rewritten between solves.

The minimizers run projected Gauss-Newton: the initial values are clamped
into the box; at each iteration the components that sit at a bound with the
gradient pointing outward are held (they leave the linear solve, their step
is zero); a free component the solved step would still push through its
bound is held as well and the system solved again (an active-set refinement,
rarely more than one extra solve); every trial step, line-search steps
included, is clamped into the box. Constant states are never projected (a
measured initial state outside its bounds stays as it is). The RANSAC
minimizers do not enforce bounds and reject a problem with bounded states.

Compared with bounds as AL constraints (``BoundFactorBatch`` below), this is
exact at every iteration, needs no outer iterations, and keeps the
Gauss-Newton step well-defined when many bounds are active at once (saturated
controls). Prefer it for limits on vector states.

===============================================================================
Constraint batches
===============================================================================

``ConstraintFactorBatch(inner, kind, scale=1)``
   Turns every residual row :math:`r` of ``inner`` into the constraint
   :math:`s\,r(x) = 0` (``ConstraintKind.Equality``) or :math:`s\,r(x) \le 0`
   (``ConstraintKind.Inequality``). The scale :math:`s > 0` makes the
   tolerance mean the same for rows in meters, radians or newtons. Examples:
   a prior on the last pose wrapped as an equality ("end at the goal"), a
   signed-distance factor wrapped as an inequality ("keep clear of the
   obstacle"). The wrapper does not own ``inner``.

``BoundFactorBatchN(lower, upper, capacity, scale=1)`` (C++ ``BoundFactorBatch<N>``)
   Box constraints on an N-dimensional vector state, with per-factor,
   per-component bounds; :math:`\pm\infty` leaves a side unbounded. An
   inequality constraint batch by itself (no wrapper needed). The same
   semantics as ``set_bounds``, through the AL loop: use it where only some
   factors of a state should be bounded or bounds should be soft until the
   AL loop converges; otherwise prefer ``set_bounds``.

``HalfspaceFactorBatchN(normals, offsets, capacity)`` (C++ ``HalfspaceFactorBatch<N>``)
   The signed value :math:`a^\top x - b` of a vector state: wrap it as an
   inequality for :math:`a^\top x \le b` (lanes, polygonal free space as
   several halfspaces) or as an equality for a hyperplane.

Subclasses of ``ConstraintFactorBatchBase`` (C++) hold one multiplier per row
and one penalty per factor (``multipliers_ptr`` / ``penalties_ptr`` in
Python). Only ``AugmentedLagrangianMinimizer`` updates them: the Gauss-Newton,
Levenberg-Marquardt and RANSAC minimizers reject a problem with constraint
batches (``ValueError``, C++ ``std::invalid_argument``), and so do
``WeightedFactorBatch`` / ``InformationFactorBatch`` when asked to wrap one
(scale a constraint with its own ``scale``).

**Robust losses.** A robust loss on an *objective* factor works as with the
plain minimizers (the inner solver applies it). A constraint takes no loss: it
must hold exactly, and a loss would down-weight large violations and corrupt
the multiplier update, so ``AugmentedLagrangianMinimizer`` rejects a constraint
batch registered with a robust loss.

===============================================================================
The method
===============================================================================

With multipliers :math:`\lambda` (equalities), :math:`\mu \ge 0`
(inequalities) and penalty :math:`\rho`, each constraint row contributes the
least-squares residual

.. math::

   r_E = \sqrt\rho\,(c + \lambda/\rho), \qquad
   r_I = \sqrt\rho\,\max(0,\ c + \mu/\rho),

whose squared norm is, up to a constant, the AL term of the row. The inner
solve is therefore an ordinary least-squares solve. The outer loop:

1. minimize the AL cost (warm-started, ``inner_iterations`` iterations, with
   a line search);
2. update the multipliers: :math:`\lambda \leftarrow \lambda + \rho c`,
   :math:`\mu \leftarrow \max(0, \mu + \rho c)`;
3. multiply the penalty of a constraint batch by ``penalty_increase`` when its
   violation did not drop below ``violation_decrease`` times the previous one
   (up to ``max_penalty``); after an inner solve cut off by its iteration cap,
   only when the violation is stagnating (above 0.9 times the previous one):
   a violation still falling is limited by the short solve, not by the
   penalty, and a larger penalty would only worsen the conditioning;
4. once every running subproblem is feasible, run the inner solve to
   convergence (``final_inner_iterations``); a subproblem that is feasible
   after an inner solve that converged on its own (not at the iteration cap)
   is done, and keeps its multipliers while the others continue.

The violation of a row is :math:`|c|` (equality) or :math:`\max(0, c)`
(inequality); a subproblem is feasible when every row is within
``constraint_tolerance`` (default ``1e-4``).

Batched problems
   With a problem partition (``Problem.set_problem_partition``), every
   subproblem has its own penalty per constraint batch and stops on its own;
   updates run on the device with one small read-back per outer iteration.

Line search
   The inequality rows make the cost piecewise quadratic: a Gauss-Newton step
   can activate rows the model did not see and overshoot. The inner solves
   therefore use a backtracking line search (``inner_line_search_steps``,
   see ``MinimizerOptions.max_line_search_steps``): any step that decreases
   the cost, possibly halved, is taken. Gauss-Newton with this line search is
   usually the better inner solver for strongly curved constraints; with
   Levenberg-Marquardt the damping, dominated by the constraint rows at large
   penalties, slows progress along them.

Status
   ``Converged`` (every subproblem feasible and stationary),
   ``MaxOuterIterations``, or ``MaxPenalty`` (a subproblem's violation stopped
   decreasing at the penalty cap: typically an infeasible problem).
   ``summary.final_cost`` is the objective: the cost of the non-constraint
   factors.

Real time
   ``options.real_time = True``: exactly ``max_outer_iterations`` outer
   iterations of ``inner_iterations`` inner iterations, every decision on the
   GPU, one read-back per call (the summary's costs are NaN). With
   ``warm_start`` and ``reuse_structure`` the penalties and violation history
   continue from call to call. ``solver.options = o`` switches the options between calls
   and keeps the warm-start state (a converged first solve, then a real-time
   budget).

Structure reuse
   ``options.reuse_structure = True``: the problem's structure (batches,
   connectivity, active and constant counts, partition) is unchanged since the
   previous call; the inner minimizer skips its setup (index expansion, Hessian
   pattern, the linear solver's analysis). A size check falls back to the full
   setup.

Warm start
   ``options.warm_start = True`` starts a solve from the previous solve's
   multipliers and penalties (same constraint batches and sizes), as in
   receding-horizon control. The penalties are lowered by one
   ``penalty_increase`` step (not below ``initial_penalty``) so that a long
   closed loop does not ratchet them up to ``max_penalty``.

===============================================================================
C++
===============================================================================

.. code-block:: cpp

   #include "cunls/cunls.h"

   cunls::BoundFactorBatch<3> bounds(lower, upper, n);          // device buffers
   bounds.SetNumActiveFactors(n);
   cunls::HalfspaceFactorBatch<3> plane(normals, offsets, n);
   plane.SetNumActiveFactors(n);
   cunls::ConstraintFactorBatch on_plane(&plane, cunls::ConstraintKind::kEquality);
   problem.AddFactorBatch(&bounds, ptrs);
   problem.AddFactorBatch(&on_plane, ptrs);

   cunls::LevenbergMarquardtMinimizer inner;
   cunls::AugmentedLagrangianMinimizerOptions options;  // constraint_tolerance, ...
   cunls::AugmentedLagrangianMinimizer solver(inner, options);
   cunls::AugmentedLagrangianMinimizerSummary summary = solver.Minimize(stream, problem);
