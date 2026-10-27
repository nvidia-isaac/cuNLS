###############################################################################
Linear systems and solvers
###############################################################################

Every Gauss-Newton or Levenberg-Marquardt iteration solves the normal
equations of the linearized problem for the step :math:`\delta`:

.. math::

   (H + \lambda D)\,\delta = b, \qquad H = J^T J, \quad b = -J^T f,

with the damping :math:`\lambda D` of Levenberg-Marquardt (zero for
Gauss-Newton), optionally column-scaled (see :doc:`api/minimizer`). Two options
of ``MinimizerOptions`` decide how this system is formed and solved:

- ``linear_system`` (``LinearSystemType``): ``Assembled`` (default) assembles
  the full system; ``SchurComplement`` eliminates the landmark states of a
  bundle adjustment first and solves a much smaller reduced system.
- ``sparse_linear_solver_type`` (``SparseLinearSolverType``): the solver of the
  assembled system. The default, ``BlockSparseCholesky``, is cuNLS's own
  supernodal sparse Cholesky factorization; the Schur complement always uses
  it for its reduced system.

This page explains how to choose, then how the two main pieces work.

===============================================================================
Choosing
===============================================================================

.. list-table::
   :header-rows: 1
   :widths: 40 22 38

   * - Problem
     - ``linear_system``
     - ``sparse_linear_solver_type``
   * - Bundle adjustment: poses and landmarks, each landmark observed by
       several poses
     - ``SchurComplement``
     - (not used: ``BlockSparseCholesky``)
   * - Visual-inertial bundle adjustment (keyframe poses, velocities, biases
       and landmarks)
     - ``SchurComplement``
     - (not used)
   * - Pose graphs, trajectories, SLAM back ends: graphs whose edges are
       mostly local (odometry, loop closures between nearby places)
     - ``Assembled``
     - ``BlockSparseCholesky`` (default)
   * - Graphs with many long-range random edges (no small separators), and
       very large well-conditioned systems of small blocks
     - ``Assembled``
     - ``BlockSparsePCG``
   * - Stage-ordered problems (MPC, trajectories with
       ``Problem::SetStateStages``)
     - ``Assembled``
     - ``BlockTridiagonal``
   * - A single small dense problem (one pose: PnP, registration)
     - ``Assembled``
     - ``DenseLDLT`` or ``DenseCholesky``

``BlockSparseCholesky`` and the Schur complement compute the exact step (up to
float32 rounding). ``BlockSparsePCG`` stops at a relative residual (``1e-3`` by
default): its steps are inexact, which costs Levenberg-Marquardt iterations on
ill-conditioned problems and can stop it short (bundle adjustment with long
tracks, visual-inertial windows). Its strength is graphs on which any direct
factorization fills in: when many edges connect distant states at random, the
factor of :math:`H` becomes nearly dense, while conjugate gradient only
multiplies by :math:`H`. ``cuDSS`` (NVIDIA's sparse direct solver) is an
optional dependency; ``BlockSparseCholesky`` is faster on the problems above.

``cuDSS`` factorizes the matrix as symmetric indefinite (not SPD), because
constant states can make the Hessian singular or ill-conditioned.
``cuDSSLinearSolverOptions::pivoting`` selects the pivoting strategy: ``true``
(default) is diagonal pivoting; ``false`` disables pivoting, which makes the
factorization about 20-40% faster, but a near-singular or badly scaled system
can then return a wrong solution without any error. The factorization status
(``CUDSS_DATA_INFO``) is read back after every solve only when
``MinimizerOptions::disable_safety_checks`` is ``false``, since reading it costs a
stream synchronization.

Measured end to end (Levenberg-Marquardt to convergence, RTX A6000, every call
including the symbolic setup):

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Problem
     - BlockSparseCholesky
     - BlockSparsePCG
     - cuDSS
   * - SE(3) pose graph, 2 500 poses, grid-like loop closures
     - **15 ms**
     - 75 ms
     - 50 ms
   * - SE(3) pose graph, 100 000 poses, grid-like loop closures
     - **541 ms**
     - 592 ms
     - 2 124 ms
   * - SE(3) pose graph, 5 000 poses, 2 000 random loop closures
     - 106 ms
     - **52 ms**
     - 329 ms
   * - 256 independent SE(2) graphs of 12 poses (one solve)
     - **4.6 ms**
     - 49 ms
     - 8.1 ms
   * - Bundle adjustment, full system (119 real problems)
     - **281 ms**
     - 661 ms
     - 345 ms
   * - Visual-inertial BA, 501 keyframes, full system
     - **23 ms**
     - 19 ms, 100x the final cost
     - 46 ms
   * - Visual-inertial BA, 501 keyframes, ``SchurComplement``
     - **14 ms**
     -
     -

===============================================================================
BlockSparseCholesky
===============================================================================

``BlockSparseCholeskySolver`` factors :math:`P A P^T = L L^T` (:math:`P` a
fill-reducing permutation) and solves with :math:`L`, entirely on the GPU and
without external dependencies. The matrices of nonlinear least squares are
block structured (one dense block per pair of connected states), which the
solver exploits throughout: it works on supervariables (the columns of one
state) and on dense panels of :math:`L`.

-------------------------------------------------------------------------------
Setup: symbolic analysis (``Initialize``)
-------------------------------------------------------------------------------

Run once per sparsity pattern (every ``minimize`` call, unless
``MinimizerOptions::reuse_structure`` vouches for an unchanged structure).
It reads the pattern only:

1. **Supervariables.** Consecutive columns with identical patterns (a state's
   tangent block, or several states always seen together) become one vertex.
   They and their graph are found on the GPU; only the graph comes to the
   host.
2. **Ordering.** Two fill-reducing orderings of the supervariable graph are
   computed and the one with the lower estimated factorization time kept (a
   fixed latency per sequential panel step plus the flops):

   - *approximate minimum degree* on the quotient graph (an eliminated vertex
     becomes an element standing for the clique of its neighbors, so the fill
     is never formed explicitly): the least fill on irregular graphs, such as
     covisibility, bundle adjustment and random loop closures;
   - *nested dissection* with balanced breadth-first level-set separators and
     minimum-degree leaves: short schedules and little fill on chains and
     grids, such as trajectories, visual-inertial windows and pose graphs with
     local loop closures.

3. **Supernodes and panels.** The elimination tree gives the fundamental
   supernodes (columns with nested structure). A supernode is merged into its
   parent when that saves a sequential step and keeps the explicit zeros at
   no more than 50%. Supernodes are cut into **panels** of at most 64
   columns: dense column-major blocks holding the panel's diagonal block and
   all its rows below.
4. **Schedule.** A panel's level is one more than the highest level among the
   panels that update it, so the panels of a level are independent. The work
   items of every level, and the maps from the nonzeros of :math:`A` into the
   panels and from each panel's rows into the panels it updates, are built
   (the maps on the GPU).

Setup takes a few milliseconds for thousands of states (2-4 ms for a
camera system with 700 poses, 10 ms for a full bundle adjustment with 10 000
landmarks) and grows to about 60 ms for 40 000 poses.

-------------------------------------------------------------------------------
Numeric phase (``Solve``)
-------------------------------------------------------------------------------

Every solve factors the current values and solves; nothing is synchronized
with the host unless safety checks are on:

1. **Assembly.** The lower triangle of :math:`PAP^T` is scattered into the
   panels and the right-hand side is permuted.
2. **Factorization and forward substitution, level by level.** One kernel
   launch per level factors all its panels, one CTA per panel and 64 rows
   below its diagonal block. The CTA stacks the diagonal block :math:`D`, its
   rows :math:`B` and the panel's part of the right-hand side :math:`y^T` in
   shared memory and factors them in blocks of 16 columns: one warp factors
   the 16 x 16 diagonal block (in groups of 4 columns, each group with two
   warp round trips), one thread per row solves the rows below it, and all
   threads apply the rank-16 update to the remaining columns. This yields
   :math:`L_D`, :math:`B L_D^{-T}` and, from the last row, the panel's step of
   the forward substitution :math:`z = L^{-1} b` at no extra cost. A second
   launch applies each panel's update :math:`L_B L_B^T` to the panels it
   targets, one CTA per 64 x 64 tile; atomics are used only on levels where two
   panels update the same target.
3. **Backward substitution**, levels in reverse: each CTA multiplies its rows'
   :math:`L_B^T` by the solution below, and the last CTA of a panel solves the
   diagonal block.

The cost of a level is mostly latency for small problems (each level is two
dependent launches) and throughput for large ones, so the number of levels
matters as much as the flops; this is what the cost model of the ordering
weighs.

-------------------------------------------------------------------------------
Rank deficiency and failures
-------------------------------------------------------------------------------

A pivot within :math:`10^{-5}` of its original diagonal entry (relative, either
sign) is treated as a rank deficiency rather than a failure: its unknown is
fixed at zero. This solves consistent semidefinite systems, such as
Gauss-Newton on a pose graph without an anchored pose (a free gauge), the
way conjugate gradient does. A clearly negative or non-finite pivot is a
failure: with safety checks on (``disable_safety_checks = false``) the solve
returns an error, otherwise the step becomes non-finite and is rejected.
Levenberg-Marquardt's damping keeps the system positive definite.

-------------------------------------------------------------------------------
Limits
-------------------------------------------------------------------------------

- The factor and its update maps are limited to :math:`2^{31}` entries each
  (``std::runtime_error`` at setup). Graphs with many long-range random edges
  reach it first: their factors are nearly dense. Use ``BlockSparsePCG`` there.
- Scalar CSR input only (no block storage): the assembly of the normal
  equations is scalar for this backend, at no measurable cost.
- Small, latency-bound problems (a few hundred poses) solve in about 0.1 ms
  per factorization; when the structure is the same at every call, set
  ``reuse_structure`` to skip the setup.

===============================================================================
Schur complement
===============================================================================

In a bundle adjustment every factor connects one landmark with one or a few
poses, and landmarks outnumber poses by orders of magnitude. Ordering the
unknowns as retained states :math:`r` (poses, and any other states) and
eliminated landmarks :math:`e`, the normal equations read

.. math::

   \begin{bmatrix} C & W \\ W^T & D \end{bmatrix}
   \begin{bmatrix} \delta_r \\ \delta_e \end{bmatrix} =
   \begin{bmatrix} b_r \\ b_e \end{bmatrix},

where :math:`D` is **block diagonal**: no factor touches two landmarks, so each
landmark has its own small block :math:`D_i` (3 x 3 for a point). Eliminating
the landmarks gives the **reduced system** of the retained states

.. math::

   S\,\delta_r = g, \qquad S = C - W D^{-1} W^T, \qquad g = b_r - W D^{-1} b_e,

and each landmark's step follows from its own block:

.. math::

   \delta_{e,i} = D_i^{-1} \big(b_{e,i} - W_i^T \delta_r\big).

The step is the exact solution of the full system. :math:`S` has one block per
pair of poses that observe a common landmark: dense when every camera sees
most of the scene, banded for a visual-inertial trajectory, and in any case
far smaller than the full system.

-------------------------------------------------------------------------------
Usage
-------------------------------------------------------------------------------

Declare the landmark state batches as eliminated, and select the Schur
complement:

.. code-block:: cpp

   problem.SetEliminatedStateBatches({&point_states});
   cunls::LevenbergMarquardtMinimizerOptions options;
   options.base_options.linear_system = cunls::LinearSystemType::SchurComplement;
   cunls::LevenbergMarquardtMinimizer minimizer(options);
   minimizer.Minimize(stream, problem);

.. code-block:: python

   problem.set_eliminated_state_batches([point_states])
   options = pycunls.LevenbergMarquardtMinimizerOptions()
   options.base_options.linear_system = pycunls.LinearSystemType.SchurComplement
   pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)

Complete programs: ``examples/sparse_bundle_adjustment`` (C++) and
``python/examples/sparse_bundle_adjustment.py``; visual-inertial bundle
adjustment: ``python/examples/imu_bundle_adjustment.py``.

Requirements, checked when the minimizer sets up the structure
(``std::invalid_argument`` / ``ValueError`` otherwise):

- every factor touches at most one non-constant state of the eliminated
  batches (constant states are not eliminated);
- the eliminated states share one tangent size of 1 to 3;
- the retained states that share factors with them share one tangent size of
  6 (poses), 3 or 9. Other factors, between retained states only (IMU,
  priors, odometry), may have any shape.

-------------------------------------------------------------------------------
How it works
-------------------------------------------------------------------------------

The full Hessian is never formed. The setup (once per structure, on the GPU)
numbers the landmarks and the retained states, finds every (retained state,
landmark) pair that shares a factor, orders the pairs by retained state
("pose-major"), lays the landmark factors' Jacobians out in that order, and
builds the block pattern of :math:`S` and the maps that write into it.

Each iteration, with the column scaling, damping and mask of the minimizer
applied to the full system before the elimination:

1. **Assembly** of :math:`b`, :math:`\operatorname{diag}(H)` and the landmark
   blocks :math:`D_i`, in one pass over the factors.
2. **Pair pass** (one thread per pair, pose-major): the pair's Jacobians are
   contracted into :math:`W_{pi}` and :math:`A_{pi} = J_p^T J_p`, the
   landmark block is factored, :math:`D_i = L_i L_i^T`, and
   :math:`V_{pi} = W_{pi} L_i^{-T}` and :math:`h_i = L_i^{-1} b_{e,i}` are
   formed. The diagonal blocks of :math:`S` receive
   :math:`A_{pi} - V_{pi} V_{pi}^T`, written per pair so no large term
   cancels another, and :math:`g` receives :math:`-V_{pi} h_i`. Warp-segmented
   sums leave one atomic per retained state and warp.
3. **Off-diagonal pass** (one warp per landmark track): the blocks
   :math:`-V_{pi} V_{qi}^T` for every pair of poses observing landmark
   :math:`i`.
4. **Reduced solve**: :math:`S` (expanded to scalar CSR) is factored by
   ``BlockSparseCholesky``.
5. **Back-substitution**, one thread per landmark:
   :math:`\delta_{e,i} = L_i^{-T}\big(h_i - \sum_p V_{pi}^T \delta_{r,p}\big)`.

A landmark block that is not positive definite (for example a point seen
along a single ray) is regularized with a small diagonal shift. With safety
checks on, a block that still cannot be factored makes the solve fail;
without them its landmark's step is non-finite and the minimizer rejects the
step.

Per solve of the reduced system (RTX A6000): 1.6 ms for a 723-pose cuSFM
bundle adjustment (dense covisibility, :math:`S` of size 4 338), 0.6 ms for a
2 001-keyframe visual-inertial window (:math:`S` of size 30 009, banded).
