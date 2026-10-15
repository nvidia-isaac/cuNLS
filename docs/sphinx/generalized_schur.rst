Generalized Schur solver
========================

Select the backend using the existing minimizer options::

   options.sparse_linear_solver_type = cunls::SparseLinearSolverType::GeneralizedSchur;

Python exposes the same ``SparseLinearSolverType.GeneralizedSchur`` value.
There are no additional public options or required state annotations.

For example, with Python Levenberg--Marquardt:

.. code-block:: python

   options = pycunls.LevenbergMarquardtMinimizerOptions()
   options.base_options.sparse_linear_solver_type = (
       pycunls.SparseLinearSolverType.GeneralizedSchur
   )
   summary = pycunls.LevenbergMarquardtMinimizer(options).minimize(stream, problem)

The C++ and Python sparse bundle adjustment examples, and the Python
visual-inertial bundle adjustment example, select this backend explicitly.

How it works
------------

The solver builds a graph of free tangent-state blocks and automatically selects
an independent set using connectivity, block dimension, and deterministic
tie-breaking. Selection crosses state-batch boundaries: batches are storage
groups, not elimination levels. The implementation uses one elimination level;
it does not recursively eliminate each batch. It accepts general block-sparse
graphs and mixed state dimensions, without identifying cameras or landmarks.

For the partitioned system ``[D B; B^T C]``, it factors the independent diagonal
blocks of ``D``, solves the reduced system ``S = C - B^T D^-1 B``, and recovers
the eliminated variables. Small reduced systems use an internal Cholesky solve.
Larger systems use an internal block-preconditioned conjugate-gradient iteration.
This backend owns these operations and never calls another cuNLS solver backend.

The reduced matrix is explicitly assembled when its dimension, estimated
assembly work, and available memory permit. Otherwise, an implicit operator
avoids materializing fill. Local factors, transformed cross blocks, and explicit
Schur accumulation use double precision; the input matrix and Krylov vectors
remain float. Symbolic work can be reused through the existing
``reuse_structure`` option when the state layout and sparsity pattern are fixed.
Numeric factors are refreshed for each solve, including damping changes.

The intended input is the symmetric positive-definite working system supplied
by the minimizer. General indefinite systems and unresolved gauge singularities
are not supported. Exactly zero-information rows with a consistent zero right
hand side receive zero updates. Invalid pivots cause failure rather than a
fallback to another backend. A full-system residual check requires relative
residual at most ``1e-3``; up to three refinement corrections are available.
The internal reduced iteration limit is 200. These are internal constants,
independent of PCG configuration options.

Elimination and recovery equations
----------------------------------

Let ``H x = b`` be the working linear system after the minimizer has applied
damping and column scaling. A permutation places the selected states first:

.. math::

   \begin{bmatrix}D & B\\ B^T & C\end{bmatrix}
   \begin{bmatrix}x_e\\x_r\end{bmatrix}
   = \begin{bmatrix}b_e\\b_r\end{bmatrix}.

No two selected states share a structural Hessian edge, so ``D`` is block
diagonal. Each selected state can therefore be factored independently on the
GPU. Substitution gives the reduced right hand side and recovery formula:

.. math::

   g = b_r - B^T D^{-1}b_e,\qquad
   S x_r = g,\qquad S = C - B^T D^{-1}B,

.. math::

   x_e = D^{-1}(b_e-Bx_r).

This is an algebraic elimination within a linear solve. It does not remove
states from ``Problem`` or create persistent marginalization priors. The full
update is returned in the caller's original ordering. When ``H`` is positive
definite, both the pivot blocks and the Schur complement are positive definite.

Automatic selection and multiple batches
----------------------------------------

The planner uses complete tangent states as vertices, not individual BSR tiles
or entire batches. For example, a six-dimensional pose remains one vertex even
when stored using three-dimensional BSR tiles. Constant states have already
been removed from the free-variable system. Raw matrix solves without problem
metadata use scalar CSR entries or BSR tiles as vertices instead.

Four GPU selection rounds favor low-connectivity, small-dimensional states.
The score is the number of stored scalar columns in the state's first row
multiplied by its tangent dimension, with a deterministic hash to break ties.
Each round selects local priority
minima and excludes their neighbors from subsequent selection. Any undecided
states are retained after the fourth round; maximality is not required for a
valid Schur decomposition. The heuristic does not promise minimum fill.

In ordinary SBA, landmarks tend to be selected because they are small and have
few neighbors. On other graphs, poses or other state types may be selected.
States from several batches can be eliminated together whenever they are
independent. Connected states cannot both enter this diagonal elimination set.
Repeated elimination per batch would introduce dependencies and extra fill;
this implementation instead solves all retained states together in one reduced
system. Multiple elimination levels are not implemented.

Storage, GPU work, and numerical checks
---------------------------------------

Symbolic initialization validates the block pattern, builds the partition and
row maps, and groups pivot work by dimension. Common sizes have specialized
kernels; a generic path handles other dimensions. CSR rows belonging to a
logical dense block must have consistent sorted column patterns, as produced
by the library's Hessian assembler.

Each numeric solve refreshes local Cholesky factors, their small dense inverses,
and transformed cross blocks ``W = D^-1 B``. Explicit assembly adds each
eliminated state's contribution to its retained-neighbor clique. It is chosen
only for at most 4096 retained scalar dimensions, an estimated clique-work
budget of 512 million scalar pairs, and a conservative allocation estimate
below one quarter of currently free GPU memory. These internal heuristics
avoid allocating a large dense reduced matrix for general sparse graphs.

The implicit path evaluates ``S v = C v - B^T (W v)`` from sparse storage.
Both iterative paths use block Jacobi formed from the actual diagonal blocks
of ``S``, including the eliminated-state contributions. Explicit systems of
at most 64 retained dimensions use the internal direct Cholesky path. An empty
retained system only needs independent local solves and recovery.

Double-precision local work and Schur accumulation reduce cancellation when
forming ``C - B^T D^-1 B``. The reduced iterative stopping target is tighter
than the final full-system target. After recovery, the backend recomputes
``b - Hx`` using the original matrix. If necessary, it solves for a correction
with the same factors and checks again. Failure to meet the original-system
residual target is reported as failure, including when advisory safety checks
are disabled. This is a global residual criterion, not a separate accuracy
guarantee for each disconnected component.

Reusing structure reuses the partition and allocations, not stale numerical
factors. Changing the state layout or sparsity requires symbolic
reinitialization under the existing solver lifecycle. Small or strongly damped
problems can favor BlockSparsePCG because Schur's numeric setup costs more than
a few full-system iterations; the measurements below expose that tradeoff.

GPU implementation and Jetson Orin
----------------------------------

The backend uses CUDA kernels directly; it does not use Thrust. Selection
metadata is copied once for a single host pass that builds stable state prefixes,
size buckets, and the fill estimate. One coalesced GPU pass expands the prefixes
into scalar row maps. Graph edges stay on the GPU. This avoids separate
selection-library compactions and their scratch allocations and synchronizations.

The numeric path is organized around the following costs:

* Three-dimensional pivots with at most 64 stored scalar columns stage their
  cross blocks in shared memory. Threads reuse those values for the clique
  update instead of repeatedly loading and indexing sparse rows. Wider pivots
  and other dimensions retain a general path.
* Explicit assembly computes only the upper triangle, then mirrors it. This
  approximately halves its dot products and atomic updates while preserving
  symmetry. Double-precision accumulation remains in place.
* Transformed cross blocks are packed by eliminated row. There is no unused
  transformed storage for retained rows, and neighboring lanes read contiguous
  transformed values. Dimension-specialized local inversions use compile-time
  loop bounds.
* Sparse products use warp reductions. The original-system residual assigns a
  warp to short rows and a full thread block to rows wider than 128 scalar
  columns, avoiding a single warp serializing a highly connected state.

The implementation uses 32-thread warps, ordinary shared memory, shuffle
reductions, and native double atomics. It does not depend on SM 90 features,
thread-block clusters, or desktop-only CUDA libraries. The optimized build was
compiled for both SM 86 and Orin's SM 87; its binary contains both targets.
The largest static shared-memory allocation is 32 KiB for the small terminal
factorization. Orin runtime correctness and performance still require the
Jetson runner; compiling an SM 87 target on x86 does not establish either.
FP64 work is deliberately retained for numerical stability and remains an
important cost on GPUs with limited FP64 throughput.

Validation and reproduction
---------------------------

The dedicated generalized-Schur tests cover mixed state dimensions, CSR/BSR
storage, explicit and implicit operators, numeric reuse, zero right hand sides,
invalid pivots, wide cliques, and long residual rows. Python tests exercise the
backend enum, scaling, projected bounds, and numeric Jacobians. The C++ SBA,
Python SBA, and Python visual-inertial BA examples select the solver explicitly.

Build and test using the standard repository scripts::

   EXTRA_CMAKE_ARGS="-DBUILD_TESTING=ON" ./scripts/build_cunls_in_docker.sh Release ./build_docker
   ./scripts/test_cunls_in_docker.sh ./build_docker

The optional ``scripts/benchmark_schur_in_docker.sh`` script compares this
backend with BlockSparsePCG and saves its results under the build output
folder. It records the actual CUDA device and checks matching problem sizes,
linear residuals, and final nonlinear costs. Performance depends on the graph,
damping, GPU, and reuse of symbolic work; Schur setup can cost more than a few
PCG iterations on small or strongly damped systems.
