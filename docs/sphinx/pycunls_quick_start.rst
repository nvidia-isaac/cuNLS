###############################################################################
Python Quick Start
###############################################################################

This section shows a minimal end-to-end Python setup:

1. Install pycunls
2. Write a tiny script
3. Run it

===============================================================================
Step 1: Install pycunls
===============================================================================

Use :doc:`pycunls_installation` and make sure pycunls is importable.

===============================================================================
Step 2: Create a minimal script
===============================================================================

Create ``minimal.py``. The script solves the simplest possible nonlinear
least-squares problem: a single scalar variable :math:`x` pulled toward a
target value :math:`o = 2` by a prior factor with residual :math:`r = x - o`.
The cost is :math:`\tfrac{1}{2}\|x - o\|^2`, so the optimal solution is
:math:`x^* = 2`.

**Imports and GPU data.**
All GPU memory is managed via CuPy arrays. A `CudaStream` is required by
every pycunls API call — it controls asynchronous GPU execution.

.. code-block:: python

   import cupy as cp
   import pycunls

   stream = pycunls.CudaStream()

   # Initial guess: x = 0.  Target observation: o = 2.
   state_gpu = cp.array([0.0], dtype=cp.float32)
   obs_gpu   = cp.array([2.0], dtype=cp.float32)

**Create the state batch.**
A `VectorStateBatch1` wraps the device memory as a batch of 1-dimensional
Euclidean state blocks (see :doc:`api/state`).

.. code-block:: python

   state_batch = pycunls.VectorStateBatch1(state_gpu, 1)

**Create the factor batch.**
A `PriorVectorFactorBatch1` computes the residual :math:`r = x - o` and
Jacobian :math:`J = I` for each factor (see :doc:`api/factor`).

.. code-block:: python

   prior = pycunls.PriorVectorFactorBatch1(obs_gpu, 1)

**Wire state pointers and assemble the problem.**
The state-pointer list tells the solver which state block each factor
reads. For a prior factor with one state input, there is exactly one
pointer per factor. `Problem` collects all state and factor batches into a
single factor graph (see :doc:`api/minimizer`).

.. code-block:: python

   state_ptrs = [state_batch.state_block_device_ptr(0)]

   problem = pycunls.Problem()
   problem.add_state_batch(state_batch)
   problem.add_factor_batch(prior, state_ptrs)

**Run the solver.**
`LevenbergMarquardtMinimizer` solves the damped normal equations at each
iteration and adapts the damping parameter :math:`\lambda` based on step
quality. ``minimize`` updates the state memory in-place and returns a
`MinimizerSummary` with solve statistics.

.. code-block:: python

   minimizer = pycunls.LevenbergMarquardtMinimizer()
   summary   = minimizer.minimize(stream, problem)

**Inspect results.**

.. code-block:: python

   print(f"Iterations:   {summary.num_iterations}")
   print(f"Initial cost: {summary.initial_cost}")
   print(f"Final cost:   {summary.final_cost}")
   print(f"Solution x:   {cp.asnumpy(state_gpu)}")

===============================================================================
Step 3: Run
===============================================================================

.. code-block:: bash

   python minimal.py

You should see the final cost decrease toward zero and the solution converge
to :math:`x = 2`.
