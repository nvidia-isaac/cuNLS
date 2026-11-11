###############################################################################
Quick Start
###############################################################################

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``set_num_active_factors(n)`` /
   ``set_num_active_states(n)`` before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

This section shows a minimal end-to-end Python setup with ``pycunls``.

.. tip::

   **The shortest path.** Fitting a model to data is one call,
   ``pycunls.fit(model, x, y, p0)``; pose graphs and other problems with
   built-in factors take a few lines with ``pycunls.ProblemBuilder``. Both build
   everything below for you: see :doc:`simple_api`. Check an install with
   ``python -m pycunls doctor``.

.. note::

   **Using cuNLS from C++.** The same example written against the C++ API,
   with a CMake project to build it, is in :doc:`quick_start`.

Steps:

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
GPU data lives in `NVIDIA Warp <https://github.com/NVIDIA/warp>`_ arrays
(``wp.array``; any array with ``__cuda_array_interface__``, such as a PyTorch
or CuPy array, works too). A `CudaStream` is required by every solve — it
controls asynchronous GPU execution.

.. code-block:: python

   import warp as wp
   import pycunls

   stream = pycunls.CudaStream()

   # Initial guess: x = 0.  Target observation: o = 2.
   state_gpu = wp.array([0.0], dtype=wp.float32)
   obs_gpu   = wp.array([2.0], dtype=wp.float32)

**Create the state batch.**
A `VectorStateBatch1` wraps the device memory as a batch of 1-dimensional
Euclidean states (see :doc:`api/state`). The second argument is the
capacity (states the buffer holds). A batch starts with 0 active states:
``set_num_active_states`` sets how many the solver uses, before the first
solve.

.. code-block:: python

   state_batch = pycunls.VectorStateBatch1(state_gpu, 1)
   state_batch.set_num_active_states(1)

**Create the factor batch.**
A `PriorVectorFactorBatch1` computes the residual :math:`r = x - o` and
Jacobian :math:`J = I` for each factor (see :doc:`api/factor`). It is
also constructed with its capacity and starts with 0 active factors.

.. code-block:: python

   prior = pycunls.PriorVectorFactorBatch1(obs_gpu, 1)
   prior.set_num_active_factors(1)

**Wire state pointers and assemble the problem.**
The state-pointer list tells the solver which state each factor
reads. For a prior factor with one state input, there is exactly one
pointer per factor. `Problem` collects all state and factor batches into a
single factor graph (see :doc:`api/minimizer`).

.. code-block:: python

   state_ptrs = [state_batch.state_device_ptr(0)]

   problem = pycunls.Problem()
   problem.add_state_batch(state_batch)
   problem.add_factor_batch(prior, state_ptrs)

**Run the solver.**
`LevenbergMarquardtMinimizer` solves the damped normal equations at each
iteration and adapts the damping parameter :math:`\lambda` based on step
quality. ``minimize`` updates the state memory in-place and returns a
`MinimizerSummary` with solve statistics. The minimizer allocates GPU
workspace during initialization before the first linearized solve.

The uploads above ran on Warp's stream, the solve runs on ``stream``:
synchronize Warp's stream first so the solver reads the uploaded values (or
upload inside ``with wp.ScopedStream(wp.Stream(wp.get_cuda_device(),
cuda_stream=stream.get_stream())):``). ``minimize`` synchronizes ``stream``
before it returns, so the results can be read right after it.

.. code-block:: python

   wp.synchronize_stream(wp.get_stream())
   minimizer = pycunls.LevenbergMarquardtMinimizer()
   summary   = minimizer.minimize(stream, problem)

**Inspect results.**

.. code-block:: python

   print(f"Iterations:   {summary.num_iterations}")
   print(f"Initial cost: {summary.initial_cost}")
   print(f"Final cost:   {summary.final_cost}")
   print(f"Solution x:   {state_gpu.numpy()}")

===============================================================================
Step 3: Run
===============================================================================

.. code-block:: bash

   python minimal.py

You should see the final cost decrease toward zero and the solution converge
to :math:`x = 2`.

Next: :doc:`pycunls_tutorial` for complete examples.
