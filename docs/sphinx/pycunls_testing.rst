###############################################################################
Python Tests
###############################################################################

pycunls tests are written with `pytest` and live in ``python/tests/``.
For the tests of the C++ library, see :doc:`testing` (C++ Tests).

===============================================================================
Prerequisites
===============================================================================

Install the test dependencies:

.. code-block:: bash

   cd python
   pip install ".[test]"

This installs ``pytest`` and ``cupy-cuda12x`` (used only by
``test_cupy_interop.py``; ``warp-lang`` is a runtime dependency).

===============================================================================
Run all tests
===============================================================================

.. code-block:: bash

   cd python
   pytest tests/ -v

===============================================================================
Run a specific test module
===============================================================================

.. code-block:: bash

   pytest tests/test_minimizer.py -v
   pytest tests/test_warp_factor.py -v

===============================================================================
Test modules
===============================================================================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Coverage
   * - ``test_cupy_interop.py``
     - Interop with arrays from another library (CuPy) and raw pointers in
       pycunls constructors.
   * - ``test_state_batches.py``
     - State batch creation, state pointer access, and constant-state
       marking for all Euclidean and Lie group state types.
   * - ``test_factor_batches.py``
     - Built-in factor batch creation and connectivity for all factor
       types (priors, between, reprojection, ICP).
   * - ``test_minimizer.py``
     - End-to-end Gauss-Newton and Levenberg-Marquardt solves on small
       problems; verifies cost reduction and convergence.
   * - ``test_loss_functions.py``
     - All robust loss function batches (Huber, Cauchy, Arctan,
       SoftLOne, Tolerant, Tukey, Trivial, Scaled).
   * - ``test_warp_factor.py``
     - Custom `WarpFactorBatch` subclass end-to-end test.
   * - ``test_warp_state.py``
     - Custom `WarpStateBatch` subclass end-to-end test.
