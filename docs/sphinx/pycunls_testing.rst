###############################################################################
Python Tests
###############################################################################

pycunls tests are written with `pytest` and live in ``python/tests/``.

===============================================================================
Prerequisites
===============================================================================

Install the test dependencies:

.. code-block:: bash

   cd python
   pip install ".[test]"

This installs ``pytest``, ``cupy-cuda12x``, and ``warp-lang``.

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
     - CuPy array and raw pointer interop with pycunls constructors.
   * - ``test_state_batches.py``
     - State batch creation, block pointer access, and constant-state
       marking for all Euclidean and Lie group state types.
   * - ``test_factor_batches.py``
     - Built-in factor batch creation and connectivity for all factor
       types (priors, between, reprojection, ICP).
   * - ``test_minimizer.py``
     - End-to-end Gauss-Newton and Levenberg-Marquardt solves on small
       problems; verifies cost reduction and convergence.
   * - ``test_loss_functions.py``
     - All robust loss function batches (Huber, Cauchy, Arctan,
       SoftLOne, Tolerant, Tukey, Trivial).
   * - ``test_warp_factor.py``
     - Custom `WarpFactorBatch` subclass end-to-end test.
   * - ``test_warp_state.py``
     - Custom `WarpStateBatch` subclass end-to-end test.
