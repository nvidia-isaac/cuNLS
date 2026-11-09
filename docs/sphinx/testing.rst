###############################################################################
C++ Tests
###############################################################################

The cuNLS C++ library tests (GoogleTest) are built through CMake when
``BUILD_TESTING=ON``. For the Python (``pycunls``) tests, see
:doc:`pycunls_testing`.

===============================================================================
Step-by-step
===============================================================================

1. Configure:

.. code-block:: bash

   cmake -S . -B build/tests \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_TESTING=ON

2. Build:

.. code-block:: bash

   cmake --build build/tests -j

3. Run all tests with CTest:

.. code-block:: bash

   ctest --test-dir build/tests --output-on-failure

===============================================================================
Alternative: Run the aggregated binary directly
===============================================================================

.. code-block:: bash

   ./build/tests/bin/nls_tests

===============================================================================
Coverage mode
===============================================================================

The repository supports a coverage build path through `scripts/build_cunls.sh`
with `CMAKE_BUILD_TYPE=Coverage` (it enables `BUILD_TESTING`). The script configures the source
one level above the build directory, so use a top-level build directory:

.. code-block:: bash

   ./scripts/build_cunls.sh build_coverage Coverage

This runs the test suite under the `coverage` target and emits report assets under
`build_coverage/bin/coverage_report`.
