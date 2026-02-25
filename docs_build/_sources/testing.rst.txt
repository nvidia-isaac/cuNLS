###############################################################################
Build and Run Tests
###############################################################################

cuNLS tests are built through CMake when `BUILD_TESTING=ON`.

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
with `CMAKE_BUILD_TYPE=Coverage`.

.. code-block:: bash

   ./scripts/build_cunls.sh build/coverage Coverage

This triggers the `coverage` target and emits report assets under:
`build/coverage/bin/coverage_report`.
