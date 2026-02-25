###############################################################################
Quick Start
###############################################################################

This section shows a minimal end-to-end setup:

1. Install cuNLS
2. Write a tiny app
3. Compile and run it against the installed library

===============================================================================
Step 1: Install cuNLS
===============================================================================

Use :doc:`installation` and make sure you have an install prefix (example:
`/tmp/cunls_install`).

===============================================================================
Step 2: Create a minimal source file
===============================================================================

Create `main.cpp`. The program solves the simplest possible nonlinear
least-squares problem: a single scalar variable :math:`x` pulled toward a
target value :math:`o = 2` by a prior factor with residual :math:`r = x - o`.
The cost is :math:`\tfrac{1}{2}\|x - o\|^2`, so the optimal solution is
:math:`x^* = 2`.

**Includes and CUDA stream.**
All cuNLS public headers are available through the umbrella header
`cunls/cunls.h`. A CUDA stream is required by every cuNLS API call — it
controls asynchronous GPU execution.

.. code-block:: cpp

   #include <cuda_runtime.h>
   #include <iostream>
   #include <vector>
   #include "cunls/cunls.h"

   int main() {
     // cuNLS operations are asynchronous; a CUDA stream serializes GPU work.
     cudaStream_t stream = nullptr;
     cudaStreamCreate(&stream);

**Prepare host data and upload to the GPU.**
We define one scalar state :math:`x = 0` (the initial guess) and one
observation :math:`o = 2` (the target). `dvector` (see :doc:`api/common`)
is a thin RAII wrapper around `cudaMalloc` / `cudaMemcpy` that uploads
host vectors to device memory on construction.

.. code-block:: cpp

     // Initial guess: x = 0.  Target observation: o = 2.
     std::vector<float> h_state = {0.0f};
     std::vector<float> h_obs   = {2.0f};

     // Upload both vectors to GPU memory.
     cunls::dvector<float> d_state(h_state);
     cunls::dvector<float> d_obs(h_obs);

**Create the state batch.**
A `VectorStateBatch<1>` (see :doc:`api/state`) wraps the device memory as a
batch of 1-dimensional Euclidean state blocks. The template argument ``1``
means each block has one float. The second constructor argument is the
number of state blocks (here just one).

.. code-block:: cpp

     // Wrap the device state memory in a VectorStateBatch with one block of
     // dimension 1.  The solver will update this memory in-place.
     cunls::VectorStateBatch<1> state_batch(d_state.data(), /*num_blocks=*/1);

**Create the factor batch.**
A `PriorVectorFactorBatch<1>` (see :doc:`api/factor`) computes the residual
:math:`r = x - o` and Jacobian :math:`J = I` for each factor. The
constructor takes a device pointer to the observation vectors and the
number of factors.

.. code-block:: cpp

     // Build a prior factor that penalizes deviation from the observation.
     // Residual: r = x - o,  Jacobian: J = I.
     cunls::PriorVectorFactorBatch<1> prior(
         reinterpret_cast<const cunls::Vector<1>*>(d_obs.data()),
         /*num_factors=*/1);

**Wire state pointers and assemble the problem.**
The state-pointer vector tells the solver which state block each factor
reads. For a prior factor with one state input, there is exactly one
pointer per factor. `Problem` (see :doc:`api/minimizer`) collects all
state and factor batches into a single factor graph.

.. code-block:: cpp

     // Each factor needs a list of device pointers to its input state blocks.
     // The prior factor reads one state block, so we provide one pointer.
     std::vector<float*> state_ptrs = {state_batch.StateBlockDevicePtr(0)};

     // Assemble the factor graph.
     cunls::Problem problem;
     problem.AddStateBatch(&state_batch);
     problem.AddFactorBatch(&prior, state_ptrs);

**Run the solver.**
`LevenbergMarquardtMinimizer` (see :doc:`api/minimizer`) solves the damped
normal equations :math:`(J^T J + \lambda D)\,\Delta x = -J^T r` at each
iteration, adapting :math:`\lambda` based on step quality. `Minimize`
updates the state memory in-place and returns a `MinimizerSummary` with
solve statistics.

.. code-block:: cpp

     // Solve with default LM settings.  The state (d_state) is updated
     // in-place on the GPU.
     cunls::LevenbergMarquardtMinimizer minimizer;
     auto summary = minimizer.Minimize(stream, problem);

**Inspect results.**

.. code-block:: cpp

     std::cout << "Iterations: "   << summary.num_iterations << "\n";
     std::cout << "Initial cost: " << summary.initial_cost   << "\n";
     std::cout << "Final cost: "   << summary.final_cost     << "\n";

     cudaStreamDestroy(stream);
     return 0;
   }

===============================================================================
Step 3: Create CMakeLists.txt
===============================================================================

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.24)
   project(cunls_quick_start LANGUAGES CXX CUDA)

   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   if(NOT DEFINED CUNLS_INSTALL_DIR)
     message(FATAL_ERROR "Set CUNLS_INSTALL_DIR to cuNLS install prefix.")
   endif()

   find_package(CUDAToolkit REQUIRED)
   find_library(CUNLS_LIBRARY cunls PATHS "${CUNLS_INSTALL_DIR}/lib" REQUIRED NO_DEFAULT_PATH)

   add_executable(minimal main.cpp)
   target_include_directories(minimal PRIVATE "${CUNLS_INSTALL_DIR}/include")
   target_link_libraries(minimal PRIVATE "${CUNLS_LIBRARY}" CUDA::cudart)
   set_target_properties(minimal PROPERTIES
     BUILD_RPATH "${CUNLS_INSTALL_DIR}/lib"
     INSTALL_RPATH "${CUNLS_INSTALL_DIR}/lib"
   )

===============================================================================
Step 4: Build and run
===============================================================================

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUNLS_INSTALL_DIR=/tmp/cunls_install
   cmake --build build -j
   ./build/minimal

You should see the final cost decrease toward zero.
