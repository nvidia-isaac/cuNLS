###############################################################################
cuNLS Documentation
###############################################################################

cuNLS provides GPU-accelerated nonlinear least-squares optimization for
batched geometric estimation problems. It is used primarily from **Python**
through the ``pycunls`` package (CuPy arrays, custom kernels with NVIDIA
Warp), and also offers a native **C++/CUDA** API.

.. raw:: html

   <div style="display:flex; justify-content:center; margin: 0.5rem 0 1.0rem 0;">
     <div style="max-width: 360px; width: 100%;">
      <img class="only-light" src="_static/cuNLS_logo_light.png" alt="cuNLS logo" style="width:100%; height:auto;">
      <img class="only-dark" src="_static/cuNLS_logo_dark.png" alt="cuNLS logo" style="width:100%; height:auto;">
     </div>
   </div>

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``set_num_active_factors(n)`` /
   ``set_num_active_states(n)`` (C++: ``SetNumActiveFactors`` /
   ``SetNumActiveStates``) before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

===============================================================================
Getting Started
===============================================================================

.. toctree::
   :maxdepth: 2

   introduction
   pycunls_installation
   pycunls_quick_start
   pycunls_tutorial

===============================================================================
Guides
===============================================================================

.. toctree::
   :maxdepth: 2

   ransac
   custom_factors_and_states
   numeric_jacobians

===============================================================================
C++ API
===============================================================================

.. toctree::
   :maxdepth: 2

   installation
   quick_start
   tutorial

===============================================================================
Testing
===============================================================================

.. toctree::
   :maxdepth: 2

   pycunls_testing
   testing

===============================================================================
Licensing
===============================================================================

.. toctree::
   :maxdepth: 1

   licensing

===============================================================================
API Reference
===============================================================================

.. toctree::
   :maxdepth: 2

   api/index
