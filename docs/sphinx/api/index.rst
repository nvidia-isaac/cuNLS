###############################################################################
API Reference
###############################################################################

This reference documents the cuNLS public and module-level classes, key
configuration structs, and major helper abstractions for both the C++ library
and the **pycunls** Python package.

Each section presents the shared theory once, then shows the C++ and Python
signatures side-by-side.  Python classes live in the ``pycunls`` package and
accept CuPy arrays (or raw ``int`` device pointers) wherever the C++ API
takes ``const float*``.  See :doc:`../pycunls_installation` for setup
instructions.

.. toctree::
   :maxdepth: 2

   minimizer
   state
   factor
   robustifier
   linear_solver
   common
   math
