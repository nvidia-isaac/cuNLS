###############################################################################
API Reference
###############################################################################

This reference documents the cuNLS Python package **pycunls** and the C++
library it is built on: solver classes, key configuration structs, and major
helper abstractions.

Each page presents the shared theory once, then the Python API, then the C++
API. Python classes live in the ``pycunls`` package and accept CuPy arrays (or
raw ``int`` device pointers) wherever the C++ API takes ``const float*``. See
:doc:`../pycunls_installation` for setup instructions. The
:doc:`linear_solver` and :doc:`math` pages document C++-only APIs.

.. toctree::
   :maxdepth: 2

   minimizer
   state
   factor
   robustifier
   common
   linear_solver
   math
