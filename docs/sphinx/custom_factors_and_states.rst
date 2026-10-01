###############################################################################
Custom Factors and States (C++ and Python)
###############################################################################

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / states their buffers hold) and
   start with **zero** active entries: call ``SetNumActiveFactors(n)`` /
   ``SetNumActiveStates(n)`` (Python: ``set_num_active_factors`` /
   ``set_num_active_states``) before solving, and again whenever the problem
   size changes. See :ref:`capacity-and-active-count`.

cuNLS ships many factor and state types, but most applications need at least
one of their own. This page explains, step by step, how to write a custom
**factor batch** (residuals and Jacobians) and a custom **state batch** (a
manifold and its ``Plus``) in C++ and in Python, so that they work with
**every** minimizer: Gauss-Newton, Levenberg-Marquardt, and the RANSAC
minimizers (:doc:`ransac`).

.. contents:: On this page
   :local:
   :depth: 2

===============================================================================
The two contracts in one minute
===============================================================================

A custom type implements one GPU method. Both methods have two extra
parameters that the regular minimizers leave at their defaults and the RANSAC
minimizers use to evaluate hundreds of hypotheses in one call.

**FactorBatch::Evaluate** evaluates *items*. An item is one factor of the
batch evaluated at one set of states:

- the call evaluates :math:`n` items (``num_factor_ids``, or ``NumActiveFactors()``
  when 0);
- item :math:`t` reads **the measurement of factor** :math:`f(t)` =
  ``factor_ids[t]``, or :math:`t \bmod N` when ``factor_ids`` is null;
- item :math:`t` reads **its own state pointers**
  ``state_pointers[t * B + b]`` and writes **its own output row** :math:`t`.

**StateBatch::Plus** updates *replicas*. The arrays hold ``num_replicas``
contiguous copies of the batch, so it must process ``num_replicas *
NumActiveStates()`` states.

That is all. The rule that makes a kernel correct:

.. important::

   In ``Evaluate``, index **measurements** (observations, constants, per-factor
   data) by :math:`f(t)`, and index **everything else** (state pointers,
   residuals, Jacobians) by the item :math:`t`. In ``Plus``, loop over
   ``num_replicas * NumActiveStates()`` states.

With the default arguments (``factor_ids = nullptr``, ``num_factor_ids = 0``,
``num_replicas = 1``) both reduce to the familiar behavior: item :math:`t` is
factor :math:`t`, and ``Plus`` updates the batch once.

===============================================================================
Factors: the item contract in detail
===============================================================================

Notation for one factor batch:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`N`
     - ``NumActiveFactors()``: number of active factors (measurements) in the batch,
       at most ``Capacity()``; 0 until ``SetNumActiveFactors`` is called.
   * - :math:`B`
     - ``StateSizes().size()``: states one factor reads.
   * - :math:`m`
     - ``ResidualsSize()``: residual dimension of one factor.
   * - :math:`J`
     - sum of ``StateSizes()``: Jacobian columns of one factor (tangent
       dimensions, state 0 first).
   * - :math:`n`
     - number of items in this call: ``num_factor_ids``, or :math:`N` if 0.
   * - :math:`f(t)`
     - factor of item :math:`t`: ``factor_ids[t]``, or :math:`t \bmod N` if
       ``factor_ids`` is null.

The C++ signature is

.. code-block:: cpp

   bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                 cudaStream_t stream, const int *factor_ids = nullptr,
                 size_t num_factor_ids = 0) const;

and its arguments are:

``residuals`` [out]
  :math:`n \cdot m` floats. Item :math:`t` writes
  ``residuals[t * m + r]``, :math:`r \in [0, m)`.

``jacobians`` [out]
  :math:`n \cdot m \cdot J` floats, or null when only residuals are needed
  (always check). Item :math:`t` writes a row-major :math:`m \times J` block:
  element :math:`(r, c)` is ``jacobians[(t * m + r) * J + c]``. Columns are the
  tangent coordinates of state 0, then state 1, and so on.

``state_pointers`` [in]
  :math:`n \cdot B` device pointers. Item :math:`t` reads state :math:`b` at
  ``state_pointers[t * B + b]``. Different items may point to the same memory
  (every PnP factor reads the one camera pose).

``stream`` [in]
  Enqueue all work on it; do not synchronize unless you must.

``factor_ids`` [in]
  Null (the default): :math:`f(t) = t \bmod N`. Otherwise :math:`n` device
  ints in :math:`[0, N)`, in any order, with repeats.

``num_factor_ids`` [in]
  :math:`n`, the number of items; 0 (the default) means :math:`N`.

How the minimizers call it, for a batch of :math:`N = 3` factors with one
state each:

.. code-block:: text

   Regular minimizers: Evaluate(res, jac, ptrs, stream)                 n = 3
     item t        0      1      2
     factor f(t)   0      1      2
     ptrs[t]       x      x      x        (all factors read the same state x)
     res rows      [0,m)  [m,2m) [2m,3m)

   RANSAC, two hypotheses P and Q, all factors:
   Evaluate(res, jac, ptrs, stream, nullptr, 6)                          n = 6
     item t        0   1   2   3   4   5
     factor f(t)   0   1   2   0   1   2      (t % 3)
     ptrs[t]       P   P   P   Q   Q   Q

   RANSAC, minimal samples {2, 0} for P and {2, 1} for Q:
   Evaluate(res, jac, ptrs, stream, ids = {2, 0, 2, 1}, 4)               n = 4
     item t        0   1   2   3
     factor f(t)   2   0   2   1
     ptrs[t]       P   P   Q   Q

Requirements:

1. Item :math:`t` must produce exactly what a plain evaluation produces for
   factor :math:`f(t)` at item :math:`t`'s states. Built-in factors are
   bitwise identical; yours should at least be identical up to rounding.
2. Do not assume :math:`n = N` or :math:`f(t) = t`. Size any internal scratch
   buffer for :math:`n` items (it may change from call to call).
3. Support ``jacobians == nullptr``.
4. Return ``false`` only on failure.

===============================================================================
States: the replica contract in detail
===============================================================================

.. code-block:: cpp

   void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
             size_t num_replicas = 1);

computes :math:`x \oplus \delta` for every state. With :math:`N` =
``NumActiveStates()``, :math:`A` = ``AmbientSize()`` (floats stored per state),
:math:`T` = ``TangentSize()`` (floats per update) and :math:`R` =
``num_replicas``, the arrays hold :math:`R \cdot N` states:

.. code-block:: text

   N = 2 states, R = 3 replicas:
     global state i     0     1  |  2     3  |  4     5
     replica r          0     0  |  1     1  |  2     2
     state within r     0     1  |  0     1  |  0     1
   state i of x / x_plus_delta at  i * A,   of delta at  i * T

``x``, ``delta`` [in]
  :math:`R N A` and :math:`R N T` floats.

``x_plus_delta`` [out]
  :math:`R N A` floats; never overlaps the inputs.

``num_replicas`` [in]
  :math:`R \ge 1`, default 1. The RANSAC minimizers keep one replica per
  hypothesis and update all of them in one call.

Each state is updated independently, so the simplest correct
implementation treats the arrays as one batch of :math:`R \cdot N` states.

===============================================================================
C++
===============================================================================

The running example is a robust **line fit**: estimate :math:`(a, b)` of
:math:`y = a x + b` from points :math:`(x_i, y_i)`, many of which are
outliers. Each factor has residual :math:`r_i = a x_i + b - y_i`
(:math:`m = 1`) and reads one 2D state (:math:`B = 1`, :math:`J = 2`), with
Jacobian :math:`[x_i,\ 1]`.

-------------------------------------------------------------------------------
Step 1: the kernel
-------------------------------------------------------------------------------

One thread per item. Measurements by ``f``, everything else by ``t``:

.. code-block:: cuda

   __global__ void LineFitKernel(const float *xs, const float *ys, const int *factor_ids,
                                 int num_factors, int num_items,
                                 float const *const *state_pointers,
                                 float *residuals, float *jacobians) {
     const int t = blockIdx.x * blockDim.x + threadIdx.x;   // item
     if (t >= num_items) return;
     const int f = factor_ids != nullptr ? factor_ids[t] : t % num_factors;  // measurement

     const float *ab = state_pointers[t];                   // item t's state (B = 1)
     residuals[t] = ab[0] * xs[f] + ab[1] - ys[f];          // item t's row (m = 1)
     if (jacobians != nullptr) {                            // row-major 1 x 2 block
       jacobians[2 * t + 0] = xs[f];                        // d r / d a
       jacobians[2 * t + 1] = 1.f;                          // d r / d b
     }
   }

-------------------------------------------------------------------------------
Step 2: the factor batch class
-------------------------------------------------------------------------------

Derive from ``SizedFactorBatch<m, state sizes...>``, which fixes
``ResidualsSize()`` and ``StateSizes()`` at compile time, and pass it the
capacity: the number of measurements your buffers hold. The base keeps the
active count ``NumActiveFactors()``, which starts at 0 and is set with
``SetNumActiveFactors(n)`` (any ``n`` up to the capacity), so the same batch serves
problems of any size without reallocation. Implement ``Evaluate``:

.. code-block:: cpp

   #include "cunls/cunls.h"

   class LineFitFactorBatch : public cunls::SizedFactorBatch<1, 2> {
    public:
     // xs, ys: device arrays of capacity floats; must outlive the batch.
     LineFitFactorBatch(const float *xs, const float *ys, size_t capacity)
         : SizedFactorBatch(capacity), xs_(xs), ys_(ys) {}

     bool Evaluate(float *residuals, float *jacobians, float const *const *state_pointers,
                   cudaStream_t stream, const int *factor_ids = nullptr,
                   size_t num_factor_ids = 0) const override {
       const size_t num_factors = NumActiveFactors();  // the active count
       const size_t num_items = num_factor_ids == 0 ? num_factors : num_factor_ids;
       if (num_items == 0 || num_factors == 0) return true;
       const int block = 256;
       const int grid = static_cast<int>((num_items + block - 1) / block);
       LineFitKernel<<<grid, block, 0, stream>>>(xs_, ys_, factor_ids,
                                                 static_cast<int>(num_factors),
                                                 static_cast<int>(num_items), state_pointers,
                                                 residuals, jacobians);
       return cudaGetLastError() == cudaSuccess;
     }

    private:
     const float *xs_;
     const float *ys_;
   };

-------------------------------------------------------------------------------
Step 3 (optional): a custom state batch
-------------------------------------------------------------------------------

The line parameters are an ordinary vector, so ``VectorStateBatch<2>`` is all
the example needs. A custom state is needed when the variable lives on a
manifold cuNLS does not ship. As an illustration, here is a **positive
scalar** parametrized multiplicatively, :math:`x \oplus \delta = x\,e^{\delta}`
(ambient 1, tangent 1). Derive from ``SizedStateBatch<A, T>``, which provides
storage, state pointers, constant states and the active sizes (capacity in
the constructor, ``SetNumActiveStates`` for the active counts), and implement
``Plus`` over the active states:

.. code-block:: cuda

   __global__ void PositivePlusKernel(const float *x, const float *delta, float *out,
                                      size_t num_states) {
     const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
     if (i < num_states) out[i] = x[i] * expf(delta[i]);   // state i: ambient 1, tangent 1
   }

   class PositiveScalarStateBatch : public cunls::SizedStateBatch<1, 1> {
    public:
     using cunls::SizedStateBatch<1, 1>::SizedStateBatch;  // (device_ptr, capacity[, ...])

     void Plus(const float *x, const float *delta, float *x_plus_delta, cudaStream_t stream,
               size_t num_replicas = 1) override {
       const size_t n = NumActiveStates() * num_replicas;   // every state of every replica
       if (n == 0) return;
       PositivePlusKernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(
           x, delta, x_plus_delta, n);
     }
   };

-------------------------------------------------------------------------------
Step 4: use it with any minimizer
-------------------------------------------------------------------------------

.. code-block:: cpp

   // Device data: xs, ys (num_points floats each) and the line state (2 floats).
   cunls::dvector<float> d_xs(xs), d_ys(ys), d_ab(std::vector<float>{0.f, 0.f});
   cunls::VectorStateBatch<2> line(d_ab.data(), 1);
   LineFitFactorBatch fit(d_xs.data(), d_ys.data(), num_points);
   line.SetNumActiveStates(1);  // batches start with 0 active entries
   fit.SetNumActiveFactors(num_points);

   cunls::Problem problem;
   problem.AddStateBatch(&line);
   problem.AddFactorBatch(&fit, std::vector<float *>(num_points, line.StateDevicePtr(0)));

   // Plain least squares:
   //   cunls::LevenbergMarquardtMinimizer().Minimize(stream, problem);

   // Robust, with outliers:
   cunls::RansacMinimizerOptions options;
   options.default_inlier_threshold = 0.05f;  // ~3 sigma of the y noise
   cunls::RansacGaussNewtonMinimizer ransac(options);
   cunls::RansacSummary summary = ransac.Minimize(stream, problem);
   // d_ab now holds (a, b); ransac.InlierMask(0) the classification.

The free dimension is :math:`D = 2` and :math:`m = 1`, so each hypothesis is
fitted to :math:`s = 2` points: exactly the textbook RANSAC line fit.

===============================================================================
Python
===============================================================================

Python custom types subclass ``pycunls.CustomFactorBatch`` /
``pycunls.CustomStateBatch`` and override ``evaluate`` / ``plus``. cuNLS calls
them with raw device pointers (``int``) and the CUDA stream handle, from the
minimizer's thread with the GIL held. Launch GPU kernels on that stream with
CuPy, NVIDIA Warp (``pycunls.warp``), or any other library.

The Python signatures mirror C++, with one difference: ``num_factor_ids`` is
always the **actual** item count :math:`n > 0` (the binding resolves the
C++ default 0 to ``NumActiveFactors()``).

.. code-block:: python

   def evaluate(self, residuals_ptr, jacobians_ptr, state_pointers_ptr,
                stream_handle, factor_ids_ptr, num_factor_ids) -> bool: ...

   def plus(self, x_ptr, delta_ptr, x_plus_delta_ptr, stream_handle, num_replicas) -> None: ...

``jacobians_ptr`` and ``factor_ids_ptr`` are ``0`` when null.

-------------------------------------------------------------------------------
With CuPy raw kernels
-------------------------------------------------------------------------------

The same line fit, with the kernel written in CUDA C++ and launched through
``cupy.RawKernel``. The state pointers arrive as an array of 64-bit
addresses. ``cupy_stream`` makes CuPy launch on cuNLS's stream, so the kernel
is ordered with the rest of the minimizer's work:

.. code-block:: python

   import cupy as cp
   import numpy as np
   import pycunls

   class _StreamHandle:
       """Exposes a raw cudaStream_t handle through the CUDA stream protocol."""

       def __init__(self, handle):
           self.handle = handle

       def __cuda_stream__(self):
           return (0, self.handle)


   def cupy_stream(handle):
       """CuPy stream wrapping cuNLS's cudaStream_t (use as a context manager)."""
       if hasattr(cp.cuda.Stream, "from_external"):  # CuPy >= 14
           return cp.cuda.Stream.from_external(_StreamHandle(handle))
       return cp.cuda.ExternalStream(handle)

   _line_kernel = cp.RawKernel(r"""
   extern "C" __global__
   void line(const float* xs, const float* ys, const int* factor_ids, int num_factors,
             const unsigned long long* state_ptrs, float* res, float* jac, int n) {
     int t = blockIdx.x * blockDim.x + threadIdx.x;          // item
     if (t >= n) return;
     int f = factor_ids ? factor_ids[t] : t % num_factors;   // measurement of item t
     const float* ab = (const float*)state_ptrs[t];           // state of item t
     res[t] = ab[0] * xs[f] + ab[1] - ys[f];                  // row t
     if (jac) { jac[2 * t] = xs[f]; jac[2 * t + 1] = 1.f; }
   }
   """, "line")


   class LineFitFactorBatch(pycunls.CustomFactorBatch):
       def __init__(self, xs, ys):
           # residual size 1, one state of tangent size 2, capacity len(xs)
           super().__init__(1, [2], len(xs))
           self.xs, self.ys = xs, ys          # cupy arrays; keep them alive

       def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle,
                    factor_ids_ptr, num_factor_ids):
           n = num_factor_ids                 # number of items, always > 0
           with cupy_stream(stream_handle):
               _line_kernel(((n + 127) // 128,), (128,),
                            (self.xs, self.ys, cp.uint64(factor_ids_ptr),
                             cp.int32(self.num_active_factors), cp.uint64(sp_ptr),
                             cp.uint64(res_ptr), cp.uint64(jac_ptr), cp.int32(n)))
           return True

A custom state works the same way. Here, a 2D Euclidean state written by
hand (a manifold would change only the kernel body):

.. code-block:: python

   _plus_kernel = cp.RawKernel(r"""
   extern "C" __global__
   void plus(const float* x, const float* d, float* out, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) out[i] = x[i] + d[i];
   }
   """, "plus")


   class LineState(pycunls.CustomStateBatch):
       def __init__(self, data):
           super().__init__(data, 2, 2, 1)    # ambient 2, tangent 2, capacity 1

       def plus(self, x_ptr, delta_ptr, out_ptr, stream_handle, num_replicas):
           n = 2 * self.num_active_states * num_replicas   # every float of every replica
           with cupy_stream(stream_handle):
               _plus_kernel(((n + 127) // 128,), (128,),
                            (cp.uint64(x_ptr), cp.uint64(delta_ptr), cp.uint64(out_ptr),
                             cp.int32(n)))

Using them with RANSAC:

.. code-block:: python

   ab = cp.zeros(2, dtype=cp.float32)
   state = LineState(ab)
   factor = LineFitFactorBatch(cp.asarray(xs), cp.asarray(ys))
   state.set_num_active_states(1)          # active sizes start at 0
   factor.set_num_active_factors(len(xs))
   problem = pycunls.Problem()
   problem.add_state_batch(state)
   problem.add_factor_batch(factor, [state.state_device_ptr(0)] * len(xs))

   options = pycunls.RansacMinimizerOptions()
   options.default_inlier_threshold = 0.05
   ransac = pycunls.RansacGaussNewtonMinimizer(options)
   summary = ransac.minimize(pycunls.CudaStream(), problem)
   a, b = cp.asnumpy(ab)
   mask = ransac.inlier_mask(0)

This exact code is exercised by ``python/tests/test_ransac.py``.

-------------------------------------------------------------------------------
With NVIDIA Warp
-------------------------------------------------------------------------------

``pycunls.warp.WarpFactorBatch`` and ``WarpStateBatch`` wrap the raw pointers
as ``warp.array`` objects. ``WarpFactorBatch.factor_ids(factor_ids_ptr, n)``
returns the factor of every item as an ``int32`` array, so a kernel can always
read ``ids[t]``, whether or not the caller passed factor ids:

.. code-block:: python

   import warp as wp
   from pycunls.warp import WarpFactorBatch

   @wp.kernel
   def line_kernel(xs: wp.array(dtype=wp.float32), ys: wp.array(dtype=wp.float32),
                   ids: wp.array(dtype=wp.int32), ab: wp.array(dtype=wp.vec2),
                   res: wp.array(dtype=wp.float32), jac: wp.array(dtype=wp.float32),
                   write_jac: int):
       t = wp.tid()                 # item
       f = ids[t]                   # measurement of item t
       res[t] = ab[t][0] * xs[f] + ab[t][1] - ys[f]
       if write_jac != 0:
           jac[2 * t] = xs[f]
           jac[2 * t + 1] = 1.0

   class WarpLineFit(WarpFactorBatch):
       def __init__(self, xs, ys):
           super().__init__(residual_size=1, state_sizes=[2], capacity=xs.shape[0])
           self.xs, self.ys = xs, ys

       def evaluate(self, res_ptr, jac_ptr, sp_ptr, stream_handle, factor_ids_ptr,
                    num_factor_ids):
           n = num_factor_ids
           ids = self.factor_ids(factor_ids_ptr, n)
           ab = gather_states(sp_ptr, n, stream_handle)  # item t's state -> ab[t] (see note)
           ...                             # wrap res / jac, launch line_kernel with dim=n
           return True

Warp kernels cannot dereference the ``float*`` table directly, so the Warp
examples first gather the item states into a contiguous array with a small
CuPy kernel on cuNLS's stream (``gather_state_values`` / ``gather_state_pairs``
in ``python/examples/example_utils/gpu.py``).
Gather **per item** (``n * B`` pointers), not per factor. Complete Warp
versions: ``python/examples/custom_warp_factor.py`` and
``python/examples/custom_warp_state.py``, and the :doc:`pycunls_tutorial`.

===============================================================================
Checklist and common mistakes
===============================================================================

Before using a custom type with RANSAC, check:

- [ ] The kernel launches ``num_items`` threads, not ``NumActiveFactors()``.
- [ ] Measurements are read at ``f = factor_ids ? factor_ids[t] : t % N``.
- [ ] State pointers are read at ``t * B + b`` and outputs written at row
  ``t``, never at ``f``.
- [ ] Scratch buffers are sized for ``num_items`` (resize on demand).
- [ ] ``jacobians == nullptr`` (``jac_ptr == 0``) skips the Jacobian.
- [ ] ``Plus`` processes ``num_replicas * NumActiveStates()`` states.

Common mistakes and their symptoms:

``residuals[f * m + r] = ...``
  Writing rows at the factor index. Regular minimizers still work (there
  :math:`f = t`), but RANSAC overwrites rows of other hypotheses: wrong
  inlier counts, random results.

``state_pointers[f * B + b]``
  Reading the state of the wrong item: every hypothesis is evaluated at
  hypothesis 0's state. RANSAC finds no consensus.

Launching ``NumActiveFactors()`` threads
  Items beyond :math:`N` are never evaluated; most hypotheses keep stale
  residuals.

``Plus`` over ``NumActiveStates()`` only
  Only the first hypothesis moves; all others stay at the initial guess.

**A quick self-test.** Evaluate your batch once normally, then with
``num_factor_ids = k * NumActiveFactors()`` and ``factor_ids = nullptr`` on a
pointer table repeated :math:`k` times: the :math:`k` copies of the output
must equal the plain output. Then evaluate random ``factor_ids`` with a
matching pointer table and compare row by row. cuNLS runs exactly this check
on every built-in factor (``tests/evaluate_items_check.h``).

**Numeric Jacobians.** Factors registered with ``JacobianMode::kNumeric``
(:doc:`numeric_jacobians`) work with the regular minimizers but are rejected by
the RANSAC minimizers for now; provide an analytic Jacobian for RANSAC.
