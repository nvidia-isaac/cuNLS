################################################################################
Robustifier API
################################################################################

The robustifier module defines GPU-batched robust loss functions used by the
minimizer to reduce the influence of outliers in non-linear least squares.
This page explains what the loss outputs mean, then documents the Python loss
classes, then the C++ classes together with the formula of each loss.

- **Python** — ``pycunls``
- **C++** — ``cunls/robustifier``

================================================================================
Overview
================================================================================

**What robustifier functions are for**
  In least squares, a few bad measurements (outliers) can pull the solution
  away from the true optimum. A robust loss function :math:`\rho(s)` replaces
  the squared residual term :math:`s = \|f_i\|^2` with :math:`\rho(s)`, so that
  large residuals contribute less to the cost. The solver then minimizes
  :math:`\frac{1}{2}\sum_i \rho(\|f_i\|^2)` instead of
  :math:`\frac{1}{2}\sum_i \|f_i\|^2`. Typical choices of :math:`\rho` behave
  like :math:`s` for small :math:`s` (inliers) and grow sublinearly for large
  :math:`s` (outliers), so inliers are fitted normally and outliers are
  down-weighted.

**Inputs and outputs**
  Each robustifier evaluates at **squared residual** values :math:`s = \|f\|^2`
  (one :math:`s` per residual vector). For each :math:`s` it returns a
  :cpp:type:`float3` :math:`(\rho(s), \rho'(s), \rho''(s))`:

  - **First component** :math:`\rho(s)`: robustified cost value; the term
    contributed to the total cost is :math:`\frac{1}{2}\rho(s)`.
  - **Second component** :math:`\rho'(s)`: first derivative of :math:`\rho`
    with respect to :math:`s`; used to scale residuals and Jacobians in the
    robustified Gauss-Newton step.
  - **Third component** :math:`\rho''(s)`: second derivative of :math:`\rho`
    with respect to :math:`s`; used in the Triggs correction when forming
    the robustified normal equations.

  Calling the evaluator with negative :math:`s` is invalid; implementations
  need not handle that case. Common choices of :math:`\rho` satisfy
  :math:`\rho(0)=0`, :math:`\rho'(0)=1`, and in the outlier region
  :math:`\rho'(s) < 1` and :math:`\rho''(s) < 0`.

Theory — How robustifier outputs are used in optimization
---------------------------------------------------------

The non-linear least squares problem with robustification is

.. math::

   \min_{\mathbf{x}} \quad \frac{1}{2}\sum_i \rho_i\bigl(\|f_i(x_{i_1},\ldots,x_{i_k})\|^2\bigr),

where :math:`f_i` are residual vectors and :math:`\rho_i` are loss functions.
Let :math:`s = \|f\|^2` and write :math:`\rho`, :math:`\rho'`, :math:`\rho''`
for the loss and its derivatives at :math:`s`. The contribution of one term
to the total cost is :math:`\frac{1}{2}\rho(s)`; the robustifier API returns
:math:`(\rho(s), \rho'(s), \rho''(s))` so the solver can form the robustified
gradient and Gauss-Newton system without recomputing :math:`\rho`.

**Robustified gradient**
  For a single residual block, the gradient of :math:`\frac{1}{2}\rho(\|f(x)\|^2)`
  with respect to the parameters :math:`x` is

  .. math::

     g = \rho'\, J^\top f,

  where :math:`J` is the Jacobian of :math:`f` with respect to :math:`x`.
  So :math:`\rho'` (the second component of the robustifier output) scales the
  gradient and thus down-weights the contribution of large residuals.

**Robustified Gauss-Newton Hessian**
  The Gauss-Newton approximation to the Hessian of :math:`\frac{1}{2}\rho(\|f\|^2)`
  involves both :math:`\rho'` and :math:`\rho''`. With
  :math:`r = f(x)` and :math:`s = \|f\|^2 = r^\top r`, the Hessian contribution
  (ignoring second derivatives of :math:`f`) is

  .. math::

     H = J^\top \left( \rho'\, I + 2\rho''\, r r^\top \right) J.

  When :math:`\rho'' < 0` (typical for robust losses in the outlier region),
  :math:`H` can be indefinite. To keep a positive-definite approximation and
  still use a Jacobian-based solver, the implementation rescales the residual
  and Jacobian so that the robustified problem looks like a standard least
  squares problem in the rescaled variables.

**Rescaling (Triggs correction)**
  Let :math:`\alpha` be a root of

  .. math::

     \frac{1}{2}\alpha^2 - \alpha - \frac{\rho''}{\rho'}\|f\|^2 = 0.

  Then the rescaled residual and Jacobian

  .. math::

     \tilde{f} = \frac{\sqrt{\rho'}}{1-\alpha}\, f,\qquad
     \tilde{J} = \sqrt{\rho'}\,\left(I - \alpha\, \frac{f f^\top}{\|f\|^2}\right) J

  When :math:`s > 0` and :math:`\rho'' > 0`, the root
  :math:`\alpha = 1 - \sqrt{1 + 2 s \rho''/\rho'}` is real and below 1, and
  the rescaled residual and Jacobian yield a Gauss-Newton step equivalent to
  that of the robustified Gauss-Newton Hessian above.

  Otherwise (:math:`s = 0`, or :math:`\rho'' \le 0`, the usual case for a
  robust loss in its outlier region, where :math:`H` can be indefinite) the
  solver uses :math:`\alpha = 0`: residual and Jacobian are scaled by
  :math:`\sqrt{\rho'}` only. This drops the :math:`2\rho''\, r r^\top`
  curvature term, so it is an approximation that keeps the system positive
  semi-definite, not an exact equivalent of the robustified problem.

  The robustifier output :math:`(\rho(s), \rho'(s), \rho''(s))` is used to
  compute :math:`\alpha` and the scaling factors :math:`\sqrt{\rho'}` and
  :math:`(1-\alpha)^{-1}` applied to residuals and Jacobians in the solver.
  This is the standard "Triggs correction" used by robust nonlinear
  least-squares solvers to keep a Gauss-Newton-style Jacobian approximation
  valid under a robust loss.

================================================================================
Python API (``pycunls``)
================================================================================

All Python loss function batches share the same base class
``pycunls.LossFunctionBatch``.  The formulas and parameters are identical to
the C++ versions documented in :ref:`robustifier-cpp-api` (linked from the
**Formula** column below).  Pass a loss function instance to
``Problem.add_factor_batch`` to apply robustification:

.. code-block:: python

   loss = pycunls.HuberLossFunctionBatch(delta=1.0)
   problem.add_factor_batch(factor_batch, loss, state_pointers)

.. list-table::
   :header-rows: 1
   :widths: 35 40 25

   * - Python class
     - Constructor
     - Formula
   * - ``TrivialLossFunctionBatch``
     - ``TrivialLossFunctionBatch()``
     - :ref:`TrivialLossFunctionBatch <cpp-trivial-loss-function-batch>`
   * - ``HuberLossFunctionBatch``
     - ``HuberLossFunctionBatch(delta: float)``
     - :ref:`HuberLossFunctionBatch <cpp-huber-loss-function-batch>`
   * - ``CauchyLossFunctionBatch``
     - ``CauchyLossFunctionBatch(b: float, c: float)``
     - :ref:`CauchyLossFunctionBatch <cpp-cauchy-loss-function-batch>`
   * - ``ArctanLossFunctionBatch``
     - ``ArctanLossFunctionBatch(a: float, b: float)``
     - :ref:`ArctanLossFunctionBatch <cpp-arctan-loss-function-batch>`
   * - ``SoftLOneLossFunctionBatch``
     - ``SoftLOneLossFunctionBatch(b: float, c: float)``
     - :ref:`SoftLOneLossFunctionBatch <cpp-soft-lone-loss-function-batch>`
   * - ``TolerantLossFunctionBatch``
     - ``TolerantLossFunctionBatch(a: float, b: float)``
     - :ref:`TolerantLossFunctionBatch <cpp-tolerant-loss-function-batch>`
   * - ``TukeyLossFunctionBatch``
     - ``TukeyLossFunctionBatch(a: float)``
     - :ref:`TukeyLossFunctionBatch <cpp-tukey-loss-function-batch>`
   * - ``ScaledLossFunctionBatch``
     - ``ScaledLossFunctionBatch(loss_function: LossFunctionBatch, a: float)``
     - :ref:`ScaledLossFunctionBatch <cpp-scaled-loss-function-batch>`

.. _robustifier-cpp-api:

================================================================================
C++ API
================================================================================

.. _cpp-loss-function-batch:

LossFunctionBatch
-----------------

Abstract base (:code:`cunls/robustifier/loss_function_batch.h`).

.. cpp:function:: bool Evaluate(float* s, float3* out, int num_losses, cudaStream_t stream) const

  Evaluates the loss for a batch of squared residuals.

  :param s: [in] Device pointer to squared residual values :math:`s = \|f\|^2`.
  :param out: [out] Device pointer to :cpp:type:`float3` tuples
    :math:`(\rho(s), \rho'(s), \rho''(s))` for each input.
  :param num_losses: [in] Number of residual values to process.
  :param stream: [in] CUDA stream for asynchronous execution.
  :returns: ``true`` on success.

.. _cpp-trivial-loss-function-batch:

TrivialLossFunctionBatch
------------------------

Header: :code:`cunls/robustifier/trivial_loss_function_batch.h`

.. cpp:function:: TrivialLossFunctionBatch()

  :returns: Constructor has no return value.

**Formula (unscaled)**

.. math::

   \rho(s) = s,\qquad \rho'(s) = 1,\qquad \rho''(s) = 0.

Identity loss: no robustification; equivalent to standard least squares.

.. _cpp-huber-loss-function-batch:

HuberLossFunctionBatch
----------------------

Header: :code:`cunls/robustifier/huber_loss_function_batch.h`

.. cpp:function:: HuberLossFunctionBatch(float delta)

  :param delta: [in] Inlier/outlier threshold (scale); quadratic for
    :math:`s \le \delta^2`, linear for :math:`s > \delta^2`.
  :returns: Constructor has no return value.

**Formula (scaled with :math:`\delta`)**

.. math::

   \rho(s) = \begin{cases}
     s & s \le \delta^2 \\
     2\delta\sqrt{s} - \delta^2 & s > \delta^2
   \end{cases}

.. math::

   \rho'(s) = \begin{cases}
     1 & s \le \delta^2 \\
     \delta/\sqrt{s} & s > \delta^2
   \end{cases}
   ,\qquad
   \rho''(s) = \begin{cases}
     0 & s \le \delta^2 \\
     -\rho'(s)/(2s) & s > \delta^2
   \end{cases}.

.. _cpp-cauchy-loss-function-batch:

CauchyLossFunctionBatch
-----------------------

Header: :code:`cunls/robustifier/cauchy_loss_function_batch.h`

.. cpp:function:: CauchyLossFunctionBatch(float b, float c)

  :param b: [in] Output scale parameter.
  :param c: [in] Shape parameter (larger :math:`c` makes the loss grow more slowly).
  :returns: Constructor has no return value.

**Formula**

.. math::

   \rho(s) = b\,\ln(1 + c\,s),\qquad
   \rho'(s) = \frac{b\,c}{1 + c\,s},\qquad
   \rho''(s) = -\frac{c^2 b}{(1+c\,s)^2}.

Unscaled case: :math:`\rho(s) = \ln(1+s)` (e.g. :math:`b=1`, :math:`c=1`).

.. _cpp-arctan-loss-function-batch:

ArctanLossFunctionBatch
-----------------------

Header: :code:`cunls/robustifier/arctan_loss_function_batch.h`

.. cpp:function:: ArctanLossFunctionBatch(float a, float b)

  :param a: [in] Scale parameter (argument scale in :math:`\arctan(s/a)`).
  :param b: [in] Shape parameter, typically :math:`1/a^2` for derivative scaling.
  :returns: Constructor has no return value.

**Formula**

  With :math:`s` the squared residual, the implementation uses
  :math:`\rho(s) = a\,\arctan(s/a)` and
  :math:`\rho'(s) = 1/(1 + s^2 b)` with :math:`b = 1/a^2`:

.. math::

   \rho(s) = a\,\arctan\frac{s}{a},\qquad
   \rho'(s) = \frac{1}{1 + (s/a)^2},\qquad
   \rho''(s) = -\frac{2s/a^2}{(1+(s/a)^2)^2}.

Unscaled case: :math:`\rho(s) = \arctan(s)` (e.g. :math:`a=1`, :math:`b=1`).

.. _cpp-soft-lone-loss-function-batch:

SoftLOneLossFunctionBatch
-------------------------

Header: :code:`cunls/robustifier/soft_lone_loss_function_batch.h`

.. cpp:function:: SoftLOneLossFunctionBatch(float b, float c)

  :param b: [in] Scale parameter.
  :param c: [in] Shape parameter (larger :math:`c` makes the loss grow more slowly).
  :returns: Constructor has no return value.

**Formula**

.. math::

   \rho(s) = 2b\left(\sqrt{1 + c\,s} - 1\right),\qquad
   \rho'(s) = \frac{b\,c}{\sqrt{1+c\,s}},\qquad
   \rho''(s) = -\frac{c^2 b}{2(1+c\,s)^{3/2}}.

Unscaled case: :math:`\rho(s) = 2(\sqrt{1+s}-1)` (e.g. :math:`b=1`, :math:`c=1`).

.. _cpp-tolerant-loss-function-batch:

TolerantLossFunctionBatch
-------------------------

Header: :code:`cunls/robustifier/tolerant_loss_function_batch.h`

.. cpp:function:: TolerantLossFunctionBatch(float a, float b)

  :param a: [in] Offset parameter (soft threshold).
  :param b: [in] Scale parameter (smoothing).
  :returns: Constructor has no return value.

**Formula**

  With :math:`c = b\,\ln(1 + e^{-a/b})` so that :math:`\rho(0)=0`:

.. math::

   \rho(s) = b\,\ln\left(1 + e^{(s-a)/b}\right) - c,\qquad
   \rho'(s) = \frac{e^{(s-a)/b}}{1 + e^{(s-a)/b}},\qquad
   \rho''(s) = \frac{1}{4b\,\cosh^2\bigl((s-a)/(2b)\bigr)}.

.. _cpp-tukey-loss-function-batch:

TukeyLossFunctionBatch
----------------------

Header: :code:`cunls/robustifier/tukey_loss_function_batch.h`

.. cpp:function:: TukeyLossFunctionBatch(float a)

  :param a: [in] Cutoff threshold; residuals with :math:`s > a^2` get zero weight.
  :returns: Constructor has no return value.

**Formula**

  With :math:`s` the squared residual and :math:`a^2` the squared cutoff:

.. math::

   \rho(s) = \begin{cases}
     \displaystyle\frac{a^2}{3}\left(1 - \left(1 - \frac{s}{a^2}\right)^3\right)
     & s \le a^2 \\[0.5em]
     \displaystyle\frac{a^2}{3} & s > a^2
   \end{cases}

.. math::

   \rho'(s) = \begin{cases}
     \displaystyle\left(1 - \frac{s}{a^2}\right)^2 & s \le a^2 \\
     0 & s > a^2
   \end{cases}
   ,\qquad
   \rho''(s) = \begin{cases}
     \displaystyle -\frac{2}{a^2}\left(1 - \frac{s}{a^2}\right) & s \le a^2 \\
     0 & s > a^2
   \end{cases}.

.. _cpp-scaled-loss-function-batch:

ScaledLossFunctionBatch
-----------------------

Header: :code:`cunls/robustifier/scaled_loss_function_batch.h`

.. cpp:function:: template <class T> ScaledLossFunctionBatch(float a, Args&&... loss_args)

  :param a: [in] Positive scale factor applied to all loss outputs.
  :param loss_args: [in] Arguments forwarded to the wrapped loss function
    constructor (e.g. ``delta`` when ``T`` is ``HuberLossFunctionBatch``).
  :returns: Constructor has no return value.
  :throws std\:\:invalid_argument: if ``a <= 0``.

  ``T`` must derive from ``LossFunctionBatch``. The inner loss is owned by
  value and constructed from the forwarded arguments, following the same
  decorator pattern as ``InformationFactorBatch<T>``.

**Formula**

  Given an inner loss :math:`f(s)` and a positive scalar :math:`a`:

.. math::

   \rho(s) = a\,f(s),\qquad
   \rho'(s) = a\,f'(s),\qquad
   \rho''(s) = a\,f''(s).

**C++ example**

.. code-block:: cpp

   // Scale Huber loss by 0.5 — the delta=1.0 argument is forwarded
   // to the HuberLossFunctionBatch constructor.
   cunls::ScaledLossFunctionBatch<cunls::HuberLossFunctionBatch> loss(0.5f, 1.0f);
