################################################################################
Robustifier API
################################################################################

The ``cunls/robustifier`` module defines GPU-batched robust loss (robustifier)
functions used by :cpp:class:`ResidualBatch` to reduce the influence of
outliers in non-linear least squares.

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

TrivialLossFunctionBatch
------------------------

Header: :code:`cunls/robustifier/trivial_loss_function_batch.h`

.. cpp:function:: TrivialLossFunctionBatch()

  :returns: Constructor has no return value.

**Formula (unscaled)**

.. math::

   \rho(s) = s,\qquad \rho'(s) = 1,\qquad \rho''(s) = 0.

Identity loss: no robustification; equivalent to standard least squares.

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

Theory — How robustifier outputs are used in optimization
==========================================================

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

  yield a Gauss-Newton step equivalent to the robustified problem. When
  :math:`2\rho''\|f\|^2 + \rho' \lesssim 0`, :math:`\alpha` is capped (e.g.
  :math:`\alpha \le 1-\epsilon`) to avoid numerical issues. The robustifier
  output :math:`(\rho(s), \rho'(s), \rho''(s))` is used to compute
  :math:`\alpha` and the scaling factors :math:`\sqrt{\rho'}` and
  :math:`(1-\alpha)^{-1}` applied to residuals and Jacobians in the solver.
  For more detail, see the Ceres Solver documentation on
  `LossFunction <http://ceres-solver.org/nnls_modeling.html#lossfunction>`_
  and the references therein (e.g. Triggs).
