###############################################################################
Numeric (finite-difference) Jacobians
###############################################################################

Every shipped cuNLS factor computes its Jacobian analytically: a hand-derived
closed form, evaluated in the same fused CUDA kernel as the residual. That's
the fastest option, but deriving a correct closed-form Jacobian by hand isn't
always worth the effort — especially while prototyping a new factor, or for
a residual that's awkward to differentiate.

cuNLS can compute the Jacobian for you instead, via finite differences. A
factor that implements **only** a residual (``Evaluate`` never has to write
to its ``jacobians`` argument) can be optimized exactly like any other
factor — you just tell the minimizer to differentiate it numerically.

===============================================================================
Enabling numeric Jacobians
===============================================================================

The switch is :cpp:enum:`JacobianMode` (``cunls/minimizer/jacobian_mode.h``),
with two values: ``kAnalytic`` (default) and ``kNumeric``. It can be set two
ways:

- **Globally**, via :code:`MinimizerOptions::jacobian_mode` — applies to
  every factor batch in the problem unless overridden.
- **Per factor group**, via the optional last argument of
  :cpp:func:`Problem::AddFactorBatch` — overrides the global default for
  just that one factor batch. This lets you mix modes in a single
  ``Problem``: for example, keep cuNLS's shipped, analytically-differentiated
  factors on the fast path while a new factor you're still prototyping uses
  numeric differentiation.

.. code-block:: cpp

   #include "cunls/cunls.h"

   // Option A: set the global default for every factor batch in the problem.
   cunls::MinimizerOptions options;
   options.jacobian_mode = cunls::JacobianMode::kNumeric;

   // Option B: override just one factor group, leaving everything else
   // (including shipped factors) on the global default (kAnalytic here).
   cunls::Problem problem;
   problem.AddFactorBatch(&my_factor, state_pointers, cunls::JacobianMode::kNumeric);

A factor doesn't need any special marker to be eligible for numeric
differentiation — every :cpp:class:`FactorBatch` must already support
residual-only evaluation (``jacobians == nullptr``, used for cost-only
evaluation), and that's the only requirement. See :ref:`factor-inputs` in
:doc:`api/factor` and the worked example below.

===============================================================================
How it works
===============================================================================

Numeric differentiation is manifold-aware: for each tangent-space
coordinate of each state block a factor references, cuNLS perturbs the
state via that state batch's own :cpp:func:`StateBatch::Plus` (the same
retraction the minimizer uses to apply solved steps), evaluates the
residual at the perturbed state, and differences against a second
evaluation (forward difference: the unperturbed baseline; central
difference: the same perturbation in the opposite direction). This means
numeric Jacobians are correct on SO2/SO3/SE2/SE3/Sim2/Sim3/SL4 states, not
just Euclidean ``Vector<Dim>`` states — there's no need to reason about
exponential maps or local parameterizations yourself.

:cpp:struct:`NumericDiffOptions` controls the scheme:

- **method**: ``kCentral`` (default, two-sided,
  :math:`(f(x+\epsilon)-f(x-\epsilon))/(2\epsilon)`, more accurate) or
  ``kForward`` (one-sided, :math:`(f(x+\epsilon)-f(x))/\epsilon`, cheaper).
- **relative_step_size**: the per-tangent-coordinate perturbation
  :math:`\epsilon`. Default: ``1e-4``.

===============================================================================
Accuracy and performance
===============================================================================

cuNLS is float32 throughout, so numeric Jacobians are finite-difference
*approximations*, not exact derivatives — expect agreement with an analytic
Jacobian to roughly 1e-2–1e-3 relative accuracy with the default central
difference, not machine precision. Loosen `MinimizerOptions::cost_tolerance`
slightly for problems solved entirely with numeric Jacobians if you see
convergence stall just short of an analytic run's final cost.

Numeric differentiation costs more than an analytic Jacobian: computing it
requires evaluating the factor's residual multiple times per tangent
coordinate (twice per coordinate for central differences), whereas an
analytic factor computes residual and Jacobian together in one kernel.
Internal benchmarking across PGO/SBA/PnP-scale problems shows numeric-diff
Jacobian evaluation taking roughly 3-12x longer than the equivalent analytic
kernel, with the gap widening for factors that touch more state-block tangent
dimensions. In practice this cost is often small relative to the sparse
linear solve that dominates most iterations — but for factors that run at
scale and are worth the extra effort, prefer writing an analytic Jacobian.
Numeric differentiation is best suited to prototyping, one-off factors, or
residuals where a closed-form derivative genuinely isn't worth deriving.

===============================================================================
Worked example
===============================================================================

``examples/custom_factor/`` solves the same toy problem two ways: once with
a hand-derived analytic Jacobian, once with only a residual. The
residual-only factor:

.. code-block:: cpp

   // Same residual as the analytic version: r_i = (x_{i+1} - x_i) - m_i.
   // No Jacobian code path at all -- `jacobians` is simply never touched.
   __global__ void ScalarDifferenceResidualOnlyKernel(
       const float *measurements, float const *const *state_pointers,
       float *residuals, size_t num_factors) {
     const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
     if (idx >= num_factors) return;
     const float *left = state_pointers[idx * 2];
     const float *right = state_pointers[idx * 2 + 1];
     if (residuals != nullptr) {
       residuals[idx] = (right[0] - left[0]) - measurements[idx];
     }
   }

   class ScalarDifferenceResidualOnlyFactorBatch
       : public cunls::SizedFactorBatch<1, 1, 1> {
    public:
     ScalarDifferenceResidualOnlyFactorBatch(const float *measurements, size_t num_factors)
         : measurements_(measurements), num_factors_(num_factors) {}

     bool Evaluate(float *residuals, float * /*jacobians*/,
                   float const *const *state_pointers, cudaStream_t stream) const final {
       constexpr int kBlockSize = 256;
       const int grid_size = static_cast<int>((num_factors_ + kBlockSize - 1) / kBlockSize);
       ScalarDifferenceResidualOnlyKernel<<<grid_size, kBlockSize, 0, stream>>>(
           measurements_, state_pointers, residuals, num_factors_);
       THROW_ON_CUDA_ERROR(cudaGetLastError());
       return true;
     }

     size_t NumFactors() const final { return num_factors_; }

    private:
     const float *measurements_;
     size_t num_factors_;
   };

Registering it with the per-group override (while an anchor
``PriorFactorBatch`` in the same problem stays analytic):

.. code-block:: cpp

   problem.AddFactorBatch(&numeric_difference_factor, diff_state_pointers,
                          cunls::JacobianMode::kNumeric);
   problem.AddFactorBatch(&anchor_factor, anchor_state_pointers);  // stays analytic

See ``examples/custom_factor/README.md`` for the full walkthrough and an
"analytic vs. numeric" comparison table.
