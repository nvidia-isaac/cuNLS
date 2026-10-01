###############################################################################
Robust Estimation with RANSAC
###############################################################################

cuNLS ships two RANSAC minimizers, ``RansacGaussNewtonMinimizer`` and
``RansacLevenbergMarquardtMinimizer`` (C++ header
:code:`cunls/minimizer/ransac_minimizer.h`; Python ``pycunls``). They solve the
same :cpp:class:`Problem` as the regular minimizers, but they assume that some
measurements are **gross outliers**: wrong data association, not just noise.
They find the estimate that most measurements agree with, refine it on those
measurements, and report which measurements were inliers.

This page explains when to use them, the theory behind them, exactly what the
implementation does, how to call it from C++ and Python, and how to tune it.
How to write custom factors and states that work with RANSAC is covered in
:doc:`custom_factors_and_states`. Complete runnable programs are in
``examples/ransac_pnp`` (C++) and ``python/examples/ransac_pnp.py``.

.. contents:: On this page
   :local:
   :depth: 2

===============================================================================
When to use RANSAC
===============================================================================

A least-squares solver weighs every residual by its square. A single
measurement that is wrong by 100 noise standard deviations contributes as
much as 10,000 good ones, so a few outliers move the solution arbitrarily far.
There are two families of remedies:

**Robust losses** (Huber, Cauchy, Tukey, ...; see :doc:`api/robustifier`)
  down-weight large residuals inside the ordinary solver. They are cheap and
  work well when the outlier ratio is moderate *and* the initial guess is
  already close: the loss decides what "large" means from the current
  estimate, so a poor start can make the outliers look like the inliers.

**RANSAC** (RANdom SAmple Consensus) does not trust the initial estimate
  to tell inliers from outliers. It generates many candidate solutions
  ("hypotheses") from tiny random subsets of the measurements, keeps the one
  that most measurements agree with, and only then refines it. It tolerates
  very high outlier ratios (80–90% in the PnP benchmarks) and returns an
  explicit inlier/outlier classification.

Use the RANSAC minimizers when:

- a significant fraction of the measurements can be completely wrong
  (feature mismatches, wrong loop closures, spurious returns);
- you need the inlier set, not just the estimate;
- the **free state is small**: the sum of the tangent dimensions of all
  non-constant state blocks must be at most ``kMaxRansacTangentDim = 64``
  (a camera pose is 6, a pose plus a focal length is 7, a rig of 10 poses
  is 60). Constant blocks (e.g. known 3D points) do not count, however many
  there are, and the number of factors is unlimited.

Typical problems: PnP (pose from 3D-2D matches), point-cloud registration
(pose from 3D-3D matches), relative pose / extrinsic calibration, fitting a
low-dimensional model (line, plane, homography-like) to data, a multi-camera
rig pose with known extrinsics.

===============================================================================
Theory
===============================================================================

-------------------------------------------------------------------------------
The problem
-------------------------------------------------------------------------------

The regular minimizers solve

.. math::
   \min_x \; \frac{1}{2} \sum_i \rho_i\!\left(\|r_i(x)\|^2\right)

over the free state :math:`x` (dimension :math:`D`) for residual blocks
:math:`r_i` of dimension :math:`m_i`. RANSAC assumes the residual blocks are
of two kinds:

- **inliers**: :math:`r_i(x^\star)` at the true state :math:`x^\star` is small,
  of the size of the measurement noise;
- **outliers**: :math:`r_i(x^\star)` is arbitrary.

Which is which is unknown. The goal is the state that explains as many
measurements as possible within the noise level, and that partition.

In cuNLS every residual batch has a **role**:

``kSampled`` (Python ``RansacRole.sampled``)
  Data factors that may be outliers. RANSAC samples from them and classifies
  every one of them. Each sampled batch has an **inlier threshold**
  :math:`\tau`: factor :math:`i` is an inlier of state :math:`x` iff
  :math:`\|r_i(x)\| \le \tau`.

``kAlwaysOn`` (Python ``RansacRole.always_on``)
  Trusted factors: priors, motion models, known extrinsic constraints. They
  are part of every solve and never classified.

-------------------------------------------------------------------------------
Hypotheses from minimal samples
-------------------------------------------------------------------------------

A **minimal sample** is a set of :math:`s` sampled factors that, together
with the always-on factors, determines the state. Each factor contributes
:math:`m` equations, so :math:`s = \lceil D / m_{\min} \rceil` factors are
enough in general (:math:`m_{\min}` is the smallest residual dimension among
the sampled batches). For PnP, :math:`D = 6` and :math:`m = 2`, so
:math:`s = 3` correspondences.

If all :math:`s` factors of a sample are inliers, the state fitted to them is
close to :math:`x^\star`, and most other inliers will agree with it. If any is
an outlier, the fitted state is essentially random and few factors agree with
it. RANSAC therefore draws many samples and keeps the hypothesis with the
most support.

Classic RANSAC fits each sample with a problem-specific closed-form
"minimal solver" (e.g. P3P for PnP). cuNLS instead fits each sample with a
few **Gauss-Newton or Levenberg-Marquardt iterations from the initial
guess**, using the problem's own factors. This needs no problem-specific
code, so *any* factor (built-in or custom) works, at the price of needing an
initial guess inside the convergence basin of the sample problem. In the PnP
benchmarks this basin is wide: rotations off by 0.3 rad and translations off
by 10% of the depth converge reliably.

-------------------------------------------------------------------------------
Scoring
-------------------------------------------------------------------------------

Every hypothesis :math:`x_k` is scored against **all** sampled factors.
The default rule is MSAC (M-estimator SAmple Consensus), a truncated quadratic:

.. math::
   \mathrm{score}(x_k) = \sum_{i \in \text{sampled}} \min\!\left(\|r_i(x_k)\|^2,\ \tau_i^2\right)
   \quad (+\ 2 \cdot \mathrm{cost}_{\text{always-on}}(x_k))

Lower is better. Inliers contribute their (small) squared residual, outliers
a constant :math:`\tau^2`. Compared with counting inliers, MSAC also prefers
the hypothesis whose inliers fit more tightly, which breaks ties between
hypotheses with the same support. ``RansacScoring::kInlierCount`` counts
inliers instead (ties broken by MSAC). When ``score_always_on`` is set (the
default), the always-on cost is added so that hypotheses violating a trusted
prior are penalized.

**Informative inliers.** Some factors report a zero residual for
configurations they cannot evaluate. ``PnPFactorBatch``, for example,
returns zero residual and zero Jacobian for points behind the camera. A
hypothesis that puts every point behind the camera would then look perfect.
With ``require_informative_inliers`` (default on), a factor counts as an
inlier only if its Jacobian has a non-zero entry on a free state block.

-------------------------------------------------------------------------------
How many hypotheses?
-------------------------------------------------------------------------------

Let :math:`w` be the inlier ratio. A random sample of :math:`s` factors is
all-inlier with probability :math:`w^s`. To draw at least one all-inlier
sample with probability :math:`p` (``confidence``), one needs

.. math::
   K \ \ge\ \frac{\log(1 - p)}{\log(1 - w^s)}

hypotheses. For :math:`p = 0.999`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - inlier ratio :math:`w`
     - :math:`s = 2`
     - :math:`s = 3`
     - :math:`s = 4`
     - :math:`s = 6`
   * - 0.9
     - 5
     - 6
     - 7
     - 10
   * - 0.5
     - 25
     - 52
     - 108
     - 439
   * - 0.3
     - 74
     - 253
     - 850
     - 9,473
   * - 0.2
     - 170
     - 861
     - 4,314
     - 107,931
   * - 0.1
     - 688
     - 6,905
     - 69,075
     - :math:`6.9 \cdot 10^6`

:math:`w` is unknown in advance, so cuNLS works in **rounds** of
``hypotheses_per_round`` (default 256) hypotheses and stops adaptively: after
each round it sets :math:`w` to the best inlier ratio found so far and stops
as soon as the hypotheses drawn reach the bound above (or the inlier ratio
reaches ``early_stop_inlier_ratio``, or ``max_rounds`` is hit). The table
shows why small samples matter: prefer factors with larger residual
dimension (fewer factors per sample) and keep :math:`D` small.

-------------------------------------------------------------------------------
Choosing the inlier threshold
-------------------------------------------------------------------------------

:math:`\tau` is in the units of the raw residual (before any loss function),
and it is the single most important parameter. If the residual of an inlier
is Gaussian with per-component standard deviation :math:`\sigma`, then
:math:`\|r\|^2 / \sigma^2` follows a :math:`\chi^2` distribution with
:math:`m` degrees of freedom, and

.. math::
   \tau = \sigma \sqrt{\chi^2_m(q)}

keeps a fraction :math:`q` of the true inliers:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - residual dim :math:`m`
     - :math:`q = 0.95`
     - :math:`q = 0.99`
     - :math:`q = 0.999`
   * - 1
     - :math:`1.96\sigma`
     - :math:`2.58\sigma`
     - :math:`3.29\sigma`
   * - 2
     - :math:`2.45\sigma`
     - :math:`3.03\sigma`
     - :math:`3.72\sigma`
   * - 3
     - :math:`2.80\sigma`
     - :math:`3.37\sigma`
     - :math:`4.03\sigma`
   * - 6
     - :math:`3.55\sigma`
     - :math:`4.10\sigma`
     - :math:`4.74\sigma`

Too small a threshold rejects good measurements and makes the estimate noisy;
too large a threshold accepts outliers that lie close to the model. If the
noise is not isotropic, wrap the factor batch in an ``InformationFactorBatch``
with the square-root information matrix: the residual is then whitened
(:math:`\sigma = 1`) and :math:`\tau = \sqrt{\chi^2_m(q)}` directly.

-------------------------------------------------------------------------------
Refinement
-------------------------------------------------------------------------------

The best hypothesis was fitted to only :math:`s` factors, so it carries the
noise of those few measurements. The final step classifies all sampled
factors at the best hypothesis and runs ``final_iterations`` of
Gauss-Newton / Levenberg-Marquardt on **the inliers plus the always-on
factors**, which averages the noise of all inliers. The inliers are then
re-classified at the refined state, and the result and its inlier mask are
returned. If the refined state scores worse than the best hypothesis (rare;
it can happen when the hypothesis was poor), the hypothesis is returned
instead and ``RansacSummary::refinement_reverted`` is set.

===============================================================================
What the implementation does
===============================================================================

One call to ``Minimize(stream, problem)``:

1. **Validate and lay out** the problem (see :ref:`ransac-limits`): find the
   free blocks and their tangent columns (:math:`D` total), the role of every
   residual batch, the sample size :math:`s`.
2. **Rounds** (at most ``max_rounds``), each with :math:`K` =
   ``hypotheses_per_round`` hypotheses solved **in parallel on the GPU**:

   a. Every hypothesis gets its own copy ("replica") of the free state
      batches, initialized from the problem's current state values.
   b. Hypothesis :math:`k` draws its sample: the first :math:`s` entries of a
      keyed pseudo-random permutation of all sampled factors (key = seed,
      round, :math:`k`). No sorting, no RNG state; the same seed gives the
      same samples.
   c. ``hypothesis_iterations`` (default 5) GN or LM iterations per
      hypothesis on its sample plus the always-on factors. Each iteration
      builds the small dense normal equations
      :math:`J^\top J\,\delta = -J^\top r` per hypothesis, solves them with a
      pivoted LDLᵀ (default) or Cholesky, and applies the step through the
      state batch's ``Plus``. A hypothesis whose first solve fails (degenerate
      sample) is marked invalid.
   d. Every hypothesis is scored on all sampled factors (MSAC). For large
      problems, **two-stage scoring** (when there are more than
      2 × ``scoring_subset_size`` sampled factors) first scores all
      hypotheses on a random subset of 16,384 factors and then scores only
      the best ``scoring_finalists`` (default 4) on all factors.
   e. The best hypothesis so far is kept on the device. The adaptive stopping
      rule decides whether another round is needed.

3. **Refine** the best hypothesis on its inliers (``final_iterations``,
   default 20), re-classify, write the state back into the problem's state
   batches, and fill the summary.

**All hypotheses of a round are evaluated together.** The minimizer does not
call your factor once per hypothesis. It calls ``FactorBatch::Evaluate``
once per residual batch for *all* hypotheses, using the item parameters
``factor_ids`` / ``num_factor_ids`` to say which factor each output row
belongs to and which hypothesis's state it reads. Likewise it calls
``StateBatch::Plus`` once per state batch for all hypotheses, using
``num_replicas``. This is why custom factors and states must honor those
parameters (:doc:`custom_factors_and_states`). Everything else (normal
equations, solves, scoring, selection) runs in a handful of fused kernels
without atomics.

**Deterministic.** Every reduction runs in a fixed order and sampling is
counter-based, so the same problem and the same ``seed`` give bitwise
identical results, run after run.

**Synchronization.** The host reads a few scalars once per round (to decide
whether to stop) and at the end. Everything else is asynchronous on the given
stream.

===============================================================================
Usage
===============================================================================

.. important::

   **Capacity vs. active count.** Factor and state batches are constructed with
   their *capacity* (how many factors / state blocks their buffers hold) and
   start with **zero** active entries: call ``SetNumFactors(n)`` /
   ``SetNumStateBlocks(n)`` (Python: ``set_num_factors`` /
   ``set_num_state_blocks``) before solving, and again whenever the problem size
   changes. See :ref:`capacity-and-active-count`.

Build the problem exactly as for the regular minimizers. Then choose the
roles and thresholds, run the minimizer, and read the inlier mask. The
example below is the PnP problem from ``examples/ransac_pnp``: one SE(3)
pose state and one ``PnPFactorBatch`` with a factor per 3D-2D
correspondence, all pointing to the pose.

-------------------------------------------------------------------------------
C++
-------------------------------------------------------------------------------

.. code-block:: cpp

   #include "cunls/cunls.h"

   // ... states, factors and problem built as usual:
   //   pose_state.SetNumStateBlocks(1);
   //   pnp.SetNumFactors(num_matches);
   //   problem.AddStateBatch(&pose_state);
   //   problem.AddFactorBatch(&pnp, pointers);          // residual batch 0

   cunls::RansacLevenbergMarquardtMinimizerOptions options;
   cunls::RansacMinimizerOptions &ransac = options.base_options;
   // One entry per residual batch, in the order they were added to the problem.
   ransac.factor_batches = {{cunls::RansacRole::kSampled, /*inlier_threshold=*/0.01f}};
   ransac.seed = 1;

   cunls::RansacLevenbergMarquardtMinimizer minimizer(options);
   cunls::RansacSummary summary = minimizer.Minimize(stream, problem);

   // The estimate is now in the problem's state batches (here: pose_state).
   // Inlier mask of residual batch 0: device memory, one byte per factor.
   std::vector<uint8_t> mask(minimizer.InlierMaskSize(0));
   cudaMemcpy(mask.data(), minimizer.InlierMask(0), mask.size(), cudaMemcpyDeviceToHost);

   printf("%zu rounds, %zu inliers (%.1f%%)\n", summary.num_rounds, summary.num_inliers,
          100.f * summary.inlier_ratio);

``RansacGaussNewtonMinimizer`` takes a ``RansacMinimizerOptions`` directly:

.. code-block:: cpp

   cunls::RansacMinimizerOptions options;
   options.default_inlier_threshold = 0.01f;   // every batch sampled with this threshold
   cunls::RansacGaussNewtonMinimizer minimizer(options);

-------------------------------------------------------------------------------
Python
-------------------------------------------------------------------------------

.. code-block:: python

   import pycunls

   # ... states, factors and problem built as usual (residual batch 0 = PnP).

   options = pycunls.RansacLevenbergMarquardtMinimizerOptions()
   ransac = options.base_options          # a reference: edits change `options`
   ransac.factor_batches = [               # assign a whole list, one entry per batch
       pycunls.RansacFactorBatchOptions(pycunls.RansacRole.sampled, 0.01),
   ]
   ransac.seed = 1

   minimizer = pycunls.RansacLevenbergMarquardtMinimizer(options)
   summary = minimizer.minimize(stream, problem)     # writes the estimate back
   mask = minimizer.inlier_mask(0)                   # numpy uint8, 1 = inlier
   print(summary)                                     # RansacSummary(rounds=..., inliers=...)

.. note::

   ``options.factor_batches`` returns a copy of the list:
   ``options.factor_batches.append(...)`` has no effect. Always assign a
   complete list.

-------------------------------------------------------------------------------
Roles in practice
-------------------------------------------------------------------------------

- **One data batch, nothing else** (PnP, registration, model fitting): leave
  ``factor_batches`` empty and set ``default_inlier_threshold``; every batch is
  then sampled with that threshold.
- **Data plus priors** (e.g. a pose prior from odometry): mark the prior
  batches ``kAlwaysOn``. They take part in every hypothesis solve and the
  refinement and, with ``score_always_on``, in the score. If the prior may be
  wrong by much more than its stated uncertainty, set ``score_always_on =
  false`` so it does not veto the correct hypothesis.
- **Several data batches** (e.g. one per camera of a rig): mark each
  ``kSampled`` with its own threshold. Samples are drawn from the union of all
  sampled factors, and ``InlierMask(i)`` gives the mask of batch ``i``.
- **Known quantities** (3D landmarks in PnP, rig extrinsics): put them in
  state batches with constant blocks (``const_state_ids``); they cost nothing
  towards :math:`D`.

-------------------------------------------------------------------------------
Gauss-Newton or Levenberg-Marquardt?
-------------------------------------------------------------------------------

Both variants share every option and step. ``RansacGaussNewtonMinimizer``
takes full Gauss-Newton steps and is the fastest. ``RansacLevenbergMarquardt
Minimizer`` damps each hypothesis's steps with its own :math:`\lambda`, which
widens the convergence basin when the initial guess is far off or the problem
is strongly nonlinear, at some extra cost per iteration. Start with GN; switch
to LM if hypotheses fail to converge from your initial guesses.

===============================================================================
Options reference and tuning
===============================================================================

Defaults are tuned on PnP and work for most small problems.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Option
     - Default
     - Meaning and advice
   * - ``factor_batches``
     - empty
     - Role and inlier threshold per residual batch (same order as
       ``Problem::GetResidualBatches()``). Empty: all sampled with
       ``default_inlier_threshold``.
   * - ``default_inlier_threshold``
     - 1.0
     - Threshold :math:`\tau` used when ``factor_batches`` is empty. Set it from
       the noise level (see above); the default is only a placeholder.
   * - ``hypotheses_per_round``
     - 256
     - Hypotheses solved in parallel per round. Larger rounds use the GPU
       better but can overshoot the number actually needed; 256–1024 is a good
       range.
   * - ``max_rounds``
     - 8
     - Upper bound on rounds. The maximum number of hypotheses is
       ``hypotheses_per_round * max_rounds``; raise it for very high outlier
       ratios (see the table above).
   * - ``confidence``
     - 0.999
     - Target probability of having drawn at least one all-inlier sample.
   * - ``early_stop_inlier_ratio``
     - 1.0
     - Stop as soon as the best inlier ratio reaches this value (1 disables).
   * - ``sample_size``
     - 0
     - Factors per minimal sample; 0 = :math:`\lceil D / m_{\min} \rceil`.
       Increase it if minimal samples are often degenerate (e.g. collinear
       points), at the price of more hypotheses.
   * - ``seed``
     - 0
     - Sampler seed. Same seed and problem, bitwise identical result.
   * - ``scoring``
     - MSAC
     - ``kMSAC`` or ``kInlierCount``.
   * - ``score_always_on``
     - true
     - Add the always-on cost to the score.
   * - ``require_informative_inliers``
     - true
     - Count a factor as an inlier only if its Jacobian is non-zero on a free
       block. Disable only for factors that never report zero residuals for
       invalid configurations; it saves one Jacobian evaluation per scored
       factor.
   * - ``scoring_subset_size``
     - 16384
     - Two-stage scoring for problems with more than twice this many sampled
       factors. 0 always scores every hypothesis on every factor.
   * - ``scoring_finalists``
     - 4
     - Hypotheses re-scored on all factors in two-stage scoring (at most 64).
   * - ``scoring_memory_budget_bytes``
     - 64 MiB
     - Device memory for scoring buffers; bounds how many hypotheses are
       scored at once. Results do not depend on it.
   * - ``hypothesis_iterations``
     - 5
     - GN / LM iterations per hypothesis. More helps a poor initial guess.
   * - ``final_iterations``
     - 20
     - Iterations of the final refinement (stops early on convergence).
   * - ``state_tolerance``, ``cost_tolerance``
     - 1e-10, 1e-7
     - Per-hypothesis convergence on the squared step norm and the relative
       cost decrease.
   * - ``linear_solver``
     - LDLT
     - ``kLDLT`` gives rank-deficient directions a zero step, which suits the
       near-singular systems minimal samples produce. ``kCholesky`` marks such
       hypotheses invalid.

``RansacLevenbergMarquardtMinimizerOptions`` adds the damping parameters of
the regular LM minimizer (``initial_lambda`` = 1e-3, ``lambda_upscale`` = 2,
``lambda_downscale`` = 0.5, ``lambda_min``/``lambda_max``,
``step_accept_threshold``, ``lambda_downscale_threshold``), applied per
hypothesis.

**Reading the summary.** ``RansacSummary`` extends ``MinimizerSummary``
(``initial_cost`` over all factors, ``final_cost`` of the refinement over the
inliers and always-on factors, ``num_iterations`` and ``iteration_costs`` of
the refinement) with ``num_rounds``, ``num_hypotheses``,
``num_valid_hypotheses``, ``num_inliers``, ``inlier_ratio``, ``best_score``
and ``refinement_reverted``.

**Troubleshooting.**

- *Too few inliers / wrong estimate*: check the threshold against the actual
  residual noise (evaluate the residuals at a known good state); raise
  ``max_rounds`` for low inlier ratios; use the LM variant or more
  ``hypothesis_iterations`` if the initial guess is far.
- *Many invalid hypotheses* (``num_valid_hypotheses`` much smaller than
  ``num_hypotheses``): samples are often degenerate. Increase
  ``sample_size`` by one or two.
- *Inliers that are clearly wrong*: the threshold is too large, or a factor
  returns zero residuals for invalid configurations without
  ``require_informative_inliers``.

.. _ransac-limits:

===============================================================================
Limits and requirements
===============================================================================

``Minimize`` throws ``std::invalid_argument`` (Python ``ValueError``) with an
explanatory message when:

- the free tangent dimension :math:`D` is 0 or exceeds 64;
- no residual batch is sampled, or the sample size exceeds the number of
  sampled factors;
- ``factor_batches`` is non-empty but its length differs from the number of
  residual batches, or a sampled batch has a non-positive threshold;
- a factor references more than 8 state blocks, or a state block that
  belongs to no registered state batch;
- a residual batch uses numeric Jacobians (``JacobianMode::kNumeric``), which
  RANSAC does not support yet;
- ``Problem::CheckConsistency()`` fails, or an option is out of range
  (``hypotheses_per_round``, ``max_rounds`` or ``hypothesis_iterations`` of 0,
  ``confidence`` outside (0, 1)).

Every factor and state batch must support the item / replica parameters of
``Evaluate`` and ``Plus``. All built-in batches do; for your own types see
:doc:`custom_factors_and_states`.

===============================================================================
Performance
===============================================================================

PnP, 30% outliers, one round of 256 hypotheses, median wall time (NVIDIA RTX
A6000):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - correspondences
     - LM + Cauchy loss
     - RANSAC-GN
     - RANSAC-LM
   * - 1,000
     - 0.7 ms
     - 1.4 ms
     - 1.4 ms
   * - 100,000
     - 3.7 ms
     - 7.3 ms
     - 7.5 ms
   * - 1,000,000
     - 31.6 ms
     - 14.9 ms
     - 47.3 ms

In the same benchmarks RANSAC succeeds in 100% of trials from 0% to 90%
uniformly random outliers, from both small and large initial errors, and
against coherent outliers (a second, competing pose); plain GN/LM fail from
10% outliers and LM with a Huber loss from 60%.
