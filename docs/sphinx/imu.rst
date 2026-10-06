###############################################################################
IMU factor
###############################################################################

``ImuFactorBatch`` ties two keyframes together through the raw gyroscope and
accelerometer samples recorded between them. It is the inertial part of
visual-inertial bundle adjustment, inertial PnP and visual-inertial odometry
back ends. The keyframe poses it reads are the same pose states that
``ReprojectionFactorBatch`` and ``PnPFactorBatch`` read.

Unlike the classical *preintegrated* IMU factor, it does not cache integrated
deltas at a reference bias. Every evaluation integrates the samples at the
current bias, treats the states between samples as variables of an inner
least-squares problem, and **eliminates them exactly** (a Schur complement).
The outer solver sees a dense factor on the two keyframes whose normal
equations are exactly those of the full sample-by-sample chain at the current
linearization point. There is no reference bias, no first-order bias
correction and no re-integration policy to tune.

===============================================================================
When to use it
===============================================================================

- **Visual-inertial bundle adjustment** (keyframes, landmarks, IMU between
  consecutive keyframes): estimate poses, velocities and IMU biases together
  with the structure. See :ref:`imu-example`.
- **Inertial PnP / tracking**: two frames joined by an IMU factor, one of them
  with prior information from the previous solve, landmarks fixed. See
  :ref:`imu-inertial-pnp-example`.
- **Inertial pose graphs and smoothing**: keyframes with IMU factors and other
  pose factors (priors, between factors, position priors).

The factor needs a **known gravity vector** in the world frame (a parameter,
not estimated) and **at least one sample per keyframe pair**.

===============================================================================
Conventions
===============================================================================

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Quantity
     - Convention
   * - Pose state :math:`T`
     - ``SE3StateBatch``, **world_from_rig** (the pose of the body or rig in
       the world), the same state as ``ReprojectionFactorBatch`` and
       ``PnPFactorBatch`` read (:ref:`pose-convention`). Perturbed on the
       right, :math:`T\,\mathrm{Exp}(\xi)`, :math:`\xi = [\varphi;\ \rho]` in
       the rig frame.
   * - Extrinsic
     - ``ImuParameters.body_from_imu`` :math:`T_{bi}` (rig_from_imu, an
       ``SE3Transform``). The IMU's pose in the world is
       :math:`T_{wi} = T\,T_{bi}`. Identity when the IMU defines the rig
       frame.
   * - Velocity :math:`v`
     - ``VectorStateBatch3``: world-frame velocity **of the IMU origin** (the
       rig velocity when the IMU sits at the rig origin). With this definition
       a non-zero extrinsic translation needs no lever-arm approximation.
   * - Bias :math:`b`
     - ``VectorStateBatch6``: :math:`[b_g;\ b_a]`, gyroscope [rad/s] then
       accelerometer [m/s²], in the IMU frame.
   * - Gravity :math:`g`
     - ``ImuParameters.gravity``, world frame, default
       :math:`(0, 0, -9.80665)` (+Z up, ROS REP-103). For an OpenCV-style world
       with +Y down, use :math:`(0, 9.81, 0)`.
   * - Samples
     - Gyroscope :math:`\tilde\omega` [rad/s] and specific force
       :math:`\tilde a` [m/s²] in the IMU frame, and the step duration
       :math:`\Delta t` [s] over which they are applied.

**Specific force.** An accelerometer at rest on a table reads :math:`+9.81`
m/s² along the up axis: :math:`\tilde a = R^\top (a - g) + b_a`, with :math:`a`
the true world acceleration. Feed the raw accelerometer output.

===============================================================================
Theory
===============================================================================

-------------------------------------------------------------------------------
Measurement model and kinematics
-------------------------------------------------------------------------------

With :math:`(R, v, p)` the IMU's orientation, velocity and position in the
world, the sensor measures

.. math::

   \tilde\omega = \omega + b_g + \eta_g, \qquad
   \tilde a = R^\top (a - g) + b_a + \eta_a,

where :math:`\omega` is the body angular velocity and :math:`\eta_g, \eta_a`
are white noise. The kinematics are

.. math::

   \dot R = R\,[\tilde\omega - b_g - \eta_g]_\times, \qquad
   \dot v = R(\tilde a - b_a - \eta_a) + g, \qquad
   \dot p = v.

-------------------------------------------------------------------------------
The discrete chain
-------------------------------------------------------------------------------

The :math:`N` samples between keyframes :math:`a` and :math:`b` define :math:`N`
Euler steps between states :math:`x_0 = x_a, x_1, \dots, x_N = x_b`. The bias of
keyframe :math:`a` is held over the interval:

.. math::

   \begin{aligned}
   R_{k+1} &= R_k \,\mathrm{Exp}\big((\tilde\omega_k - b_g)\,\Delta t_k\big) \\
   v_{k+1} &= v_k + g\,\Delta t_k + R_k(\tilde a_k - b_a)\,\Delta t_k \\
   p_{k+1} &= p_k + v_k\,\Delta t_k + \tfrac12 g\,\Delta t_k^2
              + \tfrac12 R_k(\tilde a_k - b_a)\,\Delta t_k^2
   \end{aligned}

Sample :math:`k` is applied over :math:`[t_k, t_{k+1})`; the durations add up
to the time between the keyframes. Write :math:`f_k(x_k)` for the right-hand
side. The intermediate states use the tangent
:math:`[\delta\varphi;\ \delta v;\ \delta p]` (right rotation perturbation,
world-frame velocity and position).

-------------------------------------------------------------------------------
Noise model
-------------------------------------------------------------------------------

The noise densities are continuous-time, as on IMU data sheets and in Kalibr
calibrations: a sample averaged over :math:`\Delta t` has variance
:math:`\sigma^2 / \Delta t`, and a random walk grows by :math:`\sigma^2 \Delta
t`. The defect of step :math:`k`,

.. math::

   d_k = x_{k+1} \ominus f_k(x_k) =
   \begin{bmatrix} \mathrm{Log}(f_R^\top R_{k+1}) \\ v_{k+1} - f_v \\ p_{k+1} - f_p \end{bmatrix},

then has covariance

.. math::

   Q_k = \begin{bmatrix}
     \sigma_g^2 \Delta t\, J_r J_r^\top & 0 & 0 \\
     0 & \sigma_a^2 \Delta t\, I & \tfrac12 \sigma_a^2 \Delta t^2\, I \\
     0 & \tfrac12 \sigma_a^2 \Delta t^2\, I &
       \big(\tfrac14 \sigma_a^2 \Delta t^3 + \sigma_i^2 \Delta t\big) I
   \end{bmatrix},

with :math:`J_r = J_r((\tilde\omega_k - b_g)\Delta t_k)` the SO(3) right
Jacobian. The velocity and position blocks are isotropic, so :math:`R_k` drops
out. The last term, ``integration_noise_density`` :math:`\sigma_i`, models the
Euler step's own error (GTSAM's integration covariance). It must be positive:
velocity and position noise both come from :math:`\eta_a`, so without it a
one-sample chain would have a singular covariance.

The biases follow a random walk across the interval, which gives six more rows

.. math::

   r_{\text{bias}} = \frac{b_b - b_a}{\sigma_b \sqrt{T}}, \qquad
   T = \textstyle\sum_k \Delta t_k,

with :math:`\sigma_{bg}` for the gyroscope bias and :math:`\sigma_{ba}` for the
accelerometer bias.

-------------------------------------------------------------------------------
Eliminating the chain
-------------------------------------------------------------------------------

Consider the chain as its own least-squares problem. The keyframe states
:math:`(T_a, v_a, b_a, T_b, v_b, b_b)` *and* the :math:`N-1` intermediate states
are variables, and every step contributes :math:`\|d_k\|^2_{Q_k^{-1}}`. It is
linearized at the states integrated forward from keyframe :math:`a`, where
every interior defect is zero. In the Gauss-Newton system

.. math::

   \begin{bmatrix} H_{BB} & H_{BI} \\ H_{IB} & H_{II} \end{bmatrix}
   \begin{bmatrix} \delta_B \\ \delta_I \end{bmatrix}
   = - \begin{bmatrix} g_B \\ g_I \end{bmatrix}

(:math:`B`: keyframe variables, :math:`I`: intermediate states), eliminating
:math:`\delta_I` leaves the **Schur complement**

.. math::

   \bar H = H_{BB} - H_{BI} H_{II}^{-1} H_{IB}, \qquad
   \bar g = g_B - H_{BI} H_{II}^{-1} g_I .

Solving the outer problem with :math:`(\bar H, \bar g)` gives the same keyframe
step as keeping every intermediate state in the problem. This is variable
elimination (Dellaert and Kaess), known as condensing in multiple shooting
(Bock and Plitt). The factor returns a residual :math:`r` and Jacobian
:math:`J` with

.. math::

   J^\top J = \bar H, \qquad J^\top r = \bar g, \qquad
   \|r\|^2 = \min_{\delta_I}\ (\text{linearized chain cost}),

so the outer minimizer also sees the chain's marginal cost, which line search
and the Levenberg-Marquardt gain ratio use. The marginal has rank 15: given
the start state and the bias, the chain predicts the 9-dimensional end state
up to noise, and the bias random walk adds 6. Hence 15 residual rows, not 30.

-------------------------------------------------------------------------------
Computing the marginal: covariance form
-------------------------------------------------------------------------------

The elimination is a Riccati recursion along the chain. Its *information*
form (a block-tridiagonal Cholesky factorization of :math:`H_{II}`) loses about
five digits in float32 over 200 samples. Each step subtracts the large
per-step information to leave the much smaller accumulated information. The
factor uses the equivalent *covariance* form, which only accumulates positive
terms. It propagates the covariance of the predicted end state,

.. math::

   \Sigma_{k+1} = \Phi_k \Sigma_k \Phi_k^\top + Q_k, \qquad \Sigma_0 = 0,
   \qquad
   \Phi_k = \begin{bmatrix}
     E_k^\top & 0 & 0 \\
     -R_k [\tilde a_k - b_a]_\times \Delta t_k & I & 0 \\
     -\tfrac12 R_k [\tilde a_k - b_a]_\times \Delta t_k^2 & \Delta t_k I & I
   \end{bmatrix},

with :math:`E_k = \mathrm{Exp}((\tilde\omega_k - b_g)\Delta t_k)`. For the
Jacobian it also propagates the sensitivities of the prediction to the start
rotation and to the bias. Measured against a float64 elimination of the
explicit chain at 200 samples, the float32 marginal Hessian is accurate to
about :math:`2 \cdot 10^{-6}` (relative). The information form is off by
:math:`10^{-1}`.

Let :math:`\hat x_b = (\hat R, \hat v, \hat p)` be the predicted IMU state at
keyframe :math:`b`, :math:`x_b = (R_b, v_b, p_b)` the IMU state given by
keyframe :math:`b`'s states, and :math:`\Sigma = L L^\top` (Cholesky). The
chain rows are

.. math::

   e = \begin{bmatrix}
     \mathrm{Log}(\hat R^\top R_b) \\ v_b - \hat v \\ p_b - \hat p
   \end{bmatrix}, \qquad
   r_{0:9} = L^{-1} e, \qquad
   J_{0:9} = L^{-1} D^{-1} \frac{\partial e}{\partial x}, \quad
   D = \mathrm{diag}\big(J_l^{-1}(e_R), I, I\big).

:math:`D` accounts for the rotation residual's tangent at the prediction.
Since :math:`J_l(e_R)\,e_R = e_R`, it leaves :math:`r` unchanged, and with it
the identities above hold exactly at any state. The test suite checks them
against the explicit chain. As usual in Gauss-Newton, the whitening is held
constant when differentiating.

-------------------------------------------------------------------------------
Jacobians
-------------------------------------------------------------------------------

A state perturbation :math:`T\,\mathrm{Exp}([\varphi;\ \rho])` acts in the rig
frame. With :math:`T = (R, t)` and :math:`T_{bi} = (R_{bi}, t_{bi})` it moves
the IMU by

.. math::

   \varphi_{\text{imu}} = R_{bi}^\top \varphi, \qquad
   \delta p_{\text{imu}} = -R\,[t_{bi}]_\times \varphi + R\,\rho.

Before whitening, :math:`\partial e/\partial(\cdot)` has these blocks (rows:
rotation; velocity; position):

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - State
     - Block
   * - :math:`T_a`
     - :math:`\big[-\Psi_\varphi R_{bi}^\top + [0;\ 0;\ R_a[t_{bi}]_\times]
       \ \big|\ [0;\ 0;\ -R_a]\big]`, where :math:`\Psi_\varphi =
       [\Delta R^\top;\ -R_0[\Delta v]_\times;\ -R_0[\Delta p]_\times]`
   * - :math:`v_a`
     - :math:`-[0;\ I;\ T\,I]`
   * - :math:`b_a`
     - :math:`-[G_g \mid G_a]`, the bias sensitivities of the prediction
   * - :math:`T_b`
     - :math:`\big[[\hat R^\top R_b;\ 0;\ -R_b[t_{bi}]_\times]\ \big|\ [0;\ 0;\ R_b]\big]`
   * - :math:`v_b`
     - :math:`[0;\ I;\ 0]`
   * - :math:`b_b`
     - 0 (in the bias rows only)

Here :math:`R_a, R_b` are the rotations of :math:`T_a, T_b`, :math:`R_0 = R_a
R_{bi}` is the IMU orientation at keyframe :math:`a`, and :math:`(\Delta R,
\Delta v, \Delta p)` are the integrated deltas in the frame of :math:`R_0`
(without gravity). Because :math:`\varphi` rotates a keyframe about its own
origin, the rotation columns contain only the extrinsic lever arm
:math:`|t_{bi}|`, not the distance from the world origin.

-------------------------------------------------------------------------------
Comparison with preintegration
-------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * -
     - Preintegrated factor (Forster et al.)
     - ``ImuFactorBatch``
   * - Deltas at the current bias
     - First-order correction from a reference bias; re-integration past a
       threshold
     - Exact: integrated at every evaluation
   * - Covariance
     - Computed once, at the reference bias
     - Recomputed at every evaluation
   * - Residual
     - :math:`9 + 6` (combined factor)
     - :math:`9 + 6`, the exact marginal of the sample chain
   * - Cost per evaluation
     - :math:`O(1)`, plus :math:`O(N)` re-integrations
     - :math:`O(N)`, on the GPU, parallel over factors and within chains

At the reference bias both give the same linearization. They differ as the
bias estimate moves away from it.

-------------------------------------------------------------------------------
Parallel evaluation
-------------------------------------------------------------------------------

The sweep along a chain is sequential, but it is **associative**. A run of
samples reduces to a summary in the frame of its first state: the deltas
:math:`(\Delta R, \Delta v, \Delta p)` without gravity, the duration
:math:`T`, the accumulated noise :math:`Q` and the bias sensitivities
:math:`G`. The run's transition is

.. math::

   \Phi = \begin{bmatrix}
     \Delta R^\top & 0 & 0 \\ -[\Delta v]_\times & I & 0 \\ -[\Delta p]_\times & T I & I
   \end{bmatrix}.

Two consecutive runs :math:`a, b` compose like preintegrated deltas:

.. math::

   \Delta R_{ab} = \Delta R_a \Delta R_b, \qquad
   \Delta v_{ab} = \Delta v_a + \Delta R_a \Delta v_b, \qquad
   \Delta p_{ab} = \Delta p_a + T_b \Delta v_a + \Delta R_a \Delta p_b,

.. math::

   Q_{ab} = \tilde\Phi_b Q_a \tilde\Phi_b^\top + D Q_b D^\top, \qquad
   \tilde\Phi_b = D \Phi_b D^\top, \qquad D = \mathrm{diag}(I, \Delta R_a, \Delta R_a).

Each factor's chain therefore runs on 1 to 32 GPU threads, which integrate
consecutive segments and combine the summaries in a binary tree. Batches
with fewer factors than the GPU has threads split chains over a warp; large
batches give each thread a whole chain. The split depends on the batch size,
the number of SMs and the typical chain length (the ``num_samples``
argument). Each factor uses at most one thread per 4 of its samples.

===============================================================================
API
===============================================================================

-------------------------------------------------------------------------------
States and residual
-------------------------------------------------------------------------------

One factor per keyframe pair, ``SizedFactorBatch<15, 6, 3, 6, 6, 3, 6>``:

.. list-table::
   :header-rows: 1
   :widths: 10 40 15 35

   * - Slot
     - State
     - Tangent
     - Jacobian columns
   * - 0
     - :math:`T_a`, ``SE3StateBatch`` (world_from_rig)
     - 6, :math:`[\varphi; \rho]`
     - 0–5
   * - 1
     - :math:`v_a`, ``VectorStateBatch3``
     - 3
     - 6–8
   * - 2
     - :math:`b_a = [b_g; b_a]`, ``VectorStateBatch6``
     - 6
     - 9–14
   * - 3
     - :math:`T_b`
     - 6
     - 15–20
   * - 4
     - :math:`v_b`
     - 3
     - 21–23
   * - 5
     - :math:`b_b`
     - 6
     - 24–29

Residual rows 0–8 are the whitened chain defect :math:`L^{-1} e`. Rows 9–14
are the bias random walk, gyroscope first.

-------------------------------------------------------------------------------
Sample buffer
-------------------------------------------------------------------------------

All samples of all factors go into one device buffer, back to back, with CSR
offsets (capacity + 1 of them):

.. code-block:: text

   imu_samples    = [ω_x ω_y ω_z a_x a_y a_z Δt] x num_samples   (float32)
   sample_offsets = [0, n_0, n_0 + n_1, ..., num_samples]        (int32)

Factor :math:`f` uses samples ``[sample_offsets[f], sample_offsets[f + 1])``.
The first starts at keyframe :math:`a` and the last ends at keyframe :math:`b`.
A sensor that time-stamps each sample at the *end* of its interval (sample
:math:`m` covering :math:`(t_{m-1}, t_m]`) maps directly: store
:math:`(\tilde\omega_m, \tilde a_m, t_m - t_{m-1})`. The buffers are read at
every evaluation and must outlive the factor; they may be rewritten between
solves.

-------------------------------------------------------------------------------
``ImuParameters``
-------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 18 38

   * - Field
     - Unit
     - Default
     - Meaning
   * - ``gravity`` (``Vector<3>``)
     - m/s²
     - (0, 0, −9.80665)
     - World-frame gravity.
   * - ``gyro_noise_density``
     - rad/s/√Hz
     - 1.6968e-4
     - :math:`\sigma_g`, gyroscope white noise.
   * - ``accel_noise_density``
     - m/s²/√Hz
     - 2.0e-3
     - :math:`\sigma_a`, accelerometer white noise.
   * - ``integration_noise_density``
     - m/√s
     - 1e-4
     - :math:`\sigma_i`, Euler step error on the position. Must be positive.
   * - ``gyro_bias_random_walk``
     - rad/s²/√Hz
     - 1.9393e-5
     - :math:`\sigma_{bg}`.
   * - ``accel_bias_random_walk``
     - m/s³/√Hz
     - 3.0e-3
     - :math:`\sigma_{ba}`.
   * - ``body_from_imu`` (``SE3Transform``)
     - —
     - identity
     - Pose of the IMU in the rig frame (rig_from_imu).

The noise defaults are those of the ADIS16448 in the EuRoC MAV dataset. All
five densities must be positive. Read :ref:`Float32 conditioning <imu-float32>` before using
realistic values together with vision.

-------------------------------------------------------------------------------
C++
-------------------------------------------------------------------------------

Header: ``cunls/factor/imu_factor_batch.h``.

.. code-block:: cpp

   ImuFactorBatch(const float *imu_samples, const int *sample_offsets,
                  size_t num_samples, const ImuParameters &parameters,
                  size_t capacity);

- ``imu_samples``, ``sample_offsets``: device buffers as above (not null).
- ``num_samples``: number of samples the buffer holds. Only sizes the work
  split: ``num_samples / capacity`` is taken as the typical chain length.
- ``parameters``: copied at construction; ``Parameters()`` returns them.
- ``capacity``: keyframe pairs the buffers hold. Call
  ``SetNumActiveFactors(n)`` before solving.

Throws ``std::invalid_argument`` for a null buffer or a non-positive noise
density. Evaluation follows the ``FactorBatch`` item contract (item and plain
evaluations are bitwise equal), so the factor also works with the RANSAC
minimizers.

-------------------------------------------------------------------------------
Python
-------------------------------------------------------------------------------

.. code-block:: python

   p = pycunls.ImuParameters()          # the fields above
   p.gravity = [0.0, 9.81, 0.0]         # lists; body_from_imu: 16 floats, row-major
   imu = pycunls.ImuFactorBatch(imu_samples, sample_offsets, num_samples, p, capacity)
   imu.set_num_active_factors(capacity)

``imu_samples`` must be a float32 array and ``sample_offsets`` an int32 array
(``TypeError`` otherwise). Raw integer device pointers are accepted unchecked.
The factor keeps references to both arrays.

.. _imu-example:

===============================================================================
Example: visual-inertial bundle adjustment
===============================================================================

``python/examples/imu_bundle_adjustment.py`` simulates ten keyframes of a
smooth trajectory, 40 IMU samples between consecutive keyframes (200 Hz) and
300 landmarks observed by nearby keyframes. IMU and reprojection factors share
the pose states. Velocities and biases start at zero, poses and landmarks
start perturbed, and keyframe 0 is fixed (the gauge). Output on an RTX A6000:

.. code-block:: text

   10 keyframes, 40 IMU samples each, 300 landmarks, 1320 observations
   LM: 8 iterations, cost 2.51e+05 -> 0.000287
   max position error   1.23e-03 m
   max velocity error   9.68e-04 m/s
   gyro bias  [ 0.00999385 -0.01996889  0.01499784]  (true [ 0.01  -0.02   0.015])
   accel bias [ 0.10012887 -0.04981745  0.07999814]  (true [ 0.1  -0.05  0.08])

The problem setup:

.. literalinclude:: ../../python/examples/imu_bundle_adjustment.py
   :language: python
   :start-after: # --- Sensor model ---
   :end-before: # --- Compare with the ground truth ---

The IMU factor in C++, with the states and buffers allocated as for any
problem:

.. code-block:: cpp

   #include "cunls/cunls.h"

   ImuParameters params;                         // gravity (0, 0, -9.80665)
   params.gyro_bias_random_walk = 1e-2f;
   params.accel_bias_random_walk = 1e-1f;

   // d_samples: num_samples x 7 floats; d_offsets: K ints (K - 1 pairs + 1).
   ImuFactorBatch imu(d_samples, d_offsets, num_samples, params, K - 1);
   imu.SetNumActiveFactors(K - 1);

   std::vector<float *> imu_ptrs;
   for (int k = 0; k + 1 < K; ++k) {
     for (int j : {k, k + 1}) {
       imu_ptrs.push_back(pose_states.StateDevicePtr(j));   // SE3StateBatch
       imu_ptrs.push_back(vel_states.StateDevicePtr(j));    // VectorStateBatch<3>
       imu_ptrs.push_back(bias_states.StateDevicePtr(j));   // VectorStateBatch<6>
     }
   }
   problem.AddFactorBatch(&imu, imu_ptrs);
   // The reprojection factors read the same pose_states.

.. _imu-inertial-pnp-example:

===============================================================================
Example: visual-inertial odometry with RANSAC
===============================================================================

``python/examples/tartan_vio.py`` is a small RGB-D inertial odometry on the
TartanGround ``OldTownFall`` anymal sequence P2000 (82 m in 129 s, a camera
at 10 Hz with depth, an IMU at 100 Hz). It tracks KLT features (OpenCV) and
creates each track's landmark from the depth image at its first frame. Every
frame solves one problem on a two-frame window: the pose, velocity and bias of
the previous and the current frame (30 free tangent dimensions), one
``ImuFactorBatch`` between them, priors on the previous frame from the last
solve, and one ``PnPFactorBatch`` factor per tracked landmark.
``RansacLevenbergMarquardtMinimizer`` samples the matches (two per hypothesis,
since the IMU predicts the motion) and keeps the IMU factor and the priors
always on. The buffers, the problem and the minimizers are created once;
every frame rewrites the buffers, the active count and the connectivity
(``Problem.set_state_pointers``).

The example adds noise and biases to the dataset's ideal IMU and injects
swapped matches, coherently shifted matches (repetitive texture), a camera
blackout and 75% clutter into the 2D matches. Visual-only RANSAC PnP, the
same window solved by Levenberg-Marquardt with a Huber loss, and IMU dead
reckoning run on the same data. ``--rrd`` / ``--spawn`` log the run to Rerun.
Output on an RTX A6000:

.. code-block:: text

   inertial RANSAC     : final position error   0.717 m (0.87% of the path)
   visual RANSAC       : final position error   2.360 m (2.87% of the path)
   inertial LM + Huber : final position error 4364.330 m (diverges in the clutter)
   inertial RANSAC: kept 116905 of 197972 genuine matches, accepted 5 of 58106 injected outliers

===============================================================================
Practical notes
===============================================================================

**Gauge and observability.** With known gravity, IMU factors make scale, roll
and pitch observable, but not the global position or yaw. Fix a keyframe (a
constant state) or add a prior. Biases need some rotation and acceleration
over the window. On static or straight-line segments, add a bias prior (a
weighted ``PriorVectorFactorBatch6`` on :math:`b`).

**Initialization.** Velocities and biases can start at zero when the poses are
reasonable (from vision or the previous solve). Gravity must be known: align
the world frame with it first.

**Weighting.** Wrap the factor in ``WeightedFactorBatch`` (uniform or
per-factor weights) to down-weight IMU factors, for example at the boundary of
a sliding window or in frame-to-frame tracking.

**Extrinsics and timing.** ``body_from_imu`` must be calibrated (for example
with Kalibr). Time offsets between camera and IMU are not modeled: shift the
timestamps before building the sample buffer.

.. _imu-float32:

**Float32 conditioning.** cuNLS solves in float32. Realistic IMU noise makes
the IMU rows very stiff compared with vision and with the weakly observed
directions (absolute biases, velocity along the motion). The bias random-walk
rows weigh :math:`1/(\sigma_b\sqrt T)`, about :math:`10^5` for the default
gyroscope value over 0.2 s. The linear solve then loses the weak directions,
and Levenberg-Marquardt converges slowly or stalls. In the tests and the
example:

- A 6-keyframe trajectory with pose priors stalls with the default bias random
  walks and converges with :math:`(\sigma_{bg}, \sigma_{ba}) = (10^{-2}, 10^{-1})`.
- The example with the default (EuRoC) chain noise reaches cost 0.02 after 100
  iterations but is still 2–3 cm off. With the noise in the example it
  converges in 8 iterations to about 1 mm.

When this happens, loosen the noise densities or down-weight the IMU factors.
This trades some statistical efficiency for convergence.

**Empty chains.** A factor without samples (``offsets[f] == offsets[f + 1]``),
or whose samples have no positive duration, has zero covariance and carries no
information. It evaluates to all-zero residual and Jacobian rows, so it adds
nothing to the solve. States constrained only by such a factor (for example the
velocity of the next keyframe) are then unobserved; give each pair at least
one sample.

===============================================================================
Limits
===============================================================================

- Gravity is a parameter; it is not estimated.
- The bias is constant between two keyframes (a random walk across
  keyframes only).
- Euler integration: its error is well below the noise at typical IMU rates
  (100–1000 Hz), but grows at low rates or high angular velocities.
- No measurements on the intermediate states (barometer, wheel odometry,
  zero-velocity updates), although the elimination would support them.
- No camera-IMU time offset.

===============================================================================
Performance
===============================================================================

Time per evaluation, with Jacobians / residuals only (RTX A6000):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Samples per factor
     - 16 factors
     - 1,024 factors
     - 65,536 factors
   * - 20
     - 29 / 11 µs
     - 31 / 12 µs
     - 418 / 135 µs
   * - 200
     - 33 / 15 µs
     - 44 / 29 µs
     - 2.0 / 1.9 ms
   * - 1,000
     - 58 / 37 µs
     - 159 / 131 µs
     - —

Large batches run at about 0.15 ns per sample. In a visual-inertial bundle
adjustment with 100 keyframes (200 samples per pair), 10,000 landmarks and
41,600 reprojections, the IMU kernels take 1.9% of the GPU time. The
preconditioned conjugate gradient solve takes most of the rest.

===============================================================================
References
===============================================================================

- C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza, "On-Manifold
  Preintegration for Real-Time Visual-Inertial Odometry," *IEEE Transactions
  on Robotics* 33(1), 2017. Measurement model, noise propagation and the
  preintegrated factor this one replaces.
- T. Lupton, S. Sukkarieh, "Visual-Inertial-Aided Navigation for High-Dynamic
  Motion in Built Environments Without Initial Conditions," *IEEE Transactions
  on Robotics* 28(1), 2012. Origin of IMU preintegration.
- L. Carlone, Z. Kira, C. Beall, V. Indelman, F. Dellaert, "Eliminating
  Conditionally Independent Sets in Factor Graphs: A Unifying Perspective
  based on Smart Factors," *IEEE ICRA*, 2014. Marginalizing variables inside a
  factor.
- F. Dellaert, M. Kaess, "Square Root SAM: Simultaneous Localization and
  Mapping via Square Root Information Smoothing," *International Journal of
  Robotics Research* 25(12), 2006. Variable elimination in factor graphs.
- H. G. Bock, K. J. Plitt, "A Multiple Shooting Algorithm for Direct Solution
  of Optimal Control Problems," *IFAC Proceedings* 17(2), 1984. Condensing the
  intermediate states of a shooting chain.
- J. Solà, J. Deray, D. Atchuthan, "A micro Lie theory for state estimation in
  robotics," arXiv:1812.01537, 2018. Right perturbations and the SO(3) / SE(3)
  Jacobians.
- M. Burri et al., "The EuRoC micro aerial vehicle datasets," *International
  Journal of Robotics Research* 35(10), 2016. Source of the default noise
  densities.
- GTSAM: ``PreintegrationParams`` (``integrationCovariance``) and
  ``CombinedImuFactor``. The integration-noise term and the combined factor
  with bias rows.
