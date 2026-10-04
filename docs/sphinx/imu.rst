###############################################################################
IMU factor
###############################################################################

``ImuFactorBatch`` connects two keyframes through the raw IMU samples between
them. The samples define a chain of Euler steps between intermediate states;
instead of preintegrating them, the factor keeps the chain and **marginalizes
its intermediate states inside every evaluation** (a Schur complement onto the
keyframe states). The marginal is exact at the current linearization point, so
there is no reference bias, no first-order bias correction and no
reintegration policy. Design: ``docs/design/imu_schur_complement_factor.md``.

===============================================================================
Model
===============================================================================

States (in order): pose :math:`T_a` (``SE3StateBatch``, world from body),
velocity :math:`v_a` (``VectorStateBatch3``, world-frame velocity of the IMU),
bias :math:`b_a = [b_g; b_a]` (``VectorStateBatch6``), then :math:`T_b, v_b,
b_b`. The IMU pose is :math:`T\,T_{bi}` with the extrinsic ``body_from_imu``.

Each sample :math:`(\tilde\omega_k, \tilde a_k, \Delta t_k)` (IMU frame, the
step starting at the sample) is one Euler step with the bias of keyframe a:

.. math::

   R_{k+1} &= R_k\,\mathrm{Exp}((\tilde\omega_k - b_g)\Delta t_k) \\
   v_{k+1} &= v_k + g\,\Delta t_k + R_k(\tilde a_k - b_a)\Delta t_k \\
   p_{k+1} &= p_k + v_k\,\Delta t_k + \tfrac12 g\,\Delta t_k^2
              + \tfrac12 R_k(\tilde a_k - b_a)\Delta t_k^2

with step noise from the gyro and accelerometer noise densities and a small
position (integration) noise. The factor integrates forward from keyframe a,
propagates the covariance :math:`\Sigma` of the prediction (the covariance form
of the Riccati recursion that eliminates the intermediate states) and returns

.. math::

   r_{0:9} = L^{-1} e, \quad \Sigma = L L^\top, \qquad
   r_{9:15} = \frac{b_b - b_a}{\sigma_b \sqrt{\textstyle\sum_k \Delta t_k}}

with :math:`e = [\mathrm{Log}(\hat R^\top R_b);\ v_b - \hat v;\ p_b - \hat p]`.
Its :math:`J^\top J` and :math:`J^\top r` equal the Schur complement of the
explicit chain (every intermediate state a variable) onto the keyframes, and
:math:`|r|^2` is the chain's cost minimized over the intermediate states. The
marginal has rank 15, hence 15 residual rows.

===============================================================================
Usage
===============================================================================

.. code-block:: python

   import cupy as cp, numpy as np, pycunls

   # samples: (total, 7) float32 rows (wx, wy, wz, ax, ay, az, dt)
   # offsets: int32, factor f uses samples[offsets[f]:offsets[f + 1]]
   p = pycunls.ImuParameters()       # EuRoC ADIS16448 noise, gravity -Z
   imu = pycunls.ImuFactorBatch(cp.asarray(samples), cp.asarray(offsets), len(samples),
                                p, pairs)
   imu.set_num_active_factors(pairs)
   ptrs = []
   for k in range(pairs):
       for j in (k, k + 1):
           ptrs += [poses.state_device_ptr(j), vels.state_device_ptr(j),
                    biases.state_device_ptr(j)]
   problem.add_factor_batch(imu, ptrs)

Notes:

- **Gravity** defaults to :math:`(0, 0, -9.80665)` (+Z up, ROS REP-103). A
  wrong direction (a Z-down / NED world) still converges, to a wrong answer.
- **Float32 conditioning.** Realistic bias random walks make the bias rows
  very stiff (weight :math:`1 / (\sigma_b \sqrt{T})`, about :math:`10^5` for
  the default gyro value over 0.2 s). The float32 outer linear solve then
  loses the weakest directions and Levenberg-Marquardt stalls. If it does,
  loosen ``gyro_bias_random_walk`` / ``accel_bias_random_walk`` (about
  1e-2 / 1e-1).
- Every factor needs at least one sample with a positive :math:`\Delta t`.
- Each chain runs on 1 to 32 threads: runs of samples reduce to summaries
  that compose like preintegrated deltas, so small batches split chains over
  a warp. ``num_samples`` (the size of the sample buffer) gives the typical
  chain length for that split. RTX A6000, 200 samples per factor, with
  Jacobians: 33 µs for 16 factors, 47 µs for 1024, 2 ms for 65536 (0.15 ns
  per sample).
