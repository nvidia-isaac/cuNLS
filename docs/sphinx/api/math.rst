################################################################################
Math API
################################################################################

`cunls/math` contains low-level batched Lie-group operations used by
state/factor implementations.

This page documents the C++ API only: ``pycunls`` does not expose these
functions. They are used internally by the built-in state and factor batches
that Python wraps.

Headers:

- `cunls/math/so_se_lie_math.h` (SO(2), SO(3), SE(2), SE(3))
- `cunls/math/sim_lie_math.h` (Sim(2), Sim(3))
- `cunls/math/sl_lie_math.h` (SL(4))

Common parameter semantics
--------------------------

Most batched math functions in this module follow the same conventions.

Parameters
^^^^^^^^^^

- `stream` - [in] CUDA stream used for asynchronous kernel/library execution.
- `twist` / `rotation` / `transform` / `matrix` - [in] Device pointer to input
  batched data.
- `*_pitch` - [in] Leading dimension (row stride) for each matrix in the batch.
- `*_stride` - [in] Stride between consecutive batch elements.
- `size` - [in] Number of batch elements.
- output pointer parameters (for example `twist`, `transform`, `jacobian`)
  - [out] Device pointer to destination buffer.

Returns
^^^^^^^

- All functions in this module return `void`.

Lie math API (`so_se_lie_math.h`, `sim_lie_math.h`, `sl_lie_math.h`)
---------------------------------------------------------------------

SO(3) family
^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeLogSO3(cudaStream_t stream, const float* rotation,
                      size_t rotation_pitch, size_t rotation_stride,
                      size_t twist_stride, size_t size, float* twist)

   void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float* twist,
                                       size_t twist_stride,
                                       size_t jacobian_inv_pitch,
                                       size_t jacobian_inv_stride, size_t size,
                                       float* jacobian_inv)

SE(3) family
^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeExpSE3(cudaStream_t stream, const float* twist,
                      size_t twist_stride, size_t transform_pitch,
                      size_t transform_stride, size_t size, float* transform)

   void ComputeLogSE3(cudaStream_t stream, const float* transform,
                      size_t transform_pitch, size_t transform_stride,
                      size_t twist_stride, size_t size, float* twist)

   void ComputeJacobianLeftSE3(cudaStream_t stream, const float* twist,
                               size_t twist_stride, size_t jacobian_pitch,
                               size_t jacobian_stride, size_t size,
                               float* jacobian)

   void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float* twist,
                                       size_t twist_stride, size_t jacobian_pitch,
                                       size_t jacobian_stride, size_t size,
                                       float* jacobian)
