################################################################################
Math API
################################################################################

`cunls/math` contains low-level Lie-group and dense-matrix operations used by
state/factor implementations.

Headers:

- `cunls/math/lie_math.h`
- `cunls/math/dense_matrix_ops.h`

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
- `size` / `num_matrices` - [in] Number of batch elements.
- output pointer parameters (for example `skew`, `jacobian`, `inverse_transform`)
  - [out] Device pointer to destination buffer.

Returns
^^^^^^^

- All functions in this module return `void`.

Lie math API (`lie_math.h`)
---------------------------

SO(3) family
^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeSkewSO3(cudaStream_t stream, const float* twist,
                       size_t twist_stride, size_t skew_pitch,
                       size_t skew_stride, size_t size, float* skew)

   void ComputeExpSO3(cudaStream_t stream, const float* twist,
                      size_t twist_stride, size_t rotation_pitch,
                      size_t rotation_stride, size_t size, float* rotation)

   void ComputeLogSO3(cudaStream_t stream, const float* rotation,
                      size_t rotation_pitch, size_t rotation_stride,
                      size_t twist_stride, size_t size, float* twist)

   void ComputeJacobianLeftSO3(cudaStream_t stream, const float* twist,
                               size_t twist_stride, size_t jacobian_pitch,
                               size_t jacobian_stride, size_t size,
                               float* jacobian)

   void ComputeJacobianRightSO3(cudaStream_t stream, const float* twist,
                                size_t twist_stride, size_t jacobian_pitch,
                                size_t jacobian_stride, size_t size,
                                float* jacobian)

   void ComputeJacobianLeftInverseSO3(cudaStream_t stream, const float* twist,
                                      size_t twist_stride,
                                      size_t jacobian_inv_pitch,
                                      size_t jacobian_inv_stride, size_t size,
                                      float* jacobian_inv)

   void ComputeJacobianRightInverseSO3(cudaStream_t stream, const float* twist,
                                       size_t twist_stride,
                                       size_t jacobian_inv_pitch,
                                       size_t jacobian_inv_stride, size_t size,
                                       float* jacobian_inv)

SE(3) family
^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeInverseSE3(cudaStream_t stream, const float* transform,
                          size_t transform_pitch, size_t transform_stride,
                          size_t inverse_pitch, size_t inverse_stride,
                          size_t size, float* inverse_transform)

   void ComputeExpSE3(cudaStream_t stream, const float* twist,
                      size_t twist_stride, size_t transform_pitch,
                      size_t transform_stride, size_t size, float* transform)

   void ComputeLogSE3(cudaStream_t stream, const float* transform,
                      size_t transform_pitch, size_t transform_stride,
                      size_t twist_stride, size_t size, float* twist)

   void ComputeAdjointSE3(cudaStream_t stream, const float* transform,
                          size_t transform_pitch, size_t transform_stride,
                          size_t adjoint_pitch, size_t adjoint_stride,
                          size_t size, float* adjoint)

   void ComputeInverseAdjointSE3(cudaStream_t stream, const float* transform,
                                 size_t transform_pitch, size_t transform_stride,
                                 size_t inv_adjoint_pitch,
                                 size_t inv_adjoint_stride, size_t size,
                                 float* inv_adjoint)

   void ComputeJacobianLeftSE3(cudaStream_t stream, const float* twist,
                               size_t twist_stride, size_t jacobian_pitch,
                               size_t jacobian_stride, size_t size,
                               float* jacobian)

   void ComputeJacobianLeftInverseSE3(cudaStream_t stream, const float* twist,
                                      size_t twist_stride, size_t jacobian_pitch,
                                      size_t jacobian_stride, size_t size,
                                      float* jacobian)

   void ComputeJacobianRightSE3(cudaStream_t stream, const float* twist,
                                size_t twist_stride, size_t jacobian_pitch,
                                size_t jacobian_stride, size_t size,
                                float* jacobian)

   void ComputeJacobianRightInverseSE3(cudaStream_t stream, const float* twist,
                                       size_t twist_stride, size_t jacobian_pitch,
                                       size_t jacobian_stride, size_t size,
                                       float* jacobian)

General utility
^^^^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeNegateMatrix(cudaStream_t stream, const float* matrix,
                            size_t rows, size_t cols, size_t pitch,
                            size_t stride, size_t size,
                            float* negated_matrix)

Additional parameter docs for `ComputeNegateMatrix`
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- `rows` - [in] number of rows in each matrix.
- `cols` - [in] number of columns in each matrix.

Sim(3) family
^^^^^^^^^^^^^

.. code-block:: cpp

   void ComputeAdjointSim3(cudaStream_t stream, const float* transforms,
                           size_t transform_stride, float* adjoints,
                           size_t adjoint_stride, size_t size)

ComputeAdjointSim3 parameters
""""""""""""""""""""""""""""""

- `transforms` - [in] Sim(3) transforms as row-major 4x4 matrices
  (``[[R, t], [0, 1/s]]``).
- `transform_stride` - [in] stride between consecutive transforms (in floats).
- `adjoints` - [out] 7x7 adjoint matrices (row-major, flattened).
- `adjoint_stride` - [in] stride between consecutive adjoints (in floats).

Dense matrix API (`dense_matrix_ops.h`)
---------------------------------------

.. code-block:: cpp

   void ComputeSqrtMatrix(cuBLASHandle& cublas_handle, cudaStream_t stream,
                          float* spd_matrix, size_t matrix_size,
                          size_t pitch, size_t num_matrices)

ComputeSqrtMatrix parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `cublas_handle` - [in] external cuBLAS handle wrapper.
- `stream` - [in] CUDA stream.
- `spd_matrix` - [in,out] Device pointer to SPD matrices; replaced in-place by
  square-root factors.
- `matrix_size` - [in] matrix dimension (`N` for `N x N`).
- `pitch` - [in] leading dimension for matrix storage.
- `num_matrices` - [in] batch size.

.. code-block:: cpp

   void ScatterToRightBlock(cudaStream_t stream, const float* src,
                            size_t block_dim, size_t src_stride, float* dst,
                            size_t dst_pitch, size_t dst_stride,
                            size_t num_blocks)

ScatterToRightBlock parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `src` - [in] source dense NxN blocks (device pointer).
- `block_dim` - [in] block dimension N.
- `src_stride` - [in] stride between consecutive source blocks.
- `dst` - [out] destination pointer (start of right sub-block).
- `dst_pitch` - [in] leading dimension (row stride) of the destination.
- `dst_stride` - [in] stride between consecutive destination blocks.
- `num_blocks` - [in] number of blocks in the batch.
