# Module: robustifier

## Purpose

Implements GPU-batched robust losses applied to squared residual norms.

## Core interface

- `loss_function_batch.h`: abstract `void LossFunctionBatch::Evaluate(float* s, float3* out,
  int num_losses, cudaStream_t stream)`

## Implementations

All in `loss_function_batch.cu` (one kernel template, one functor per loss); headers:

- `trivial_loss_function_batch.h`: identity loss
- `huber_loss_function_batch.h`
- `cauchy_loss_function_batch.h`
- `arctan_loss_function_batch.h`
- `soft_lone_loss_function_batch.h`
- `tolerant_loss_function_batch.h`
- `tukey_loss_function_batch.h`
- `scaled_loss_function_batch.h`: `ScaledLossFunctionBatch(const LossFunctionBatch&, float a)`,
  scales any loss by a > 0; the inner loss is referenced, not owned

## Semantics

- Output `float3` encodes `(rho(s), rho'(s), rho''(s))`.
- Used by `ResidualBatch` to robustify cost, residual scaling, and Jacobian
  scaling.
