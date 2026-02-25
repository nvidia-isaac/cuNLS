add_library(cunls_state OBJECT
  state_batch_ops.cu
  se2_state_batch.cu
  se3_state_batch.cu
  similarity2_state_batch.cu
  similarity3_state_batch.cu
  so2_state_batch.cu
  so3_state_batch.cu
  vector_state_batch.cu
)
