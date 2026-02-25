add_library(cunls_robustifier OBJECT
  arctan_loss_function_batch.cu
  cauchy_loss_function_batch.cu
  huber_loss_function_batch.cu
  soft_lone_loss_function_batch.cu
  tolerant_loss_function_batch.cu
  trivial_loss_function_batch.cu
  tukey_loss_function_batch.cu
)
