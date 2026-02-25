add_library(cunls_factor OBJECT
  point_to_plane_factor_batch.cu
  point_to_point_factor_batch.cu
  prior_vector_factor_batch.cu
  reprojection_factor_batch.cu
  se2_prior_factor_batch.cu
  se3_between_factor_batch.cu
  se3_prior_factor_batch.cu
  similarity2_prior_factor_batch.cu
  similarity3_prior_factor_batch.cu
  so2_prior_factor_batch.cu
  so3_prior_factor_batch.cu
  symmetric_point_to_plane_factor_batch.cu
)
