add_subdirectory(common)
add_subdirectory(factor)
add_subdirectory(linear_solver)
add_subdirectory(math)
add_subdirectory(minimizer)
add_subdirectory(state)
add_subdirectory(robustifier)

set(CUNLS_OBJECT_LIBS
  cunls_common
  cunls_factor
  cunls_linear_solver
  cunls_math
  cunls_minimizer
  cunls_state
  cunls_robustifier
  PARENT_SCOPE
)
