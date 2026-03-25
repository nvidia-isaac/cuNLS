# Merges one or more static dependency archives into a target's archive,
# producing a single "fat" static library that contains all symbols.
#
# Usage:
#   bundle_static_dependencies(my_lib dep1 dep2 ...)
#
# Each dep must be a CMake target whose $<TARGET_FILE:...> resolves to a
# static archive (.a).  The merge is performed as a POST_BUILD step using
# an ar MRI script.
function(bundle_static_dependencies target)
  set(mri_file  "${CMAKE_BINARY_DIR}/merge_${target}.mri")
  set(merged_file "${CMAKE_BINARY_DIR}/lib${target}_merged.a")

  set(content "CREATE ${merged_file}\nADDLIB $<TARGET_FILE:${target}>\n")
  foreach(dep IN LISTS ARGN)
    string(APPEND content "ADDLIB $<TARGET_FILE:${dep}>\n")
  endforeach()
  string(APPEND content "SAVE\nEND\n")

  file(GENERATE OUTPUT "${mri_file}" CONTENT "${content}")

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_AR} -M < "${mri_file}"
    COMMAND ${CMAKE_COMMAND} -E rename "${merged_file}" "$<TARGET_FILE:${target}>"
    COMMENT "Bundling static dependencies into $<TARGET_FILE_NAME:${target}>"
  )
endfunction()
