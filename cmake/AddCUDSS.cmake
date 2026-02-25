# Function to add cuDSS library to the project
#
# Usage:
#   add_cudss(CUDSS_BUILD_PATH "/path/to/cudss/build")
#
# This function will:
# - Check if a local cuDSS build exists at CUDSS_BUILD_PATH
# - If found, use the local build (sets CUDSS_NEW_API compile definition)
# - If not found, download a prebuilt version using FetchContent
# - Create an imported target 'cudss' that can be linked against
#
# Parameters:
#   CUDSS_BUILD_PATH - Path to the cuDSS build directory (optional, defaults to empty)
function(add_cudss)
  # Parse arguments
  set(options "")
  set(oneValueArgs CUDSS_BUILD_PATH)
  set(multiValueArgs "")
  cmake_parse_arguments(CUDSS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Create imported target
  add_library(cudss STATIC IMPORTED)

  # Check if local build path was provided and exists
  set(LOCAL_CUDSS_BUILD OFF)
  if(CUDSS_CUDSS_BUILD_PATH)
    set(CUDSS_LIB_PATH "${CUDSS_CUDSS_BUILD_PATH}/libcudss_static.a")
    set(CUDSS_INCLUDE_PATH "${CUDSS_CUDSS_BUILD_PATH}/include")
    
    if(EXISTS "${CUDSS_LIB_PATH}" AND EXISTS "${CUDSS_INCLUDE_PATH}")
      set(LOCAL_CUDSS_BUILD ON)
      message(STATUS "Using local cuDSS build from ${CUDSS_CUDSS_BUILD_PATH}")
      set_target_properties(cudss PROPERTIES
        IMPORTED_LOCATION "${CUDSS_LIB_PATH}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_PATH}"
      )
    endif()
  endif()

  # If local build not found, use prebuilt version
  if(NOT LOCAL_CUDSS_BUILD)
    message(STATUS "Using prebuilt cuDSS")
    set(CUDSS_URL_PREFIX "https://developer.download.nvidia.com/compute/cudss/redist/libcudss/")
    message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
      set(CUDSS_URL_AARCH64 ${CUDSS_URL_PREFIX}linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda13-archive.tar.xz)
      set(CUDSS_URL_X86_64 ${CUDSS_URL_PREFIX}linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda13-archive.tar.xz)
      message(STATUS "Using CUDA 13.0 or newer")
    else()
      set(CUDSS_URL_AARCH64 ${CUDSS_URL_PREFIX}linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz)
      set(CUDSS_URL_X86_64 ${CUDSS_URL_PREFIX}linux-x86_64/libcudss-linux-x86_64-0.7.1.4_cuda12-archive.tar.xz)
      message(STATUS "Using CUDA 12.0 or older")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(AARCH64)")
      set(CUDSS_URL ${CUDSS_URL_AARCH64})
    else()
      set(CUDSS_URL ${CUDSS_URL_X86_64})
    endif()

    include(FetchContent)
    FetchContent_Declare(
      cudss
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      URL ${CUDSS_URL}
    )
    FetchContent_Populate(cudss)

    set_target_properties(cudss PROPERTIES
      IMPORTED_LOCATION "${cudss_SOURCE_DIR}/lib/libcudss_static.a"
      INTERFACE_INCLUDE_DIRECTORIES "${cudss_SOURCE_DIR}/include"
    )
  endif()

  # Set parent scope variable to indicate if local build is used
  set(LOCAL_CUDSS_BUILD ${LOCAL_CUDSS_BUILD} PARENT_SCOPE)
endfunction()
