find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nanobind)

find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_SOURCE_DIR}/cmake)

nanobind_add_module(_pycunls_core
  src/pycunls_module.cpp
  src/bind_types.cpp
  src/bind_state.cpp
  src/py_factor_wrappers.cpp
  src/bind_factor.cpp
  src/bind_loss.cpp
  src/bind_minimizer.cpp
  src/bind_ransac.cpp
  src/bind_problem.cpp
)

target_include_directories(_pycunls_core PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(_pycunls_core PRIVATE cunls CUDA::cudart)

# When cunls is static, INTERFACE does not include bundled deps; link CUDA
# toolkit libs explicitly. spdlog is already inside libcunls.a; cuDSS is never
# linked at all (it is dlopen()'d at runtime, see cudss_dynamic.cpp) so no
# pycunls wheel carries a cuDSS binary or a load-time dependency on one.
if(NOT BUILD_SHARED_LIBS)
  target_link_libraries(_pycunls_core PRIVATE
    CUDA::cusparse CUDA::cublas CUDA::cusolver)
endif()

# Find the CUDA libraries from NVIDIA's pip wheels (nvidia-cuda-runtime, nvidia-cublas, ...,
# nvidia-cudss-cu12/13) without LD_LIBRARY_PATH. The paths are relative to
# site-packages/pycunls/, where the module is installed. The loader tries LD_LIBRARY_PATH
# first, then these directories, then the system library cache, so a system CUDA toolkit
# is still used when the pip wheels are absent. glibc dlopen() also searches the caller's
# RUNPATH, which covers the runtime-loaded libcudss.so.0. CUDA 13 wheels share one directory;
# CUDA 12 wheels use one directory per library (cuDSS goes to nvidia/cu12/lib).
if(CUDAToolkit_VERSION_MAJOR VERSION_GREATER_EQUAL 13)
  set(_pycunls_pip_cuda_rpath "$ORIGIN/../nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib")
else()
  set(_pycunls_pip_cuda_rpath
    "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cublas/lib:$ORIGIN/../nvidia/cusparse/lib:$ORIGIN/../nvidia/cusolver/lib:$ORIGIN/../nvidia/cu12/lib")
endif()
set_target_properties(_pycunls_core PROPERTIES INSTALL_RPATH "${_pycunls_pip_cuda_rpath}")

install(TARGETS _pycunls_core LIBRARY DESTINATION pycunls)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pycunls/ DESTINATION pycunls
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "*.md"
    PATTERN "py.typed"
    PATTERN "_build_provenance.json")
