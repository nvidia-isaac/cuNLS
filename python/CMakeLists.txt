find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

FetchContent_Declare(
  nanobind
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
  src/bind_factor.cpp
  src/bind_loss.cpp
  src/bind_minimizer.cpp
  src/bind_problem.cpp
)

target_include_directories(_pycunls_core PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(_pycunls_core PRIVATE cunls CUDA::cudart)

# When cunls is static, its PRIVATE CUDA dependencies don't propagate.
# Link them explicitly so the extension module is self-contained.
if(NOT BUILD_SHARED_LIBS)
  target_link_libraries(_pycunls_core PRIVATE
    CUDA::cusparse CUDA::cublas CUDA::cusolver cudss)
endif()

install(TARGETS _pycunls_core LIBRARY DESTINATION pycunls)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pycunls/ DESTINATION pycunls
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "py.typed")
