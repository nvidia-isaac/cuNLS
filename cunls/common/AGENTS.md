# Module: common

## Purpose

Shared infrastructure used across cuNLS modules:

- device memory containers
- CUDA/cuBLAS/cuSPARSE/cuSOLVER/cuDSS handle wrappers
- profiler wrappers
- common math/sparse data types

## Key files

- `device_vector.h`: `DeviceVector<T>` (`dvector<T>` alias in `types.h`)
- `types.h`: `Vector`, `Matrix`, `SE3Transform`, `CSRSparseMatrix`, `BSRSparseMatrix`,
  `PerFactorJacobians`
- `cuda_stream.h`: RAII CUDA stream wrapper; `StreamBoundHandle`, the base of the per-stream
  cuBLAS / cuSPARSE / cuSOLVER / cuDSS handles
- `cublas_helper.h`, `cusparse_helper.h`, `cusolver_helper.h`, `cudss_helper.h`
- `profiler.h`: NVTX range/domain wrappers
- `cudss_dynamic.h`: cuDSS entry points resolved with dlopen() from libcudss.so
  (cuDSS is an optional run-time dependency)
- `pinned_vector.h`: RAII page-locked host buffer
- `log.h`: verbosity levels, `SetLoggerOptions(Verbosity)`, `LogMessage` / `LogError`
- `helper.h`: CUDA/cuBLAS/cuSPARSE/cuSOLVER error-checking macros (`THROW_ON_*`)
- `manifold.h`: empty tag types naming each Lie group at compile time
- `type_traits.h`: SFINAE helpers used by the factor facades

## Contracts

- Most wrappers are RAII and non-copyable.
- Many APIs require non-null CUDA stream handles.
- `DeviceVector` owns device memory; observation/state pointers in other modules
  usually remain user-owned.
