cmake_minimum_required(VERSION 2.8)
include_directories(cpu_gpu_common)

option(ENABLE_NATIVE_TUNING "Enables support for optimized code generation (-march=native)" ON)
option(ENABLE_FAST_MATH "Enables -fast-math option (breaks IEEE754 callbacks)" ON)

if (${ENABLE_NATIVE_TUNING})
  set(TUNING_FLAGS "-march=native -mtune=native")
  message(WARNING " Instructing gcc to use your native supported instruction set. This will break .so portability and may raise the ILLEGAL_INSTRUCTION signal on incompatible chipsets")
else (${ENABLE_NATIVE_TUNING})
  set(TUNING_FLAGS "")
endif (${ENABLE_NATIVE_TUNING})

if (${ENABLE_FAST_MATH})
  message("-ffast-math is enabled")
  set(FAST_MATH_FLAGS "-ffast-math")
else (${ENABLE_FAST_MATH})
  set(FAST_MATH_FLAGS "")
endif (${ENABLE_FAST_MATH})

add_subdirectory(cpu_algorithm)

option(SUPPORTED_GPU_ARCHITECTURES "Supported GPU architectures (-gencode switches) to be supplied to NVCC" "")
if (DEFINED SUPPORTED_GPU_ARCHITECTURES)  
  message("Instructing NVCC to build for architectures '${SUPPORTED_GPU_ARCHITECTURES}'")
  add_subdirectory(gpu_algorithm)
else (DEFINED SUPPORTED_GPU_ARCHITECTURES)
  message("No supporting architectures given with SUPPORTED_GPU_ARCHITECTURES. Will not build GPU-powered backend.")
endif (DEFINED SUPPORTED_GPU_ARCHITECTURES)

add_subdirectory(benchmarks) #this should probably be last (or at least after all the libraries have been built!
