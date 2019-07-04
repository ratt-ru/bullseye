project(gpu_imaging64)

#declare executable
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)
include_directories(/usr/include/casacore/ /usr/local/include/casacore)

if($ENV{DISABLE_GPU_FILTER_CACHING})
  set(FILTER_CACHING_OPTION "-DBULLSEYE_DISABLE_GPU_FILTER_CACHING")
else($ENV{DISABLE_GPU_FILTER_CACHING})
  set(FILTER_CACHING_OPTION "")
endif($ENV{DISABLE_GPU_FILTER_CACHING})

set(CMAKE_CXX_FLAGS "-DBULLSEYE_DOUBLE ${FILTER_CACHING_OPTION}")
SET(CUDA_NVCC_FLAGS "-DBULLSEYE_DOUBLE ${FILTER_CACHING_OPTION} -O3 -rdc=true -Xcompiler \"-fno-strict-aliasing -pthread -fopenmp\" ${SUPPORTED_GPU_ARCHITECTURES} -Xptxas -dlcm=ca -lineinfo")

cuda_add_library(gpu_imaging64 SHARED ../wrapper.cu ../../cpu_gpu_common/fft_shift_utils.cpp ../../cpu_gpu_common/fft_and_repacking_routines.cpp)
#link external libraries
CUDA_ADD_CUFFT_TO_TARGET(gpu_imaging64)
target_link_libraries(gpu_imaging64 casa_casa gomp fftw3 fftw3f)