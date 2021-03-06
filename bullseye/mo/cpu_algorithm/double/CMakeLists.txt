project(cpu_imaging64)

#declare executable
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)
include_directories(/usr/include/casacore/ /usr/local/include/casacore)
set(CMAKE_BUILD_TYPE Release)
if($VECTORIZE)
  set(INTRINSICS_SUPPORT "-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx")
else($VECTORIZE)
  set(INTRINSICS_SUPPORT "")
endif($VECTORIZE)
set(CMAKE_CXX_FLAGS "-DBULLSEYE_DOUBLE -Wall -fno-strict-aliasing -pthread -fopenmp -O3 --std=c++11 ${INTRINSICS_SUPPORT}")
SET(CUDA_NVCC_FLAGS "-DBULLSEYE_DOUBLE -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 --use_fast_math -Xptxas -dlcm=ca -lineinfo ${INTRINSICS_SUPPORT}")
cuda_add_library(cpu_imaging64 SHARED ../wrapper.cpp ../../cpu_gpu_common/fft_shift_utils.cpp ../../cpu_gpu_common/fft_and_repacking_routines.cpp)
#link external libraries
CUDA_ADD_CUFFT_TO_TARGET(cpu_imaging64)
target_link_libraries(cpu_imaging64 casa_casa gomp fftw3 fftw3f)
