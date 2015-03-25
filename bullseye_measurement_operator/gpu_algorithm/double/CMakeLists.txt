project(gpu_imaging64)

#declare executable
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)
include_directories(/usr/include/casacore/ /usr/local/include/casacore)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-DBULLSEYE_DOUBLE")
SET(CUDA_NVCC_FLAGS "-DBULLSEYE_DOUBLE -O3 -Xcompiler \"-pthread -fopenmp\" -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 --use_fast_math -Xptxas -dlcm=ca -lineinfo")
cuda_add_library(gpu_imaging64 SHARED ../wrapper.cu ../../algorithms/fft_shift_utils.cpp ../../algorithms/fft_and_repacking_routines.cpp)
#link external libraries
CUDA_ADD_CUFFT_TO_TARGET(gpu_imaging64)
target_link_libraries(gpu_imaging64 casa_casa gomp fftw3 fftw3f)