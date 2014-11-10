project(gpu_imaging)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
#declare executable
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenMP)
SET(CUDA_NVCC_FLAGS "-O3 -Xcompiler \"-pthread -fopenmp\" -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 --use_fast_math -Xptxas -dlcm=ca")
cuda_add_library(gpu_imaging SHARED wrapper.cu)
#link external libraries
CUDA_ADD_CUFFT_TO_TARGET(gpu_imaging)
target_link_libraries(gpu_imaging casa_casa gomp)
set(CMAKE_BUILD_TYPE Release)
