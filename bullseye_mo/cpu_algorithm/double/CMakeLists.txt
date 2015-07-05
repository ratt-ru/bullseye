project(cpu_imaging64)

#declare executable
project(cpu_imaging64)

#declare executable
FIND_PACKAGE(OpenMP)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)
include_directories(/usr/include/casacore/ /usr/local/include/casacore)
set(CMAKE_BUILD_TYPE Release)

if($ENV{VECTORIZE})
  set(INTRINSICS_SUPPORT "-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx")
else($ENV{VECTORIZE})
  set(INTRINSICS_SUPPORT "")
endif($ENV{VECTORIZE})

set(CMAKE_CXX_FLAGS "-DBULLSEYE_DOUBLE -Wall -O3 -fno-strict-aliasing -pthread -fopenmp --std=c++11 ${INTRINSICS_SUPPORT}")

add_library(cpu_imaging64 SHARED ../wrapper.cpp ../../cpu_gpu_common/fft_shift_utils.cpp ../../cpu_gpu_common/fft_and_repacking_routines.cpp)
#link external libraries
target_link_libraries(cpu_imaging64 casa_casa gomp fftw3 fftw3f)
