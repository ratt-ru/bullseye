cmake_minimum_required(VERSION 2.8)
project (imaging32)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)
find_package(FFTW REQUIRED COMPONENTS single double)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(/usr/include/casacore/ /usr/local/include/casacore)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-lfftw3 -std=c++0x -O3 -pthread -fopenmp -DBULLSEYE_SINGLE")
add_library(imaging32 SHARED ../wrapper.cpp ../fft_and_repacking_routines.cpp ../fft_shift_utils.cpp)
target_link_libraries(imaging32 casa_casa fftw3 fftw3f)