cmake_minimum_required(VERSION 2.8)

project (imaging)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x -O3")
include_directories(/usr/include/python2.7 /usr/lib/python2.7/dist-packages/numpy/core/include/numpy /usr/include/casacore/ /usr/local/include/casacore)

add_library(imaging SHARED wrapper.cpp)
target_link_libraries(imaging casa_casa)
