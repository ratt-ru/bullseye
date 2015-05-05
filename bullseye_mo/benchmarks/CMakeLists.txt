cmake_minimum_required(VERSION 2.8)

project (benchmark)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -pg -pthread -fopenmp -DBULLSEYE_SINGLE")
include_directories(/usr/include/casacore/ /usr/local/include/casacore)
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark casa_casa)

target_link_libraries(benchmark cpu_imaging32)
# target_link_libraries(benchmark gpu_imaging)

