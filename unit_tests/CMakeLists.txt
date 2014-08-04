cmake_minimum_required(VERSION 2.8)

project (backend_tests)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(CasaCore REQUIRED COMPONENTS measures scimath tables)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x -O3")
include_directories(/usr/include/casacore/ /usr/local/include/casacore)

add_executable(backend_tests unit_tests.cpp)
target_link_libraries(backend_tests casa_casa)

