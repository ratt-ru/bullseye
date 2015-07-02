cmake_minimum_required(VERSION 2.8)
include_directories(cpu_gpu_common)
add_subdirectory(gpu_algorithm)
add_subdirectory(cpu_algorithm)
add_subdirectory(benchmarks) #this should probably be last (or at least after all the libraries have been built!
