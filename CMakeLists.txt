cmake_minimum_required(VERSION 3.20)
project(parallax-samples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find parallax-runtime and parallax-compiler
find_library(PARALLAX_RUNTIME parallax-runtime PATHS ${CMAKE_SOURCE_DIR}/../parallax-runtime/build)
find_library(PARALLAX_COMPILER parallax-plugin PATHS ${CMAKE_SOURCE_DIR}/../parallax-compiler/build)

include_directories(${CMAKE_SOURCE_DIR}/../parallax-runtime/include)
include_directories(${CMAKE_SOURCE_DIR}/../parallax-compiler/include)
include_directories(${Vulkan_INCLUDE_DIRS})

# GPU kernel test
add_executable(gpu_kernel_test basic/gpu_kernel_test.cpp)
target_link_libraries(gpu_kernel_test ${PARALLAX_RUNTIME} ${Vulkan_LIBRARIES})

# Comprehensive benchmark
add_executable(comprehensive_bench basic/comprehensive_bench.cpp)
target_link_libraries(comprehensive_bench ${PARALLAX_RUNTIME} ${Vulkan_LIBRARIES})

# Automatic lambda compilation benchmark (full pipeline test)
if(PARALLAX_COMPILER)
    add_executable(auto_lambda_bench basic/auto_lambda_bench.cpp)
    target_link_libraries(auto_lambda_bench 
        ${PARALLAX_RUNTIME} 
        ${PARALLAX_COMPILER} 
        ${Vulkan_LIBRARIES})
        
    # Simpler compiler-only test (no runtime integration needed)
    add_executable(compiler_test basic/compiler_test.cpp)
    target_link_libraries(compiler_test 
        ${PARALLAX_COMPILER})
endif()
