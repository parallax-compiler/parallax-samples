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

# Execution policy test
add_executable(execution_policy_test basic/execution_policy_test.cpp)
target_link_libraries(execution_policy_test ${PARALLAX_RUNTIME} ${PARALLAX_COMPILER} ${Vulkan_LIBRARIES})
