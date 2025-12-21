cmake_minimum_required(VERSION 3.20)
project(parallax-samples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find parallax-runtime and parallax-compiler
find_library(PARALLAX_RUNTIME parallax-runtime
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../parallax-runtime/build
        ${CMAKE_CURRENT_SOURCE_DIR}/../parallax-runtime/build/lib
)
find_library(PARALLAX_COMPILER parallax-plugin PATHS ${CMAKE_SOURCE_DIR}/../parallax-compiler/build)

# Find LLVM (needed for compiler samples)
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

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
    
    # Handle LLVM 21+ monolithic library
    if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21.0")
        target_link_libraries(auto_lambda_bench 
            ${PARALLAX_RUNTIME} 
            ${PARALLAX_COMPILER} 
            ${Vulkan_LIBRARIES}
            LLVM)
    else()
        llvm_map_components_to_libnames(llvm_libs support core irreader)
        target_link_libraries(auto_lambda_bench 
            ${PARALLAX_RUNTIME} 
            ${PARALLAX_COMPILER} 
            ${Vulkan_LIBRARIES}
            ${llvm_libs})
    endif()
        
    # Simpler compiler-only test (no runtime integration needed)
    add_executable(compiler_test basic/compiler_test.cpp)
    
    if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21.0")
        target_link_libraries(compiler_test 
            ${PARALLAX_COMPILER}
            LLVM)
    else()
        target_link_libraries(compiler_test 
            ${PARALLAX_COMPILER}
            ${llvm_libs})
    endif()
endif()
