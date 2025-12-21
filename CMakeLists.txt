cmake_minimum_required(VERSION 3.20)
project(parallax-samples)

# Find Parallax runtime
find_library(PARALLAX_RUNTIME parallax-runtime 
    HINTS ${CMAKE_SOURCE_DIR}/../parallax-runtime/build
)

if(NOT PARALLAX_RUNTIME)
    message(STATUS "Parallax runtime not found, building from source...")
    add_subdirectory(${CMAKE_SOURCE_DIR}/../parallax-runtime ${CMAKE_BINARY_DIR}/parallax-runtime)
    set(PARALLAX_RUNTIME parallax-runtime)
endif()

# Vector multiply example
add_executable(vector_multiply basic/vector_multiply.cpp)
target_include_directories(vector_multiply PRIVATE 
    ${CMAKE_SOURCE_DIR}/../parallax-runtime/include
)
target_link_libraries(vector_multiply PRIVATE ${PARALLAX_RUNTIME})
target_compile_features(vector_multiply PRIVATE cxx_std_20)

# Install
install(TARGETS vector_multiply RUNTIME DESTINATION bin)
