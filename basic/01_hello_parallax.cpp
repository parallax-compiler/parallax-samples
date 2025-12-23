/**
 * @file 01_hello_parallax.cpp
 * @brief Hello World example for Parallax GPU offload
 *
 * Demonstrates the simplest possible use of Parallax to offload
 * a std::for_each operation to the GPU.
 *
 * NEW in v1.0: No custom allocator needed! Compiler automatically
 * injects GPU-accessible memory allocators. Pure ISO C++20!
 */

#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>

int main() {
    std::cout << "=== Parallax Hello World (v1.0) ===" << std::endl;
    std::cout << "Creating standard C++ vector..." << std::endl;

    // Standard C++ vector - compiler auto-injects GPU allocator!
    constexpr size_t N = 1000;
    std::vector<float> data(N, 1.0f);

    std::cout << "Before: data[0] = " << data[0] << std::endl;

    // Run on GPU with std::execution::par
    // Lambda will be automatically compiled to GPU kernel
    // Memory is automatically made GPU-accessible by the compiler!
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) {
                     x = x * 2.0f;
                 });

    std::cout << "After:  data[0] = " << data[0] << std::endl;

    // Verify results
    bool success = (data[0] == 2.0f);
    std::cout << std::endl;
    std::cout << "Result: " << (success ? "✓ SUCCESS" : "✗ FAILED") << std::endl;

    return success ? 0 : 1;
}
