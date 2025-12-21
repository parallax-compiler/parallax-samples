#include <parallax/execution_policy.hpp>
#include <parallax/runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Test std::for_each with parallax::par
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Parallax std::execution::par Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    const size_t N = 1000000;
    
    // Allocate unified memory
    float* data = (float*)parallax_umalloc(N * sizeof(float), 0);
    if (!data) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return 1;
    }
    
    // Initialize
    for (size_t i = 0; i < N; i++) {
        data[i] = static_cast<float>(i);
    }
    
    std::cout << "Testing std::for_each with parallax::par..." << std::endl;
    std::cout << "Data size: " << N << " elements" << std::endl;
    std::cout << std::endl;
    
    // CPU baseline
    std::vector<float> cpu_data(data, data + N);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::seq, cpu_data.begin(), cpu_data.end(),
                  [](float& x) { x *= 2.0f; });
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    std::cout << "CPU (std::execution::seq): " << cpu_time << " ms" << std::endl;
    
    // GPU with parallax::par
    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::for_each(parallax::par, data, data + N,
                  [](float& x) { x *= 2.0f; });
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "GPU (parallax::par):       " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup:                   " << (cpu_time / gpu_time) << "x" << std::endl;
    std::cout << std::endl;
    
    // Verify results
    bool correct = true;
    size_t errors = 0;
    for (size_t i = 0; i < N && errors < 10; i++) {
        if (std::abs(data[i] - cpu_data[i]) > 1e-5f) {
            correct = false;
            errors++;
            if (errors == 1) {
                std::cout << "First error at index " << i << ": "
                          << "GPU=" << data[i] << " CPU=" << cpu_data[i] << std::endl;
            }
        }
    }
    
    if (correct) {
        std::cout << "✓ Results verified - all " << N << " elements match!" << std::endl;
    } else {
        std::cout << "✗ Verification failed - " << errors << " errors found" << std::endl;
    }
    
    parallax_ufree(data);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return correct ? 0 : 1;
}
