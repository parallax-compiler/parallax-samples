#include <parallax/runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <chrono>

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Parallax Vector Multiply Demo" << std::endl;
    std::cout << "Standard C++ with std::execution" << std::endl;
    std::cout << "==================================" << std::endl;
    
    const size_t N = 1'000'000;
    const float multiplier = 2.0f;
    
    std::cout << "\nAllocating unified memory for " << N << " floats..." << std::endl;
    
    // Allocate unified memory (accessible from CPU and GPU)
    // Coherence is managed automatically by Parallax runtime
    float* data = (float*)parallax_umalloc(N * sizeof(float), 0);
    
    if (!data) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return 1;
    }
    
    // Initialize data - just use it like normal memory
    std::cout << "Initializing data..." << std::endl;
    for (size_t i = 0; i < N; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // CPU baseline for comparison
    std::cout << "\nRunning CPU baseline..." << std::endl;
    std::vector<float> cpu_result(N);
    std::copy(data, data + N, cpu_result.begin());
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::seq, 
                  cpu_result.begin(), cpu_result.end(),
                  [multiplier](float& x) { x *= multiplier; });
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    std::cout << "CPU time: " << cpu_time / 1000.0 << " ms" << std::endl;
    
    // GPU execution with std::execution::par
    // Parallax automatically:
    // 1. Detects this is a parallel algorithm
    // 2. Transfers dirty blocks to GPU
    // 3. Executes kernel
    // 4. Marks GPU blocks as dirty
    std::cout << "\nRunning GPU version with std::execution::par..." << std::endl;
    std::cout << "[Note: Full compiler integration in progress]" << std::endl;
    std::cout << "Simulating with CPU for demonstration..." << std::endl;
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::seq,  // Will be ::par when compiler ready
                  data, data + N,
                  [multiplier](float& x) { x *= multiplier; });
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    std::cout << "GPU time: " << gpu_time / 1000.0 << " ms" << std::endl;
    
    // Verify results - memory is automatically coherent
    std::cout << "\nVerifying results..." << std::endl;
    bool correct = true;
    for (size_t i = 0; i < N && i < 10; i++) {
        if (std::abs(data[i] - cpu_result[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << data[i] << " vs " << cpu_result[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Results match! First 10 elements verified." << std::endl;
        std::cout << "  data[0] = " << data[0] << " (expected " << cpu_result[0] << ")" << std::endl;
        std::cout << "  data[9] = " << data[9] << " (expected " << cpu_result[9] << ")" << std::endl;
    }
    
    // Cleanup - no manual synchronization needed!
    parallax_ufree(data);
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "Demo complete!" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
