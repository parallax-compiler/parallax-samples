#include <parallax/runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Parallax Vector Multiply Demo" << std::endl;
    std::cout << "==================================" << std::endl;
    
    const size_t N = 1'000'000;
    const float multiplier = 2.0f;
    
    std::cout << "\nAllocating unified memory for " << N << " floats..." << std::endl;
    
    // Allocate unified memory
    float* input = (float*)parallax_umalloc(N * sizeof(float), 0);
    float* output = (float*)parallax_umalloc(N * sizeof(float), 0);
    
    if (!input || !output) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return 1;
    }
    
    // Initialize input data
    std::cout << "Initializing data..." << std::endl;
    for (size_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i);
    }
    
    // Sync to device
    std::cout << "Syncing to GPU..." << std::endl;
    parallax_sync(input, 0); // HOST_TO_DEVICE
    
    // CPU baseline for comparison
    std::cout << "\nRunning CPU baseline..." << std::endl;
    std::vector<float> cpu_result(N);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        cpu_result[i] = input[i] * multiplier;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    std::cout << "CPU time: " << cpu_time / 1000.0 << " ms" << std::endl;
    
    // Note: GPU kernel execution would go here
    // For now, simulate with CPU to demonstrate the API
    std::cout << "\n[GPU kernel execution - implementation in progress]" << std::endl;
    std::cout << "Simulating with CPU for demonstration..." << std::endl;
    
    for (size_t i = 0; i < N; i++) {
        output[i] = input[i] * multiplier;
    }
    
    // Sync back from device
    std::cout << "Syncing from GPU..." << std::endl;
    parallax_sync(output, 1); // DEVICE_TO_HOST
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    bool correct = true;
    for (size_t i = 0; i < N && i < 10; i++) {
        if (std::abs(output[i] - cpu_result[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << output[i] << " vs " << cpu_result[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ Results match! First 10 elements verified." << std::endl;
        std::cout << "  input[0] = " << input[0] << " -> output[0] = " << output[0] << std::endl;
        std::cout << "  input[9] = " << input[9] << " -> output[9] = " << output[9] << std::endl;
    }
    
    // Cleanup
    parallax_ufree(input);
    parallax_ufree(output);
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "Demo complete!" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return 0;
}
