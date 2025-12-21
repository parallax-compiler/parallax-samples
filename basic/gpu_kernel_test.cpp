#include <parallax/runtime.h>
#include <parallax/kernel_launcher.hpp>
#include <parallax/shaders/vector_multiply.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "Parallax GPU Kernel Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    const size_t N = 1'000'000;
    const float multiplier = 2.0f;
    
    std::cout << "\nAllocating unified memory for " << N << " floats..." << std::endl;
    
    // Allocate unified memory
    float* data = (float*)parallax_umalloc(N * sizeof(float), 0);
    
    if (!data) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return 1;
    }
    
    // Initialize data
    std::cout << "Initializing data..." << std::endl;
    for (size_t i = 0; i < N; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // CPU baseline
    std::cout << "\nRunning CPU baseline..." << std::endl;
    std::vector<float> cpu_result(N);
    std::copy(data, data + N, cpu_result.begin());
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        cpu_result[i] *= multiplier;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    std::cout << "CPU time: " << cpu_time / 1000.0 << " ms" << std::endl;
    
    // GPU execution
    std::cout << "\nRunning GPU kernel..." << std::endl;
    
    // Get backend and memory manager (internal access for testing)
    extern std::unique_ptr<parallax::VulkanBackend> g_backend;
    extern std::unique_ptr<parallax::MemoryManager> g_memory_manager;
    
    if (!g_backend || !g_memory_manager) {
        std::cerr << "Parallax runtime not initialized" << std::endl;
        parallax_ufree(data);
        return 1;
    }
    
    // Create kernel launcher
    parallax::KernelLauncher launcher(g_backend.get(), g_memory_manager.get());
    
    // Load SPIR-V kernel
    if (!launcher.load_kernel("vector_multiply", 
                               parallax::shaders::VECTOR_MULTIPLY_SPV,
                               parallax::shaders::VECTOR_MULTIPLY_SPV_SIZE)) {
        std::cerr << "Failed to load kernel" << std::endl;
        parallax_ufree(data);
        return 1;
    }
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    if (!launcher.launch("vector_multiply", data, N, multiplier)) {
        std::cerr << "Failed to launch kernel" << std::endl;
        parallax_ufree(data);
        return 1;
    }
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
    
    std::cout << "GPU time: " << gpu_time / 1000.0 << " ms" << std::endl;
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    bool correct = true;
    size_t errors = 0;
    for (size_t i = 0; i < N && errors < 10; i++) {
        if (std::abs(data[i] - cpu_result[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": " 
                      << data[i] << " vs " << cpu_result[i] << std::endl;
            correct = false;
            errors++;
        }
    }
    
    if (correct) {
        std::cout << "✓ Results match! All " << N << " elements verified." << std::endl;
        std::cout << "  data[0] = " << data[0] << " (expected " << cpu_result[0] << ")" << std::endl;
        std::cout << "  data[" << N-1 << "] = " << data[N-1] << " (expected " << cpu_result[N-1] << ")" << std::endl;
        
        if (gpu_time < cpu_time) {
            float speedup = static_cast<float>(cpu_time) / gpu_time;
            std::cout << "\n🚀 GPU is " << speedup << "x faster than CPU!" << std::endl;
        }
    } else {
        std::cout << "❌ Verification failed (" << errors << " errors)" << std::endl;
    }
    
    // Cleanup
    parallax_ufree(data);
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "Test complete!" << std::endl;
    std::cout << "==================================" << std::endl;
    
    return correct ? 0 : 1;
}
