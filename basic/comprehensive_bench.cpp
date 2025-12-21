#include <parallax/runtime.h>
#include <parallax/kernel_launcher.hpp>
#include <parallax/shaders/vector_multiply.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

extern std::unique_ptr<parallax::VulkanBackend> g_backend;
extern std::unique_ptr<parallax::MemoryManager> g_memory_manager;

struct BenchmarkResult {
    size_t size;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    bool correct;
};

BenchmarkResult run_benchmark(size_t N) {
    BenchmarkResult result;
    result.size = N;
    
    // Allocate unified memory
    float* data = (float*)parallax_umalloc(N * sizeof(float), 0);
    if (!data) {
        std::cerr << "Failed to allocate " << N << " floats" << std::endl;
        result.correct = false;
        return result;
    }
    
    // Initialize
    for (size_t i = 0; i < N; i++) {
        data[i] = static_cast<float>(i);
    }
    
    const float multiplier = 2.0f;
    
    // CPU baseline
    std::vector<float> cpu_result(N);
    std::copy(data, data + N, cpu_result.begin());
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        cpu_result[i] *= multiplier;
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    result.cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU execution
    if (!g_backend || !g_memory_manager) {
        std::cerr << "Parallax runtime not initialized" << std::endl;
        parallax_ufree(data);
        result.correct = false;
        return result;
    }
    
    parallax::KernelLauncher launcher(g_backend.get(), g_memory_manager.get());
    
    if (!launcher.load_kernel("vector_multiply",
                               parallax::shaders::VECTOR_MULTIPLY_SPV,
                               parallax::shaders::VECTOR_MULTIPLY_SPV_SIZE)) {
        std::cerr << "Failed to load kernel" << std::endl;
        parallax_ufree(data);
        result.correct = false;
        return result;
    }
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    if (!launcher.launch("vector_multiply", data, N, multiplier)) {
        std::cerr << "Failed to launch kernel" << std::endl;
        parallax_ufree(data);
        result.correct = false;
        return result;
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    result.gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    // Verify
    result.correct = true;
    size_t errors = 0;
    for (size_t i = 0; i < N && errors < 10; i++) {
        if (std::abs(data[i] - cpu_result[i]) > 1e-5f) {
            result.correct = false;
            errors++;
        }
    }
    
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    parallax_ufree(data);
    return result;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Parallax Comprehensive Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test sizes: 1K, 10K, 100K, 1M, 10M, 100M
    std::vector<size_t> sizes = {
        1024,           // 1K
        10240,          // 10K
        102400,         // 100K
        1024000,        // 1M
        10240000,       // 10M
        102400000       // 100M
    };
    
    std::cout << std::setw(12) << "Size"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Status"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    
    for (size_t N : sizes) {
        auto result = run_benchmark(N);
        
        std::string size_str;
        if (N >= 1000000) {
            size_str = std::to_string(N / 1000000) + "M";
        } else if (N >= 1000) {
            size_str = std::to_string(N / 1000) + "K";
        } else {
            size_str = std::to_string(N);
        }
        
        std::cout << std::setw(12) << size_str
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.cpu_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.gpu_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::setw(12) << (result.correct ? "✓ PASS" : "✗ FAIL")
                  << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
