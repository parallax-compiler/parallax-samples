#include <parallax/execution_policy.hpp>
#include <parallax/execution_policy_impl.hpp> // Includes template definitions
#include <parallax/lambda_compiler.hpp>
#include <parallax/runtime.h>
#include <parallax/vulkan_backend.hpp>
#include <parallax/unified_buffer.hpp>
#include <algorithm>
#include <vector>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <parallax/runtime.hpp>

// Full pipeline test: pSTL → ExecutionPolicyImpl → LambdaCompiler → SPIR-V → GPU
// NO pre-compiled shaders!

// We use the shared runtime components from the library
// static std::unique_ptr<parallax::VulkanBackend> g_backend;
// static std::unique_ptr<parallax::MemoryManager> g_memory_manager;

struct BenchConfig {
    size_t size;
    int iterations;
    std::string name;
};

struct BenchResult {
    std::string name;
    size_t size;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    bool correct;
};

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

BenchResult bench_for_each(const BenchConfig& config) {
    BenchResult result;
    result.name = "for_each";
    result.size = config.size;
    
    // Allocate unified memory
    float* data = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    if (!data) {
        std::cerr << "Failed to allocate memory" << std::endl;
        result.correct = false;
        return result;
    }
    
    // Initialize
    for (size_t i = 0; i < config.size; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // CPU baseline
    std::vector<float> cpu_data(data, data + config.size);
    Timer timer;
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::for_each(cpu_data.begin(), cpu_data.end(),
                     [](float& x) { x = x * 2.0f + 1.0f; });
    }
    result.cpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    // GPU with automatic lambda compilation
    // This uses: Lambda → LambdaCompiler → SPIR-V → GPU
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        parallax::ExecutionPolicyImpl::instance().for_each_impl(
            data, data + config.size,
            [](float& x) { x = x * 2.0f + 1.0f; }
        );
    }
    result.gpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    // Verify
    result.correct = true;
    for (size_t i = 0; i < std::min(size_t(1000), config.size); i++) {
        if (std::abs(data[i] - cpu_data[i]) > 1e-4f) {
            result.correct = false;
            break;
        }
    }
    
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    parallax_ufree(data);
    return result;
}

BenchResult bench_transform(const BenchConfig& config) {
    BenchResult result;
    result.name = "transform";
    result.size = config.size;
    
    float* input = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    float* output = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    
    if (!input || !output) {
        std::cerr << "Failed to allocate memory" << std::endl;
        result.correct = false;
        return result;
    }
    
    for (size_t i = 0; i < config.size; i++) {
        input[i] = static_cast<float>(i + 1);
    }
    
    // CPU baseline
    std::vector<float> cpu_input(input, input + config.size);
    std::vector<float> cpu_output(config.size);
    Timer timer;
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::transform(cpu_input.begin(), cpu_input.end(), cpu_output.begin(),
                      [](float x) { return std::sqrt(x) * 2.0f; });
    }
    result.cpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    // GPU with automatic lambda compilation
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        parallax::ExecutionPolicyImpl::instance().transform_impl(
            input, input + config.size, output,
            [](float x) { return std::sqrt(x) * 2.0f; }
        );
    }
    result.gpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    // Verify
    result.correct = true;
    for (size_t i = 0; i < std::min(size_t(1000), config.size); i++) {
        if (std::abs(output[i] - cpu_output[i]) > 1e-3f) {
            result.correct = false;
            break;
        }
    }
    
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    parallax_ufree(input);
    parallax_ufree(output);
    return result;
}

void print_results(const std::vector<BenchResult>& results) {
    std::cout << std::setw(15) << "Benchmark"
              << std::setw(12) << "Size"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Status"
              << std::endl;
    std::cout << std::string(81, '-') << std::endl;
    
    for (const auto& r : results) {
        std::string size_str;
        if (r.size >= 1000000) {
            size_str = std::to_string(r.size / 1000000) + "M";
        } else if (r.size >= 1000) {
            size_str = std::to_string(r.size / 1000) + "K";
        } else {
            size_str = std::to_string(r.size);
        }
        
        std::cout << std::setw(15) << r.name
                  << std::setw(12) << size_str
                  << std::setw(15) << std::fixed << std::setprecision(3) << r.cpu_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << r.gpu_time_ms
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.speedup << "x"
                  << std::setw(12) << (r.correct ? "✓ PASS" : "✗ FAIL")
                  << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Parallax Automatic Lambda Compilation" << std::endl;
    std::cout << "Full Pipeline: Lambda → SPIR-V → GPU" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Initialize Parallax (via shared runtime)
    auto* backend = parallax::get_global_backend();
    auto* memory_manager = parallax::get_global_memory_manager();
    if (!backend || !memory_manager) {
        std::cerr << "Initialization Failed" << std::endl;
        return 1;
    }

    // Initialize execution policy with backend
    parallax::ExecutionPolicyImpl::instance().initialize(backend, memory_manager);
    
    std::cout << "Parallax initialized on: " << backend->device_name() << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchConfig> configs = {
        {1000000, 10, "1M"},
        {10000000, 5, "10M"},
        {100000000, 1, "100M"}
    };
    
    std::vector<BenchResult> results;
    
    std::cout << "Running for_each benchmarks (automatic lambda compilation)..." << std::endl;
    for (const auto& config : configs) {
        auto result = bench_for_each(config);
        results.push_back(result);
        std::cout << "  " << config.name << ": " 
                  << result.speedup << "x speedup" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Running transform benchmarks (automatic lambda compilation)..." << std::endl;
    for (const auto& config : configs) {
        auto result = bench_transform(config);
        results.push_back(result);
        std::cout << "  " << config.name << ": " 
                  << result.speedup << "x speedup" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    print_results(results);
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "All SPIR-V generated automatically!" << std::endl;
    std::cout << "No pre-compiled shaders used." << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    parallax::ExecutionPolicyImpl::instance().shutdown();
    
    return 0;
}
