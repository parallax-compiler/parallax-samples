#include <iostream>
#include <iomanip>
#include <cmath>
#include <execution>
#include <algorithm>
#include <vector>
#include <numeric>
#include <chrono>

#include <parallax/runtime.h>
#include <parallax/runtime.hpp>
#include <parallax/vulkan_backend.hpp>
#include <parallax/unified_buffer.hpp>
#include <parallax/execution_policy.hpp>
#include <parallax/execution_policy_impl.hpp>

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
    
    float* data = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    for (size_t i = 0; i < config.size; i++) data[i] = static_cast<float>(i);
    
    std::vector<float> cpu_data(data, data + config.size);
    Timer timer;
    
    // CPU baseline
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::for_each(std::execution::par, cpu_data.begin(), cpu_data.end(),
                     [](float& x) { x = x * 2.0f + 1.0f; });
    }
    result.cpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    // GPU - Intercepted!
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::for_each(std::execution::par, data, data + config.size,
                     [](float& x) { x = x * 2.0f + 1.0f; });
    }
    result.gpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    result.correct = true;
    for (size_t i = 0; i < std::min(size_t(1000), config.size); i++) {
        if (std::abs(data[i] - cpu_data[i]) > 1e-4f) { result.correct = false; break; }
    }
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    parallax_ufree(data);
    return result;
}

BenchResult bench_transform(const BenchConfig& config) {
    BenchResult result;
    result.name = "transform";
    result.size = config.size;
    
    float* in = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    float* out = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    for (size_t i = 0; i < config.size; i++) in[i] = static_cast<float>(i + 1);
    
    std::vector<float> cpu_in(in, in + config.size);
    std::vector<float> cpu_out(config.size);
    Timer timer;
    
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::transform(std::execution::par, cpu_in.begin(), cpu_in.end(), cpu_out.begin(),
                      [](float x) { return std::sqrt(x) * 2.0f; });
    }
    result.cpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    timer.start();
    for (int i = 0; i < config.iterations; i++) {
        std::transform(std::execution::par, in, in + config.size, out,
                      [](float x) { return std::sqrt(x) * 2.0f; });
    }
    result.gpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    result.correct = true;
    for (size_t i = 0; i < std::min(size_t(1000), config.size); i++) {
        if (std::abs(out[i] - cpu_out[i]) > 1e-3f) { result.correct = false; break; }
    }
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    parallax_ufree(in);
    parallax_ufree(out);
    return result;
}

BenchResult bench_reduce(const BenchConfig& config) {
    BenchResult result;
    result.name = "reduce";
    result.size = config.size;
    
    float* data = (float*)parallax_umalloc(config.size * sizeof(float), 0);
    for (size_t i = 0; i < config.size; i++) data[i] = 1.0f;
    
    Timer timer;
    timer.start();
    float cpu_res = 0;
    for (int i = 0; i < config.iterations; i++) {
        cpu_res = std::reduce(std::execution::par, data, data + config.size, 0.0f);
    }
    result.cpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    timer.start();
    float gpu_res = 0;
    for (int i = 0; i < config.iterations; i++) {
        gpu_res = std::reduce(std::execution::par, data, data + config.size, 0.0f);
    }
    result.gpu_time_ms = timer.elapsed_ms() / config.iterations;
    
    result.correct = (std::abs(cpu_res - gpu_res) < 1e-1f);
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    parallax_ufree(data);
    return result;
}

void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(12) << r.name 
              << std::setw(10) << (r.size >= 1000000 ? std::to_string(r.size/1000000) + "M" : std::to_string(r.size))
              << std::fixed << std::setprecision(3)
              << std::setw(12) << r.cpu_time_ms
              << std::setw(12) << r.gpu_time_ms
              << std::setw(10) << r.speedup
              << (r.correct ? "✓ PASS" : "✗ FAIL") << std::endl;
}

int main() {
    std::cout << "Parallax v0.5.0 Alpha - ISO C++ Automatic Offloading" << std::endl;
    std::cout << "Target: NVIDIA GeForce GTX 980M" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto* backend = parallax::get_global_backend();
    auto* memory = parallax::get_global_memory_manager();
    if (backend && memory) parallax::ExecutionPolicyImpl::instance().initialize(backend, memory);

    std::vector<BenchConfig> configs = {
        {1000000, 10, "1M"},
        {10000000, 5, "10M"},
        {100000000, 1, "100M"}
    };
    
    std::cout << std::left << std::setw(12) << "Algorithm" << std::setw(10) << "Size" 
              << std::setw(12) << "CPU (ms)" << std::setw(12) << "GPU (ms)" 
              << std::setw(10) << "Speedup" << "Status" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (const auto& c : configs) print_result(bench_for_each(c));
    for (const auto& c : configs) print_result(bench_transform(c));
    for (const auto& c : configs) print_result(bench_reduce(c));

    if (backend) parallax::ExecutionPolicyImpl::instance().shutdown();
    return 0;
}
