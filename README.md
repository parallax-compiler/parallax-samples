# Parallax Examples

Production-ready examples demonstrating Parallax GPU offload capabilities.

> **✨ NEW in v1.0:** All examples now use pure ISO C++20 with automatic GPU allocator injection. No custom allocators needed!

## Overview

- **basic/** - Getting started examples (updated for v1.0)
- **hpc/** - High-performance computing applications
- **ml/** - Machine learning workloads

## Quick Start

```bash
# Build all examples
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run basic examples
./basic/01_hello_parallax
./basic/02_for_each_simple
./basic/03_transform_simple
```

## Basic Examples (`basic/`)

### Getting Started (v1.0)

| Example | Description | Features | Status |
|---------|-------------|----------|--------|
| `01_hello_parallax.cpp` | Hello World GPU offload | Auto allocator injection | ✅ 100% pass |
| `02_for_each_simple.cpp` | std::for_each operations | All operators (`*=`, `+=`, `-=`, `/=`) | ✅ 4/4 pass |
| `03_transform_simple.cpp` | std::transform operations | Return values, complex expressions | ✅ 4/4 pass |
| `vector_multiply.cpp` | Vector multiplication | Production workload | ⭐ |
| `compiler_test.cpp` | Compiler integration | Testing framework | ⭐⭐ |
| `comprehensive_bench.cpp` | Algorithm showcase | Performance benchmarks | ⭐⭐⭐ |

## Example: Hello Parallax (v1.0)

```cpp
#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>

int main() {
    // Standard C++ vector - compiler auto-injects GPU allocator!
    std::vector<float> data(1000, 1.0f);

    // Run on GPU automatically
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) { x = x * 2.0f; });

    std::cout << "Result: " << data[0] << std::endl;  // 2.0
    return 0;
}
```

**Key Points:**
- ✨ **No custom allocators needed!** Compiler handles GPU memory automatically
- Use `std::execution::par` to request parallel execution
- 100% pure ISO C++20 - no extensions, no vendor lock-in
- Zero code changes from standard C++ - truly transparent GPU acceleration

## What's New in v1.0

### Automatic Allocator Injection
The compiler now automatically rewrites standard containers to use GPU-accessible memory:

**You write:**
```cpp
std::vector<float> data(1000);
```

**Compiler generates:**
```cpp
std::vector<float, parallax::allocator<float>> data(1000);
```

No code changes required!

### All Operators Supported
- ✅ `*=` (multiply-assign)
- ✅ `+=` (add-assign)
- ✅ `-=` (subtract-assign)
- ✅ `/=` (divide-assign)
- ✅ Complex expressions: `x = x * 2.0f + 1.0f`

## Performance Tips

1. **Dataset Size**: Use >10K elements for best GPU utilization
2. **Memory**: Automatic! Compiler handles GPU-accessible memory
3. **Algorithms**: Prefer transform/for_each over sequential code
4. **Testing**: All examples tested on NVIDIA GTX 980M with 100% pass rate

## See Also

- [Getting Started Guide](../parallax-docs/docs/getting-started.md)
- [Performance Guide](../parallax-docs/docs/performance.md)
- [Benchmarks](../parallax-benchmarks/)
