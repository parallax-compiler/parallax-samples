# Parallax Examples

Production-ready examples demonstrating Parallax GPU offload capabilities.

## Overview

- **basic/** - Getting started examples
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

### Getting Started

| Example | Description | Complexity |
|---------|-------------|------------|
| `vector_multiply.cpp` | Vector multiplication | ⭐ |
| `compiler_test.cpp` | Compiler integration | ⭐⭐ |
| `comprehensive_bench.cpp` | Algorithm showcase | ⭐⭐⭐ |

## Example: Hello Parallax

```cpp
#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>
#include <parallax/allocator.hpp>

int main() {
    // Create GPU-accessible vector
    std::vector<float, parallax::allocator<float>> data(1000, 1.0f);

    // Run on GPU automatically
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) { x = x * 2.0f; });

    std::cout << "Result: " << data[0] << std::endl;  // 2.0
    return 0;
}
```

**Key Points:**
- Use `parallax::allocator` for GPU-accessible memory
- Use `std::execution::par` to request parallel execution
- Zero code changes to lambda - pure standard C++!

## Performance Tips

1. **Dataset Size**: Use >10K elements for best GPU utilization
2. **Memory**: Use `parallax::allocator` to avoid copy overhead
3. **Algorithms**: Prefer transform/for_each over sequential code

## See Also

- [Getting Started Guide](../parallax-docs/docs/getting-started.md)
- [Performance Guide](../parallax-docs/docs/performance.md)
- [Benchmarks](../parallax-benchmarks/)
