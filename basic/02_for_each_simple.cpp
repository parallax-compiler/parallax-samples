/**
 * @file 02_for_each_simple.cpp
 * @brief Simple std::for_each examples with various lambda patterns
 *
 * NEW in v1.0: Uses standard C++ containers with automatic GPU
 * allocator injection. No custom allocators needed!
 */

#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>
#include <iomanip>

void print_test(const char* name, float expected, float actual) {
    bool pass = (expected == actual);
    std::cout << std::setw(30) << std::left << name;
    std::cout << " Expected: " << std::setw(8) << expected;
    std::cout << " Got: " << std::setw(8) << actual;
    std::cout << (pass ? " ✓" : " ✗") << std::endl;
}

int main() {
    std::cout << "=== std::for_each GPU Tests ===" << std::endl << std::endl;

    constexpr size_t N = 10000;
    int passed = 0, total = 0;

    // Test 1: Multiply
    {
        std::vector<float> data(N, 5.0f);  // Standard C++!
        std::for_each(std::execution::par, data.begin(), data.end(),
                     [](float& x) { x *= 2.0f; });
        print_test("Multiply (*=)", 10.0f, data[0]);
        passed += (data[0] == 10.0f);
        total++;
    }

    // Test 2: Add
    {
        std::vector<float> data(N, 5.0f);  // Standard C++!
        std::for_each(std::execution::par, data.begin(), data.end(),
                     [](float& x) { x += 3.0f; });
        print_test("Add (+=)", 8.0f, data[0]);
        passed += (data[0] == 8.0f);
        total++;
    }

    // Test 3: Complex expression
    {
        std::vector<float> data(N, 2.0f);  // Standard C++!
        std::for_each(std::execution::par, data.begin(), data.end(),
                     [](float& x) { x = x * 3.0f + 1.0f; });
        print_test("Complex (x*3+1)", 7.0f, data[0]);
        passed += (data[0] == 7.0f);
        total++;
    }

    // Test 4: Division
    {
        std::vector<float> data(N, 10.0f);  // Standard C++!
        std::for_each(std::execution::par, data.begin(), data.end(),
                     [](float& x) { x /= 2.0f; });
        print_test("Divide (/=)", 5.0f, data[0]);
        passed += (data[0] == 5.0f);
        total++;
    }

    std::cout << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;

    return (passed == total) ? 0 : 1;
}
