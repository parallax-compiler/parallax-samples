/**
 * @file 03_transform_simple.cpp
 * @brief Simple std::transform examples with return values
 */

#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>
#include <iomanip>
#include <parallax/allocator.hpp>

void print_test(const char* name, float expected, float actual) {
    bool pass = (expected == actual);
    std::cout << std::setw(30) << std::left << name;
    std::cout << " Expected: " << std::setw(8) << expected;
    std::cout << " Got: " << std::setw(8) << actual;
    std::cout << (pass ? " ✓" : " ✗") << std::endl;
}

int main() {
    std::cout << "=== std::transform GPU Tests ===" << std::endl << std::endl;

    constexpr size_t N = 10000;
    int passed = 0, total = 0;

    // Test 1: Simple multiply
    {
        std::vector<float, parallax::allocator<float>> input(N, 3.0f);
        std::vector<float, parallax::allocator<float>> output(N);

        std::transform(std::execution::par,
                      input.begin(), input.end(), output.begin(),
                      [](float x) { return x * 2.0f; });

        print_test("Multiply (x*2)", 6.0f, output[0]);
        passed += (output[0] == 6.0f);
        total++;
    }

    // Test 2: Complex expression
    {
        std::vector<float, parallax::allocator<float>> input(N, 3.0f);
        std::vector<float, parallax::allocator<float>> output(N);

        std::transform(std::execution::par,
                      input.begin(), input.end(), output.begin(),
                      [](float x) { return x * 2.0f + 1.0f; });

        print_test("Complex (x*2+1)", 7.0f, output[0]);
        passed += (output[0] == 7.0f);
        total++;
    }

    // Test 3: Division
    {
        std::vector<float, parallax::allocator<float>> input(N, 10.0f);
        std::vector<float, parallax::allocator<float>> output(N);

        std::transform(std::execution::par,
                      input.begin(), input.end(), output.begin(),
                      [](float x) { return x / 2.0f; });

        print_test("Divide (x/2)", 5.0f, output[0]);
        passed += (output[0] == 5.0f);
        total++;
    }

    // Test 4: Subtraction
    {
        std::vector<float, parallax::allocator<float>> input(N, 10.0f);
        std::vector<float, parallax::allocator<float>> output(N);

        std::transform(std::execution::par,
                      input.begin(), input.end(), output.begin(),
                      [](float x) { return x - 3.0f; });

        print_test("Subtract (x-3)", 7.0f, output[0]);
        passed += (output[0] == 7.0f);
        total++;
    }

    std::cout << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;

    return (passed == total) ? 0 : 1;
}
