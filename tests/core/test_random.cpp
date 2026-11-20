
#include "latent/core/Random.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using latent::core::RandomGenerator;
using latent::core::Tensor;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Function>
void run_test(const char* name, Function function)
{
    function();
    std::cout << "[PASS] " << name << '\n';
}

void test_uniform_range()
{
    RandomGenerator rng(42);

    for (int i = 0; i < 10000; ++i) {
        const double value = rng.uniform();

        require(std::isfinite(value),
                "Non-finite uniform value");
        require(value >= 0.0 && value < 1.0,
                "Uniform value out of range");
    }
}

void test_uniform_repeatability()
{
    RandomGenerator first(12345);
    RandomGenerator second(12345);

    for (int i = 0; i < 1000; ++i) {
        require(first.uniform() == second.uniform(),
                "Uniform sequences differ");
    }
}

void test_gaussian_repeatability()
{
    RandomGenerator first(54321);
    RandomGenerator second(54321);

    for (int i = 0; i < 1000; ++i) {
        require(first.gaussian() == second.gaussian(),
                "Gaussian sequences differ");
    }
}

void test_different_seeds()
{
    RandomGenerator first(100);
    RandomGenerator second(200);

    bool found_difference = false;

    for (int i = 0; i < 100; ++i) {
        if (first.uniform() != second.uniform()) {
            found_difference = true;
            break;
        }
    }

    require(found_difference,
            "Different seeds produced identical samples");
}

void test_gaussian_finite()
{
    RandomGenerator rng(2026);

    for (int i = 0; i < 10000; ++i) {
        require(std::isfinite(rng.gaussian()),
                "Non-finite Gaussian value");
    }
}

void test_gaussian_statistics()
{
    RandomGenerator rng(9876);

    constexpr int count = 100000;

    double sum = 0.0;
    double sum_squares = 0.0;

    for (int i = 0; i < count; ++i) {
        const double value = rng.gaussian();

        sum += value;
        sum_squares += value * value;
    }

    const double mean = sum / count;
    const double variance =
        sum_squares / count - mean * mean;

    // Broad sanity bounds, not exact-value assertions.
    require(std::abs(mean) < 0.03,
            "Gaussian mean outside tolerance");

    require(std::abs(variance - 1.0) < 0.05,
            "Gaussian variance outside tolerance");
}

void test_gaussian_tensor()
{
    RandomGenerator rng(42);

    Tensor result = rng.gaussian_tensor({2, 3, 4});

    require(result.rank() == 3,
            "Incorrect Tensor rank");

    require(result.size() == 24,
            "Incorrect Tensor size");

    require(result.all_finite(),
            "Tensor contains non-finite values");
}

void test_tensor_repeatability()
{
    RandomGenerator first(77);
    RandomGenerator second(77);

    Tensor a = first.gaussian_tensor({4, 8});
    Tensor b = second.gaussian_tensor({4, 8});

    require(a.same_shape(b), "Tensor shapes differ");

    for (std::size_t i = 0; i < a.size(); ++i) {
        require(a[i] == b[i],
                "Gaussian Tensors differ");
    }
}

void test_reseed_clears_gaussian_cache()
{
    RandomGenerator rng(100);

    const double expected = rng.gaussian();

    // Cache currently contains a second sample.
    rng.reseed(100);

    require(rng.gaussian() == expected,
            "Reseeding failed to reset Gaussian state");
}

void test_invalid_tensor_shape()
{
    RandomGenerator rng(42);

    bool rejected = false;

    try {
        (void)rng.gaussian_tensor({2, 0, 3});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    require(rejected, "Invalid Tensor shape accepted");
}

} // namespace

int main()
{
    try {
        run_test("Uniform range", test_uniform_range);
        run_test("Uniform repeatability",
                 test_uniform_repeatability);
        run_test("Gaussian repeatability",
                 test_gaussian_repeatability);
        run_test("Different seeds", test_different_seeds);
        run_test("Gaussian finite values",
                 test_gaussian_finite);
        run_test("Gaussian distribution statistics",
                 test_gaussian_statistics);
        run_test("Gaussian Tensor generation",
                 test_gaussian_tensor);
        run_test("Gaussian Tensor repeatability",
                 test_tensor_repeatability);
        run_test("Reseeding clears cached state",
                 test_reseed_clears_gaussian_cache);
        run_test("Invalid Tensor dimensions",
                 test_invalid_tensor_shape);

        std::cout << "[PASS] P0.5 RandomGenerator\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "[FAIL] P0.5 RandomGenerator: "
            << error.what() << '\n';
        return 1;
    }
}
